
import re
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy import signal, sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import pdist, squareform

try:
    import networkx as nx
except Exception:
    nx = None


st.set_page_config(page_title="VRC / HRV RRi Analyzer Pro v5.1 Kubios Mode", layout="wide")

PHASES = ["Basal"] + [f"E{i}" for i in range(1, 7)] + [f"R{i}" for i in range(1, 4)]
PHASE_GROUP = {
    "Basal": "Basal",
    **{f"E{i}": "Ejercicio" for i in range(1, 7)},
    **{f"R{i}": "Recuperación" for i in range(1, 4)},
}
PHASE_COLORS = {
    "Basal": "rgba(0,150,255,0.24)",
    "Ejercicio": "rgba(255,140,0,0.20)",
    "Recuperación": "rgba(0,200,100,0.20)",
}
PHASE_LINE_COLORS = {
    "Basal": "#0096ff",
    "Ejercicio": "#ff8c00",
    "Recuperación": "#00c864",
}

FS_INTERP = 4.0
LAMBDA_DEFAULT = 500

PARAM_GROUPS = {
    "Tiempo": ["MeanHR", "MeanRR", "SDNN", "RMSSD", "pNN50", "SD1", "SD2"],
    "Frecuencia": ["VLF", "LF", "HF", "TOTAL", "LF_HF"],
    "Complejidad": ["DFA_alpha1", "DFA_alpha2", "ApEn", "SampEn"],
    "Recurrencia": ["REC", "DET", "Lmean", "Lmax", "ShanEn"],
}
DEFAULT_MULTI = ["RMSSD", "SDNN", "SD1", "SD2", "LF", "HF"]


def sanitize_name(name):
    name = Path(str(name)).stem
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_")
    return name or "registro"


def read_rri_file(uploaded_file):
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")
    vals = []
    for line in text.replace(";", "\n").replace("\t", "\n").splitlines():
        line = line.strip().replace(",", ".")
        if not line:
            continue
        for p in line.split():
            try:
                vals.append(float(p))
            except Exception:
                pass

    rr = np.asarray(vals, dtype=float)
    rr = rr[np.isfinite(rr)]

    if len(rr) == 0:
        raise ValueError("No se han detectado RRi numéricos.")

    if np.nanmedian(rr) > 10:
        rr = rr / 1000.0

    rr = rr[(rr >= 0.3) & (rr <= 2.0)]

    if len(rr) == 0:
        raise ValueError("Tras el filtrado fisiológico no quedan RRi válidos.")

    return rr


def correct_artifacts_kubios_like(rr, level="none", window=5):
    rr = np.asarray(rr, dtype=float)
    rr_corr = rr.copy()
    n = len(rr)

    if level == "none" or n < 10:
        return rr_corr, np.zeros(n, dtype=bool), {
            "level": level,
            "n_artifacts": 0,
            "percent_artifacts": 0.0,
        }

    thresholds = {
        "very low": 0.45,
        "low": 0.35,
        "medium": 0.25,
        "strong": 0.15,
        "very strong": 0.05,
    }
    th = thresholds.get(level, 0.25)

    local = pd.Series(rr).rolling(window=window, center=True, min_periods=1).median().to_numpy()
    artifacts = np.abs(rr - local) > th

    if np.mean(artifacts) > 0.30:
        artifacts[:] = False

    idx = np.arange(n)
    good = ~artifacts

    if np.sum(good) >= 2 and np.sum(artifacts) > 0:
        rr_corr[artifacts] = np.interp(idx[artifacts], idx[good], rr[good])

    return rr_corr, artifacts, {
        "level": level,
        "n_artifacts": int(np.sum(artifacts)),
        "percent_artifacts": float(100 * np.mean(artifacts)),
    }


def cumulative_time(rr):
    return np.cumsum(rr)


def sec_to_hms(seconds):
    seconds = int(round(float(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def hms_to_sec(s):
    parts = [float(p) for p in str(s).strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def cut_segment(rr, start_s, end_s):
    t = cumulative_time(rr)
    return rr[(t >= start_s) & (t <= end_s)]


def empty_windows():
    return {ph: None for ph in PHASES}


def default_windows(t_max):
    t_max = float(max(t_max, 1.0))
    if t_max < 600:
        step = max(t_max / 10, 20)
        return {ph: [min(i * step, t_max), min((i + 1) * step, t_max)] for i, ph in enumerate(PHASES)}

    basal = [0.0, min(300.0, t_max)]
    rem_start = basal[1]
    rem = max(0.0, t_max - rem_start)
    step = rem / 9.0 if rem > 0 else 60.0
    w = {"Basal": basal}

    for i in range(1, 7):
        w[f"E{i}"] = [min(rem_start + (i - 1) * step, t_max), min(rem_start + i * step, t_max)]

    for i in range(1, 4):
        j = 6 + i
        w[f"R{i}"] = [min(rem_start + (j - 1) * step, t_max), min(rem_start + j * step, t_max)]

    return w


def smoothness_priors_detrend(y, lam=500):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return y - np.mean(y) if n else y

    I = sparse.eye(n, format="csc")
    e = np.ones(n)
    D2 = sparse.diags([e[:-2], -2 * e[:-2], e[:-2]], [0, 1, 2], shape=(n - 2, n), format="csc")
    trend = spsolve(I + (lam ** 2) * (D2.T @ D2), y)
    return y - trend


def interpolate_rr(rr, fs=FS_INTERP, apply_lambda=False, lam=500):
    t = cumulative_time(rr)
    if len(t) < 5:
        return np.array([]), np.array([])

    t = t - t[0]
    x = rr.copy()
    keep = np.r_[True, np.diff(t) > 0]
    t, x = t[keep], x[keep]

    if len(t) < 5:
        return np.array([]), np.array([])

    ti = np.arange(0, t[-1], 1 / fs)

    if len(ti) < 5:
        return np.array([]), np.array([])

    xi = CubicSpline(t, x, bc_type="natural")(ti)

    if apply_lambda:
        xi = smoothness_priors_detrend(xi, lam)

    return ti, xi


def time_metrics(rr):
    rr_ms = rr * 1000.0
    diff = np.diff(rr_ms)
    mean_rr = np.mean(rr_ms)
    sdnn = np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else np.nan
    rmssd = np.sqrt(np.mean(diff ** 2)) if len(diff) else np.nan
    nn50 = int(np.sum(np.abs(diff) > 50)) if len(diff) else 0
    pnn50 = 100 * nn50 / len(diff) if len(diff) else np.nan
    sd1 = np.sqrt(0.5) * np.std(diff, ddof=1) if len(diff) > 1 else np.nan
    sd2 = np.sqrt(max(0, 2 * sdnn ** 2 - sd1 ** 2)) if np.isfinite(sdnn) and np.isfinite(sd1) else np.nan

    return {
        "N_RRi": len(rr),
        "Duration_s": float(np.sum(rr)),
        "MeanRR": mean_rr,
        "MeanHR": 60000 / mean_rr if mean_rr > 0 else np.nan,
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "NN50": nn50,
        "pNN50": pnn50,
        "SD1": sd1,
        "SD2": sd2,
    }


def psd_metrics(rr):
    ti, xi = interpolate_rr(rr, fs=FS_INTERP, apply_lambda=True, lam=LAMBDA_DEFAULT)

    if len(xi) < 32:
        return {"VLF": np.nan, "LF": np.nan, "HF": np.nan, "TOTAL": np.nan, "LF_HF": np.nan}

    xi_ms = xi * 1000
    xi_ms = xi_ms - np.mean(xi_ms)
    nperseg = min(int(256 * FS_INTERP), len(xi_ms))
    noverlap = int(0.5 * nperseg)

    f, pxx = signal.welch(
        xi_ms,
        fs=FS_INTERP,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
    )

    def bp(lo, hi):
        mask = (f >= lo) & (f < hi)
        return np.trapezoid(pxx[mask], f[mask]) if np.any(mask) else np.nan

    vlf, lf, hf = bp(0.0033, 0.04), bp(0.04, 0.15), bp(0.15, 0.40)
    total = np.nansum([vlf, lf, hf])

    return {"VLF": vlf, "LF": lf, "HF": hf, "TOTAL": total, "LF_HF": lf / hf if pd.notna(hf) and hf > 0 else np.nan}


def _phi_apen(x, m, r):
    n = len(x)

    if n <= m + 1:
        return np.nan

    pats = np.array([x[i:i + m] for i in range(n - m + 1)])
    vals = []

    for p in pats:
        dist = np.max(np.abs(pats - p), axis=1)
        c = np.mean(dist <= r)
        if c > 0:
            vals.append(np.log(c))

    return np.mean(vals) if vals else np.nan


def apen_calc(x, m=2, r_ratio=0.2):
    x = smoothness_priors_detrend(np.asarray(x, dtype=float), LAMBDA_DEFAULT)
    r = r_ratio * np.std(x, ddof=1)

    if not np.isfinite(r) or r == 0:
        return np.nan

    return _phi_apen(x, m, r) - _phi_apen(x, m + 1, r)


def sampen_calc(x, m=2, r_ratio=0.2):
    x = smoothness_priors_detrend(np.asarray(x, dtype=float), LAMBDA_DEFAULT)
    n = len(x)

    if n <= m + 2:
        return np.nan

    r = r_ratio * np.std(x, ddof=1)

    if not np.isfinite(r) or r == 0:
        return np.nan

    def count(mm):
        pats = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        c = 0
        for i in range(len(pats) - 1):
            dist = np.max(np.abs(pats[i + 1:] - pats[i]), axis=1)
            c += np.sum(dist <= r)
        return c

    b, a = count(m), count(m + 1)

    if a == 0 or b == 0:
        return np.nan

    return -np.log(a / b)


def dfa_calc(x):
    x = np.asarray(x, dtype=float)
    n = len(x)

    if n < 50:
        return np.nan, np.nan

    y = np.cumsum(x - np.mean(x))
    scales = np.unique(np.floor(np.logspace(np.log10(4), np.log10(max(5, n // 4)), 18)).astype(int))

    ss, ff = [], []

    for s in scales:
        if s < 4 or n // s < 2:
            continue

        rms = []

        for i in range(n // s):
            seg = y[i * s:(i + 1) * s]
            t = np.arange(s)
            co = np.polyfit(t, seg, 1)
            rms.append(np.sqrt(np.mean((seg - np.polyval(co, t)) ** 2)))

        val = np.sqrt(np.mean(np.asarray(rms) ** 2))

        if val > 0:
            ss.append(s)
            ff.append(val)

    ss, ff = np.asarray(ss), np.asarray(ff)

    if len(ss) < 4:
        return np.nan, np.nan

    m1, m2 = (ss >= 4) & (ss <= 16), ss > 16

    return (
        np.polyfit(np.log(ss[m1]), np.log(ff[m1]), 1)[0] if np.sum(m1) >= 2 else np.nan,
        np.polyfit(np.log(ss[m2]), np.log(ff[m2]), 1)[0] if np.sum(m2) >= 2 else np.nan,
    )


def rqa_calc(x, emb_dim=10, tau=1, l_min=2, max_n=500):
    x = np.asarray(x, dtype=float)

    if len(x) > max_n:
        x = x[np.linspace(0, len(x) - 1, max_n).astype(int)]

    n = len(x) - (emb_dim - 1) * tau

    if n < 20:
        return {"REC": np.nan, "DET": np.nan, "Lmean": np.nan, "Lmax": np.nan, "ShanEn": np.nan}

    D = squareform(pdist(np.array([x[i:i + emb_dim * tau:tau] for i in range(n)])))
    radius = np.sqrt(emb_dim) * np.std(x, ddof=1)
    R = (D <= radius).astype(int)
    np.fill_diagonal(R, 0)
    rec = 100 * R.sum() / (n * n - n)

    lens = []

    for k in range(-n + 1, n):
        diag = np.diag(R, k=k)
        c = 0

        for val in diag:
            if val:
                c += 1
            else:
                if c >= l_min:
                    lens.append(c)
                c = 0

        if c >= l_min:
            lens.append(c)

    if not lens:
        return {"REC": rec, "DET": 0, "Lmean": 0, "Lmax": 0, "ShanEn": 0}

    lens = np.asarray(lens)
    det = 100 * lens.sum() / R.sum() if R.sum() > 0 else 0
    vals, counts = np.unique(lens, return_counts=True)
    p = counts / counts.sum()

    return {"REC": rec, "DET": det, "Lmean": np.mean(lens), "Lmax": np.max(lens), "ShanEn": -np.sum(p * np.log(p))}




def hvg_graph(x, max_nodes=500):
    if nx is None:
        return None

    x = np.asarray(x, dtype=float)
    if len(x) > max_nodes:
        idx = np.linspace(0, len(x) - 1, max_nodes).astype(int)
        x = x[idx]

    n = len(x)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n - 1):
        G.add_edge(i, i + 1)
        for j in range(i + 2, n):
            if np.max(x[i + 1:j]) < min(x[i], x[j]):
                G.add_edge(i, j)

    return G


def hvg_metrics(rr, max_nodes=500):
    if nx is None:
        return {
            "HVG_nodes": np.nan,
            "HVG_edges": np.nan,
            "HVG_degree_mean": np.nan,
            "HVG_degree_max": np.nan,
            "HVG_hubs_p90": np.nan,
            "HVG_clustering": np.nan,
            "HVG_lambda": np.nan,
            "HVG_path_length": np.nan,
            "HVG_diameter": np.nan,
        }

    G = hvg_graph(rr, max_nodes=max_nodes)
    if G is None or G.number_of_nodes() < 20:
        return {
            "HVG_nodes": G.number_of_nodes() if G is not None else 0,
            "HVG_edges": np.nan,
            "HVG_degree_mean": np.nan,
            "HVG_degree_max": np.nan,
            "HVG_hubs_p90": np.nan,
            "HVG_clustering": np.nan,
            "HVG_lambda": np.nan,
            "HVG_path_length": np.nan,
            "HVG_diameter": np.nan,
        }

    n = G.number_of_nodes()
    m = G.number_of_edges()
    deg = np.array([d for _, d in G.degree()])

    vals, counts = np.unique(deg, return_counts=True)
    p = counts / counts.sum()
    mask = (vals > 1) & (p > 0)
    lam = -np.polyfit(vals[mask], np.log(p[mask]), 1)[0] if np.sum(mask) >= 2 else np.nan

    if nx.is_connected(G):
        path_length = nx.average_shortest_path_length(G)
        diameter = nx.diameter(G)
    else:
        path_length = np.nan
        diameter = np.nan

    return {
        "HVG_nodes": n,
        "HVG_edges": m,
        "HVG_degree_mean": 2 * m / n if n else np.nan,
        "HVG_degree_max": np.max(deg) if len(deg) else np.nan,
        "HVG_hubs_p90": int(np.sum(deg >= np.percentile(deg, 90))) if len(deg) else np.nan,
        "HVG_clustering": nx.average_clustering(G) if n else np.nan,
        "HVG_lambda": lam,
        "HVG_path_length": path_length,
        "HVG_diameter": diameter,
    }


def hvg_network_figure(rr, title="HVG", max_nodes=140):
    fig = go.Figure()
    if nx is None:
        fig.update_layout(title="NetworkX no disponible")
        return fig

    G = hvg_graph(rr, max_nodes=max_nodes)
    if G is None or G.number_of_nodes() == 0:
        fig.update_layout(title="Sin grafo")
        return fig

    pos = nx.spring_layout(G, seed=42, k=0.18, iterations=60)

    edge_x, edge_y = [], []
    for a, b in G.edges():
        edge_x += [pos[a][0], pos[b][0], None]
        edge_y += [pos[a][1], pos[b][1], None]

    deg = dict(G.degree())
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_size = [6 + deg[n] * 2.5 for n in G.nodes()]
    node_text = [f"n={n}<br>grado={deg[n]}" for n in G.nodes()]

    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.5), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers", marker=dict(size=node_size), text=node_text, hoverinfo="text", showlegend=False))
    fig.update_layout(title=title, height=520, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def poincare_figure(record_data, global_windows, record_windows, phase, use_independent):
    fig = go.Figure()

    for rec, data in record_data.items():
        windows = get_record_windows(global_windows, record_windows, rec, use_independent)
        w = windows.get(phase)
        if w is None:
            continue

        seg = cut_segment(data["rr"], w[0], w[1])
        if len(seg) < 3:
            continue

        rr_ms = seg * 1000
        x = rr_ms[:-1]
        y = rr_ms[1:]
        diff = np.diff(rr_ms)
        sdnn = np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else np.nan
        sd1 = np.sqrt(0.5) * np.std(diff, ddof=1) if len(diff) > 1 else np.nan
        sd2 = np.sqrt(max(0, 2 * sdnn ** 2 - sd1 ** 2)) if np.isfinite(sdnn) and np.isfinite(sd1) else np.nan

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name=f"{rec} · SD1={sd1:.1f}, SD2={sd2:.1f}",
            marker=dict(size=6, opacity=0.65)
        ))

    fig.update_layout(
        title=f"Poincaré comparativo · {phase}",
        height=560,
        xaxis_title="RR(n) ms",
        yaxis_title="RR(n+1) ms",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig



def calculate_all(rr, include_rqa=True, include_hvg=False):
    rr_ms = rr * 1000
    out = {}
    out.update(time_metrics(rr))
    out.update(psd_metrics(rr))
    a1, a2 = dfa_calc(rr_ms)
    out["DFA_alpha1"], out["DFA_alpha2"] = a1, a2
    out["ApEn"] = apen_calc(rr_ms)
    out["SampEn"] = sampen_calc(rr_ms)

    if include_rqa:
        out.update(rqa_calc(rr_ms))

    if include_hvg:
        out.update(hvg_metrics(rr))

    return out


def get_record_windows(global_windows, record_windows, rec, use_independent):
    if use_independent:
        return record_windows.get(rec, global_windows)
    return global_windows


def calculate_record(rr, windows, active_phases, min_rr, include_rqa, include_hvg=False):
    rows, segments, valid = [], {}, {}

    for ph in PHASES:
        w = windows.get(ph)
        if w is None:
            segments[ph] = np.array([])
            valid[ph] = False
            continue

        s, e = w
        seg = cut_segment(rr, s, e)
        segments[ph] = seg
        valid[ph] = len(seg) >= min_rr and ph in active_phases

        if valid[ph]:
            res = calculate_all(seg, include_rqa=include_rqa, include_hvg=include_hvg)
            res["Fase"] = ph
            rows.append(res)

    return (pd.DataFrame(rows).set_index("Fase") if rows else pd.DataFrame()), segments, valid


def build_long(records_results):
    rows = []

    for rec, df in records_results.items():
        if df is None or df.empty:
            continue

        tmp = df.copy()
        tmp.insert(0, "Registro", rec)
        tmp.insert(1, "Fase", tmp.index)
        rows.append(tmp.reset_index(drop=True))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_windows_to_fig(fig, windows):
    for ph, w in windows.items():
        if w is None:
            continue

        s, e = w
        group = PHASE_GROUP.get(ph, ph)
        fig.add_vrect(
            x0=s / 60,
            x1=e / 60,
            fillcolor=PHASE_COLORS.get(group, "rgba(180,180,180,.15)"),
            line_width=0,
            annotation_text=ph,
            annotation_position="top left",
        )


def rr_plot(record_data, global_windows, record_windows, view_mode, selected_record, use_independent):
    fig = go.Figure()
    names = [selected_record] if view_mode == "Registro principal" else list(record_data.keys())

    for name in names:
        rr = record_data[name]["rr"]
        t = cumulative_time(rr) / 60

        if np.any(record_data[name].get("artifact_mask", np.array([]))):
            rr_raw = record_data[name]["rr_raw"]
            t_raw = cumulative_time(rr_raw) / 60
            mask = record_data[name]["artifact_mask"]

            fig.add_trace(go.Scatter(x=t_raw, y=rr_raw * 1000, mode="lines", name=f"{name} original", opacity=0.25))
            fig.add_trace(go.Scatter(x=t, y=rr * 1000, mode="lines", name=f"{name} corregido"))

            if len(mask) == len(rr_raw):
                fig.add_trace(go.Scatter(x=t_raw[mask], y=rr_raw[mask] * 1000, mode="markers", name=f"{name} artefactos", marker=dict(symbol="x", size=9)))
        else:
            fig.add_trace(go.Scatter(x=t, y=rr * 1000, mode="lines", name=name))

    if view_mode == "Registro principal":
        windows = get_record_windows(global_windows, record_windows, selected_record, use_independent)
        add_windows_to_fig(fig, windows)

    # Trazas invisibles de ayuda para que la selección con recuadro capture el rango X completo.
    # Plotly/Streamlit devuelve puntos seleccionados, no las coordenadas exactas del recuadro.
    # Estas líneas invisibles hacen que el rango X sea más estable aunque el recuadro no toque muchos puntos RRi.
    all_durations = [data["duration"] for data in record_data.values()]
    max_x_min = max(all_durations) / 60 if all_durations else 1
    helper_x = np.linspace(0, max_x_min, 1200)

    y_values = []
    for data in record_data.values():
        if len(data["rr"]) > 0:
            y_values.extend(list(data["rr"] * 1000))

    if y_values:
        y_min, y_max = float(np.nanmin(y_values)), float(np.nanmax(y_values))
        if y_max > y_min:
            for y0 in np.linspace(y_min, y_max, 12):
                fig.add_trace(go.Scatter(
                    x=helper_x,
                    y=np.full_like(helper_x, y0),
                    mode="markers",
                    marker=dict(size=3, opacity=0.01),
                    name="_selector_helper",
                    hoverinfo="skip",
                    showlegend=False,
                ))

    fig.update_layout(height=520, xaxis_title="Tiempo acumulado (min)", yaxis_title="RRi (ms)", hovermode="x unified", dragmode="select")
    fig.update_xaxes(rangeslider_visible=True)

    return fig


def comparison_bar_line(pivot, variable):
    fig = go.Figure()
    phases = list(pivot.index)

    # Barras con etiqueta fase + registro en el eje X para que el nombre quede debajo de cada columna.
    for rec in pivot.columns:
        x_labels = [f"{ph}<br>{rec}" for ph in phases]
        y = pivot[rec].astype(float).values
        fig.add_trace(go.Bar(
            x=x_labels,
            y=y,
            name=f"{rec} · barras",
            opacity=0.68,
            text=[rec for _ in x_labels],
            textposition="outside"
        ))
        fig.add_trace(go.Scatter(
            x=x_labels,
            y=y,
            mode="lines+markers",
            name=f"{rec} · tendencia",
            line=dict(width=3)
        ))

    fig.update_layout(
        height=540,
        title=f"{variable}: columnas por fase y registro + tendencia",
        xaxis_title="Fase · registro",
        yaxis_title=variable,
        barmode="group",
        hovermode="x unified",
        bargap=0.22,
        bargroupgap=0.08,
    )
    return fig

def dashboard_compare(long_df, phases, params):
    params = [p for p in params if p in long_df.columns]

    if len(params) == 0:
        return go.Figure()

    cols = 2
    rows = int(np.ceil(len(params) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=params)

    for idx, p in enumerate(params):
        r = idx // cols + 1
        c = idx % cols + 1
        pivot = long_df[long_df["Fase"].isin(phases)].pivot_table(index="Fase", columns="Registro", values=p, aggfunc="first").reindex(phases)

        for rec in pivot.columns:
            fig.add_trace(go.Bar(x=list(pivot.index), y=pivot[rec], name=f"{rec} · {p}", opacity=0.60, showlegend=(idx == 0)), row=r, col=c)
            fig.add_trace(go.Scatter(x=list(pivot.index), y=pivot[rec], mode="lines+markers", name=f"{rec} tendencia", showlegend=False), row=r, col=c)

    fig.update_layout(height=max(440, rows * 340), barmode="group", title="Dashboard comparativo: barras + tendencia por parámetro")

    return fig


def phase_rr_overlay(record_data, global_windows, record_windows, phase, use_independent):
    fig = go.Figure()

    for rec, data in record_data.items():
        windows = get_record_windows(global_windows, record_windows, rec, use_independent)
        w = windows.get(phase)

        if w is None:
            continue

        s, e = w
        seg = cut_segment(data["rr"], s, e)

        if len(seg) < 3:
            continue

        t = cumulative_time(seg)
        t = t - t[0]
        fig.add_trace(go.Scatter(x=t / 60, y=seg * 1000, mode="lines", name=rec))

    fig.update_layout(height=440, title=f"RRi superpuesto dentro de {phase}", xaxis_title="Tiempo dentro de fase (min)", yaxis_title="RRi (ms)")

    return fig


def windows_table(global_windows, record_windows, records, record_data, records_segments, records_valid, use_independent):
    rows = []

    for ph in PHASES:
        row = {"Fase": ph}

        if not use_independent:
            w = global_windows.get(ph)
            if w is None:
                row.update({"Inicio": "", "Fin": "", "Duración_min": np.nan})
            else:
                row.update({"Inicio": sec_to_hms(w[0]), "Fin": sec_to_hms(w[1]), "Duración_min": round((w[1] - w[0]) / 60, 2)})

        for rec in records:
            w = get_record_windows(global_windows, record_windows, rec, use_independent).get(ph)
            if use_independent:
                row[f"{rec}_inicio"] = sec_to_hms(w[0]) if w else ""
                row[f"{rec}_fin"] = sec_to_hms(w[1]) if w else ""

            row[f"{rec}_N"] = len(records_segments[rec][ph])
            row[f"{rec}_OK"] = records_valid[rec][ph]

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# APP
# ============================================================

st.title("VRC / HRV RRi Analyzer Pro v5.1 · Kubios Mode")
st.caption("Segmentación tipo Kubios, selección con ratón mejorada, Poincaré, HVG/grafos y comparación con etiquetas por registro.")

with st.sidebar:
    uploaded_files = st.file_uploader("Sube uno o varios CSV/TXT con RRi", type=["csv", "txt"], accept_multiple_files=True)
    min_rr = st.number_input("Mínimo RRi por ventana", min_value=10, max_value=300, value=30, step=5)
    include_rqa = st.checkbox("Calcular RQA", value=False, help="Puede tardar en ventanas largas.")
    include_hvg = st.checkbox("Calcular HVG/grafos", value=False, help="Más lento. Actívalo cuando ya tengas las ventanas definidas.")
    artifact_level = st.selectbox(
        "Corrección de artefactos",
        ["none", "very low", "low", "medium", "strong", "very strong"],
        index=0,
        help="Aproximada tipo Kubios: mediana local + interpolación lineal.",
    )
    st.caption("Consejo: para ventanas de ~30 s usa mínimo RRi 20-30; para 5 min usa 30-110 según el caso.")

if not uploaded_files:
    st.info("Sube uno o varios registros RRi.")
    st.stop()

record_data = {}
errors = []

for uf in uploaded_files:
    try:
        rr_raw = read_rri_file(uf)
        rr, artifact_mask, artifact_info = correct_artifacts_kubios_like(rr_raw, level=artifact_level)
        name = sanitize_name(uf.name)
        base, k = name, 2

        while name in record_data:
            name = f"{base}_{k}"
            k += 1

        record_data[name] = {
            "rr": rr,
            "rr_raw": rr_raw,
            "artifact_mask": artifact_mask,
            "artifact_info": artifact_info,
            "duration": float(np.sum(rr)),
            "filename": uf.name,
        }
    except Exception as e:
        errors.append(f"{uf.name}: {e}")

if errors:
    st.error("\n".join(errors))

if not record_data:
    st.stop()

records = list(record_data.keys())
selected_record = st.sidebar.selectbox("Registro principal", records)
t_max = record_data[selected_record]["duration"]

if "selected_record_v50" not in st.session_state or st.session_state.selected_record_v50 != selected_record:
    st.session_state.selected_record_v50 = selected_record

if "global_windows_v50" not in st.session_state:
    st.session_state.global_windows_v50 = empty_windows()

if "record_windows_v50" not in st.session_state:
    st.session_state.record_windows_v50 = {rec: empty_windows() for rec in records}

for rec in records:
    st.session_state.record_windows_v50.setdefault(rec, empty_windows())

if "pending_selection_v50" not in st.session_state:
    st.session_state.pending_selection_v50 = None

if "active_phases_v50" not in st.session_state:
    st.session_state.active_phases_v50 = ["Basal"]

with st.sidebar.expander("Segmentación", expanded=True):
    use_independent = st.checkbox("Ventanas independientes por registro", value=False)
    active_phases = st.multiselect("Fases activas para calcular", PHASES, default=st.session_state.active_phases_v50)
    st.session_state.active_phases_v50 = active_phases

    if st.button("Limpiar todas las ventanas"):
        st.session_state.global_windows_v50 = empty_windows()
        st.session_state.record_windows_v50 = {rec: empty_windows() for rec in records}
        st.session_state.pending_selection_v50 = None
        st.rerun()

    if st.button("Autodividir todo el registro"):
        if use_independent:
            st.session_state.record_windows_v50[selected_record] = default_windows(t_max)
        else:
            st.session_state.global_windows_v50 = default_windows(t_max)
        st.session_state.active_phases_v50 = PHASES.copy()
        st.rerun()

    if use_independent and st.button("Copiar ventanas del registro principal a todos"):
        base_w = st.session_state.record_windows_v50.get(selected_record, empty_windows())
        st.session_state.record_windows_v50 = {rec: {ph: (list(base_w[ph]) if base_w[ph] is not None else None) for ph in PHASES} for rec in records}
        st.rerun()

if artifact_level != "none":
    with st.sidebar.expander("Resumen artefactos", expanded=True):
        for rec, data in record_data.items():
            info = data.get("artifact_info", {})
            st.write(f"**{rec}**: {info.get('n_artifacts', 0)} ({info.get('percent_artifacts', 0):.2f}%)")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["1) Segmentar tipo Kubios", "2) HRV", "3) Comparar", "4) Poincaré / Grafos", "5) Dashboard", "6) Exportar"])

# central calculation
records_results, records_segments, records_valid = {}, {}, {}

for rec, data in record_data.items():
    w = get_record_windows(st.session_state.global_windows_v50, st.session_state.record_windows_v50, rec, use_independent)
    df, segs, valid = calculate_record(data["rr"], w, active_phases, min_rr, include_rqa, include_hvg=include_hvg)
    records_results[rec], records_segments[rec], records_valid[rec] = df, segs, valid

metrics_df = records_results[selected_record]
long_df = build_long(records_results)

with tab1:
    st.subheader("Segmentación tipo Kubios")
    st.write(
        "1) Encuadra una región con el ratón. "
        "2) Pulsa **Guardar selección**. "
        "3) Pulsa **Asignar a Basal/E1/E2...**. "
        "Sólo se calcularán las fases activas."
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        view_mode = st.radio("Vista", ["Registro principal", "Todos superpuestos"], index=1)
    with c2:
        st.info("Para comparar dos registros del mismo paciente, usa 'Todos superpuestos' y asigna las ventanas que quieras comparar.")

    fig = rr_plot(
        record_data,
        st.session_state.global_windows_v50,
        st.session_state.record_windows_v50,
        view_mode,
        selected_record,
        use_independent,
    )

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode=("box", "lasso"),
        key="rr_select_v50",
    )

    if event and getattr(event, "selection", None):
        pts = event.selection.get("points", [])
        xs = [p.get("x") for p in pts if "x" in p]

        if xs:
            s_sel, e_sel = min(xs) * 60, max(xs) * 60
            st.success(f"Selección detectada: {sec_to_hms(s_sel)} - {sec_to_hms(e_sel)}")

            if st.button("Guardar selección"):
                st.session_state.pending_selection_v50 = [s_sel, e_sel]
                st.rerun()

    if st.session_state.pending_selection_v50 is not None:
        s_sel, e_sel = st.session_state.pending_selection_v50
        st.success(f"Selección guardada: {sec_to_hms(s_sel)} - {sec_to_hms(e_sel)}")

        st.markdown("### Asignar selección guardada a fase")
        phase_cols = st.columns(10)

        for idx, ph in enumerate(PHASES):
            with phase_cols[idx % 10]:
                if st.button(ph, key=f"assign_{ph}_v50"):
                    if use_independent:
                        st.session_state.record_windows_v50[selected_record][ph] = [s_sel, e_sel]
                    else:
                        st.session_state.global_windows_v50[ph] = [s_sel, e_sel]

                    if ph not in st.session_state.active_phases_v50:
                        st.session_state.active_phases_v50.append(ph)

                    st.session_state.pending_selection_v50 = None
                    st.rerun()

        if st.button("Borrar selección guardada"):
            st.session_state.pending_selection_v50 = None
            st.rerun()

    st.markdown("### Ventanas definidas")
    win_df = windows_table(
        st.session_state.global_windows_v50,
        st.session_state.record_windows_v50,
        records,
        record_data,
        records_segments,
        records_valid,
        use_independent,
    )
    st.dataframe(win_df, use_container_width=True)

    st.markdown("### Edición manual opcional")
    manual_phase = st.selectbox("Fase a editar manualmente", PHASES)
    current_w = get_record_windows(st.session_state.global_windows_v50, st.session_state.record_windows_v50, selected_record, use_independent).get(manual_phase)

    if current_w is None:
        ini_default, fin_default = "00:00:00", "00:05:00"
    else:
        ini_default, fin_default = sec_to_hms(current_w[0]), sec_to_hms(current_w[1])

    c_ini, c_fin, c_apply, c_clear = st.columns([1, 1, 1, 1])
    with c_ini:
        ini_txt = st.text_input("Inicio", ini_default)
    with c_fin:
        fin_txt = st.text_input("Fin", fin_default)
    with c_apply:
        st.write("")
        st.write("")
        if st.button("Aplicar manual"):
            try:
                s, e = hms_to_sec(ini_txt), hms_to_sec(fin_txt)
                if e <= s:
                    st.warning("El final debe ser mayor que el inicio.")
                else:
                    if use_independent:
                        st.session_state.record_windows_v50[selected_record][manual_phase] = [s, e]
                    else:
                        st.session_state.global_windows_v50[manual_phase] = [s, e]
                    if manual_phase not in st.session_state.active_phases_v50:
                        st.session_state.active_phases_v50.append(manual_phase)
                    st.rerun()
            except Exception:
                st.warning("Formato no válido. Usa HH:MM:SS.")
    with c_clear:
        st.write("")
        st.write("")
        if st.button("Borrar fase"):
            if use_independent:
                st.session_state.record_windows_v50[selected_record][manual_phase] = None
            else:
                st.session_state.global_windows_v50[manual_phase] = None
            if manual_phase in st.session_state.active_phases_v50:
                st.session_state.active_phases_v50.remove(manual_phase)
            st.rerun()

with tab2:
    st.subheader(f"HRV: {selected_record}")

    if metrics_df.empty:
        st.info("No hay ventanas válidas para el registro principal. Define ventanas, activa fases o baja el mínimo RRi.")
    else:
        for group, cols in PARAM_GROUPS.items():
            present = [c for c in cols if c in metrics_df.columns]
            if present:
                st.markdown(f"### {group}")
                st.dataframe(metrics_df[present], use_container_width=True)

with tab3:
    st.subheader("Comparar registros")

    if len(records) < 2:
        st.info("Sube dos o más registros.")
    elif long_df.empty:
        st.info("No hay datos comparables. Define ventanas, activa fases o baja el mínimo RRi.")
    else:
        valid_summary = pd.DataFrame(records_valid).T.reindex(columns=PHASES)
        st.markdown("### Ventanas válidas")
        st.dataframe(valid_summary, use_container_width=True)

        available_phases = [p for p in PHASES if p in long_df["Fase"].unique()]
        selected_phases = st.multiselect("Fases a comparar", PHASES, default=available_phases)
        numeric_vars = [c for c in long_df.columns if c not in ["Registro", "Fase"] and pd.api.types.is_numeric_dtype(long_df[c])]

        default_var = "RMSSD" if "RMSSD" in numeric_vars else numeric_vars[0]
        variable = st.selectbox("Variable principal", numeric_vars, index=numeric_vars.index(default_var))
        df_sel = long_df[long_df["Fase"].isin(selected_phases)] if selected_phases else long_df
        pivot = df_sel.pivot_table(index="Fase", columns="Registro", values=variable, aggfunc="first").reindex(selected_phases)

        st.markdown(f"### {variable}: barras agrupadas + línea de tendencia")
        st.dataframe(pivot, use_container_width=True)
        st.plotly_chart(comparison_bar_line(pivot, variable), use_container_width=True, key=f"compare_main_{variable}_{len(selected_phases)}")

        st.markdown("### Panel de varios parámetros")
        param_defaults = [p for p in DEFAULT_MULTI if p in numeric_vars]
        params = st.multiselect("Parámetros", numeric_vars, default=param_defaults)
        if params:
            st.plotly_chart(dashboard_compare(long_df, selected_phases or available_phases, params), use_container_width=True, key="compare_dashboard_params")

        ph_overlay = st.selectbox("RRi superpuesto por fase", selected_phases or available_phases)
        st.plotly_chart(
            phase_rr_overlay(record_data, st.session_state.global_windows_v50, st.session_state.record_windows_v50, ph_overlay, use_independent),
            use_container_width=True,
            key=f"phase_overlay_{ph_overlay}",
        )

        st.markdown("### Tabla completa filtrada")
        st.dataframe(df_sel, use_container_width=True)


with tab4:
    st.subheader("Poincaré y grafos comparativos")

    if len(records) < 1:
        st.info("Sube al menos un registro.")
    else:
        available_phases_pg = [p for p in PHASES if p in active_phases]
        if not available_phases_pg:
            available_phases_pg = [p for p in PHASES if any(records_valid[rec].get(p, False) for rec in records)]

        if not available_phases_pg:
            st.info("No hay fases válidas. Define ventanas y activa fases.")
        else:
            phase_pg = st.selectbox("Fase para Poincaré / grafo", available_phases_pg, key="phase_pg_v51")

            st.markdown("### Poincaré comparativo")
            st.plotly_chart(
                poincare_figure(
                    record_data,
                    st.session_state.global_windows_v50,
                    st.session_state.record_windows_v50,
                    phase_pg,
                    use_independent,
                ),
                use_container_width=True,
                key=f"poincare_{phase_pg}"
            )

            st.markdown("### Métricas HVG / grafos")
            if not include_hvg:
                st.warning("Activa 'Calcular HVG/grafos' en la barra lateral para calcular las métricas de grafos.")
            else:
                hvg_cols = [
                    "HVG_nodes", "HVG_edges", "HVG_degree_mean", "HVG_degree_max",
                    "HVG_hubs_p90", "HVG_clustering", "HVG_lambda",
                    "HVG_path_length", "HVG_diameter"
                ]
                hvg_df = long_df[long_df["Fase"] == phase_pg][["Registro", "Fase"] + [c for c in hvg_cols if c in long_df.columns]]
                st.dataframe(hvg_df, use_container_width=True)

                hvg_numeric = [c for c in hvg_cols if c in hvg_df.columns and pd.api.types.is_numeric_dtype(hvg_df[c])]
                if hvg_numeric:
                    hvg_var = st.selectbox("Métrica de grafo a comparar", hvg_numeric)
                    pivot_hvg = hvg_df.pivot_table(index="Fase", columns="Registro", values=hvg_var, aggfunc="first")
                    st.plotly_chart(comparison_bar_line(pivot_hvg, hvg_var), use_container_width=True, key=f"hvg_compare_{hvg_var}_{phase_pg}")

                st.markdown("### Grafo HVG simplificado")
                rec_graph = st.selectbox("Registro para visualizar grafo", records, key="rec_graph_v51")
                windows_graph = get_record_windows(
                    st.session_state.global_windows_v50,
                    st.session_state.record_windows_v50,
                    rec_graph,
                    use_independent
                )
                w_graph = windows_graph.get(phase_pg)
                if w_graph is not None:
                    seg_graph = cut_segment(record_data[rec_graph]["rr"], w_graph[0], w_graph[1])
                    if len(seg_graph) >= min_rr:
                        st.plotly_chart(
                            hvg_network_figure(seg_graph, title=f"HVG {rec_graph} · {phase_pg}", max_nodes=140),
                            use_container_width=True,
                            key=f"hvg_network_{rec_graph}_{phase_pg}"
                        )
                    else:
                        st.info("La fase seleccionada tiene pocos RRi para visualizar el grafo.")


with tab5:
    st.subheader("Dashboard visual")

    if long_df.empty:
        st.info("No hay datos.")
    else:
        available_phases = [p for p in PHASES if p in long_df["Fase"].unique()]
        numeric_vars = [c for c in long_df.columns if c not in ["Registro", "Fase"] and pd.api.types.is_numeric_dtype(long_df[c])]
        phases_dash = st.multiselect("Fases", PHASES, default=available_phases, key="dash_phases")
        params_dash = st.multiselect("Parámetros", numeric_vars, default=[p for p in DEFAULT_MULTI if p in numeric_vars], key="dash_params")
        if params_dash:
            st.plotly_chart(dashboard_compare(long_df, phases_dash or available_phases, params_dash), use_container_width=True, key="dashboard_tab_main")

with tab6:
    st.subheader("Exportar")

    if long_df.empty:
        st.info("No hay datos para exportar.")
    else:
        valid_summary = pd.DataFrame(records_valid).T.reindex(columns=PHASES)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            xlsx = tmpdir / "resultados_hrv_comparativa.xlsx"
            csv = tmpdir / "resultados_hrv_comparativa.csv"
            zipf = tmpdir / "resultados_hrv_comparativa.zip"

            long_df.to_csv(csv, index=False)

            with pd.ExcelWriter(xlsx) as writer:
                long_df.to_excel(writer, sheet_name="metricas", index=False)
                valid_summary.to_excel(writer, sheet_name="ventanas_validas")

                rows_w = []
                for rec in records:
                    w = get_record_windows(st.session_state.global_windows_v50, st.session_state.record_windows_v50, rec, use_independent)
                    for ph in PHASES:
                        ww = w.get(ph)
                        rows_w.append({
                            "Registro": rec,
                            "Fase": ph,
                            "Inicio": sec_to_hms(ww[0]) if ww else "",
                            "Fin": sec_to_hms(ww[1]) if ww else "",
                            "Duracion_min": (ww[1] - ww[0]) / 60 if ww else np.nan,
                            "Activa": ph in active_phases,
                        })
                pd.DataFrame(rows_w).to_excel(writer, sheet_name="ventanas", index=False)

                artifact_rows = []
                for rec, data in record_data.items():
                    info = data.get("artifact_info", {})
                    artifact_rows.append({
                        "Registro": rec,
                        "Nivel_correccion": info.get("level", "none"),
                        "Artefactos_n": info.get("n_artifacts", 0),
                        "Artefactos_pct": info.get("percent_artifacts", 0.0),
                    })
                pd.DataFrame(artifact_rows).to_excel(writer, sheet_name="artefactos", index=False)

            with zipfile.ZipFile(zipf, "w", zipfile.ZIP_DEFLATED) as z:
                z.write(xlsx, arcname=xlsx.name)
                z.write(csv, arcname=csv.name)

            st.download_button("Descargar ZIP", zipf.read_bytes(), file_name="resultados_hrv_comparativa.zip", mime="application/zip")
            st.download_button("Descargar Excel", xlsx.read_bytes(), file_name="resultados_hrv_comparativa.xlsx")
