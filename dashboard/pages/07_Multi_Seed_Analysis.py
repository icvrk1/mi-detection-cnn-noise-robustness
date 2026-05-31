"""
Multi-seed analiza - mean +/- std preko N nezavisnih treninga.

Ova stranica je kanonski prikaz rezultata koji se koriste u diplomskom radu.
Cita outputs/reports/aggregate_v1.json, aggregate_v3.json, aggregate_comparison.json
i krive treninga iz outputs/runs/{v1,v3}/seed_*/training_history_*.json.

Ako objedinjeni rezultati ne postoje, prikazuje uputstvo kako ih generisati.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

_REPORTS = _ROOT / "outputs" / "reports"
_RUNS    = _ROOT / "outputs" / "runs"

AGG_V1 = _REPORTS / "aggregate_v1.json"
AGG_V3 = _REPORTS / "aggregate_v3.json"
AGG_CP = _REPORTS / "aggregate_comparison.json"

NOISE_LABELS = {
    "gaussian":         "Gaussov bijeli sum",
    "baseline_wander":  "Kolebanje osnovne linije",
    "powerline":        "Mrezna frekvencija (50 Hz)",
    "muscle_artifact":  "Misicni artefakt",
    "electrode_motion": "Pomicanje elektrode",
}
NOISE_COLORS = {
    "gaussian":         "#1f77b4",
    "baseline_wander":  "#ff7f0e",
    "powerline":        "#2ca02c",
    "muscle_artifact":  "#d62728",
    "electrode_motion": "#9467bd",
}
SNR_LEVELS = [-6, 0, 6, 12, 18, 24]


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def fmt_ms(d: dict, digits: int = 3) -> str:
    if d is None or d.get("mean") is None:
        return "—"
    return f"{d['mean']:.{digits}f} ± {d['std']:.{digits}f}"


def p_stars(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def load_histories(variant: str) -> list[dict]:
    base = _RUNS / variant
    if not base.exists():
        return []
    out = []
    for sd in sorted(base.iterdir()):
        if not (sd.is_dir() and sd.name.startswith("seed_")):
            continue
        hp = sd / f"training_history_{variant}.json"
        if hp.exists():
            with open(hp) as fh:
                out.append(json.load(fh))
    return out


def pad_mat(series_list: list[list[float]]) -> np.ndarray:
    if not series_list:
        return np.zeros((0, 0))
    max_len = max(len(s) for s in series_list)
    M = np.full((len(series_list), max_len), np.nan, dtype=float)
    for i, s in enumerate(series_list):
        M[i, :len(s)] = s
    return M


def _hex_to_rgba(color: str, alpha: float) -> str:
    """Pretvori #RRGGBB ili rgb(...) u rgba string sa zadatom alfom."""
    c = color.strip()
    if c.startswith("#"):
        c = c.lstrip("#")
        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    if c.startswith("rgb("):
        inner = c[4:].rstrip(")")
        return f"rgba({inner},{alpha})"
    if c.startswith("rgba("):
        return c
    return f"rgba(120,120,120,{alpha})"


def mean_std_band_fig(matrices: dict[str, tuple[np.ndarray, str]],
                      ylabel: str, title: str, ylim=None) -> go.Figure:
    fig = go.Figure()
    for label, (mat, color) in matrices.items():
        if mat.size == 0:
            continue
        m = np.nanmean(mat, axis=0)
        s = np.nanstd(mat, axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(m)
        x = np.arange(1, mat.shape[1] + 1)
        # std band - poluprozirna ispuna (hex -> rgba sa alfom)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([m + s, (m - s)[::-1]]),
            fill="toself", fillcolor=_hex_to_rgba(color, 0.18),
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
            showlegend=False, name=f"{label} std",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=m, mode="lines", name=label, line=dict(color=color, width=2.5),
            hovertemplate=f"{label}<br>epoha %{{x}}<br>%{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        title=title, xaxis_title="Epoha", yaxis_title=ylabel,
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,35,50,0.6)", height=380,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    if ylim:
        fig.update_layout(yaxis=dict(range=list(ylim)))
    return fig


# ============================================================ stranica
st.title("Multi-seed analiza (srednja vrijednost ± standardna devijacija)")

if not (AGG_V1.exists() and AGG_V3.exists() and AGG_CP.exists()):
    st.warning(
        "Objedinjeni rezultati nisu pronađeni. Da generišeš rezultate iz više nezavisnih treninga, pokreni:"
    )
    st.code(
        "python scripts/10_run_multi_seed.py --num-seeds 10 --start-seed 42\n"
        "python scripts/11_aggregate_runs.py\n"
        "python scripts/12_aggregate_training_curves.py  # opcionalno",
        language="bash",
    )
    st.stop()

agg_v1   = load_json(str(AGG_V1))
agg_v3   = load_json(str(AGG_V3))
cmp_data = load_json(str(AGG_CP))

n_v1 = agg_v1["n_runs"]; n_v3 = agg_v3["n_runs"]

st.markdown(
    f"Sve vrijednosti na ovoj stranici su **srednja vrijednost ± standardna devijacija** "
    f"preko **{n_v1} (V1)** i **{n_v3} (V3)** nezavisnih treninga sa razlicitim slucajnim "
    f"sjemenima (seedovi {agg_v1['seeds']} / {agg_v3['seeds']}). "
    f"P-vrijednosti dolaze iz uparenog t-testa po istom sjemenu (V1 vs V3)."
)

tab_clean, tab_robust, tab_compare, tab_train = st.tabs(
    ["Cisti test skup", "Robustnost", "V1 vs V3", "Krive treninga"]
)

# ============ TAB 1: Cisti test skup
with tab_clean:
    st.subheader("Performanse na čistom test skupu (srednja vrijednost ± standardna devijacija)")
    rows = []
    for label, key in [("Tacnost","accuracy"), ("Preciznost","precision"),
                       ("Osjetljivost (recall)","recall"), ("Specificnost","specificity"),
                       ("F1-mjera","f1"), ("AUC-ROC","auc_roc"), ("AUC-PR","auc_pr")]:
        rows.append({
            "Metrika":         label,
            "V1 (sr. vr. ± st. dev.)": fmt_ms(agg_v1["clean"][key], 4),
            "V3 (sr. vr. ± st. dev.)": fmt_ms(agg_v3["clean"][key], 4),
            "Δ (V3-V1)":       f"{cmp_data['clean'][key]['delta_mean']:+.4f}"
                               if cmp_data['clean'][key]['delta_mean'] is not None else "—",
            "p":               f"{cmp_data['clean'][key].get('t_p'):.4f}"
                               if cmp_data['clean'][key].get('t_p') is not None else "—",
            "":                p_stars(cmp_data['clean'][key].get('t_p')),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Oznake p-vrijednosti: *** p<0.001, ** p<0.01, * p<0.05")

# ============ TAB 2: Robustnost
with tab_robust:
    st.subheader("F1-mjera po vrsti šuma i SNR-u (V1, srednja vrijednost ± standardna devijacija)")
    rows = []
    for nt in agg_v1["noisy"]:
        row = {"Tip suma": NOISE_LABELS.get(nt, nt)}
        for s in SNR_LEVELS:
            key = f"snr_{s}"
            d = agg_v1["noisy"][nt][key]["f1"]
            row[f"{s:+d} dB"] = f"{d['mean']:.3f} ± {d['std']:.3f}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("F1 - krive sa error barovima (V1 vs V3)")
    fig = go.Figure()
    cln_v1 = agg_v1["clean"]["f1"]
    fig.add_hline(y=cln_v1["mean"], line=dict(color="black", dash="dash", width=1),
                  annotation_text=f"V1 cist: {cln_v1['mean']:.3f}±{cln_v1['std']:.3f}",
                  annotation_position="top right")
    for nt in agg_v1["noisy"]:
        color = NOISE_COLORS.get(nt, "#888")
        v1_m = [agg_v1["noisy"][nt][f"snr_{s}"]["f1"]["mean"] for s in SNR_LEVELS]
        v1_s = [agg_v1["noisy"][nt][f"snr_{s}"]["f1"]["std"]  for s in SNR_LEVELS]
        v3_m = [agg_v3["noisy"][nt][f"snr_{s}"]["f1"]["mean"] for s in SNR_LEVELS] \
               if nt in agg_v3["noisy"] else None
        v3_s = [agg_v3["noisy"][nt][f"snr_{s}"]["f1"]["std"]  for s in SNR_LEVELS] \
               if nt in agg_v3["noisy"] else None
        fig.add_trace(go.Scatter(
            x=SNR_LEVELS, y=v1_m, mode="lines+markers",
            name=f"{NOISE_LABELS.get(nt, nt)} - V1",
            line=dict(color=color, width=2),
            error_y=dict(type="data", array=v1_s, visible=True, color=color),
        ))
        if v3_m is not None:
            fig.add_trace(go.Scatter(
                x=SNR_LEVELS, y=v3_m, mode="lines+markers",
                name=f"{NOISE_LABELS.get(nt, nt)} - V3",
                line=dict(color=color, width=2, dash="dash"),
                error_y=dict(type="data", array=v3_s, visible=True, color=color),
                opacity=0.85,
            ))
    fig.update_layout(
        xaxis_title="SNR (dB)", yaxis_title="F1 mjera (mean ± std)",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,35,50,0.6)", height=520, legend=dict(orientation="v"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# ============ TAB 3: V1 vs V3
with tab_compare:
    st.subheader("Razlika V3 - V1 po (vrsta suma, SNR) sa p-vrijednoscu")
    rows = []
    for nt in cmp_data["noisy"]:
        for s in SNR_LEVELS:
            key = f"snr_{s}"
            r = cmp_data["noisy"][nt][key]["f1"]
            stars = p_stars(r.get("t_p"))
            rows.append({
                "Tip suma":    NOISE_LABELS.get(nt, nt),
                "SNR":         f"{s:+d} dB",
                "V1":          f"{r['v1_mean']:.3f} ± {r['v1_std']:.3f}",
                "V3":          f"{r['v3_mean']:.3f} ± {r['v3_std']:.3f}",
                "Δ":           f"{r['delta_mean']:+.4f}" if r['delta_mean'] is not None else "—",
                "p":           f"{r['t_p']:.4f}" if r.get('t_p') is not None else "—",
                "":            stars,
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Pozitivno Δ znaci da V3 ima bolji F1 od V1 za tu kombinaciju.")

# ============ TAB 4: Krive treninga
with tab_train:
    st.subheader("Krive treninga (srednja vrijednost ± standardna devijacija)")
    hv1 = load_histories("v1")
    hv3 = load_histories("v3")
    if not hv1 and not hv3:
        st.info("Nema dostupnih history fajlova u outputs/runs/.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**V1 - bazni model**")
            mats = {}
            if hv1:
                mats["Trening"] = (pad_mat([h["train_loss"] for h in hv1]), "#0072B2")
                mats["Validacija"] = (pad_mat([h["val_loss"] for h in hv1]), "#D55E00")
            st.plotly_chart(
                mean_std_band_fig(mats, "Gubitak", f"V1 - gubitak (n={len(hv1)})"),
                use_container_width=True,
            )
            mats_f1 = {}
            if hv1:
                mats_f1["Trening"]   = (pad_mat([h["train_f1"] for h in hv1]), "#0072B2")
                mats_f1["Validacija"] = (pad_mat([h["val_f1"] for h in hv1]), "#D55E00")
            st.plotly_chart(
                mean_std_band_fig(mats_f1, "F1", f"V1 - F1 (n={len(hv1)})",
                                  ylim=(0.7, 1.0)),
                use_container_width=True,
            )
        with c2:
            st.markdown("**V3 - sa augmentacijom suma**")
            mats = {}
            if hv3:
                mats["Trening"] = (pad_mat([h["train_loss"] for h in hv3]), "#0072B2")
                mats["Validacija"] = (pad_mat([h["val_loss"] for h in hv3]), "#D55E00")
            st.plotly_chart(
                mean_std_band_fig(mats, "Gubitak", f"V3 - gubitak (n={len(hv3)})"),
                use_container_width=True,
            )
            mats_f1 = {}
            if hv3:
                mats_f1["Trening"]   = (pad_mat([h["train_f1"] for h in hv3]), "#0072B2")
                mats_f1["Validacija"] = (pad_mat([h["val_f1"] for h in hv3]), "#D55E00")
            st.plotly_chart(
                mean_std_band_fig(mats_f1, "F1", f"V3 - F1 (n={len(hv3)})",
                                  ylim=(0.7, 1.0)),
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("**Validacijska F1 - V1 vs V3**")
        mats_cmp = {}
        if hv1: mats_cmp[f"V1 (n={len(hv1)})"] = (pad_mat([h["val_f1"] for h in hv1]), "#0072B2")
        if hv3: mats_cmp[f"V3 (n={len(hv3)})"] = (pad_mat([h["val_f1"] for h in hv3]), "#D55E00")
        st.plotly_chart(
            mean_std_band_fig(mats_cmp, "F1", "Validacijska F1 - V1 vs V3",
                              ylim=(0.7, 1.0)),
            use_container_width=True,
        )
