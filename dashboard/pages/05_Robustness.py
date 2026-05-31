
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "dashboard"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.charts import (
    COLORS, NOISE_LABELS, NOISE_COLORS, SNR_VALUES,
    page_footer,
)

_NOISY_V1 = _ROOT / "outputs" / "reports" / "eval_noisy.json"
_CLEAN_V1 = _ROOT / "outputs" / "reports" / "eval_clean.json"
_NOISY_V3 = _ROOT / "outputs" / "reports" / "eval_noisy_v3.json"
_CLEAN_V3 = _ROOT / "outputs" / "reports" / "eval_clean_v3.json"
_AGG_V1   = _ROOT / "outputs" / "reports" / "aggregate_v1.json"
_AGG_V3   = _ROOT / "outputs" / "reports" / "aggregate_v3.json"


@st.cache_data(show_spinner=False)
def _load_agg_flat(agg_path: str) -> tuple[dict, dict, dict]:
    """Iz aggregate_*.json gradi 'flat' strukture (srednje vrijednosti) koje
    stranica ocekuje, plus paralelnu mapu standardnih devijacija.
    Vraca: (noisy_mean, clean_mean, noisy_std)."""
    agg = json.loads(Path(agg_path).read_text())
    metrics = ["accuracy", "precision", "recall", "specificity", "f1", "auc_roc", "auc_pr"]
    noisy_mean: dict = {}
    noisy_std: dict = {}
    for nt, by_snr in agg["noisy"].items():
        noisy_mean[nt] = {}
        noisy_std[nt] = {}
        for snr_key, md in by_snr.items():
            noisy_mean[nt][snr_key] = {m: md[m]["mean"] for m in metrics if m in md}
            noisy_std[nt][snr_key]  = {m: md[m]["std"]  for m in metrics if m in md}
    clean_mean = {"metrics": {m: agg["clean"][m]["mean"] for m in metrics}}
    return noisy_mean, clean_mean, noisy_std

METRIC_META = {
    "f1":       ("F1 mjera",  [0.4, 1.05]),
    "auc_roc":  ("AUC-ROC",   [0.4, 1.05]),
    "accuracy": ("Tacnost",   [0.3, 1.05]),
}


@st.cache_data(show_spinner=False)
def _load(noisy_path: str, clean_path: str) -> tuple[dict, dict]:
    noisy = json.loads(Path(noisy_path).read_text())
    clean = json.loads(Path(clean_path).read_text())
    return noisy, clean


def _build_matrix(eval_noisy: dict, metric: str) -> np.ndarray:
    noise_types = list(eval_noisy.keys())
    matrix = np.zeros((len(noise_types), len(SNR_VALUES)))
    for i, nt in enumerate(noise_types):
        for j, snr in enumerate(SNR_VALUES):
            matrix[i, j] = eval_noisy[nt][f"snr_{snr}"][metric]
    return matrix


def _key_findings(eval_noisy: dict, eval_clean: dict, metric: str,
                  metric_label: str) -> None:
  
    noise_types = list(eval_noisy.keys())
    snr0 = {nt: eval_noisy[nt]["snr_0"][metric] for nt in noise_types}
    worst_nt = min(snr0, key=snr0.get)
    best_nt  = max(snr0, key=snr0.get)
    clean_val = eval_clean["metrics"][metric]

    avg_by_snr = []
    for snr in SNR_VALUES:
        vals = [eval_noisy[nt][f"snr_{snr}"][metric] for nt in noise_types]
        avg_by_snr.append(sum(vals) / len(vals))

    threshold_snr = None
    for snr, avg in zip(SNR_VALUES, avg_by_snr):
        if avg >= 0.85:
            threshold_snr = snr
            break

    st.markdown("### Kljucni nalazi")
    kf1, kf2, kf3 = st.columns(3)
    kf1.metric(
        "Najstetnija vrsta suma (SNR=0 dB)",
        NOISE_LABELS.get(worst_nt, worst_nt),
        delta=f"{snr0[worst_nt] - clean_val:+.3f}",
        delta_color="inverse",
    )
    kf2.metric(
        "Najrobusnija vrsta suma (SNR=0 dB)",
        NOISE_LABELS.get(best_nt, best_nt),
        delta=f"{snr0[best_nt] - clean_val:+.3f}",
        delta_color="inverse",
    )
    if threshold_snr is not None:
        kf3.metric(
            f"Min SNR za prosjek {metric_label} >= 0.85",
            f"{threshold_snr} dB",
        )
    else:
        kf3.metric("Prosjek nikad ne dostize 0.85", "-")


def _render_line_plot(eval_noisy: dict, eval_clean: dict, metric: str,
                      metric_label: str, y_range: list, title_suffix: str) -> None:
    noise_types  = list(eval_noisy.keys())
    clean_val    = eval_clean["metrics"][metric]
    matrix       = _build_matrix(eval_noisy, metric)

    fig = go.Figure()
    fig.add_hline(
        y=clean_val, line_dash="dash", line_color=COLORS["muted"],
        annotation_text=f"Cisti signal ({clean_val:.3f})",
        annotation_position="bottom right",
    )
    for i, (nt, color) in enumerate(zip(noise_types, NOISE_COLORS)):
        fig.add_trace(go.Scatter(
            x=SNR_VALUES, y=matrix[i].tolist(),
            mode="lines+markers",
            name=NOISE_LABELS.get(nt, nt),
            line=dict(color=color, width=2),
            marker=dict(size=7),
        ))
    fig.update_layout(
        title=f"{metric_label} po vrsti suma i SNR-u {title_suffix}",
        xaxis_title="SNR (dB)",
        yaxis_title=metric_label,
        yaxis=dict(range=y_range),
        height=400,
        legend=dict(x=0.01, y=0.01),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["bg_plot"],
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_heatmap(eval_noisy: dict, metric: str, metric_label: str,
                    title_suffix: str) -> None:
    noise_types  = list(eval_noisy.keys())
    noise_labels = [NOISE_LABELS.get(t, t) for t in noise_types]
    snr_labels   = [f"{s} dB" for s in SNR_VALUES]
    matrix       = _build_matrix(eval_noisy, metric)

    fig = go.Figure(go.Heatmap(
        z=matrix.tolist(), x=snr_labels, y=noise_labels,
        colorscale="RdYlGn", zmin=0.4, zmax=1.0,
        text=[[f"{v:.3f}" for v in row] for row in matrix],
        texttemplate="%{text}", showscale=True,
        colorbar=dict(title=metric_label),
    ))
    fig.update_layout(
        title=f"Toplotna mapa {metric_label} {title_suffix}",
        xaxis_title="SNR", yaxis_title="Vrsta suma",
        height=280,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_table(eval_noisy: dict, metric: str) -> None:
    noise_types  = list(eval_noisy.keys())
    noise_labels = [NOISE_LABELS.get(t, t) for t in noise_types]
    snr_labels   = [f"{s} dB" for s in SNR_VALUES]
    matrix       = _build_matrix(eval_noisy, metric)
    df = pd.DataFrame(matrix, index=noise_labels, columns=snr_labels)
    df = df.map(lambda x: f"{x:.3f}")
    st.dataframe(df, use_container_width=True)


def _render_boxplot(eval_noisy: dict, metric: str, metric_label: str,
                    title_suffix: str) -> None:
    noise_types = list(eval_noisy.keys())
    matrix      = _build_matrix(eval_noisy, metric)
    snr_labels  = [f"{s} dB" for s in SNR_VALUES]

    fig = go.Figure()
    box_colors = ["#E84C4C", "#E87A4C", "#E8A44C", "#C8E84C", "#4CE874", "#4C9BE8"]
    for j, (snr_lbl, bc) in enumerate(zip(snr_labels, box_colors)):
        fig.add_trace(go.Box(
            y=matrix[:, j].tolist(), name=snr_lbl,
            boxmean=True, marker_color=bc,
        ))
    fig.update_layout(
        title=f"Distribucija {metric_label} po SNR nivou {title_suffix}",
        xaxis_title="SNR nivo", yaxis_title=metric_label,
        yaxis=dict(range=[0.3, 1.05]),
        height=320,
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["bg_plot"],
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_single(eval_noisy: dict, eval_clean: dict, metric: str,
                   metric_label: str, y_range: list, title_suffix: str) -> None:
    _key_findings(eval_noisy, eval_clean, metric, metric_label)
    st.markdown("---")
    _render_line_plot(eval_noisy, eval_clean, metric, metric_label, y_range, title_suffix)
    st.info(
        f"Linijski grafikon prikazuje kako {metric_label} opada s nizim SNR nivoima za svaku "
        "vrstu suma. Horizontalna isprekidana linija oznacava vrijednost na cistom signalu. "
        "Idealan model bi zadrzao visoke performanse cak i pri niskim SNR vrijednostima."
    )
    _render_heatmap(eval_noisy, metric, metric_label, title_suffix)
    st.info(
        "Toplotna mapa daje pregledan prikaz svih kombinacija suma i SNR-a. "
        "Crvena polja oznacavaju znacajan pad performansi; zelena polja znace robustno ponasanje. "
        "Uocljivo je da svi modeli imaju nize performanse pri SNR <= 0 dB."
    )
    with st.expander("Tabela vrijednosti"):
        _render_table(eval_noisy, metric)
    _render_boxplot(eval_noisy, metric, metric_label, title_suffix)
    st.info(
        "Box-plot prikazuje raspodjelu metrike po SNR nivou (svaka tacka = jedna vrsta suma). "
        "Pri visokim SNR nivoima (18-24 dB) sve vrste suma imaju slicne performanse - "
        "signal je dovoljno cist i vrsta suma postaje manje relevantna."
    )


def _render_comparison(noisy_v1: dict, clean_v1: dict, noisy_v3: dict, clean_v3: dict,
                        metric: str, metric_label: str) -> None:

    noise_types  = list(noisy_v1.keys())
    noise_labels = [NOISE_LABELS.get(t, t) for t in noise_types]
    snr_labels   = [f"{s} dB" for s in SNR_VALUES]
    clean_v1_val = clean_v1["metrics"][metric]
    clean_v3_val = clean_v3["metrics"][metric]


    _key_findings(noisy_v3, clean_v3, metric, metric_label)

    st.markdown("---")
    st.markdown(f"### {metric_label} - V1 (isprekidano) vs V3 (puno) po vrsti suma")
    fig_cmp = go.Figure()
    fig_cmp.add_hline(
        y=clean_v1_val, line_dash="dot", line_color=COLORS["muted"],
        annotation_text=f"V1 cist ({clean_v1_val:.3f})",
        annotation_position="bottom right",
    )
    fig_cmp.add_hline(
        y=clean_v3_val, line_dash="dot", line_color=COLORS["norm"],
        annotation_text=f"V3 cist ({clean_v3_val:.3f})",
        annotation_position="top right",
    )
    for i, (nt, color) in enumerate(zip(noise_types, NOISE_COLORS)):
        label = NOISE_LABELS.get(nt, nt)
        y_v1  = [noisy_v1[nt][f"snr_{s}"][metric] for s in SNR_VALUES]
        y_v3  = [noisy_v3[nt][f"snr_{s}"][metric] for s in SNR_VALUES]
        fig_cmp.add_trace(go.Scatter(
            x=SNR_VALUES, y=y_v1, mode="lines+markers",
            name=f"{label} - V1",
            line=dict(color=color, width=1.5, dash="dash"),
            marker=dict(size=5, symbol="circle"),
        ))
        fig_cmp.add_trace(go.Scatter(
            x=SNR_VALUES, y=y_v3, mode="lines+markers",
            name=f"{label} - V3",
            line=dict(color=color, width=2.0, dash="solid"),
            marker=dict(size=6, symbol="square"),
        ))
    fig_cmp.update_layout(
        xaxis_title="SNR (dB)", yaxis_title=metric_label,
        yaxis=dict(range=[0.4, 1.05]),
        height=460,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["bg_plot"],
        legend=dict(x=1.01, y=1, xanchor="left"),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)
    st.info(
        "V3 (pune linije) u prosjeku je bolji ili jednak V1 (isprekidane linije) u srednjem "
        "SNR opsegu (0–6 dB), gdje augmentacija šuma ima najveći uticaj i gdje je razlika "
        "statistički značajna za bazno lutanje, mišićni artefakt i pomicanje elektrode "
        "(p < 0.05, vidi „Multi-seed analiza“). Pri visokim SNR nivoima (≥ 12 dB) razlika "
        "nestaje jer se oba modela približavaju performansama na čistom signalu."
    )


    matrix_v1   = _build_matrix(noisy_v1, metric)
    matrix_v3   = _build_matrix(noisy_v3, metric)
    matrix_diff = matrix_v3 - matrix_v1
    abs_max     = max(float(np.abs(matrix_diff).max()), 0.01)

    st.markdown(f"### Razlika {metric_label}: V3 - V1 (zelena = V3 bolji)")
    fig_diff = go.Figure(go.Heatmap(
        z=matrix_diff.tolist(), x=snr_labels, y=noise_labels,
        colorscale="RdYlGn", zmin=-abs_max, zmax=abs_max,
        text=[[f"{v:+.3f}" for v in row] for row in matrix_diff],
        texttemplate="%{text}", showscale=True,
        colorbar=dict(title=f"Delta {metric_label}"),
    ))
    fig_diff.update_layout(
        xaxis_title="SNR", yaxis_title="Vrsta suma",
        height=280,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_diff, use_container_width=True)
    st.info(
        "Najveći prosječni dobitak augmentacije zabilježen je pri SNR = 0 dB za bazno lutanje "
        "(≈ +0.09), mišićni artefakt (≈ +0.08) i pomicanje elektrode (≈ +0.06); te razlike su "
        "statistički značajne (upareni t-test, p < 0.05). Pri vrlo niskom SNR-u (−6 dB) razlike "
        "su nestabilne i standardna devijacija je velika (posebno za mrežnu smetnju), pa ih "
        "treba tumačiti oprezno."
    )


    st.markdown(f"### Tabela razlike {metric_label} (V3 - V1)")
    df_diff = pd.DataFrame(matrix_diff, index=noise_labels, columns=snr_labels)
    df_diff = df_diff.map(lambda x: f"{x:+.3f}")
    st.dataframe(df_diff, use_container_width=True)



st.title("Analiza robustnosti na šum")
st.markdown(
    "Performanse modela pri različitim vrstama šuma i nivoima omjera signal-šum (SNR). "
    "Niži SNR znači više šuma i teže uvjete za klasifikaciju. "
    "Evaluacija je rađena na 1 462 test snimka za svaku od 30 kombinacija (šum × SNR)."
)

_AGG_MODE = _AGG_V1.exists() and _AGG_V3.exists()
if _AGG_MODE:
    _n1 = json.loads(_AGG_V1.read_text())["n_runs"]
    _n3 = json.loads(_AGG_V3.read_text())["n_runs"]
    st.success(
        f"Prikazane vrijednosti su **srednje vrijednosti iz {_n1} (V1) / {_n3} (V3) "
        f"nezavisnih treninga**. Za standardne devijacije, p-vrijednosti i krive sa error "
        f"barovima pogledajte stranicu „Multi-seed analiza“."
    )

if not _AGG_MODE and not _NOISY_V1.exists():
    try:
        from utils.download_assets import ensure_noisy_results
        ensure_noisy_results()
    except Exception as e:
        st.error("Preuzimanje rezultata evaluacije robustnosti nije uspjelo.")
        st.exception(e)

_USING_MOCK = (not _AGG_MODE) and (not _NOISY_V1.exists())
if _USING_MOCK:
    st.info(
        "Demo rezim: stvarni rezultati evaluacije nisu dostupni. "
        "Prikazuju se simulirani podaci za demonstraciju."
    )
    from mock_data import get_mock_noisy_eval, get_mock_clean_eval
    _mock_noisy = get_mock_noisy_eval()
    _mock_clean = get_mock_clean_eval()

v3_available = _AGG_MODE or ((not _USING_MOCK) and _NOISY_V3.exists() and _CLEAN_V3.exists())

model_options = ["V1 (bazni model)"]
if v3_available:
    model_options += ["V3 (augmentirani model)", "Poredjenje V1 i V3"]

model_choice  = st.sidebar.radio("Verzija modela", model_options)
metric_choice = st.sidebar.radio(
    "Metrika za prikaz",
    list(METRIC_META.keys()),
    format_func=lambda k: METRIC_META[k][0],
)

metric_label, y_range = METRIC_META[metric_choice]

_suffix = "(srednje vrijednosti, N=10)" if _AGG_MODE else ""

if _USING_MOCK:
    _render_single(_mock_noisy, _mock_clean, metric_choice, metric_label, y_range, "(demo)")

elif _AGG_MODE:
    nv1, cv1, _ = _load_agg_flat(str(_AGG_V1))
    nv3, cv3, _ = _load_agg_flat(str(_AGG_V3))
    if model_choice == "V1 (bazni model)":
        _render_single(nv1, cv1, metric_choice, metric_label, y_range, f"V1 {_suffix}")
    elif model_choice == "V3 (augmentirani model)":
        _render_single(nv3, cv3, metric_choice, metric_label, y_range, f"V3 {_suffix}")
    else:
        _render_comparison(nv1, cv1, nv3, cv3, metric_choice, metric_label)

elif model_choice == "V1 (bazni model)":
    noisy, clean = _load(str(_NOISY_V1), str(_CLEAN_V1))
    _render_single(noisy, clean, metric_choice, metric_label, y_range, "(V1)")

elif model_choice == "V3 (augmentirani model)":
    noisy, clean = _load(str(_NOISY_V3), str(_CLEAN_V3))
    _render_single(noisy, clean, metric_choice, metric_label, y_range, "(V3)")

else:
    noisy_v1, clean_v1 = _load(str(_NOISY_V1), str(_CLEAN_V1))
    noisy_v3, clean_v3 = _load(str(_NOISY_V3), str(_CLEAN_V3))
    _render_comparison(noisy_v1, clean_v1, noisy_v3, clean_v3, metric_choice, metric_label)

page_footer()
