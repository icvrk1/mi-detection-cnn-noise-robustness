"""
Reusable Plotly/Streamlit chart components for the dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Shared color palette ---
COLORS = {
    "primary":  "#4C9BE8",  # blue - main / primary
    "mi":       "#E84C6A",  # red-pink - MI class / danger
    "norm":     "#4CE874",  # green - normal class / good
    "warn":     "#E8A44C",  # orange - noise / warning
    "accent":   "#9467bd",  # purple - secondary accent
    "muted":    "#AAAAAA",  # gray - reference lines
    "bg_plot":  "rgba(26,35,50,0.6)",
}

# --- Shared label sets ---
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SNR_VALUES = [-6, 0, 6, 12, 18, 24]
SNR_COLORS = ["#E84C4C", "#E87A4C", "#E8A44C", "#C8E84C", "#4CE874", "#4C9BE8"]

NOISE_LABELS = {
    "gaussian":         "Gaussov bijeli sum",
    "baseline_wander":  "Bazno lutanje",
    "muscle_artifact":  "Misicni artefakt",
    "electrode_motion": "Pomicanje elektrode",
    "powerline":        "Smetnja mreze (50 Hz)",
}
NOISE_COLORS = ["#9467bd", "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"]

_LAYOUT_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,35,50,0.6)",
)


def page_footer() -> None:
    """Standard page footer."""
    st.markdown("---")
    st.caption("EKG-MI Dashboard | Zavrsni rad | Ilma Cvrk | 2026")


def plot_ecg_signal(
    signal: np.ndarray,
    fs: int = 100,
    title: str = "EKG signal",
    color: str = COLORS["primary"],
) -> go.Figure:
    t = np.arange(len(signal)) / fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=signal, mode="lines",
        line=dict(color=color, width=1.2), name="signal",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Vrijeme (s)",
        yaxis_title="Amplituda (mV)",
        height=280,
        margin=dict(l=40, r=20, t=40, b=40),
        **_LAYOUT_BASE,
    )
    return fig


def plot_training_curves(
    history: dict,
    metric: str,
    label: str,
    train_color: str = COLORS["primary"],
    val_color: str = COLORS["mi"],
    height: int = 300,
) -> go.Figure:
    """Plot train/val curve for a given metric key from a history dict."""
    epochs = list(range(1, len(history[f"train_{metric}"]) + 1))
    best_ep = history.get("best_epoch")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=history[f"train_{metric}"], mode="lines",
        name="Trening", line=dict(color=train_color, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=history[f"val_{metric}"], mode="lines",
        name="Validacija", line=dict(color=val_color, width=2, dash="dash"),
    ))
    if best_ep:
        fig.add_vline(
            x=best_ep, line_dash="dot", line_color="white",
            annotation_text=f"Ep. {best_ep}",
            annotation_position="top right",
        )
    fig.update_layout(
        xaxis_title="Epoha",
        yaxis_title=label,
        height=height,
        legend=dict(x=0.01, y=0.05),
        margin=dict(l=50, r=20, t=20, b=40),
        **_LAYOUT_BASE,
    )
    return fig


def plot_metric_heatmap(
    data: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    title: str = "Heatmap",
    colorscale: str = "Blues",
    fmt: str = ".2f",
) -> go.Figure:
    text = np.array([[f"{v:{fmt}}" for v in row] for row in data])
    fig = go.Figure(go.Heatmap(
        z=data, x=x_labels, y=y_labels,
        colorscale=colorscale, zmin=0, zmax=1,
        text=text, texttemplate="%{text}", showscale=True,
    ))
    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=100, r=20, t=50, b=60),
        **_LAYOUT_BASE,
    )
    return fig


def display_metrics_table(metrics: dict[str, float]) -> None:
    """Render a metrics dict as a styled Streamlit dataframe."""
    labels = {
        "accuracy":    "Tacnost",
        "precision":   "Preciznost",
        "recall":      "Osjetljivost (Recall)",
        "specificity": "Specificnost",
        "f1":          "F1 mjera",
        "auc_roc":     "AUC-ROC",
        "auc_pr":      "AUC-PR",
    }
    rows = [
        {"Metrika": labels.get(k, k), "Vrijednost": f"{v:.4f}"}
        for k, v in metrics.items()
        if k in labels
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
