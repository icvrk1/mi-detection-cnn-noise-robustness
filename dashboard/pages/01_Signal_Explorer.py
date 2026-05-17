
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "dashboard"))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.charts import (
    COLORS, LEAD_NAMES,
    page_footer, plot_ecg_signal,
)

FS = 100
_DATA_PATH = _ROOT / "data" / "processed" / "test_clean.npz"
_DIST_PATH = _ROOT / "outputs" / "reports" / "data_distribution.json"

CLASS_NAMES = {0: "Normalan (NORM)", 1: "Infarkt miokarda (MI)"}
CLASS_COLORS = {0: COLORS["primary"], 1: COLORS["mi"]}


@st.cache_data(show_spinner=False)
def load_test_signals() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(_DATA_PATH)
    return data["signals"], data["labels"]


@st.cache_data(show_spinner=False)
def load_distribution() -> dict:
    if _DIST_PATH.exists():
        return json.loads(_DIST_PATH.read_text())
    return {}


st.title("Istrazi EKG signale")
st.markdown(
    "Pregled stvarnih EKG snimaka iz PTB-XL test skupa. "
    "Svaki snimak sadrzi 12 odvoda, duzine 10 s (1 000 uzoraka pri 100 Hz). "
    "Binarni problem: **Normalan ritam (NORM)** vs. **Infarkt miokarda (MI)**."
)

dist = load_distribution()
if dist:
    splits = dist.get("splits", {})
    st.markdown("### Raspodjela skupa podataka")

    col_train, col_val, col_test = st.columns(3)
    for col, split_key, split_name in [
        (col_train, "train",      "Trening"),
        (col_val,   "val",        "Validacija"),
        (col_test,  "test_clean", "Test"),
    ]:
        info = splits.get(split_key, {})
        n, mi, norm = info.get("n", 0), info.get("mi", 0), info.get("norm", 0)
        fig = go.Figure(go.Bar(
            x=["NORM", "MI"],
            y=[norm, mi],
            marker_color=[COLORS["primary"], COLORS["mi"]],
            text=[str(norm), str(mi)],
            textposition="outside",
            width=0.5,
        ))
        fig.update_layout(
            title=f"{split_name} ({n:,} snimaka)",
            yaxis_range=[0, max(norm, mi) * 1.25],
            height=220,
            margin=dict(l=30, r=10, t=45, b=30),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=COLORS["bg_plot"],
            showlegend=False,
        )
        col.plotly_chart(fig, use_container_width=True)

    st.info(
        "PTB-XL je podijeljen po pacijentima na trening (fold 1-8), validacijski (fold 9) "
        "i test skup (fold 10). Omjer MI:NORM je priblizno 38:62 u svim podskupovima."
    )

st.markdown("---")

if not _DATA_PATH.exists():
    st.error(f"Podaci nisu pronadjeni: {_DATA_PATH}")
    st.stop()

signals, labels = load_test_signals()


st.sidebar.header("Pretraga signala")
klasa = st.sidebar.radio("Klasa signala", ["Svi", "Normalan (NORM)", "Infarkt miokarda (MI)"])
idx_map = {
    "Svi":                   list(range(len(labels))),
    "Normalan (NORM)":       [i for i, l in enumerate(labels) if l == 0],
    "Infarkt miokarda (MI)": [i for i, l in enumerate(labels) if l == 1],
}
available = idx_map[klasa]

odabrani = st.sidebar.selectbox(
    "Odaberite signal",
    options=available,
    format_func=lambda i: f"Signal {i+1} - {CLASS_NAMES[int(labels[i])]}",
)
odvod_idx = st.sidebar.selectbox(
    "Odvod",
    options=list(range(12)),
    format_func=lambda i: LEAD_NAMES[i],
    index=1,
)


signal_1d = signals[odabrani, odvod_idx]
klasa_str = CLASS_NAMES[int(labels[odabrani])]
boja      = CLASS_COLORS[int(labels[odabrani])]
odvod_ime = LEAD_NAMES[odvod_idx]

st.markdown(f"### Signal {odabrani + 1} - {klasa_str} | Odvod {odvod_ime}")

info_cols = st.columns(4)
info_cols[0].metric("Klasa",          klasa_str)
info_cols[1].metric("Trajanje",       f"{len(signal_1d) / FS:.1f} s")
info_cols[2].metric("Min amplituda",  f"{signal_1d.min():.4f}")
info_cols[3].metric("Max amplituda",  f"{signal_1d.max():.4f}")

fig = plot_ecg_signal(
    signal_1d, fs=FS,
    title=f"Signal {odabrani+1} - {klasa_str} | Odvod {odvod_ime}",
    color=boja,
)
fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

stat_cols = st.columns(2)
stat_cols[0].metric("Srednja vrijednost", f"{signal_1d.mean():.4f}")
stat_cols[1].metric("Std devijacija",     f"{signal_1d.std():.4f}")

if int(labels[odabrani]) == 1:
    st.info(
        "MI signal moze pokazivati karakteristicne EKG promjene: elevaciju ili depresiju "
        "ST-segmenta, patoloske Q-talase i promjene T-talasa. Ove promjene su razlog "
        "zasto automatska analiza EKG-a moze biti korisna kao podrska u dijagnostici."
    )
else:
    st.info(
        "Normalni EKG signali prikazuju pravilan sinusni ritam bez patoloskih promjena. "
        "Precizno razlikovanje normalnog od MI ritma je kljucno za smanjenje lazno "
        "pozitivnih dijagnoza (visoka specificnost modela)."
    )

st.markdown(f"---")
st.markdown(f"### Galerija - 6 nasumicnih signala ({klasa})")

rng         = np.random.default_rng(42)
preview_idx = rng.choice(available, size=min(6, len(available)), replace=False).tolist()
rows_grid   = [preview_idx[i : i + 2] for i in range(0, len(preview_idx), 2)]

for row_idxs in rows_grid:
    cols = st.columns(2)
    for col, idx in zip(cols, row_idxs):
        s = signals[idx, odvod_idx]
        l = int(labels[idx])
        fig_s = plot_ecg_signal(
            s, fs=FS,
            title=f"Signal {idx+1} - {CLASS_NAMES[l]} | {odvod_ime}",
            color=CLASS_COLORS[l],
        )
        fig_s.update_layout(height=200, margin=dict(l=20, r=10, t=30, b=20))
        col.plotly_chart(fig_s, use_container_width=True)

page_footer()
