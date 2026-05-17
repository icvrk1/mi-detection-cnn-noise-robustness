
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "dashboard"))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ekg_mi.noise.injection import add_noise
from components.charts import (
    COLORS, LEAD_NAMES, SNR_VALUES, SNR_COLORS,
    NOISE_LABELS, page_footer, plot_ecg_signal,
)

FS = 100
_DATA_PATH = _ROOT / "data" / "processed" / "test_clean.npz"

NOISE_DESCRIPTIONS = {
    "gaussian":         "Bijeli sum ravnomjerno rasporedjenih frekvencija - modelira elektronski sum senzora.",
    "baseline_wander":  "Niskofrekvencijsko kretanje bazne linije uzrokovano disanjem pacijenta.",
    "muscle_artifact":  "Visokofrekv. mioelektricna aktivnost misica - najcesci artefakt u klinickoj praksi.",
    "electrode_motion": "Nagle promjene signala zbog pomicanja elektrode ili kontaktnih problema.",
    "powerline":        "Smetnja elektricne mreze na 50 Hz - tipicna u slabo zastitenim okruzenjima.",
}
CLASS_NAMES = {0: "Normalan (NORM)", 1: "Infarkt miokarda (MI)"}


@st.cache_data(show_spinner=False)
def load_test_signals() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(_DATA_PATH)
    return data["signals"], data["labels"]


@st.cache_data(show_spinner=False)
def _noisy(signal_1d: np.ndarray, noise_type: str, snr_db: float) -> np.ndarray:
    return add_noise(signal_1d, noise_type=noise_type, snr_db=snr_db)


st.title("Laboratorija suma")
st.markdown(
    "Dodajte razlicite vrste suma na EKG signal i vizualizirajte kako "
    "**omjer signal-sum (SNR)** utice na oblik signala. "
    "Visi SNR znaci cistiji signal; nizi SNR znaci vise suma."
)

if not _DATA_PATH.exists():
    st.error(f"Podaci nisu pronadjeni: {_DATA_PATH}")
    st.stop()

signals, labels = load_test_signals()

# --- Sidebar ---
st.sidebar.header("Podesavanja")
signal_idx = st.sidebar.selectbox(
    "Signal",
    options=list(range(len(labels))),
    format_func=lambda i: f"Signal {i+1} - {CLASS_NAMES[int(labels[i])]}",
)
noise_label = st.sidebar.selectbox("Vrsta suma", list(NOISE_LABELS.values()))
noise_type  = {v: k for k, v in NOISE_LABELS.items()}[noise_label]
snr_db      = st.sidebar.select_slider("SNR nivo (dB)", options=SNR_VALUES, value=6)
odvod_idx   = st.sidebar.selectbox(
    "Odvod",
    options=list(range(12)),
    format_func=lambda i: LEAD_NAMES[i],
    index=1,
)

signal_1d = signals[signal_idx, odvod_idx]
odvod_ime = LEAD_NAMES[odvod_idx]

with st.spinner("Primjena suma..."):
    noisy = _noisy(signal_1d, noise_type=noise_type, snr_db=float(snr_db))


st.info(f"**{noise_label}** - {NOISE_DESCRIPTIONS[noise_type]}")


p_clean = float(np.mean(signal_1d ** 2))
p_noisy = float(np.mean(noisy    ** 2))
m1, m2, m3 = st.columns(3)
m1.metric("SNR nivo",               f"{snr_db} dB")
m2.metric("Snaga cistog signala",   f"{p_clean:.5f}")
m3.metric("Snaga zasumljenog sig.", f"{p_noisy:.5f}")


st.markdown("### Cisti signal vs. zasumljeni signal")
col_clean, col_noisy = st.columns(2)

with col_clean:
    fig_c = plot_ecg_signal(
        signal_1d, fs=FS,
        title=f"Cisti EKG | Odvod {odvod_ime}",
        color=COLORS["primary"],
    )
    st.plotly_chart(fig_c, use_container_width=True)

with col_noisy:
    fig_n = plot_ecg_signal(
        noisy, fs=FS,
        title=f"{noise_label} | SNR = {snr_db} dB | Odvod {odvod_ime}",
        color=COLORS["warn"],
    )
    st.plotly_chart(fig_n, use_container_width=True)


st.markdown("### Efekat SNR nivoa na isti signal")
fig_multi = go.Figure()
fig_multi.add_trace(go.Scatter(
    x=np.arange(len(signal_1d)) / FS, y=signal_1d,
    mode="lines", name="Cisti signal",
    line=dict(color="white", width=1.5, dash="dot"),
))
for snr, color in zip(SNR_VALUES, SNR_COLORS):
    n = _noisy(signal_1d, noise_type=noise_type, snr_db=float(snr))
    t = np.arange(len(n)) / FS
    fig_multi.add_trace(go.Scatter(
        x=t, y=n, mode="lines",
        name=f"SNR = {snr} dB",
        line=dict(color=color, width=1.0),
    ))
fig_multi.update_layout(
    title=f"Poredenje pri svim SNR nivoima - {noise_label}",
    xaxis_title="Vrijeme (s)",
    yaxis_title="Amplituda (mV)",
    height=380,
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=COLORS["bg_plot"],
    legend=dict(x=1.01, y=1, xanchor="left"),
)
st.plotly_chart(fig_multi, use_container_width=True)

st.info(
    "Pri SNR = -6 dB sum dominira nad signalom i moze u potpunosti maskirati klinicke "
    "karakteristike kao sto su ST-elevacija ili Q-talasi. "
    "Na SNR >= 12 dB signal je vidno cist i model moze pouzdano klasificirati. "
    "Ovo objasnjava znacajan pad performansi modela pri niskim SNR nivoima - vise detalja "
    "na stranici 'Analiza robustnosti'."
)

page_footer()
