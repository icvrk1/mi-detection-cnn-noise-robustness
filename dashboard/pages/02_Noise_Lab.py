from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "dashboard"))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.charts import (
    COLORS,
    LEAD_NAMES,
    SNR_VALUES,
    SNR_COLORS,
    NOISE_LABELS,
    page_footer,
    plot_ecg_signal,
)
from utils.download_assets import ensure_test_data, ensure_noisy_test_data


FS = 100
_DATA_PATH = _ROOT / "data" / "processed" / "test_clean.npz"

NOISE_DESCRIPTIONS = {
    "gaussian": "Bijeli Gaussov šum predstavlja generički oblik nasumične smetnje.",
    "baseline_wander": "Kolebanje osnovne linije je niskofrekventna smetnja najčešće povezana s disanjem i pomjeranjem pacijenta.",
    "muscle_artifact": "Mišićni artefakt nastaje zbog električne aktivnosti skeletnih mišića i često se javlja u kliničkim uslovima.",
    "electrode_motion": "Pomicanje elektrode uzrokuje nagle promjene signala zbog pomjeranja elektrode ili slabog kontakta s kožom.",
    "powerline": "Smetnja iz električne mreže predstavlja periodičnu smetnju, najčešće na 50 Hz u evropskim sistemima.",
}

CLASS_NAMES = {
    0: "Normalan (NORM)",
    1: "Infarkt miokarda (MI)",
}


@st.cache_data(show_spinner=False)
def load_npz_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["signals"], data["labels"]


def load_noisy_dataset(
    noise_type: str,
    snr_db: int | float,
) -> tuple[np.ndarray, np.ndarray, str]:
    noisy_path = ensure_noisy_test_data(noise_type, snr_db)
    noisy_signals, noisy_labels = load_npz_dataset(noisy_path)
    return noisy_signals, noisy_labels, noisy_path


st.title("Laboratorija šuma")
st.markdown(
    "Odaberite EKG signal, vrstu šuma i SNR vrijednost kako biste vidjeli kako šum "
    "mijenja oblik signala. Veći SNR znači čišći signal, dok niži SNR znači jače "
    "narušavanje signala."
)


# Učitavanje čistog test skupa
try:
    clean_path = ensure_test_data()
except Exception as e:
    st.error("Preuzimanje čistog test skupa sa GitHub Release-a nije uspjelo.")
    st.exception(e)
    st.stop()

if not _DATA_PATH.exists():
    st.error(f"Test skup nije pronađen: {_DATA_PATH}")
    st.stop()

signals, labels = load_npz_dataset(str(_DATA_PATH))


# Sidebar
st.sidebar.header("Podešavanja")

signal_idx = st.sidebar.selectbox(
    "Signal",
    options=list(range(len(labels))),
    format_func=lambda i: f"Signal {i + 1} - {CLASS_NAMES[int(labels[i])]}",
)

noise_label = st.sidebar.selectbox(
    "Vrsta šuma",
    list(NOISE_LABELS.values()),
)

noise_type = {v: k for k, v in NOISE_LABELS.items()}[noise_label]

snr_db = st.sidebar.select_slider(
    "SNR nivo (dB)",
    options=SNR_VALUES,
    value=6,
)

odvod_idx = st.sidebar.selectbox(
    "Odvod",
    options=list(range(12)),
    format_func=lambda i: LEAD_NAMES[i],
    index=1,
)

show_all_snr = st.sidebar.checkbox(
    "Prikaži poređenje svih SNR nivoa",
    value=False,
    help="Ova opcija može potrajati jer preuzima više zašumljenih test skupova.",
)


# Priprema čistog i zašumljenog signala
signal_1d = signals[signal_idx, odvod_idx].copy()
true_label = int(labels[signal_idx])
odvod_ime = LEAD_NAMES[odvod_idx]

try:
    noisy_signals, noisy_labels, noisy_path = load_noisy_dataset(noise_type, snr_db)
    noisy_1d = noisy_signals[signal_idx, odvod_idx].copy()
except Exception as e:
    st.error(
        "Nije moguće preuzeti ili učitati zašumljeni test skup. "
        "Provjeri da li release `v1.0-noisy-data` sadrži traženi `.npz` fajl."
    )
    st.exception(e)
    st.stop()


# Informacije o šumu i osnovne metrike signala
st.info(f"**{noise_label}** - {NOISE_DESCRIPTIONS[noise_type]}")

noise_only = noisy_1d - signal_1d

p_clean = float(np.mean(signal_1d ** 2))
p_noise = float(np.mean(noise_only ** 2))
p_noisy = float(np.mean(noisy_1d ** 2))

m1, m2, m3 = st.columns(3)
m1.metric("SNR nivo", f"{snr_db} dB")
m2.metric("Snaga čistog signala", f"{p_clean:.5f}")
m3.metric("Snaga dodanog šuma", f"{p_noise:.5f}")

st.caption(
    f"Prikazan je signal {signal_idx + 1}, odvod {odvod_ime}, "
    f"stvarna klasa: **{CLASS_NAMES[true_label]}**."
)


# Čist signal vs. zašumljeni signal
st.markdown("### Čist signal vs. zašumljeni signal")

col_clean, col_noisy = st.columns(2)

with col_clean:
    fig_c = plot_ecg_signal(
        signal_1d,
        fs=FS,
        title=f"Čist EKG | Odvod {odvod_ime}",
        color=COLORS["primary"],
    )
    st.plotly_chart(fig_c, use_container_width=True)

with col_noisy:
    fig_n = plot_ecg_signal(
        noisy_1d,
        fs=FS,
        title=f"{noise_label} | SNR = {snr_db} dB | Odvod {odvod_ime}",
        color=COLORS["warn"],
    )
    st.plotly_chart(fig_n, use_container_width=True)


# Prikaz samo dodanog šuma
st.markdown("### Dodani šum")

fig_noise = plot_ecg_signal(
    noise_only,
    fs=FS,
    title=f"Izdvojena komponenta šuma | {noise_label} | SNR = {snr_db} dB",
    color=COLORS["warn"],
)

fig_noise.update_layout(height=300)
st.plotly_chart(fig_noise, use_container_width=True)


# Poređenje svih SNR nivoa
if show_all_snr:
    st.markdown("### Efekat SNR nivoa na isti signal")

    fig_multi = go.Figure()

    t_clean = np.arange(len(signal_1d)) / FS

    fig_multi.add_trace(
        go.Scatter(
            x=t_clean,
            y=signal_1d,
            mode="lines",
            name="Čist signal",
            line=dict(color="white", width=1.5, dash="dot"),
        )
    )

    for snr, color in zip(SNR_VALUES, SNR_COLORS):
        try:
            snr_signals, _, _ = load_noisy_dataset(noise_type, snr)
            snr_signal_1d = snr_signals[signal_idx, odvod_idx].copy()

            t = np.arange(len(snr_signal_1d)) / FS

            fig_multi.add_trace(
                go.Scatter(
                    x=t,
                    y=snr_signal_1d,
                    mode="lines",
                    name=f"SNR = {snr} dB",
                    line=dict(color=color, width=1.0),
                )
            )
        except Exception as e:
            st.warning(f"Nije moguće učitati noisy skup za SNR = {snr} dB.")
            st.exception(e)

    fig_multi.update_layout(
        title=f"Poređenje svih SNR nivoa - {noise_label}",
        xaxis_title="Vrijeme (s)",
        yaxis_title="Amplituda (mV)",
        height=420,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["bg_plot"],
        legend=dict(x=1.01, y=1, xanchor="left"),
    )

    st.plotly_chart(fig_multi, use_container_width=True)
else:
    st.caption(
        "Poređenje svih SNR nivoa nije automatski uključeno jer zahtijeva preuzimanje "
        "više zašumljenih test skupova. Uključite opciju u bočnom meniju ako želite taj prikaz."
    )


st.info(
    "Pri SNR = -6 dB šum je jači od korisnog signala i može značajno maskirati "
    "dijagnostički važne obrasce. Pri SNR = 0 dB signal i šum imaju približno jednaku "
    "snagu, dok pri SNR vrijednostima od 12 dB i više korisni signal uglavnom dominira "
    "nad šumom. Zbog toga modeli obično pokazuju veći pad performansi pri nižim SNR nivoima."
)

page_footer()