from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "dashboard"))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from ekg_mi.models.baseline_cnn import BaselineCNN
from components.charts import (
    COLORS,
    LEAD_NAMES,
    SNR_VALUES,
    NOISE_LABELS,
    page_footer,
    plot_ecg_signal,
)
from utils.download_assets import (
    ensure_test_data,
    ensure_model_v1,
    ensure_model_v3,
    ensure_noisy_test_data,
)


FS = 100

_MODEL_V1_PATH = _ROOT / "outputs" / "models" / "best_model.pt"
_MODEL_V3_PATH = _ROOT / "outputs" / "models" / "best_model_v3.pt"
_DATA_PATH = _ROOT / "data" / "processed" / "test_clean.npz"

CLASS_NAMES = {
    0: "Normalan (NORM)",
    1: "Infarkt miokarda (MI)",
}

CLASS_COLORS = {
    0: COLORS["norm"],
    1: COLORS["mi"],
}

DEFAULT_THR = {
    "V1 (bazni model)": 0.5,
    "V3 (augmentirani model)": 0.51,
}


@st.cache_resource(show_spinner=False)
def _load_model(path: str) -> tuple[BaselineCNN, bool]:
    model = BaselineCNN(num_channels=12)
    model_path = Path(path)

    if not model_path.exists():
        model.eval()
        return model, False

    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model.load_state_dict(state_dict)
    model.eval()

    return model, True


@st.cache_data(show_spinner=False)
def _load_signals(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["signals"], data["labels"]


def _predict(
    model: BaselineCNN,
    signal_12ch: np.ndarray,
    threshold: float,
) -> tuple[int, float]:
    tensor = torch.from_numpy(signal_12ch).float().unsqueeze(0)

    with torch.no_grad():
        prob_mi = float(torch.sigmoid(model(tensor).squeeze()).item())

    pred = int(prob_mi >= threshold)
    return pred, prob_mi


st.title("Live predikcija")
st.markdown(
    "Odaberite EKG snimak iz stvarnog test skupa, opcijsku vrstu šuma "
    "i pokrenite klasifikaciju. Model vraća vjerovatnoću infarkta miokarda."
)

with st.expander("Koji model se koristi i kako radi predikcija?", expanded=False):
    st.markdown(
        """
**Koji model (run)?** Ova stranica koristi **jedan fiksni, reprezentativan istrenirani model**
po verziji — `outputs/models/best_model.pt` (V1) odnosno `best_model_v3.pt` (V3). To je
pojedinačan model (jedno slučajno sjeme), a **ne** prosjek 10 runova. Razlog je interaktivnost:
predikcija mora biti deterministička i trenutna za odabrani signal. Objedinjena analiza
performansi preko 10 nezavisnih treninga (srednja vrijednost ± standardna devijacija, p-vrijednosti) nalazi se na stranici
**„Multi-seed analiza“** — ona kvantifikuje pouzdanost, dok je ova stranica demonstracija rada
modela na pojedinačnom snimku.

**Na osnovu čega predikcija?** Model je 1D konvolucijska mreža koja iz 12-kanalnog signala
(oblik 12 × 1000) računa jedan **logit**. Na njega se primijeni **sigmoidna** funkcija da se
dobije vjerovatnoća pripadnosti klasi MI:

$$P(\\text{MI}) = \\sigma(\\text{logit}) = \\frac{1}{1 + e^{-\\text{logit}}}$$

Konačna odluka donosi se poređenjem sa **pragom** (prag = 0.5 za V1, 0.51 za V3):
ako je $P(\\text{MI}) \\geq \\text{prag}$ → klasa **MI**, inače **NORM**. Kada se doda šum,
isti model klasifikuje zašumljenu verziju istog snimka, pa se vidi kako šum utiče na odluku.
        """
    )


try:
    ensure_test_data()
    ensure_model_v1()
except Exception as e:
    st.error("Nije moguće preuzeti test podatke ili V1 model iz GitHub Release-a.")
    st.exception(e)
    st.stop()

try:
    ensure_model_v3()
except Exception:
    pass


if not _DATA_PATH.exists():
    st.error(f"Test skup nije pronađen: {_DATA_PATH}")
    st.stop()

if not _MODEL_V1_PATH.exists():
    st.error(f"V1 best model nije pronađen: {_MODEL_V1_PATH}")
    st.stop()

if not _MODEL_V3_PATH.exists():
    st.warning("V3 best model nije pronađen. Bit će dostupan samo V1 model.")


signals, labels = _load_signals(str(_DATA_PATH))


st.sidebar.header("Podešavanja")

model_options = ["V1 (bazni model)"]
if _MODEL_V3_PATH.exists():
    model_options.append("V3 (augmentirani model)")

model_choice = st.sidebar.radio("Verzija modela", model_options)

signal_idx = st.sidebar.selectbox(
    "Signal",
    options=list(range(len(labels))),
    format_func=lambda i: f"Signal {i + 1} - {CLASS_NAMES[int(labels[i])]}",
)

add_noise_flag = st.sidebar.checkbox("Dodaj šum", value=False)

noise_label = st.sidebar.selectbox(
    "Vrsta šuma",
    list(NOISE_LABELS.values()),
    disabled=not add_noise_flag,
)

snr_db = st.sidebar.select_slider(
    "SNR (dB)",
    options=SNR_VALUES,
    value=6,
    disabled=not add_noise_flag,
)

odvod_idx = st.sidebar.selectbox(
    "Prikazani odvod",
    options=list(range(12)),
    format_func=lambda i: LEAD_NAMES[i],
    index=1,
)


model_path = _MODEL_V3_PATH if "V3" in model_choice else _MODEL_V1_PATH
model, loaded = _load_model(str(model_path))
threshold = DEFAULT_THR.get(model_choice, 0.5)

if loaded:
    st.success(f"Model {model_choice} učitan. Prag klasifikacije = {threshold}.")
else:
    st.warning(
        f"Težine modela {model_choice} nisu pronađene. "
        "Koriste se nasumične težine, pa predikcija nije validna."
    )


noise_type = {v: k for k, v in NOISE_LABELS.items()}[noise_label]

signal_12ch = signals[signal_idx].copy()
true_label = int(labels[signal_idx])
odvod_ime = LEAD_NAMES[odvod_idx]

if add_noise_flag:
    try:
        noisy_path = ensure_noisy_test_data(noise_type, snr_db)
        noisy_signals, noisy_labels = _load_signals(str(noisy_path))

        signal_12ch = noisy_signals[signal_idx].copy()
        true_label = int(noisy_labels[signal_idx])

    except Exception as e:
        st.error(
            "Nije moguće preuzeti ili učitati zašumljeni test skup. "
            "Provjeri da li je release v1.0-noisy-data objavljen i da li sadrži traženi .npz fajl."
        )
        st.exception(e)
        st.stop()

    signal_title = f"Signal {signal_idx + 1} + {noise_label} | SNR = {snr_db} dB"
    sig_color = COLORS["warn"]
else:
    signal_title = f"Signal {signal_idx + 1} - {CLASS_NAMES[true_label]}"
    sig_color = CLASS_COLORS[true_label]


fig = plot_ecg_signal(
    signal_12ch[odvod_idx],
    fs=FS,
    title=f"{signal_title} | Odvod {odvod_ime}",
    color=sig_color,
)

fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Stvarna klasa signala: **{CLASS_NAMES[true_label]}** | "
    f"Model: {model_choice} | "
    f"Prag: {threshold}"
    + (f" | Šum: {noise_label} pri {snr_db} dB" if add_noise_flag else "")
)


if st.button("Pokreni klasifikaciju", type="primary", use_container_width=True):
    with st.spinner("Klasificiranje..."):
        pred, prob_mi = _predict(model, signal_12ch, threshold)
        prob_norm = 1.0 - prob_mi

    st.markdown("---")

    res_col, bar_col = st.columns([1, 2])

    with res_col:
        st.markdown("#### Rezultat klasifikacije")

        pred_label = CLASS_NAMES[pred]
        pred_color = CLASS_COLORS[pred]

        st.markdown(
            f"<h2 style='color:{pred_color}; margin-bottom:4px;'>{pred_label}</h2>",
            unsafe_allow_html=True,
        )

        correct = pred == true_label

        if correct:
            st.success("Tačna predikcija")
        else:
            st.error(
                f"Pogrešna predikcija. "
                f"Stvarna klasa: **{CLASS_NAMES[true_label]}**"
            )

        confidence = max(prob_mi, prob_norm)

        if confidence >= 0.9:
            conf_label = "Visoko uvjerljiva predikcija"
        elif confidence >= 0.75:
            conf_label = "Umjereno uvjerljiva predikcija"
        else:
            conf_label = "Niska uvjerenost, graničan slučaj"

        st.caption(conf_label)

    with bar_col:
        st.markdown("#### Vjerovatnoća klasa")

        fig_bar = go.Figure(
            go.Bar(
                x=["Normalan (NORM)", "Infarkt miokarda (MI)"],
                y=[prob_norm, prob_mi],
                marker_color=[COLORS["norm"], COLORS["mi"]],
                text=[f"{prob_norm:.1%}", f"{prob_mi:.1%}"],
                textposition="outside",
                width=0.45,
            )
        )

        fig_bar.update_layout(
            yaxis=dict(range=[0, 1.2], tickformat=".0%"),
            height=270,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=COLORS["bg_plot"],
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=40),
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### Detalji")

    d1, d2, d3, d4, d5 = st.columns(5)

    d1.metric("Predikcija", pred_label)
    d2.metric("Model", model_choice.split()[0])
    d3.metric("Prag", f"{threshold}")
    d4.metric("P(NORM)", f"{prob_norm:.4f}")
    d5.metric("P(MI)", f"{prob_mi:.4f}")

    if add_noise_flag and not correct:
        st.warning(
            f"Pogrešna predikcija pri šumu '{noise_label}' i SNR={snr_db} dB. "
            "Pokušajte viši SNR nivo za čišći signal ili odaberite V3 model."
        )
    elif add_noise_flag and correct:
        st.info(
            f"Tačna predikcija pri šumu '{noise_label}' i SNR={snr_db} dB. "
            "Za detaljnu analizu pogledajte stranicu 'Robustness'."
        )

page_footer()