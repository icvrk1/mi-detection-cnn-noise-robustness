
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "dashboard"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from ekg_mi.models.baseline_cnn import BaselineCNN
from components.charts import COLORS, page_footer

_MODEL_V1   = _ROOT / "outputs" / "models"   / "best_model.pt"
_ARCH_PATH  = _ROOT / "outputs" / "reports"  / "model_architecture.txt"
_HIST_V1    = _ROOT / "outputs" / "logs"     / "training_history.json"
_HIST_V3    = _ROOT / "outputs" / "logs"     / "training_history_v3.json"
_EVAL_V1    = _ROOT / "outputs" / "reports"  / "eval_clean.json"
_EVAL_V3    = _ROOT / "outputs" / "reports"  / "eval_clean_v3.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}



st.title("Pregled arhitekture modela")
st.markdown(
    "**BaselineCNN** je jednodimenzionalna konvolucijska neuronska mreza za binarnu "
    "klasifikaciju 12-kanalnih EKG signala (NORM vs. MI). "
    "Isti arhitekturalni dizajn koristi i V1 (bazni model) i V3 (model s augmentacijom suma)."
)


model = BaselineCNN(num_channels=12)
if _MODEL_V1.exists():
    model.load_state_dict(torch.load(_MODEL_V1, map_location="cpu", weights_only=True))
    st.success("Trenirani model V1 ucitan (best_model.pt).")
else:
    st.warning("Model nije pronadjen - prikazuju se nasumicno inicijalizirane tezine.")
model.eval()

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

p1, p2, p3 = st.columns(3)
p1.metric("Ukupno parametara",          f"{total_params:,}")
p2.metric("Trenabilnih parametara",     f"{trainable_params:,}")
p3.metric("Ulaznih kanala",             "12 (svi EKG odvodi)")

st.markdown("---")


st.markdown("### Arhitektura - konvolucijski blokovi")
st.markdown(
    "Svaki **ConvBlock** se sastoji od: `Conv1d` -> `BatchNorm1d` -> `ReLU` -> `MaxPool1d(2)`. "
    "Nakon cetiri bloka slijede globalni prosjek i potpuno povezani klasifikator."
)

arch_data = [
    ("Ulaz",       "12 x 1000",  "-",      "-",        "-"),
    ("ConvBlock 1","12 x 1000",  "32",     "16",       "6 240"),
    ("ConvBlock 2","32 x 500",   "64",     "5",        "10 432"),
    ("ConvBlock 3","64 x 250",   "128",    "5",        "41 344"),
    ("ConvBlock 4","128 x 125",  "256",    "5",        "164 608"),
    ("AdaptAvgPool","256 x 62",  "-",      "-",        "-"),
    ("Flatten + Dropout(0.5)","256", "-",  "-",        "-"),
    ("Linear(256, 64) + ReLU",  "256", "-","-",        "16 448"),
    ("Izlaz Linear(64, 1)",      "64", "-","-",        "65"),
]
df_arch = pd.DataFrame(arch_data,
    columns=["Sloj", "Ulazni oblik", "Filteri", "Kernel", "Parametri"])
st.dataframe(df_arch, use_container_width=True, hide_index=True)

st.info(
    "Gubitak: BCEWithLogitsLoss s automatskim pos_weight radi kompenzacije neravnomjerne "
    "raspodjele klasa. Optimizator: Adam (lr=0.001). "
    "Rani zaustavljac (patience=7) cuva model s najnizim val_loss."
)

# --- torchinfo / architecture text ---
if _ARCH_PATH.exists():
    st.markdown("### Detaljan prikaz slojeva (torchinfo)")
    with st.expander("Prikazi/sakrij", expanded=False):
        st.code(_ARCH_PATH.read_text(encoding="utf-8"), language="text")

# --- Data flow diagram ---
st.markdown("### Tok podataka kroz model")

nodes  = ["Ulaz\n(12x1000)", "Conv1\n(32x500)", "Conv2\n(64x250)",
          "Conv3\n(128x125)", "Conv4\n(256x62)", "AvgPool\n(256x1)",
          "Flatten\n(256)", "FC+ReLU\n(64)", "Logit\n(1)"]
x_pos  = list(range(len(nodes)))
colors_node = [COLORS["norm"]] + [COLORS["primary"]] * 7 + [COLORS["mi"]]

fig_flow = go.Figure()
for i in range(len(nodes) - 1):
    fig_flow.add_trace(go.Scatter(
        x=[x_pos[i] + 0.15, x_pos[i + 1] - 0.15], y=[0, 0],
        mode="lines", line=dict(color=COLORS["muted"], width=1.5),
        showlegend=False,
    ))
for i, (x, label, clr) in enumerate(zip(x_pos, nodes, colors_node)):
    fig_flow.add_trace(go.Scatter(
        x=[x], y=[0],
        mode="markers+text",
        marker=dict(size=24, color=clr),
        text=[label],
        textposition="bottom center",
        showlegend=False,
    ))
fig_flow.update_layout(
    height=190,
    margin=dict(l=20, r=20, t=10, b=80),
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[-0.6, 0.4]),
)
st.plotly_chart(fig_flow, use_container_width=True)

# --- V1 vs V3 training comparison ---
st.markdown("---")
st.markdown("### Poredjenje treninga: V1 vs V3")

h1 = _load_json(_HIST_V1)
h3 = _load_json(_HIST_V3)
e1 = _load_json(_EVAL_V1).get("metrics", {})
e3 = _load_json(_EVAL_V3).get("metrics", {})

if h1 and h3:
    cmp_rows = [
        ("Broj epoha",            len(h1.get("train_loss", [])),     len(h3.get("train_loss", []))),
        ("Najbolja epoha",        h1.get("best_epoch", "-"),          h3.get("best_epoch", "-")),
        ("Najb. val F1",          f"{max(h1.get('val_f1', [0])):.4f}", f"{max(h3.get('val_f1', [0])):.4f}"),
        ("Najb. val Loss",        f"{h1.get('best_val_loss', 0):.4f}", f"{h3.get('best_val_loss', 0):.4f}"),
        ("Test F1 (cist skup)",   f"{e1.get('f1', 0):.4f}",           f"{e3.get('f1', 0):.4f}"),
        ("Test AUC-ROC",          f"{e1.get('auc_roc', 0):.4f}",      f"{e3.get('auc_roc', 0):.4f}"),
        ("Test Osjetljivost",     f"{e1.get('recall', 0):.4f}",        f"{e3.get('recall', 0):.4f}"),
    ]
    df_cmp = pd.DataFrame(cmp_rows, columns=["Metrika", "V1 (bazni)", "V3 (augment.)"])
    st.dataframe(df_cmp, use_container_width=True, hide_index=True)

    if h3.get("aug_count"):
        avg_aug = sum(h3["aug_count"]) / len(h3["aug_count"])
        st.caption(
            f"V3 trening: prosjecno {avg_aug:.0f} augmentiranih uzoraka po epohi "
            f"(od ~{len(h3['train_loss'])} epoha ukupno)."
        )

    st.info(
        "Oba modela dijele istu arhitekturu (239 137 parametara). "
        "V3 trening je trajao jednu epohu duze zbog nesto sporije konvergencije uzrokovane "
        "sumom u trenaznim primjerima. Razlika u broju parametara ne postoji - razlika je "
        "iskljucivo u procesu treninga."
    )
else:
    st.info("Pokrenite skripte 02_train.py i 02_train_v3.py da biste vidjeli podatke o treningu.")

page_footer()
