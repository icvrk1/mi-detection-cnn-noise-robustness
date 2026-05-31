
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
_AGG_V1     = _ROOT / "outputs" / "reports"  / "aggregate_v1.json"
_AGG_V3     = _ROOT / "outputs" / "reports"  / "aggregate_v3.json"
_AGG_CMP    = _ROOT / "outputs" / "reports"  / "aggregate_comparison.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _ms(d: dict, digits: int = 4) -> str:
    if not d or d.get("mean") is None:
        return "-"
    return f"{d['mean']:.{digits}f} ± {d['std']:.{digits}f}"


def _stars(p) -> str:
    if p is None:
        return ""
    if p < 0.001: return " ***"
    if p < 0.01:  return " **"
    if p < 0.05:  return " *"
    return ""



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

# --- V1 vs V3 training comparison (objedinjeni rezultati, N=10) ---
st.markdown("---")
st.markdown("### Poređenje treninga i performansi: V1 vs V3")

agg1 = _load_json(_AGG_V1)
agg3 = _load_json(_AGG_V3)
cmp  = _load_json(_AGG_CMP)

if agg1 and agg3 and cmp:
    n1, n3 = agg1["n_runs"], agg3["n_runs"]
    st.success(
        f"Vrijednosti su objedinjene iz **{n1} (V1)** i **{n3} (V3)** nezavisnih treninga "
        f"sa različitim slučajnim sjemenima — prikazane kao srednja vrijednost ± standardna "
        f"devijacija. P-vrijednost dolazi iz uparenog t-testa (V1 vs V3 po istom sjemenu)."
    )
    t1, t3 = agg1["training"], agg3["training"]
    c1, c3 = agg1["clean"], agg3["clean"]
    cc = cmp["clean"]

    def _row(name, v1s, v3s, p=None):
        return {"Metrika": name, "V1 (bazni)": v1s, "V3 (augment.)": v3s,
                "p (V1 vs V3)": ("-" if p is None else f"{p:.3f}{_stars(p)}")}

    cmp_rows = [
        _row("Broj epoha",        _ms(t1["n_epochs_run"], 1), _ms(t3["n_epochs_run"], 1)),
        _row("Najbolja epoha",    _ms(t1["best_epoch"], 1),   _ms(t3["best_epoch"], 1)),
        _row("Najb. val F1",      _ms(t1["best_val_f1"]),     _ms(t3["best_val_f1"])),
        _row("Najb. val Loss",    _ms(t1["best_val_loss"]),   _ms(t3["best_val_loss"])),
        _row("Test F1 (čist)",    _ms(c1["f1"]),      _ms(c3["f1"]),      cc["f1"].get("t_p")),
        _row("Test AUC-ROC",      _ms(c1["auc_roc"]), _ms(c3["auc_roc"]), cc["auc_roc"].get("t_p")),
        _row("Test Osjetljivost", _ms(c1["recall"]),  _ms(c3["recall"]),  cc["recall"].get("t_p")),
        _row("Test Specifičnost", _ms(c1["specificity"]), _ms(c3["specificity"]), cc["specificity"].get("t_p")),
    ]
    st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)
    st.caption("Oznake značajnosti: * p<0.05, ** p<0.01, *** p<0.001.")

    st.info(
        "Oba modela dijele istu arhitekturu (239 137 parametara) — razlika je isključivo u "
        "procesu treninga (V3 koristi augmentaciju šumom). Na čistom test skupu V1 i V3 su "
        "**statistički ekvivalentni** (p > 0.05 za sve metrike), tj. augmentacija ne narušava "
        "performanse na čistim signalima. Prednost V3 vidljiva je tek pod šumom "
        "(stranica „Analiza robusnosti“ i „Multi-seed analiza“)."
    )
else:
    # fallback na single-run ako agregati ne postoje
    h1 = _load_json(_HIST_V1); h3 = _load_json(_HIST_V3)
    e1 = _load_json(_EVAL_V1).get("metrics", {}); e3 = _load_json(_EVAL_V3).get("metrics", {})
    if h1 and h3:
        cmp_rows = [
            ("Broj epoha", len(h1.get("train_loss", [])), len(h3.get("train_loss", []))),
            ("Test F1 (cist skup)", f"{e1.get('f1', 0):.4f}", f"{e3.get('f1', 0):.4f}"),
            ("Test AUC-ROC", f"{e1.get('auc_roc', 0):.4f}", f"{e3.get('auc_roc', 0):.4f}"),
        ]
        st.dataframe(pd.DataFrame(cmp_rows, columns=["Metrika", "V1 (bazni)", "V3 (augment.)"]),
                     use_container_width=True, hide_index=True)
        st.warning("Objedinjeni rezultati (N=10) nisu pronađeni — prikazan je pojedinačni run. "
                   "Pokreni `scripts/11_aggregate_runs.py` za objedinjene vrijednosti.")
    else:
        st.info("Pokrenite skripte 02_train.py i 02_train_v3.py da biste vidjeli podatke o treningu.")

page_footer()
