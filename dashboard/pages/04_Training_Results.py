
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "dashboard"))

import numpy as np
import pandas as pd
import streamlit as st

from components.charts import COLORS, page_footer, plot_training_curves

_HIST_V1 = _ROOT / "outputs" / "logs"    / "training_history.json"
_HIST_V3 = _ROOT / "outputs" / "logs"    / "training_history_v3.json"
_EVAL_V1 = _ROOT / "outputs" / "reports" / "eval_clean.json"
_EVAL_V3 = _ROOT / "outputs" / "reports" / "eval_clean_v3.json"
_AGG_V1  = _ROOT / "outputs" / "reports" / "aggregate_v1.json"
_AGG_V3  = _ROOT / "outputs" / "reports" / "aggregate_v3.json"


def _ms(d: dict, digits: int = 4) -> str:
    if not d or d.get("mean") is None:
        return "-"
    return f"{d['mean']:.{digits}f} ± {d['std']:.{digits}f}"


@st.cache_data(show_spinner=False)
def _load(hist_path: str, eval_path: str) -> tuple[dict, dict]:
    history = json.loads(Path(hist_path).read_text())
    full    = json.loads(Path(eval_path).read_text())
    return history, full


def _confusion_fig(tp: int, tn: int, fp: int, fn: int, title: str):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    fig.patch.set_alpha(0)
    ax.set_facecolor("#1A2332")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["NORM", "MI"],
        yticklabels=["NORM", "MI"],
        ax=ax,
        annot_kws={"color": "white", "size": 13},
    )
    ax.set_xlabel("Predikcija",    color="white")
    ax.set_ylabel("Stvarna klasa", color="white")
    ax.tick_params(colors="white")
    ax.set_title(title, color="white", fontsize=11)
    fig.tight_layout()
    return fig


def _render_model_tab(
    hist_path: Path | None,
    eval_path: Path | None,
    model_name: str,
    preloaded: tuple[dict, dict] | None = None,
) -> None:

    if preloaded is not None:
        history, full = preloaded
    elif hist_path is None or not hist_path.exists() or eval_path is None or not eval_path.exists():
        st.warning(f"Fajlovi za {model_name} nisu pronadjeni.")
        return
    else:
        history, full = _load(str(hist_path), str(eval_path))
    metrics  = full["metrics"]
    cm_dict  = full["confusion_matrix"]
    n_epochs = len(history["train_loss"])
    best_ep  = history.get("best_epoch")

    # Key metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tacnost",      f"{metrics['accuracy']:.2%}")
    c2.metric("F1 mjera",     f"{metrics['f1']:.4f}")
    c3.metric("AUC-ROC",      f"{metrics['auc_roc']:.4f}")
    c4.metric("Osjetljivost", f"{metrics['recall']:.2%}")
    c5.metric("Specificnost", f"{metrics['specificity']:.2%}")

    if best_ep:
        st.caption(
            f"Reprezentativan run: {n_epochs} epoha - model sačuvan na epohi {best_ep} "
            f"(val_loss = {history.get('best_val_loss', 0):.4f}). "
            f"Metrike i matrica konfuzije ispod odnose se na ovaj pojedinačni run; "
            f"konačne objedinjene vrijednosti (srednja vrijednost ± standardna devijacija) date su u banneru na vrhu stranice."
        )

    # Training curves - 2 columns
    curve_col1, curve_col2 = st.columns(2)
    with curve_col1:
        st.markdown("**Kriva gubitka (BCE Loss)**")
        fig_loss = plot_training_curves(
            history, "loss", "Gubitak (BCE)",
            train_color=COLORS["primary"], val_color=COLORS["mi"],
        )
        st.plotly_chart(fig_loss, use_container_width=True)
    with curve_col2:
        st.markdown("**Kriva F1 mjere**")
        fig_f1 = plot_training_curves(
            history, "f1", "F1 mjera",
            train_color=COLORS["norm"], val_color=COLORS["warn"],
        )
        fig_f1.update_layout(yaxis=dict(range=[0.8, 1.0]))
        st.plotly_chart(fig_f1, use_container_width=True)

    # Learning rate info if available
    if "lr" in history:
        unique_lr = sorted(set(history["lr"]), reverse=True)
        if len(unique_lr) > 1:
            st.caption(
                f"Stopa ucenja: pocetna {unique_lr[0]}, smanjena na "
                + ", ".join(f"{lr}" for lr in unique_lr[1:])
                + " (ReduceLROnPlateau)."
            )

    # Confusion matrix + metrics table
    cm_col, tbl_col = st.columns([1, 1])
    with cm_col:
        st.markdown("**Matrica konfuzije- cist test skup**")
        fig_cm = _confusion_fig(
            cm_dict["tp"], cm_dict["tn"], cm_dict["fp"], cm_dict["fn"],
            title=model_name,
        )
        st.pyplot(fig_cm)
        import matplotlib.pyplot as plt
        plt.close(fig_cm)

    with tbl_col:
        st.markdown("**Sve metrike - cist test skup**")
        rows = [
            {"Metrika": "Tacnost",       "Vrijednost": f"{metrics['accuracy']:.4f}"},
            {"Metrika": "Preciznost",    "Vrijednost": f"{metrics['precision']:.4f}"},
            {"Metrika": "Osjetljivost",  "Vrijednost": f"{metrics['recall']:.4f}"},
            {"Metrika": "Specificnost",  "Vrijednost": f"{metrics['specificity']:.4f}"},
            {"Metrika": "F1 mjera",      "Vrijednost": f"{metrics['f1']:.4f}"},
            {"Metrika": "AUC-ROC",       "Vrijednost": f"{metrics['auc_roc']:.4f}"},
            {"Metrika": "AUC-PR",        "Vrijednost": f"{metrics['auc_pr']:.4f}"},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        tp = cm_dict["tp"]; tn = cm_dict["tn"]
        fp = cm_dict["fp"]; fn = cm_dict["fn"]
        n_total = tp + tn + fp + fn
        st.caption(
            f"Test skup: {n_total} snimaka - "
            f"TP={tp}, TN={tn}, FP={fp}, FN={fn}."
        )


# --- Page header ---
st.title("Rezultati treninga")
st.markdown(
    "Krive gubitka i F1 mjere tokom treninga, matrica konfuzije i sve klasifikacijske "
    "metrike na čistom test skupu za oba modela."
)

# --- Objedinjeni sažetak (N=10) ---
_agg1 = json.loads(_AGG_V1.read_text()) if _AGG_V1.exists() else {}
_agg3 = json.loads(_AGG_V3.read_text()) if _AGG_V3.exists() else {}
if _agg1 and _agg3:
    n1, n3 = _agg1["n_runs"], _agg3["n_runs"]
    st.success(
        f"Konačne metrike objedinjene su kroz **{n1} (V1)** i **{n3} (V3)** nezavisnih treninga "
        f"(srednja vrijednost ± standardna devijacija). Krive treninga u karticama ispod prikazuju "
        f"**jedan reprezentativan run** radi čitljivosti; krive sa pojasom standardne devijacije "
        f"nalaze se na stranici „Multi-seed analiza“."
    )
    cc1, cc3 = _agg1["clean"], _agg3["clean"]
    cols = st.columns(4)
    cols[0].metric("V1 F1 (sr.±st.dev.)",  _ms(cc1["f1"]))
    cols[1].metric("V1 AUC-ROC",           _ms(cc1["auc_roc"]))
    cols[2].metric("V3 F1 (sr.±st.dev.)",  _ms(cc3["f1"]))
    cols[3].metric("V3 AUC-ROC",           _ms(cc3["auc_roc"]))
    st.markdown("---")

# Download both V1 and V3 training files up front so v3_available is accurate
if not _HIST_V1.exists() or not _EVAL_V1.exists():
    try:
        from utils.download_assets import ensure_training_results
        ensure_training_results()
    except Exception as e:
        st.error("Preuzimanje rezultata treninga sa GitHub Release-a nije uspjelo.")
        st.exception(e)

v3_available = _HIST_V3.exists() and _EVAL_V3.exists()
v1_available = _HIST_V1.exists() and _EVAL_V1.exists()

if v3_available:
    tab1, tab2 = st.tabs(["V1 - Bazni model", "V3 - Augmentirani model"])
    with tab1:
        _render_model_tab(_HIST_V1, _EVAL_V1, "V1 - bazni model")
        st.info(
            "V1 je treniran iskljucivo na cistim signalima. "
            "Validacijska kriva gubitka prati trening krivu bez znakova prekomjernog "
            "prilagodjivanja (overfitting-a), sto potvrdjuje dobru generalnost modela."
        )
    with tab2:
        _render_model_tab(_HIST_V3, _EVAL_V3, "V3 - augmentirani model")
        st.info(
            "V3 je treniran s nasumicnom augmentacijom suma (vjerovatnoca 50%). "
            "Trening konvergira neznatno sporije, ali postize bolju osjetljivost "
            "(recall) na cistom test skupu, sto je kljucno za detekciju MI."
        )
elif v1_available:
    _render_model_tab(_HIST_V1, _EVAL_V1, "V1 - bazni model")
    st.info(
        "Validacijska kriva gubitka prati trening krivu bez znakova prekomjernog "
        "prilagodjivanja, sto potvrdjuje dobru generalnost modela na nevidjenim podacima."
    )
else:
    st.info(
        "Demo rezim: fajlovi treninga nisu dostupni. "
        "Prikazuju se simulirani rezultati za demonstraciju."
    )
    from mock_data import get_mock_training_history, get_mock_clean_eval
    _render_model_tab(None, None, "V1 - bazni model (demo)",
                      preloaded=(get_mock_training_history(), get_mock_clean_eval()))
    st.info(
        "Validacijska kriva gubitka prati trening krivu bez znakova prekomjernog "
        "prilagodjivanja, sto potvrdjuje dobru generalnost modela na nevidjenim podacima."
    )

page_footer()
