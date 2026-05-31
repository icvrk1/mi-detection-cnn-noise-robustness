"""
EKG-MI Dashboard - main Streamlit multi-page entry point.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "dashboard"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="EKG-MI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

_REPORTS = _ROOT / "outputs" / "reports"
_AGG_V1  = _REPORTS / "aggregate_v1.json"
_AGG_V3  = _REPORTS / "aggregate_v3.json"


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _fmt_ms(d: dict, digits: int = 4) -> str:
    if not d or d.get("mean") is None:
        return "—"
    return f"{d['mean']:.{digits}f} ± {d['std']:.{digits}f}"


def _snr0_avg(noisy: dict) -> float:
    vals = [noisy[nt]["snr_0"]["f1"] for nt in noisy if "snr_0" in noisy[nt]]
    return sum(vals) / len(vals) if vals else 0.0


def landing() -> None:
    st.title("EKG-MI: Detekcija infarkta miokarda iz EKG signala")
    st.markdown(
        "Automatska detekcija infarkta miokarda (IM) iz 12-kanalnog EKG signala "
        "pomocu jednodimenzionalne konvolucijske neuronske mreze (1D CNN) trenirane na "
        "PTB-XL skupu podataka. Analiza robustnosti obuhvata pet vrsta suma pri sest SNR nivoa."
    )

    metrics_v1 = _load_json(_REPORTS / "eval_clean.json").get("metrics", {})
    metrics_v3 = _load_json(_REPORTS / "eval_clean_v3.json").get("metrics", {})
    noisy_v1   = _load_json(_REPORTS / "eval_noisy.json")
    noisy_v3   = _load_json(_REPORTS / "eval_noisy_v3.json")
    dist       = _load_json(_REPORTS / "data_distribution.json")
    agg_v1     = _load_json(_AGG_V1)
    agg_v3     = _load_json(_AGG_V3)

    # --- Multi-seed objedinjeni banner ---
    if agg_v1 and agg_v3:
        n1 = agg_v1.get("n_runs", 0); n3 = agg_v3.get("n_runs", 0)
        st.success(
            f"Objedinjeni rezultati dostupni: V1 ({n1} runova), V3 ({n3} runova). "
            f"Za potpunu analizu sa srednjim vrijednostima i p-vrijednostima, "
            f"vidi stranicu **Multi-seed analiza** u meniju."
        )
        cln_v1 = agg_v1.get("clean", {})
        st.markdown(f"### Bazni model (V1) - cist test skup (srednja vrijednost ± standardna devijacija, n={n1})")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Tacnost",      _fmt_ms(cln_v1.get("accuracy"),    4))
        c2.metric("F1 mjera",     _fmt_ms(cln_v1.get("f1"),          4))
        c3.metric("AUC-ROC",      _fmt_ms(cln_v1.get("auc_roc"),     4))
        c4.metric("Osjetljivost", _fmt_ms(cln_v1.get("recall"),      4))
        c5.metric("Specificnost", _fmt_ms(cln_v1.get("specificity"), 4))
        c6.metric("AUC-PR",       _fmt_ms(cln_v1.get("auc_pr"),      4))
    elif metrics_v1:
        st.markdown("### Performanse baznog modela (V1) - cist test skup")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Tacnost",      f"{metrics_v1['accuracy']:.2%}")
        c2.metric("F1 mjera",     f"{metrics_v1['f1']:.4f}")
        c3.metric("AUC-ROC",      f"{metrics_v1['auc_roc']:.4f}")
        c4.metric("Osjetljivost", f"{metrics_v1['recall']:.2%}")
        c5.metric("Specificnost", f"{metrics_v1['specificity']:.2%}")
        c6.metric("AUC-PR",       f"{metrics_v1['auc_pr']:.4f}")
    else:
        st.warning("Metrike nisu dostupne. Pokrenite skripte 02 i 03.")

    st.markdown("---")

    # --- Dataset overview ---
    if dist:
        st.markdown("### Skup podataka - PTB-XL")
        splits = dist.get("splits", {})
        train_info = splits.get("train",      {})
        val_info   = splits.get("val",        {})
        test_info  = splits.get("test_clean", {})

        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("Ukupno (validan)", f"{dist.get('valid_total', 0):,}")
        dc2.metric("Trening",          f"{train_info.get('n', 0):,} snimaka")
        dc3.metric("Validacija",       f"{val_info.get('n', 0):,} snimaka")
        dc4.metric("Test",             f"{test_info.get('n', 0):,} snimaka")

        # Class distribution bar chart - test set
        col_chart, col_info = st.columns([2, 1])
        with col_chart:
            fig_dist = go.Figure(go.Bar(
                x=["Normalan (NORM)", "Infarkt miokarda (MI)"],
                y=[test_info.get("norm", 0), test_info.get("mi", 0)],
                marker_color=["#4C9BE8", "#E84C6A"],
                text=[str(test_info.get("norm", 0)), str(test_info.get("mi", 0))],
                textposition="outside",
                width=0.45,
            ))
            fig_dist.update_layout(
                title="Raspodjela klasa - test skup",
                yaxis_title="Broj snimaka",
                yaxis_range=[0, 1050],
                height=260,
                margin=dict(l=40, r=20, t=50, b=40),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(26,35,50,0.6)",
                showlegend=False,
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        with col_info:
            st.markdown("**Omjer klasa (test skup)**")
            n_mi   = test_info.get("mi",   0)
            n_norm = test_info.get("norm", 0)
            n_tot  = n_mi + n_norm
            if n_tot:
                st.metric("MI klasa",     f"{n_mi}/{n_tot} ({n_mi/n_tot:.1%})")
                st.metric("NORM klasa",   f"{n_norm}/{n_tot} ({n_norm/n_tot:.1%})")
            st.markdown(
                "PTB-XL sadrzi 12-kanalne EKG signale duzine 10 s "
                "(1000 uzoraka, 100 Hz). Binarni problem: "
                "Normalan ritam vs. Infarkt miokarda."
            )

    st.markdown("---")

    # --- V1 vs V3 comparison ---
    v3_available = metrics_v3 and noisy_v3
    if v3_available:
        st.markdown("### Poređenje modela: V1 (bazni) vs V3 (augmentiran sumom)")
        st.markdown(
            "V1 je bazni model treniran iskljucivo na cistim signalima. "
            "V3 koristi isti model i isti trening postupak, ali s "
            "augmentacijom suma tokom treninga (vjerovatnoca augmentacije = 50%). "
            "Oba modela su evaluirana na identicnom test skupu od 1 462 snimka."
        )

        metric_defs = [
            ("accuracy",    "Tacnost"),
            ("f1",          "F1 mjera"),
            ("auc_roc",     "AUC-ROC"),
            ("recall",      "Osjetljivost"),
            ("specificity", "Specificnost"),
            ("auc_pr",      "AUC-PR"),
        ]
        rows = []
        for key, name in metric_defs:
            v1_val = metrics_v1.get(key, 0)
            v3_val = metrics_v3.get(key, 0)
            delta  = v3_val - v1_val
            rows.append({
                "Metrika":       name,
                "V1 (bazni)":    f"{v1_val:.4f}",
                "V3 (augment.)": f"{v3_val:.4f}",
                "Delta":         f"{delta:+.4f}",
                "Bolji model":   "V3" if delta > 0 else ("V1" if delta < 0 else "="),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Noisy comparison at SNR=0 dB (hardest realistic condition)
        if noisy_v1 and noisy_v3:
            avg_v1_snr0 = _snr0_avg(noisy_v1)
            avg_v3_snr0 = _snr0_avg(noisy_v3)
            r1, r2, r3 = st.columns(3)
            r1.metric("F1 (SNR=0 dB, prosjek) - V1", f"{avg_v1_snr0:.4f}")
            r2.metric("F1 (SNR=0 dB, prosjek) - V3", f"{avg_v3_snr0:.4f}",
                      delta=f"{avg_v3_snr0 - avg_v1_snr0:+.4f}")
            r3.metric("Poboljsanje pri SNR=0 dB",
                      f"{(avg_v3_snr0 - avg_v1_snr0) * 100:+.1f} pp")

        st.info(
            "Augmentacija sumom tokom treninga poboljsava robustnost modela pri niskim SNR nivoima, "
            "posebno za bazno lutanje i misicne artefakte (Delta F1 ~ +0.10 pri SNR=0 dB). "
            "Performanse na cistom test skupu ostaju gotovo identicne (Delta F1 = +0.0053), "
            "sto potvrdjuje da augmentacija ne steti generalnoj klasifikacijskoj sposobnosti."
        )
    elif metrics_v3:
        st.info("Noisy evaluacija za V3 nije dostupna. Pokrenite skriptu 05_evaluate_noisy_v3.py.")

    # --- Navigation ---
    st.markdown("---")
    st.markdown("### Navigacija")
    st.markdown(
        """
| Stranica | Opis |
|---|---|
| **Istrazi signale** | Pregled EKG snimaka iz PTB-XL test skupa, raspodjela klasa po podskupovima |
| **Laboratorija suma** | Interaktivno dodavanje suma na signal, vizualizacija efekta SNR-a |
| **Pregled modela** | Arhitektura 1D CNN modela, broj parametara, poredjenje V1 i V3 |
| **Rezultati treninga** | Krive gubitka i F1 mjere tokom treninga - V1 i V3 |
| **Analiza robustnosti** | Performanse modela pri razlicitim vrstama suma i SNR nivoima |
| **Live predikcija** | Interaktivna klasifikacija odabranog EKG signala sa ili bez suma |
        """
    )

    st.markdown("---")
    st.caption("EKG-MI Dashboard | Zavrsni rad | Ilma Cvrk | 2026")


pages = [
    st.Page(landing, title="Pocetna"),
    st.Page(_ROOT / "dashboard" / "pages" / "01_Signal_Explorer.py",   title="Istrazi signale"),
    st.Page(_ROOT / "dashboard" / "pages" / "02_Noise_Lab.py",          title="Laboratorija suma"),
    st.Page(_ROOT / "dashboard" / "pages" / "03_Model_Overview.py",     title="Pregled modela"),
    st.Page(_ROOT / "dashboard" / "pages" / "04_Training_Results.py",   title="Rezultati treninga"),
    st.Page(_ROOT / "dashboard" / "pages" / "05_Robustness.py",         title="Analiza robustnosti"),
    st.Page(_ROOT / "dashboard" / "pages" / "06_Live_Prediction.py",    title="Live predikcija"),
    st.Page(_ROOT / "dashboard" / "pages" / "07_Multi_Seed_Analysis.py",title="Multi-seed analiza"),
]

pg = st.navigation(pages)
pg.run()
