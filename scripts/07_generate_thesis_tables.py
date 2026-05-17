"""
Generisanje LaTeX tabela za tezu (4 .tex fajla).
Zahtjeva eval_clean.json, eval_noisy.json, data_distribution.json.
Sprema u: outputs/tables/thesis/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ekg_mi.visualization.style import NOISE_LABELS_BS

REPORTS_PATH = Path("outputs/reports")
TABLES_PATH  = Path("outputs/tables/thesis")
TABLES_PATH.mkdir(parents=True, exist_ok=True)

eval_clean = json.loads((REPORTS_PATH / "eval_clean.json").read_text())
eval_noisy = json.loads((REPORTS_PATH / "eval_noisy.json").read_text())

NOISE_TYPES = list(eval_noisy.keys())
SNR_LEVELS  = [-6, 0, 6, 12, 18, 24]


#  Tabela 1: F1 robustnost - sve vrste suma + cist 
def table_main_results():
    """Glavni rezultat: F1 po vrsti suma i SNR nivou, sa cistim signalom kao prvim redom."""
    snr_cols = " & ".join([f"\\textbf{{{s:+d} dB}}" for s in SNR_LEVELS])
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{F1 mjera modela po vrsti suma i SNR nivou}",
        r"\label{tab:robustness_f1}",
        r"\begin{tabular}{l" + "r" * len(SNR_LEVELS) + "}",
        r"\toprule",
        rf"\textbf{{Vrsta suma}} & {snr_cols} \\",
        r"\midrule",
    ]
    # Cist signal kao poseban red
    clean_f1 = eval_clean["metrics"]["f1"]
    clean_row = " & ".join([f"{clean_f1:.3f}"] * len(SNR_LEVELS))
    lines.append(rf"\textbf{{Cist signal}} & {clean_row} \\")
    lines.append(r"\midrule")
    for nt in NOISE_TYPES:
        row_vals = " & ".join(
            [f"{eval_noisy[nt][f'snr_{s}']['f1']:.3f}" for s in SNR_LEVELS]
        )
        label = NOISE_LABELS_BS[nt]
        lines.append(rf"{label} & {row_vals} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path = TABLES_PATH / "table_main_results.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {path}")


#  Tabela 2: Sve metrike na cistom test skupu 
def table_clean_metrics():
    m  = eval_clean["metrics"]
    cm = eval_clean["confusion_matrix"]
    n_samples  = eval_clean["n_samples"]
    n_positive = eval_clean["n_positive"]
    n_negative = eval_clean["n_negative"]

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Metrike modela na cistom test skupu}",
        r"\label{tab:clean_metrics}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"\textbf{Metrika} & \textbf{Vrijednost} \\",
        r"\midrule",
        rf"Tacnost & {m['accuracy']:.4f} \\",
        rf"Preciznost & {m['precision']:.4f} \\",
        rf"Osjetljivost (recall) & {m['recall']:.4f} \\",
        rf"Specificnost & {m['specificity']:.4f} \\",
        rf"F1 mjera & {m['f1']:.4f} \\",
        rf"AUC-ROC & {m['auc_roc']:.4f} \\",
        rf"AUC-PR & {m['auc_pr']:.4f} \\",
        r"\midrule",
        rf"Ukupno uzoraka & {n_samples} \\",
        rf"MI uzoraka (TP+FN) & {n_positive} \\",
        rf"NORM uzoraka (TN+FP) & {n_negative} \\",
        rf"Istinito pozitivni (TP) & {cm['tp']} \\",
        rf"Istinito negativni (TN) & {cm['tn']} \\",
        rf"Lazno pozitivni (FP) & {cm['fp']} \\",
        rf"Lazno negativni (FN) & {cm['fn']} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path = TABLES_PATH / "table_clean_metrics.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {path}")


#  Tabela 3: Rezime skupa podataka 
def table_dataset_summary():
    dist_path = REPORTS_PATH / "data_distribution.json"
    if dist_path.exists():
        dist   = json.loads(dist_path.read_text())
        splits = dist.get("splits", {})
        rows   = []
        for name, info in splits.items():
            label_map = {"train": "Trening", "val": "Validacija", "test_clean": "Test"}
            friendly = label_map.get(name, name)
            total = info["n"]
            mi    = info["mi"]
            norm  = info["norm"]
            pct   = mi / total * 100 if total > 0 else 0
            rows.append(rf"{friendly} & {total} & {norm} & {mi} & {pct:.1f}\% \\")
        body = "\n".join(rows)
    else:
        body = r"-- & -- & -- & -- & -- \\"

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Rezime PTB-XL skupa podataka nakon filtriranja}",
        r"\label{tab:dataset_summary}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Skup} & \textbf{Ukupno} & \textbf{NORM} & \textbf{MI} & \textbf{MI \%} \\",
        r"\midrule",
        body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path = TABLES_PATH / "table_dataset_summary.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {path}")


#  Tabela 4: Arhitektura modela 
def table_architecture():
    rows = [
        r"Ulazni sloj & (12, 1000) & -- \\",
        r"ConvBlock 1 (Conv1d-k16 + BN + ReLU + MaxPool) & (32, 500) & 6\,240 \\",
        r"ConvBlock 2 (Conv1d-k5 + BN + ReLU + MaxPool) & (64, 250) & 10\,432 \\",
        r"ConvBlock 3 (Conv1d-k5 + BN + ReLU + MaxPool) & (128, 125) & 41\,344 \\",
        r"ConvBlock 4 (Conv1d-k5 + BN + ReLU + MaxPool) & (256, 62) & 164\,608 \\",
        r"AdaptiveAvgPool1d & (256, 1) & -- \\",
        r"Flatten + Dropout(0.5) & 256 & -- \\",
        r"Linear(256, 64) + ReLU & 64 & 16\,448 \\",
        r"Izlazni sloj Linear(64, 1) & 1 & 65 \\",
    ]
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Arhitektura 1D CNN modela}",
        r"\label{tab:architecture}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Sloj} & \textbf{Izlazna forma} & \textbf{Parametri} \\",
        r"\midrule",
    ] + rows + [
        r"\midrule",
        r"Ukupno trenabilnih parametara & & 239\,137 \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path = TABLES_PATH / "table_architecture.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {path}")


print("Generisanje LaTeX tabela ...")
table_main_results()
table_clean_metrics()
table_dataset_summary()
table_architecture()
print(f"\nSve tabele spremljene u {TABLES_PATH}")
