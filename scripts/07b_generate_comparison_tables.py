"""
Generisanje LaTeX tabela za poredjenje V1 i V3 modela.

Ucitava:
  outputs/reports/eval_clean.json      (V1)
  outputs/reports/eval_clean_v3.json    (V3)
  outputs/reports/eval_noisy.json      (V1)
  outputs/reports/eval_noisy_v3.json  (V3)

Sprema u outputs/tables/thesis/:
  table_clean_v1_vs_v3.tex        - poredjenje 7 metrika na cistom skupu
  table_robustness_v1_vs_v3.tex     - F1 poredjenje po vrsti suma i SNR
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

clean_v1 = json.loads((REPORTS_PATH / "eval_clean.json").read_text())
clean_v3 = json.loads((REPORTS_PATH / "eval_clean_v3.json").read_text())
noisy_v1 = json.loads((REPORTS_PATH / "eval_noisy.json").read_text())
noisy_v3 = json.loads((REPORTS_PATH / "eval_noisy_v3.json").read_text())

NOISE_TYPES = list(noisy_v1.keys())
SNR_LEVELS  = [-6, 0, 6, 12, 18, 24]


def _bold_better(v1: float, v3: float, decimals: int = 4) -> tuple[str, str]:
    """Vrati (v1_str, v3_str) gdje je bolji formatiran boldiranim LaTeX kodom."""
    fmt = f".{decimals}f"
    s1 = format(v1, fmt)
    s2 = format(v3, fmt)
    if v3 > v1:
        return s1, rf"\textbf{{{s2}}}"
    elif v1 > v3:
        return rf"\textbf{{{s1}}}", s2
    return s1, s2


# Tabela 1: Poredjenje metrika na cistom skupu
def table_clean_v1_vs_v3():
    METRICS = [
        ("accuracy",    "Tacnost"),
        ("precision",   "Preciznost"),
        ("recall",      "Osjetljivost"),
        ("specificity", "Specificnost"),
        ("f1",          "F1 mjera"),
        ("auc_roc",     "AUC-ROC"),
        ("auc_pr",      "AUC-PR"),
    ]
    m1 = clean_v1["metrics"]
    m3 = clean_v3["metrics"]

    rows = []
    for key, label in METRICS:
        v1_val = m1[key]
        v3_val = m3[key]
        delta  = v3_val - v1_val
        sign   = "+" if delta >= 0 else ""
        s1, s3 = _bold_better(v1_val, v3_val)
        rows.append(rf"{label} & {s1} & {s3} & {sign}{delta:.4f} \\")

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Poredjenje V1 i V3 modela na cistom test skupu. Masno: bolji rezultat.}",
        r"\label{tab:clean_v1_vs_v3}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Metrika} & \textbf{V1} & \textbf{V3} & \textbf{Delta (V3-V1)} \\",
        r"\midrule",
    ] + rows + [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path = TABLES_PATH / "table_clean_v1_vs_v3.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {path}")


# Tabela 2: F1 por edjenje po vrsti suma i SNR
def table_robustness_v1_vs_v3():
    snr_header = " & ".join([f"\\textbf{{{s:+d} dB}}" for s in SNR_LEVELS])

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{F1 mjera V1 i V3 po vrsti suma i SNR nivou. Masno: bolji rezultat.}",
        r"\label{tab:robustness_v1_vs_v3}",
        r"\begin{tabular}{ll" + "r" * len(SNR_LEVELS) + "}",
        r"\toprule",
        rf"\textbf{{Vrsta suma}} & \textbf{{Model}} & {snr_header} \\",
        r"\midrule",
    ]

    for nt in NOISE_TYPES:
        label = NOISE_LABELS_BS[nt]
        v1_row_cells = []
        v3_row_cells = []
        for snr_db in SNR_LEVELS:
            key   = f"snr_{snr_db}"
            v1_f1 = noisy_v1[nt][key]["f1"]
            v3_f1 = noisy_v3[nt][key]["f1"]
            s1, s3 = _bold_better(v1_f1, v3_f1, decimals=3)
            v1_row_cells.append(s1)
            v3_row_cells.append(s3)

        v1_vals = " & ".join(v1_row_cells)
        v3_vals = " & ".join(v3_row_cells)
        lines.append(rf"\multirow{{2}}{{*}}{{{label}}} & V1 & {v1_vals} \\")
        lines.append(rf" & V3 & {v3_vals} \\")
        lines.append(r"\midrule")

    # Uklanjanje  zadnjeg \midrule i dodavanje \bottomrule
    lines[-1] = r"\bottomrule"
    lines += [
        r"\end{tabular}",
        r"\end{table}",
    ]

    path = TABLES_PATH / "table_robustness_v1_vs_v3.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {path}")


print("Generisanje LaTeX tabela za poredjenje V1 vs V3 ...")
table_clean_v1_vs_v3()
table_robustness_v1_vs_v3()
print(f"\nSve tabele spremljene u {TABLES_PATH}")
