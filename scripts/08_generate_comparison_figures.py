"""
Generisanje figura za poredjenje V1 i V3 modela.

Ucitava:
  outputs/reports/eval_clean.json      (V1 cist)
  outputs/reports/eval_clean_v3.json   (V3 cist)
  outputs/reports/eval_noisy.json     (V1 zasumljeni)
  outputs/reports/eval_noisy_v3.json    (V3 zasumljeni)

Sprema tri figure u outputs/figures/thesis/:
  fig_robustness_curves_v1_vs_v3.{pdf,png}  - 5 panela, po jedna vrsta suma
  fig_robustness_heatmap_diff.{pdf,png}   - toplotna karta razlike V3-V1 F1
  fig_clean_metrics_comparison.{pdf,png}  
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ekg_mi.visualization.style import NOISE_COLORS, NOISE_LABELS_BS, set_thesis_style

set_thesis_style()

CONFIG_PATH = Path("configs/config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

REPORTS_PATH = Path(cfg["paths"]["reports"])
FIGURES_PATH = Path("outputs/figures/thesis")
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

NOISE_TYPES = cfg["noise"]["types"]
SNR_LEVELS  = cfg["noise"]["snr_db"]


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


clean_v1 = _load(REPORTS_PATH / "eval_clean.json")
clean_v3 = _load(REPORTS_PATH / "eval_clean_v3.json")
noisy_v1 = _load(REPORTS_PATH / "eval_noisy.json")
noisy_v3 = _load(REPORTS_PATH / "eval_noisy_v3.json")

# Slika 1: krive robustnosti V1 vs V3 - 5 panela (jedna vrsta suma po panelu)
fig1, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)

for ax, noise_type in zip(axes, NOISE_TYPES):
    color = NOISE_COLORS[noise_type]
    label = NOISE_LABELS_BS[noise_type]

    f1_v1 = [noisy_v1[noise_type][f"snr_{s}"]["f1"] for s in SNR_LEVELS]
    f1_v3 = [noisy_v3[noise_type][f"snr_{s}"]["f1"] for s in SNR_LEVELS]

    ax.plot(SNR_LEVELS, f1_v1, color=color, ls="--", lw=1.8, marker="o", ms=5,
            label="V1")
    ax.plot(SNR_LEVELS, f1_v3, color=color, ls="-",  lw=1.8, marker="s", ms=5,
            label="V3")

    # Referentne linije cistog signala
    ax.axhline(clean_v1["metrics"]["f1"], color="gray", ls=":", lw=1.0, alpha=0.7)

    ax.set_title(label, fontsize=9)
    ax.set_xlabel("SNR (dB)")
    ax.set_xticks(SNR_LEVELS)
    ax.set_ylim(0.4, 1.0)

axes[0].set_ylabel("F1 mjera")

# Zajednicki opis linija u prvom panelu
axes[0].legend(fontsize=8, loc="lower right")

fig1.suptitle(
    "Robustnost na sum: V1 (isprekidano) vs V3 sa augmentacijom (puno)",
    fontweight="bold", y=1.01,
)
fig1.tight_layout()
for ext in ("pdf", "png"):
    p = FIGURES_PATH / f"fig_robustness_curves_v1_vs_v3.{ext}"
    fig1.savefig(p, dpi=300 if ext == "png" else None)
    print(f"  -> {p}")
plt.close(fig1)

# Slika 2: toplotna karta razlike F1 (V3 - V1) po vrsti suma i SNR
matrix_diff = np.zeros((len(NOISE_TYPES), len(SNR_LEVELS)))
for i, noise_type in enumerate(NOISE_TYPES):
    for j, snr_db in enumerate(SNR_LEVELS):
        key   = f"snr_{snr_db}"
        v1_f1 = noisy_v1[noise_type][key]["f1"]
        v3_f1 = noisy_v3[noise_type][key]["f1"]
        matrix_diff[i, j] = v3_f1 - v1_f1

noise_labels = [NOISE_LABELS_BS[nt] for nt in NOISE_TYPES]
snr_labels   = [f"{s} dB" for s in SNR_LEVELS]

fig2, ax2 = plt.subplots(figsize=(10, 4))
abs_max = float(np.abs(matrix_diff).max())
abs_max = max(abs_max, 0.01)  # izbjegavam vmax=0

im = ax2.imshow(
    matrix_diff,
    cmap="RdYlGn",
    vmin=-abs_max, vmax=abs_max,
    aspect="auto",
)
cbar = fig2.colorbar(im, ax=ax2, shrink=0.8)
cbar.set_label("Delta F1 (V3 - V1)")

ax2.set_xticks(range(len(SNR_LEVELS)))
ax2.set_xticklabels(snr_labels)
ax2.set_yticks(range(len(NOISE_TYPES)))
ax2.set_yticklabels(noise_labels)
ax2.set_xlabel("SNR")
ax2.set_ylabel("Vrsta suma")
ax2.set_title("Razlika F1: V3 - V1 (zeleno = V3 bolji, crveno = V1 bolji)",
              fontweight="bold")

for i in range(len(NOISE_TYPES)):
    for j in range(len(SNR_LEVELS)):
        val = matrix_diff[i, j]
        ax2.text(j, i, f"{val:+.3f}", ha="center", va="center",
                 fontsize=8, color="black")

fig2.tight_layout()
for ext in ("pdf", "png"):
    p = FIGURES_PATH / f"fig_robustness_heatmap_diff.{ext}"
    fig2.savefig(p, dpi=300 if ext == "png" else None)
    print(f"  -> {p}")
plt.close(fig2)

# Slika 3: grafikon 7 metrika na cistom skupu
METRICS       = ["accuracy", "precision", "recall", "specificity", "f1", "auc_roc", "auc_pr"]
METRIC_LABELS = ["Tacnost", "Preciznost", "Osjetljivost", "Specificnost",
                 "F1", "AUC-ROC", "AUC-PR"]

v1_vals = [clean_v1["metrics"][m] for m in METRICS]
v3_vals = [clean_v3["metrics"][m] for m in METRICS]

x     = np.arange(len(METRICS))
width = 0.35

fig3, ax3 = plt.subplots(figsize=(11, 5))
bars_v1 = ax3.bar(x - width/2, v1_vals, width, label="V1 (bez augmentacije)",
                  color="#1f77b4", alpha=0.85)
bars_v3 = ax3.bar(x + width/2, v3_vals, width, label="V3 (sa augmentacijom suma)",
                  color="#ff7f0e", alpha=0.85)

ax3.set_ylabel("Vrijednost")
ax3.set_title("Poredenje V1 i V3 na cistom test skupu", fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels(METRIC_LABELS)
ax3.set_ylim(0.7, 1.05)
ax3.legend()

for bar in bars_v1:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., h + 0.003,
             f"{h:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)
for bar in bars_v3:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., h + 0.003,
             f"{h:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)

fig3.tight_layout()
for ext in ("pdf", "png"):
    p = FIGURES_PATH / f"fig_clean_metrics_comparison.{ext}"
    fig3.savefig(p, dpi=300 if ext == "png" else None)
    print(f"  -> {p}")
plt.close(fig3)

print("\nSve figure za poredjenje V1 vs V3 su generisane.")
