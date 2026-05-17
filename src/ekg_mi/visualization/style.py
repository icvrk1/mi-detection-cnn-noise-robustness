"""
Stil za grafove teze: rcParams i paleta boja.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl

# Boje za pet vrsta suma (colorblind-safe paleta)
NOISE_COLORS: dict[str, str] = {
    "baseline_wander":   "#1f77b4",
    "muscle_artifact":   "#ff7f0e",
    "electrode_motion":  "#2ca02c",
    "powerline":         "#d62728",
    "gaussian":          "#9467bd",
}

NOISE_LABELS_BS: dict[str, str] = {
    "gaussian":          "Gaussov sum",
    "baseline_wander":   "Lutanje bazne linije",
    "powerline":         "Mrezna frekvencija",
    "muscle_artifact":   "Misicni artefakt",
    "electrode_motion":  "Pokret elektrode",
}

CLASS_COLORS = ["#1f77b4", "#d62728"]   # NORM, MI


def set_thesis_style() -> None:
    mpl.rcParams.update({
        "figure.dpi":          150,
        "savefig.dpi":         300,
        "figure.figsize":      (8, 4),
        "font.size":           11,
        "font.family":         "serif",
        "mathtext.fontset":    "cm",
        "axes.titlesize":      12,
        "axes.labelsize":      11,
        "xtick.labelsize":     9,
        "ytick.labelsize":     9,
        "legend.fontsize":     9,
        "legend.framealpha":   0.85,
        "lines.linewidth":     1.6,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.grid":           True,
        "grid.alpha":          0.3,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.05,
    })
