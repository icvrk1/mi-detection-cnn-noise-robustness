"""
Generisanje svih figura za tezu (8 PDF+PNG fajlova).
Zahtijeva:
  data/processed/test_clean.npz
  outputs/logs/training_history.json
  outputs/reports/eval_clean.json
  outputs/reports/eval_noisy.json
Sprema u: outputs/figures/thesis/fig_*.pdf i fig_*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ekg_mi.visualization.style import (
    set_thesis_style,
    NOISE_COLORS,
    NOISE_LABELS_BS,
    CLASS_COLORS,
)

set_thesis_style()

PROCESSED_PATH = Path("data/processed")
LOGS_PATH      = Path("outputs/logs")
REPORTS_PATH   = Path("outputs/reports")
FIGURES_PATH   = Path("outputs/figures/thesis")
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

# Ucitaj podatke
test_data  = np.load(PROCESSED_PATH / "test_clean.npz")
X_test     = test_data["signals"]   # (N, 12, 1000)
y_test     = test_data["labels"]
history    = json.loads((LOGS_PATH / "training_history.json").read_text())
eval_clean = json.loads((REPORTS_PATH / "eval_clean.json").read_text())
eval_noisy = json.loads((REPORTS_PATH / "eval_noisy.json").read_text())


def _clean(key: str) -> float:
    return eval_clean["metrics"][key]


def _noisy(nt: str, snr: int, key: str):
    return eval_noisy[nt][f"snr_{snr}"][key]


def _save(fig, name: str) -> None:
    pdf_path = FIGURES_PATH / f"{name}.pdf"
    png_path = FIGURES_PATH / f"{name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"  -> {pdf_path}")


LEAD_NAMES  = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SNR_LEVELS  = [-6, 0, 6, 12, 18, 24]
NOISE_TYPES = list(eval_noisy.keys())
t_axis      = np.arange(1000) / 100.0  # sekunde


#  Figura 1: Primjeri cistih EKG snimaka 
def fig_clean_signal_examples():
    norm_idx = np.where(y_test == 0)[0]
    mi_idx   = np.where(y_test == 1)[0]
    rng = np.random.default_rng(42)
    i_norm = rng.choice(norm_idx)
    i_mi   = rng.choice(mi_idx)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    for row, (idx, label, color) in enumerate([
        (i_norm, "NORM", CLASS_COLORS[0]),
        (i_mi,   "MI",   CLASS_COLORS[1]),
    ]):
        ax = axes[row]
        sig = X_test[idx, 1]  # odvod II
        ax.plot(t_axis, sig, color=color, lw=1.4)
        ax.set_ylabel("Amplituda (mV)", fontsize=9)
        ax.set_title(f"Klasa: {label} - odvod II", fontweight="bold", color=color)
        # Okvirne anotacije P/QRS/T regija
        ax.axvspan(1.4, 2.0, alpha=0.10, color="gray", label="P")
        ax.axvspan(2.8, 3.5, alpha=0.12, color="steelblue", label="QRS")
        ax.axvspan(4.5, 6.0, alpha=0.10, color="darkorange", label="T")
        if row == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=3,
                      title="Karakteristicni valovi")

    axes[-1].set_xlabel("Vrijeme (s)")
    fig.suptitle("Primjeri EKG snimaka - odvod II", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig_clean_signal_examples")


#  Figura 2: Primjeri vrsta suma 
def fig_noise_examples():
    from ekg_mi.noise.injection import add_noise

    rng = np.random.default_rng(0)
    idx = rng.choice(np.where(y_test == 1)[0])
    clean = X_test[idx, 1]  # odvod II

    snr_show = [18, 6, 0, -6]
    n_rows = len(NOISE_TYPES)
    n_cols = len(snr_show)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 9), sharex=True)

    for row, nt in enumerate(NOISE_TYPES):
        color = NOISE_COLORS[nt]
        for col, snr in enumerate(snr_show):
            ax = axes[row, col]
            noisy = add_noise(clean, noise_type=nt, snr_db=snr)
            ax.plot(t_axis, clean, color="gray", lw=0.8, alpha=0.6, zorder=1)
            ax.plot(t_axis, noisy, color=color, lw=0.9, alpha=0.9, zorder=2)
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"SNR = {snr:+d} dB", fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(NOISE_LABELS_BS[nt], fontsize=8, rotation=90, labelpad=4)
            if row == n_rows - 1:
                ax.set_xlabel("t (s)", fontsize=8)

    fig.suptitle("Vrste suma po SNR nivou (sivi = cist signal, boja = zasumljeni)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig_noise_examples")


#  Figura 3: Krive treninga 
def fig_training_curves():
    epochs = list(range(1, len(history["train_loss"]) + 1))
    best_ep = history.get("best_epoch", None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, history["train_loss"], label="Trening", color="#0072B2")
    ax1.plot(epochs, history["val_loss"],   label="Validacija", color="#D55E00")
    if best_ep:
        ax1.axvline(best_ep, color="black", lw=1, ls=":", label=f"Najbolja ep. ({best_ep})")
    ax1.set_xlabel("Epoha")
    ax1.set_ylabel("Gubitak")
    ax1.set_title("Kriva gubitka")
    ax1.legend()

    ax2.plot(epochs, history["train_f1"], label="Trening", color="#0072B2")
    ax2.plot(epochs, history["val_f1"],   label="Validacija", color="#D55E00")
    if best_ep:
        ax2.axvline(best_ep, color="black", lw=1, ls=":", label=f"Najbolja ep. ({best_ep})")
    ax2.set_xlabel("Epoha")
    ax2.set_ylabel("F1 mjera")
    ax2.set_title("Kriva F1 mjere")
    ax2.set_ylim(0.8, 1.0)
    ax2.legend()

    fig.suptitle("Historija treninga", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig_training_curves")


#  Figura 4: Matrica konfuzije (cist test) 
def fig_confusion_matrix_clean():
    cm_dict = eval_clean["confusion_matrix"]
    tp = cm_dict["tp"]
    tn = cm_dict["tn"]
    fp = cm_dict["fp"]
    fn = cm_dict["fn"]
    n  = tp + tn + fp + fn
    cm = np.array([[tn, fp], [fn, tp]])
    classes = ["NORM", "MI"]

    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1]); ax.set_xticklabels(classes)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes)
    ax.set_xlabel("Predvidena klasa")
    ax.set_ylabel("Stvarna klasa")
    ax.set_title("Matrica konfuzije - cist test skup", fontweight="bold")

    thresh = cm.max() / 2.0
    vals = [[tn, fp], [fn, tp]]
    for i in range(2):
        for j in range(2):
            v = vals[i][j]
            pct = v / n * 100
            ax.text(j, i, f"{v}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    _save(fig, "fig_confusion_matrix_clean")


#  Figura 5: F1 robustnost po vrsti suma i SNR (KLJUCNA) 
def fig_robustness_curves():
    fig, ax = plt.subplots(figsize=(8, 5))

    clean_f1 = _clean("f1")
    ax.axhline(clean_f1, color="black", lw=1.6, ls="--",
               label=f"Cist signal (F1={clean_f1:.3f})", zorder=5)

    for nt in NOISE_TYPES:
        f1_vals = [_noisy(nt, s, "f1") for s in SNR_LEVELS]
        ax.plot(SNR_LEVELS, f1_vals,
                color=NOISE_COLORS[nt], marker="o", ms=5, lw=1.8,
                label=NOISE_LABELS_BS[nt])

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("F1 mjera")
    ax.set_title("Robustnost modela - F1 po vrsti suma i SNR-u", fontweight="bold")
    ax.set_xticks(SNR_LEVELS)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-8, 26)

    fig.tight_layout()
    _save(fig, "fig_robustness_curves")


#  Figura 6: Toplotna mapa F1 
def fig_robustness_heatmap():
    f1_matrix = np.zeros((len(NOISE_TYPES), len(SNR_LEVELS)))
    for i, nt in enumerate(NOISE_TYPES):
        for j, snr in enumerate(SNR_LEVELS):
            f1_matrix[i, j] = _noisy(nt, snr, "f1")

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(f1_matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="F1 mjera")

    ax.set_xticks(range(len(SNR_LEVELS)))
    ax.set_xticklabels([f"{s} dB" for s in SNR_LEVELS])
    ax.set_yticks(range(len(NOISE_TYPES)))
    ax.set_yticklabels([NOISE_LABELS_BS[nt] for nt in NOISE_TYPES])
    ax.set_xlabel("SNR")
    ax.set_title("Toplotna mapa F1 - robustnost modela", fontweight="bold")

    for i in range(len(NOISE_TYPES)):
        for j in range(len(SNR_LEVELS)):
            val = f1_matrix[i, j]
            color = "black" if 0.5 < val < 0.85 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=color)

    fig.tight_layout()
    _save(fig, "fig_robustness_heatmap")


#  Figura 7: Metrike vs SNR (2x2 grid) 
def fig_metric_comparison():
    metric_keys  = ["accuracy", "f1", "recall", "specificity"]
    metric_names = ["Tacnost", "F1 mjera", "Osjetljivost", "Specificnost"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=False)
    axes_flat = axes.flatten()

    for ax_idx, (mkey, mname) in enumerate(zip(metric_keys, metric_names)):
        ax = axes_flat[ax_idx]
        for nt in NOISE_TYPES:
            vals = [_noisy(nt, s, mkey) for s in SNR_LEVELS]
            ax.plot(SNR_LEVELS, vals, color=NOISE_COLORS[nt], marker="o",
                    ms=4, lw=1.5, label=NOISE_LABELS_BS[nt])
        clean_val = _clean(mkey)
        ax.axhline(clean_val, color="black", lw=1.2, ls="--",
                   label=f"Cist ({clean_val:.3f})")
        ax.set_title(mname, fontweight="bold")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(mname)
        ax.set_xticks(SNR_LEVELS)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-8, 26)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Usporedba metrika po vrsti suma i SNR-u", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, "fig_metric_comparison")


#  Figura 8: Matrice konfuzije - odabrane kombinacije 
def fig_confusion_matrices_grid():
    # Prikaz: cist + MA@6dB + EM@0dB + PLI@18dB
    combos = [
        ("clean", None,              None,  "Cist test skup"),
        ("noisy", "muscle_artifact", 6,     "Misicni artefakt (+6 dB)"),
        ("noisy", "electrode_motion", 0,    "Pokret elektrode (0 dB)"),
        ("noisy", "powerline",       18,    "Mrezna frekvencija (+18 dB)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    for ax, (src, nt, snr, title) in zip(axes, combos):
        if src == "clean":
            cd = eval_clean["confusion_matrix"]
            cm = np.array([[cd["tn"], cd["fp"]], [cd["fn"], cd["tp"]]])
        else:
            cm = np.array(_noisy(nt, snr, "confusion_matrix"))

        im = ax.imshow(cm, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["NORM", "MI"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["NORM", "MI"], fontsize=8)
        ax.set_title(title, fontsize=8, fontweight="bold")
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if cm[i, j] > thresh else "black")

    axes[0].set_ylabel("Stvarno", fontsize=9)
    fig.suptitle("Matrice konfuzije - odabrane kombinacije",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig_confusion_matrices_grid")


#  Pokretanje 
print("Generisanje figura za tezu ...")
fig_clean_signal_examples()
fig_noise_examples()
fig_training_curves()
fig_confusion_matrix_clean()
fig_robustness_curves()
fig_robustness_heatmap()
fig_metric_comparison()
fig_confusion_matrices_grid()
print(f"\nSve figure spremljene u {FIGURES_PATH}")
