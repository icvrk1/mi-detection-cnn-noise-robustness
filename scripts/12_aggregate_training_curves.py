"""
Objedinjene krive treninga preko vise runova: srednja vrijednost +/- standardna devijacija (band).

Cita:
  outputs/runs/{v1,v3}/seed_*/training_history_{v1,v3}.json

Pise:
  outputs/figures/training_curves_aggregated_{v1,v3}.{png,pdf}
  outputs/figures/training_curves_aggregated_compare.{png,pdf}   (V1 vs V3, F1)

Runovi razlicitih duzina (zbog ranog zaustavljanja) se NaN-padded poravnavaju
po epohi, pa se koristi nanmean/nanstd kako bi se pravicno racunala statistika.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT     = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "outputs" / "runs"
FIG_DIR  = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [("train_loss", "val_loss", "Funkcija gubitka"),
           ("train_acc",  "val_acc",  "Tacnost"),
           ("train_f1",   "val_f1",   "F1 mjera")]


def load_histories(variant: str) -> list[dict]:
    base = RUNS_DIR / variant
    if not base.exists():
        return []
    out = []
    for sd in sorted(base.iterdir()):
        if not (sd.is_dir() and sd.name.startswith("seed_")):
            continue
        hp = sd / f"training_history_{variant}.json"
        if hp.exists():
            with open(hp) as fh:
                out.append(json.load(fh))
    return out


def pad_to_matrix(series_list: list[list[float]]) -> np.ndarray:
    if not series_list:
        return np.zeros((0, 0))
    max_len = max(len(s) for s in series_list)
    M = np.full((len(series_list), max_len), np.nan, dtype=float)
    for i, s in enumerate(series_list):
        M[i, :len(s)] = s
    return M


def mean_std(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mat.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat, axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros(mat.shape[1])
    n    = np.sum(~np.isnan(mat), axis=0)
    return mean, std, n


def plot_variant(variant: str, hists: list[dict]) -> None:
    if not hists:
        print(f"[{variant}] nema runova, preskacem.")
        return
    n_runs = len(hists)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (tr_key, vl_key, title) in zip(axes, METRICS):
        tr_mat = pad_to_matrix([h[tr_key] for h in hists])
        vl_mat = pad_to_matrix([h[vl_key] for h in hists])
        tr_m, tr_s, _ = mean_std(tr_mat)
        vl_m, vl_s, _ = mean_std(vl_mat)
        x = np.arange(1, tr_mat.shape[1] + 1)
        ax.plot(x, tr_m, label="trening", color="#0072B2")
        ax.fill_between(x, tr_m - tr_s, tr_m + tr_s, alpha=0.2, color="#0072B2")
        ax.plot(x, vl_m, label="validacija", color="#D55E00")
        ax.fill_between(x, vl_m - vl_s, vl_m + vl_s, alpha=0.2, color="#D55E00")
        ax.set_title(title)
        ax.set_xlabel("Epoha")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Krive treninga {variant.upper()} - srednja vrijednost $\\pm$ standardna devijacija preko {n_runs} runova",
                 fontweight="bold")
    plt.tight_layout()
    base = FIG_DIR / f"training_curves_aggregated_{variant}"
    fig.savefig(str(base) + ".png", dpi=300)
    fig.savefig(str(base) + ".pdf")
    plt.close(fig)
    print(f"  spr -> {base}.png/.pdf")


def plot_compare_f1(hv1: list[dict], hv3: list[dict]) -> None:
    if not hv1 or not hv3:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, hists, color in [("V1", hv1, "#0072B2"), ("V3", hv3, "#D55E00")]:
        mat = pad_to_matrix([h["val_f1"] for h in hists])
        m, s, _ = mean_std(mat)
        x = np.arange(1, mat.shape[1] + 1)
        ax.plot(x, m, label=f"{label} val F1 (n={len(hists)})", color=color)
        ax.fill_between(x, m - s, m + s, alpha=0.2, color=color)
    ax.set_title("Validacijska F1 mjera - V1 vs V3 (srednja vrijednost $\\pm$ standardna devijacija)", fontweight="bold")
    ax.set_xlabel("Epoha")
    ax.set_ylabel("F1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    base = FIG_DIR / "training_curves_aggregated_compare"
    fig.savefig(str(base) + ".png", dpi=300)
    fig.savefig(str(base) + ".pdf")
    plt.close(fig)
    print(f"  spr -> {base}.png/.pdf")


def main() -> None:
    hv1 = load_histories("v1")
    hv3 = load_histories("v3")
    print(f"V1 runova: {len(hv1)} | V3 runova: {len(hv3)}")
    plot_variant("v1", hv1)
    plot_variant("v3", hv3)
    plot_compare_f1(hv1, hv3)


if __name__ == "__main__":
    main()
