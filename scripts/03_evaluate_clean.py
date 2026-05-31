"""
Evaluacija BaselineCNN na cistom test skupu.
Koristi prag odluke 0.5 (podrazumijevana vrijednost).
Sprema:
  outputs/reports/eval_clean.json
  outputs/figures/confusion_matrix_clean.{png,pdf}
  outputs/figures/roc_clean.{png,pdf}
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import RocCurveDisplay
from torch.utils.data import DataLoader, TensorDataset
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ekg_mi.models.baseline_cnn import BaselineCNN
from ekg_mi.evaluation.evaluator import evaluate

CONFIG_PATH = Path("configs/config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

PROCESSED_PATH = Path(cfg["paths"]["processed"])
DEFAULT_MODELS_PATH  = Path(cfg["paths"]["models"])
DEFAULT_REPORTS_PATH = Path(cfg["paths"]["reports"])
DEFAULT_FIGURES_PATH = Path(cfg["paths"]["figures"])

parser = argparse.ArgumentParser(description="Evaluacija V1 modela na cistom test skupu.")
parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODELS_PATH / "best_model.pt"),
                    help="Putanja do .pt fajla sa tezinama modela.")
parser.add_argument("--output-json", type=str, default=str(DEFAULT_REPORTS_PATH / "eval_clean.json"),
                    help="Putanja izlaznog JSON izvjestaja.")
parser.add_argument("--figures-dir", type=str, default=str(DEFAULT_FIGURES_PATH),
                    help="Direktorij za matricu konfuzije i ROC krivu (ako prazno, ne crta).")
parser.add_argument("--no-figures", action="store_true",
                    help="Preskoci crtanje figura (korisno za multi-seed sweep).")
args = parser.parse_args()

MODEL_FILE   = Path(args.model_path)
OUTPUT_JSON  = Path(args.output_json)
FIGURES_PATH = Path(args.figures_dir)
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
if not args.no_figures:
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ucitaj test skup
print("Ucitavanje cistog test skupa ...")
test_data = np.load(PROCESSED_PATH / "test_clean.npz")
X_test = torch.tensor(test_data["signals"], dtype=torch.float32)
y_test = torch.tensor(test_data["labels"],  dtype=torch.long)
n_pos  = int((y_test == 1).sum())
n_neg  = int((y_test == 0).sum())
print(f"  Test: {X_test.shape}  (MI={n_pos}, NORM={n_neg})")

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=int(cfg["training"]["batch_size"]),
    shuffle=False,
    num_workers=0,
)

# Ucitaj model
print(f"Ucitavanje modela: {MODEL_FILE}")
model = BaselineCNN(num_channels=cfg["model"]["in_channels"]).to(device)
model.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
print("  Model ucitan.")

# Evaluacija
print("Pokretanje evaluacije ...")
results = evaluate(model, test_loader)

y_true = results["y_true"].astype(int)
y_pred = results["y_pred"].astype(int)
cm_tp = int(((y_pred == 1) & (y_true == 1)).sum())
cm_tn = int(((y_pred == 0) & (y_true == 0)).sum())
cm_fp = int(((y_pred == 1) & (y_true == 0)).sum())
cm_fn = int(((y_pred == 0) & (y_true == 1)).sum())

report = {
    "metrics": {
        "accuracy":    round(results["accuracy"],    4),
        "precision":   round(results["precision"],   4),
        "recall":      round(results["recall"],      4),
        "specificity": round(results["specificity"], 4),
        "f1":          round(results["f1"],          4),
        "auc_roc":     round(results["auc_roc"],     4),
        "auc_pr":      round(results["auc_pr"],      4),
    },
    "confusion_matrix": {"tp": cm_tp, "tn": cm_tn, "fp": cm_fp, "fn": cm_fn},
    "n_samples":  int(len(y_test)),
    "n_positive": n_pos,
    "n_negative": n_neg,
    "model_path": str(MODEL_FILE),
    "evaluation_timestamp": datetime.now().isoformat(),
}

out_path = OUTPUT_JSON
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)

# Ispis
m = report["metrics"]
print("\nRezultati na cistom test skupu:")
print(f"  Tacnost      : {m['accuracy']:.4f}")
print(f"  Preciznost   : {m['precision']:.4f}")
print(f"  Osjetljivost : {m['recall']:.4f}")
print(f"  Specificnost : {m['specificity']:.4f}")
print(f"  F1           : {m['f1']:.4f}")
print(f"  AUC-ROC      : {m['auc_roc']:.4f}")
print(f"  AUC-PR       : {m['auc_pr']:.4f}")
print(f"\n  Matrica konfuzije:")
print(f"              Predvideno")
print(f"              NORM   MI")
print(f"  Stvarno NORM  {cm_tn:4d}  {cm_fp:4d}")
print(f"          MI    {cm_fn:4d}  {cm_tp:4d}")
print(f"\n  Izvještaj spremljen -> {out_path}")

if not args.no_figures:
    # Matrica konfuzije - slika
    cm_arr = np.array([[cm_tn, cm_fp], [cm_fn, cm_tp]])
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues",
                xticklabels=["NORM", "MI"], yticklabels=["NORM", "MI"], ax=ax_cm)
    ax_cm.set_xlabel("Predvidena klasa")
    ax_cm.set_ylabel("Stvarna klasa")
    ax_cm.set_title("Matrica konfuzije - cist test skup", fontweight="bold")
    fig_cm.tight_layout()
    for ext in ("png", "pdf"):
        p = FIGURES_PATH / f"confusion_matrix_clean.{ext}"
        fig_cm.savefig(p, dpi=300 if ext == "png" else None)
        print(f"  Matrica konfuzije -> {p}")
    plt.close(fig_cm)

    # ROC kriva
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        results["y_true"], results["y_scores"],
        name=f"CNN (AUC = {m['auc_roc']:.3f})",
        ax=ax_roc, color="#0072B2",
    )
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, label="Slucajni klasifikator")
    ax_roc.set_title("ROC kriva - cist test skup", fontweight="bold")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    for ext in ("png", "pdf"):
        p = FIGURES_PATH / f"roc_clean.{ext}"
        fig_roc.savefig(p, dpi=300 if ext == "png" else None)
        print(f"  ROC kriva        -> {p}")
    plt.close(fig_roc)
