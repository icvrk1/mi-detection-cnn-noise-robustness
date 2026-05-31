"""
Evaluacija modela na svim zasumljenim varijantama test skupa.
Ucitava NPZ fajlove iz data/noisy/.
Koristi prag odluke 0.5 (podrazumijevana vrijednost).
Sprema:
  outputs/reports/eval_noisy.json
  outputs/tables/results_noisy.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ekg_mi.models.baseline_cnn import BaselineCNN
from ekg_mi.evaluation.evaluator import evaluate

CONFIG_PATH = Path("configs/config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

NOISY_PATH           = Path(cfg["paths"]["noisy"])
DEFAULT_MODELS_PATH  = Path(cfg["paths"]["models"])
DEFAULT_REPORTS_PATH = Path(cfg["paths"]["reports"])
DEFAULT_TABLES_PATH  = Path(cfg["paths"]["tables"])

parser = argparse.ArgumentParser(description="Evaluacija V1 modela na zasumljenim test skupovima.")
parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODELS_PATH / "best_model.pt"))
parser.add_argument("--output-json", type=str, default=str(DEFAULT_REPORTS_PATH / "eval_noisy.json"))
parser.add_argument("--output-csv", type=str, default=str(DEFAULT_TABLES_PATH / "results_noisy.csv"))
args = parser.parse_args()

MODEL_PATH  = Path(args.model_path)
OUTPUT_JSON = Path(args.output_json)
OUTPUT_CSV  = Path(args.output_csv)
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

NOISE_TYPES = cfg["noise"]["types"]
SNR_LEVELS  = cfg["noise"]["snr_db"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ucitaj model
print(f"Ucitavanje modela: {MODEL_PATH}")
model = BaselineCNN(num_channels=cfg["model"]["in_channels"]).to(device)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)

# Evaluacija svih kombinacija
results: dict[str, dict] = {}
csv_rows: list[dict] = []
t0 = time.time()

for noise_type in NOISE_TYPES:
    print(f"\nVrsta suma: {noise_type}")
    results[noise_type] = {}

    for snr_db in SNR_LEVELS:
        fname    = f"test_{noise_type}_snr{snr_db:+03d}.npz"
        npz_path = NOISY_PATH / fname
        if not npz_path.exists():
            print(f"  NIJE PRONADJEN: {fname}")
            continue

        t_snr = time.time()
        data  = np.load(npz_path)
        X     = torch.tensor(data["signals"], dtype=torch.float32)
        y     = torch.tensor(data["labels"],  dtype=torch.long)

        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=False,
            num_workers=0,
        )
        metrics = evaluate(model, loader)

        y_true = metrics["y_true"].astype(int)
        y_pred = metrics["y_pred"].astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        entry = {
            "accuracy":    round(metrics["accuracy"],    4),
            "precision":   round(metrics["precision"],   4),
            "recall":      round(metrics["recall"],      4),
            "specificity": round(metrics["specificity"], 4),
            "f1":          round(metrics["f1"],          4),
            "auc_roc":     round(metrics["auc_roc"],     4),
            "auc_pr":      round(metrics["auc_pr"],      4),
            "confusion_matrix": [[tn, fp], [fn, tp]],
        }
        key = f"snr_{snr_db}"
        results[noise_type][key] = entry

        csv_rows.append({
            "noise_type": noise_type,
            "snr_db":     snr_db,
            **{k: entry[k] for k in ["accuracy", "precision", "recall",
                                     "specificity", "f1", "auc_roc", "auc_pr"]},
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        })

        print(
            f"  SNR={snr_db:+3d} dB | "
            f"F1={entry['f1']:.3f}  AUC-ROC={entry['auc_roc']:.3f}  "
            f"AUC-PR={entry['auc_pr']:.3f}  ({time.time()-t_snr:.1f}s)"
        )

# Spremi JSON
json_path = OUTPUT_JSON
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nJSON spremljen -> {json_path}")

# Spremi CSV
csv_path = OUTPUT_CSV
if csv_rows:
    fieldnames = ["noise_type", "snr_db", "accuracy", "precision", "recall",
                  "specificity", "f1", "auc_roc", "auc_pr", "tp", "tn", "fp", "fn"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"CSV spremljen  -> {csv_path}")

# Rezimirajuca tabela F1
elapsed = time.time() - t0
print(f"\nUkupno: {elapsed:.1f}s")
header = f"{'Vrsta suma':<22}" + "".join(f"{s:>8}" for s in SNR_LEVELS)
print(f"\n{header}")
print("-" * (22 + 8 * len(SNR_LEVELS)))
for nt in NOISE_TYPES:
    if nt not in results:
        continue
    row = f"{nt:<22}"
    for s in SNR_LEVELS:
        v = results[nt].get(f"snr_{s}", {}).get("f1", float("nan"))
        row += f"{v:>8.3f}"
    print(row)
