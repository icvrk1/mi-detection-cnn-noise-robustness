"""
Ekstrakcija kljucnih brojeva za tezu u JSON format.
Zahtjeva eval_clean.json, eval_noisy.json, data_distribution.json,
         training_history.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPORTS_PATH = Path("outputs/reports")
LOGS_PATH    = Path("outputs/logs")
THESIS_PATH  = Path("thesis")
THESIS_PATH.mkdir(parents=True, exist_ok=True)

eval_clean = json.loads((REPORTS_PATH / "eval_clean.json").read_text())
eval_noisy = json.loads((REPORTS_PATH / "eval_noisy.json").read_text())
dist       = json.loads((REPORTS_PATH / "data_distribution.json").read_text())
history    = json.loads((LOGS_PATH / "training_history.json").read_text())

NOISE_TYPES = list(eval_noisy.keys())
SNR_LEVELS  = [-6, 0, 6, 12, 18, 24]

m = eval_clean["metrics"]

# Koji SNR nivo daje F1 >= threshold za svaku vrstu suma
def degradation_snr(noise_type: str, threshold: float = 0.90) -> int | None:
    for snr in sorted(SNR_LEVELS):
        if eval_noisy[noise_type][f"snr_{snr}"]["f1"] >= threshold:
            return snr
    return None

# Najrobustnija i najosjetljivija vrsta suma (po F1 na 0 dB)
f1_at_0db = {nt: eval_noisy[nt]["snr_0"]["f1"] for nt in NOISE_TYPES}
most_robust  = max(f1_at_0db, key=f1_at_0db.get)
least_robust = min(f1_at_0db, key=f1_at_0db.get)

# F1 na -6 dB po vrsti suma
f1_at_minus6db = {nt: eval_noisy[nt]["snr_-6"]["f1"] for nt in NOISE_TYPES}

# Procijenjeno trajanje treninga (17 epoha, ~24s/epoha na CPU)
n_epochs = len(history["train_loss"])
best_epoch = history["best_epoch"]
best_val_f1 = max(history["val_f1"])

numbers = {
    "dataset": {
        "total_records"   : dist["total_raw"],
        "valid_records"   : dist["valid_total"],
        "mi_records"      : dist["mi"],
        "norm_records"    : dist["norm"],
        "excluded_records": dist["excluded"],
        "mi_percent"      : round(dist["mi"] / dist["valid_total"] * 100, 1),
        "splits"          : dist["splits"],
    },
    "training": {
        "n_epochs"          : n_epochs,
        "best_epoch"        : best_epoch,
        "best_val_loss"     : round(history["best_val_loss"], 4),
        "best_val_f1"       : round(best_val_f1, 4),
        "n_parameters"      : 239137,
        "training_time_min" : round(n_epochs * 24 / 60, 1),
    },
    "clean_test": {
        "accuracy"    : m["accuracy"],
        "precision"   : m["precision"],
        "recall"      : m["recall"],
        "specificity" : m["specificity"],
        "f1"          : m["f1"],
        "auc_roc"     : m["auc_roc"],
        "auc_pr"      : m["auc_pr"],
        "n_test"      : eval_clean["n_samples"],
        "n_mi"        : eval_clean["n_positive"],
        "n_norm"      : eval_clean["n_negative"],
    },
    "robustness": {
        "noise_types": NOISE_TYPES,
        "snr_levels" : SNR_LEVELS,
        "f1_at_0db"  : f1_at_0db,
        "f1_at_minus6db" : f1_at_minus6db,
        "most_robust_noise_type"  : most_robust,
        "least_robust_noise_type" : least_robust,
        "snr_threshold_90pct_baseline": {
            nt: degradation_snr(nt, threshold=0.90)
            for nt in NOISE_TYPES
        },
        "f1_matrix": {
            nt: {str(s): eval_noisy[nt][f"snr_{s}"]["f1"] for s in SNR_LEVELS}
            for nt in NOISE_TYPES
        },
        "auc_roc_matrix": {
            nt: {str(s): eval_noisy[nt][f"snr_{s}"]["auc_roc"] for s in SNR_LEVELS}
            for nt in NOISE_TYPES
        },
    },
}

out_path = THESIS_PATH / "numbers.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(numbers, f, indent=2, ensure_ascii=False)

print("Kljucni brojevi za tezu:")
print(f"  Cist test - F1      : {m['f1']:.4f}")
print(f"  Cist test - AUC-ROC : {m['auc_roc']:.4f}")
print(f"  Cist test - AUC-PR  : {m['auc_pr']:.4f}")
print(f"  Trening epohe       : {n_epochs}  (najbolja: {best_epoch})")
print(f"  Broj parametara     : 239,137")
print(f"  Najrobusniji sum    : {most_robust} (F1@0dB={f1_at_0db[most_robust]:.3f})")
print(f"  Najosjetljiviji     : {least_robust} (F1@0dB={f1_at_0db[least_robust]:.3f})")
print(f"\nSNR prag 90% F1 po vrsti suma:")
for nt, snr_thr in numbers["robustness"]["snr_threshold_90pct_baseline"].items():
    label = "nije dostignut" if snr_thr is None else f"{snr_thr:+d} dB"
    print(f"  {nt:<22}: {label}")
print(f"\nSpremen -> {out_path}")
