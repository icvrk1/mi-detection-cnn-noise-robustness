"""
Uporedjivanje V1 (bez augmentacije) i V3 (sa augmentacijom suma) modela.

Ucitava:
  V1 cist skup: outputs/reports/eval_clean_tuned.json  (fallback: eval_clean.json)
  V3 cist skup: outputs/reports/eval_clean_v3.json
  V1 zasumljeni: outputs/reports/eval_noisy_tuned.json  (fallback: eval_noisy.json)
  V3 zasumljeni: outputs/reports/eval_noisy_v3.json

Sprema:
  outputs/reports/comparison_v1_vs_v3.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

CONFIG_PATH = Path("configs/config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

REPORTS_PATH = Path(cfg["paths"]["reports"])

# Ucitavanje izvjestaja
def _load_json(primary: Path, fallback: Path) -> dict:
    if primary.exists():
        with open(primary) as f:
            return json.load(f)
    print(f"  {primary.name} nije pronadjen, koristim {fallback.name}")
    with open(fallback) as f:
        return json.load(f)

clean_v1  = _load_json(
    REPORTS_PATH / "eval_clean_tuned.json",
    REPORTS_PATH / "eval_clean.json",
)
clean_v3  = _load_json(
    REPORTS_PATH / "eval_clean_v3.json",
    REPORTS_PATH / "eval_clean_v3.json",
)
noisy_v1  = _load_json(
    REPORTS_PATH / "eval_noisy_tuned.json",
    REPORTS_PATH / "eval_noisy.json",
)
noisy_v3_path = REPORTS_PATH / "eval_noisy_v3.json"
with open(noisy_v3_path) as f:
    noisy_v3 = json.load(f)

# Uporedjivanje metrika na cistom skupu
CLEAN_METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "auc_roc", "auc_pr"]

clean_compare: dict[str, dict] = {}
for metric in CLEAN_METRICS:
    v1_val = clean_v1["metrics"][metric]
    v3_val = clean_v3["metrics"][metric]
    delta  = round(v3_val - v1_val, 4)
    clean_compare[metric] = {
        "v1":    v1_val,
        "v3":    v3_val,
        "delta": delta,
        "better": "v3" if delta > 0 else ("v1" if delta < 0 else "jednako"),
    }

# Uporedjivanje F1 na zasumljenim skupovima
NOISE_TYPES = cfg["noise"]["types"]
SNR_LEVELS  = cfg["noise"]["snr_db"]

noisy_compare: dict[str, dict] = {}
for noise_type in NOISE_TYPES:
    noisy_compare[noise_type] = {}
    for snr_db in SNR_LEVELS:
        key = f"snr_{snr_db}"
        if key not in noisy_v1.get(noise_type, {}):
            continue
        v1_f1 = noisy_v1[noise_type][key]["f1"]
        v3_f1 = noisy_v3[noise_type][key]["f1"]
        delta  = round(v3_f1 - v1_f1, 4)
        noisy_compare[noise_type][key] = {
            "v1_f1":  v1_f1,
            "v3_f1":  v3_f1,
            "delta":  delta,
            "better": "v3" if delta > 0 else ("v1" if delta < 0 else "jednako"),
        }

# Prosjecni dobitak V3 u zavisnosti od SNR
avg_delta_by_snr: dict[str, float] = {}
for snr_db in SNR_LEVELS:
    key    = f"snr_{snr_db}"
    deltas = [
        noisy_compare[nt][key]["delta"]
        for nt in NOISE_TYPES
        if key in noisy_compare.get(nt, {})
    ]
    avg_delta_by_snr[str(snr_db)] = round(sum(deltas) / len(deltas), 4) if deltas else 0.0

# Spremi JSON
comparison = {
    "clean_metrics": clean_compare,
    "noisy_f1":      noisy_compare,
    "avg_f1_delta_by_snr": avg_delta_by_snr,
    "v1_clean_source": "eval_clean_tuned.json ili eval_clean.json",
    "v3_clean_source": "eval_clean_v3.json",
}
out_path = REPORTS_PATH / "comparison_v1_vs_v3.json"
with open(out_path, "w") as f:
    json.dump(comparison, f, indent=2)
print(f"Uporedjivanje spremi -> {out_path}")

# Ispis rezimirajuce tabele
print("\n=== Cist test skup ===")
print(f"{'Metrika':<14} {'V1':>8} {'V3':>8} {'Delta':>8} {'Bolje':>8}")
print("-" * 50)
for metric in CLEAN_METRICS:
    c = clean_compare[metric]
    print(f"{metric:<14} {c['v1']:>8.4f} {c['v3']:>8.4f} {c['delta']:>+8.4f} {c['better']:>8}")

print("\n=== Zasumljeni test skup - F1 mjera ===")
header = f"{'Vrsta suma':<22}" + "".join(f"{s:>8}" for s in SNR_LEVELS)
print(header)
print("-" * (22 + 8 * len(SNR_LEVELS)))

for nt in NOISE_TYPES:
    v1_row = f"  V1  {nt:<18}"
    v3_row = f"  V3  {nt:<18}"
    delta_row = f"  D   {nt:<18}"
    for snr_db in SNR_LEVELS:
        key = f"snr_{snr_db}"
        c = noisy_compare.get(nt, {}).get(key, {})
        v1_f1  = c.get("v1_f1",  float("nan"))
        v3_f1  = c.get("v3_f1",  float("nan"))
        delta  = c.get("delta",  float("nan"))
        v1_row    += f"{v1_f1:>8.3f}"
        v3_row    += f"{v3_f1:>8.3f}"
        delta_row += f"{delta:>+8.3f}"
    print(v1_row)
    print(v3_row)
    print(delta_row)
    print()

print("=== Prosjecna razlika F1 (V3 - V1) po SNR ===")
for snr_db in SNR_LEVELS:
    d = avg_delta_by_snr.get(str(snr_db), 0.0)
    print(f"  SNR {snr_db:+3d} dB : {d:+.4f}")
