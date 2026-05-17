"""
Podesavanje praga odluke za V3 model (trening sa augmentacijom suma).

Isti postupak kao za V1: jedna inferencija na val.npz, zatim pretraga
pragova od 0.30 do 0.70 (korak 0.01), odabire onaj koji maksimizira F1.
Tezine modela se ne mijenjaju.

Sprema:
  outputs/reports/threshold_tuning_v3.json
  outputs/figures/thesis/fig_threshold_sweep_v3.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ekg_mi.models.baseline_cnn import BaselineCNN
from ekg_mi.visualization.style import set_thesis_style

set_thesis_style()

CONFIG_PATH = Path("configs/config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

PROCESSED_PATH = Path(cfg["paths"]["processed"])
MODELS_PATH    = Path(cfg["paths"]["models"])
REPORTS_PATH   = Path(cfg["paths"]["reports"])
FIGURES_PATH   = Path("outputs/figures/thesis")
REPORTS_PATH.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODELS_PATH / "best_model_v3.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ucitavanje validacijskog skupa
print("Ucitavanje validacijskog skupa ...")
val_data = np.load(PROCESSED_PATH / "val.npz")
X_val = torch.tensor(val_data["signals"], dtype=torch.float32)
y_val = torch.tensor(val_data["labels"],  dtype=torch.long)
print(f"  Val: {X_val.shape}  (MI={int((y_val==1).sum())}, NORM={int((y_val==0).sum())})")

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=int(cfg["training"]["batch_size"]),
    shuffle=False,
    num_workers=0,
)

# Ucitavanje V3 modela
print(f"Ucitavanje modela: {MODEL_FILE}")
model = BaselineCNN(num_channels=cfg["model"]["in_channels"]).to(device)
model.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
model.eval()
print("  Model ucitan (tezine nepromijenjene).")

# Jedna inferencija - prikupljanje vjerovatnoca
print("Prikupljanje vjerovatnoca na validacijskom skupu ...")
all_probs: list[np.ndarray] = []
all_true:  list[np.ndarray] = []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        logits = model(X_batch.to(device)).squeeze(1)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_true.append(y_batch.numpy())

y_probs = np.concatenate(all_probs)
y_true  = np.concatenate(all_true)

# Pretraga praga
thresholds      = np.arange(0.30, 0.701, 0.01)
sweep_f1        = []
default_val_f1  = float(f1_score(y_true, (y_probs >= 0.5).astype(int), zero_division=0))

best_f1  = -1.0
best_thr = 0.5

for thr in thresholds:
    y_pred = (y_probs >= thr).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    sweep_f1.append({"threshold": round(float(thr), 2), "val_f1": round(f1, 6)})
    if f1 > best_f1:
        best_f1  = f1
        best_thr = float(thr)

best_thr = round(best_thr, 2)

print(f"\nPrag odluke 0.5  : val F1 = {default_val_f1:.4f}")
print(f"Optimalni prag   : {best_thr:.2f}  (val F1 = {best_f1:.4f})")
delta = best_f1 - default_val_f1
print(f"Poboljsanje F1   : {delta:+.4f}")

# Spremi JSON
tuning = {
    "best_threshold":    best_thr,
    "best_val_f1":       round(best_f1, 6),
    "default_threshold": 0.5,
    "default_val_f1":    round(default_val_f1, 6),
    "sweep": sweep_f1,
}
json_path = REPORTS_PATH / "threshold_tuning_v3.json"
with open(json_path, "w") as f:
    json.dump(tuning, f, indent=2)
print(f"\nRezultati podesavanja spremi -> {json_path}")

# Slika: val F1 u zavisnosti od praga
thr_vals = [entry["threshold"] for entry in sweep_f1]
f1_vals  = [entry["val_f1"]    for entry in sweep_f1]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(thr_vals, f1_vals, color="#0072B2", lw=1.8, marker="o", ms=4, zorder=3,
        label="F1 mjera (V3)")
ax.axvline(0.5,      color="gray",   lw=1.2, ls="--", zorder=2,
           label=f"Podrazumijevani prag (0.5) - F1={default_val_f1:.4f}")
ax.axvline(best_thr, color="#D55E00", lw=1.4, ls="-", zorder=4,
           label=f"Optimalni prag ({best_thr:.2f}) - F1={best_f1:.4f}")
ax.scatter([best_thr], [best_f1], color="#D55E00", s=70, zorder=5)
ax.set_xlabel("Prag odluke")
ax.set_ylabel("F1 mjera")
ax.set_title("V3 - F1 mjera na validacijskom skupu u zavisnosti od praga odluke",
             fontweight="bold")
ax.legend(fontsize=8)
ax.set_xlim(0.28, 0.72)
fig.tight_layout()

for ext in ("pdf", "png"):
    p = FIGURES_PATH / f"fig_threshold_sweep_v3.{ext}"
    fig.savefig(p, dpi=300 if ext == "png" else None)
    print(f"  -> {p}")
plt.close(fig)
