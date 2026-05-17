"""
Smoke test: verifikacija cijelog pipeline-a na mock podacima.
Cilj nije kvalitet modela - 60% accuracy je sasvim OK.
Cilj je da cijeli pipeline prodje bez gresaka.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Dodaj src/ na putanju za pokretanje bez editable install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from ekg_mi.data.dataset          import EKGDataset
from ekg_mi.data.mock_dataset     import generate_mock_dataset
from ekg_mi.evaluation.evaluator  import evaluate
from ekg_mi.models.baseline_cnn   import BaselineCNN
from ekg_mi.training.trainer      import train
from ekg_mi.visualization.confusion      import plot_confusion_matrix
from ekg_mi.visualization.signals        import plot_ecg_segment
from ekg_mi.visualization.training_plots import plot_training_curves

# ---- Podesavanja ----
SEED          = 42
N_NORMAL      = 200
N_MI          = 200
SIGNAL_LENGTH = 1000
FS            = 100
BATCH_SIZE    = 32
MAX_EPOCHS    = 5

torch.manual_seed(SEED)
np.random.seed(SEED)

config = {
    "training": {
        "lr":                      1e-3,
        "max_epochs":              MAX_EPOCHS,
        "early_stopping_patience": 10,   # bez ranog zaustavljanja u smoke testu
    }
}

# ---- 1. Mock podaci ----
print("=" * 60)
print("1. Generisanje mock ECG podataka...")
signals, labels = generate_mock_dataset(
    n_normal=N_NORMAL, n_mi=N_MI,
    length=SIGNAL_LENGTH, sampling_rate=FS, seed=SEED,
)
print(f"   signals: {signals.shape}  {signals.dtype}")
print(f"   labels:  {labels.shape}   normal={int((labels == 0).sum())}  MI={int((labels == 1).sum())}")

# ---- 2. Split 70/15/15 ----
rng   = np.random.default_rng(SEED)
idx   = rng.permutation(len(labels))
n_tr  = int(0.70 * len(labels))
n_val = int(0.15 * len(labels))

train_idx = idx[:n_tr]
val_idx   = idx[n_tr : n_tr + n_val]
test_idx  = idx[n_tr + n_val :]

train_ds = EKGDataset(signals[train_idx], labels[train_idx])
val_ds   = EKGDataset(signals[val_idx],   labels[val_idx])
test_ds  = EKGDataset(signals[test_idx],  labels[test_idx])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"\n2. Split: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

# ---- 3. Model ----
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model    = BaselineCNN(num_channels=1, num_classes=2).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\n3. Model: BaselineCNN  params={n_params:,}  device={device}")

# ---- 4. Trening ----
print(f"\n4. Trening ({MAX_EPOCHS} epoha)...")
history = train(model, train_loader, val_loader, config)

# ---- 5. Ucitaj najbolji model i evaluiraj ----
print("\n5. Evaluacija na test skupu...")
best_weights = Path("outputs/models/best_model.pt")
model.load_state_dict(torch.load(best_weights, map_location=device, weights_only=True))
results = evaluate(model, test_loader)

print("\nMetrike:")
for k, v in results.items():
    if k not in ("y_true", "y_pred", "y_scores"):
        print(f"   {k:15s}: {v:.4f}")

# ---- 6. Vizualizacije ----
out_fig = Path("outputs/figures")
out_fig.mkdir(parents=True, exist_ok=True)

plot_training_curves(history, save_path=out_fig / "smoke_training_curves.png")
plot_confusion_matrix(
    results["y_true"], results["y_pred"],
    save_path=out_fig / "smoke_confusion_matrix.png",
)

# Jedan primjer normalnog i MI signala
normal_idx = int(np.where(labels == 0)[0][0])
mi_idx     = int(np.where(labels == 1)[0][0])

plot_ecg_segment(
    signals[normal_idx, 0], title="Normal ECG (mock)", fs=FS,
    save_path=out_fig / "smoke_ecg_normal.png",
)
plot_ecg_segment(
    signals[mi_idx, 0], title="MI ECG (mock)", fs=FS,
    save_path=out_fig / "smoke_ecg_mi.png",
)

# ---- 7. Provjera izlaznih fajlova ----
print("\n6. Generirani fajlovi:")
output_files = [
    "outputs/models/best_model.pt",
    "outputs/models/training_history.json",
    "outputs/figures/smoke_training_curves.png",
    "outputs/figures/smoke_confusion_matrix.png",
    "outputs/figures/smoke_ecg_normal.png",
    "outputs/figures/smoke_ecg_mi.png",
]
all_ok = True
for f in output_files:
    p = Path(f)
    if p.exists():
        print(f"   OK      {f}  ({p.stat().st_size:,} bytes)")
    else:
        print(f"   MISSING {f}")
        all_ok = False

print("\n" + "=" * 60)
if all_ok:
    print("Smoke test PROSAO - pipeline je spreman.")
else:
    print("Smoke test NEUSPJESAN - neki fajlovi nedostaju.")
    sys.exit(1)
