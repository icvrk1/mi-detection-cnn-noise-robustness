"""
Trening BaselineCNN V3 sa augmentacijom suma tokom treninga.

Ista arhitektura, isti hiperparametri i iste podjele podataka kao V1.
Jedina razlika: tokom treninga, sa vjerovatnocom AUG_PROB = 0.5, svaki
mini-batch se zasumljuje slucajno odabranom vrstom suma i SNR nivoom.
Validacijski i testni skupovi ostaju cisti.

  AUG_PROB        = 0.5
  AUG_NOISE_TYPES = svih pet tipova
  AUG_SNR_RANGE   = (0, 24) dB

Sprema:
  outputs/models/best_model_v3.pt
  outputs/models/final_model_v3.pt
  outputs/logs/training_history_v3.json
  outputs/figures/training_curves_v3.{png,pdf}
  outputs/reports/model_architecture_v3.txt
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ekg_mi.models.baseline_cnn import BaselineCNN
from ekg_mi.utils.seed import set_seed
from ekg_mi.noise.augmentation import augment_batch, sample_noise_params

CONFIG_PATH = Path("configs/config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

set_seed(cfg["seed"])

PROCESSED_PATH = Path(cfg["paths"]["processed"])
MODELS_PATH    = Path(cfg["paths"]["models"])
FIGURES_PATH   = Path(cfg["paths"]["figures"])
REPORTS_PATH   = Path(cfg["paths"]["reports"])
LOGS_PATH      = Path("outputs/logs")

MODELS_PATH.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)
REPORTS_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

# Parametri augmentacije suma
AUG_PROB        = 0.5
AUG_NOISE_TYPES = ["gaussian", "baseline_wander", "muscle_artifact",
                   "electrode_motion", "powerline"]
AUG_SNR_RANGE   = (0.0, 24.0)  # dB; izbjegavamo SNR < 0 dB tokom treninga

# Deterministicki RNG za augmentaciju (izoliran od globalnog numpy stanja)
aug_rng = np.random.default_rng(cfg["seed"])

t_cfg      = cfg["training"]
BATCH_SIZE = int(t_cfg["batch_size"])
LR         = float(t_cfg["lr"])
MAX_EPOCHS = int(t_cfg["max_epochs"])
PATIENCE   = int(t_cfg["early_stopping_patience"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Uredaj: {device}")
print(f"Augmentacija suma: prob={AUG_PROB}, SNR=[{AUG_SNR_RANGE[0]}, {AUG_SNR_RANGE[1]}] dB")

# Ucitaj podatke
print("Ucitavanje train/val skupova ...")
train_data = np.load(PROCESSED_PATH / "train.npz")
val_data   = np.load(PROCESSED_PATH / "val.npz")

X_train = torch.tensor(train_data["signals"], dtype=torch.float32)
y_train = torch.tensor(train_data["labels"],  dtype=torch.float32)
X_val   = torch.tensor(val_data["signals"],   dtype=torch.float32)
y_val   = torch.tensor(val_data["labels"],    dtype=torch.float32)

print(f"  Train: {X_train.shape}  (MI={int(y_train.sum())}, NORM={int((y_train==0).sum())})")
print(f"  Val  : {X_val.shape}  (MI={int(y_val.sum())}, NORM={int((y_val==0).sum())})")

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# pos_weight za neravnomjernost klasa
n_norm = int((y_train == 0).sum())
n_mi   = int((y_train == 1).sum())
pos_weight = torch.tensor([n_norm / n_mi], dtype=torch.float32, device=device)
print(f"  pos_weight: {pos_weight.item():.3f}  (NORM={n_norm}, MI={n_mi})")

# Model
model = BaselineCNN(num_channels=cfg["model"]["in_channels"]).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Parametri modela: {total_params:,}")

try:
    import torchinfo
    summary = torchinfo.summary(model, input_size=(1, cfg["model"]["in_channels"], 1000), verbose=0)
    arch_txt = str(summary)
except Exception:
    arch_txt = f"BaselineCNN  in_channels={cfg['model']['in_channels']}  parametri={total_params:,}"
arch_path = REPORTS_PATH / "model_architecture_v3.txt"
arch_path.write_text(arch_txt, encoding="utf-8")
print(f"  Arhitektura spr. -> {arch_path}")

optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# Trening petlja
history: dict[str, list] = {
    "train_loss": [], "val_loss": [],
    "train_acc":  [], "val_acc":  [],
    "train_f1":   [], "val_f1":   [],
    "lr":         [],
    "aug_count":  [],  # broj augmentiranih batcheva po epohi
}
best_val_loss    = float("inf")
patience_counter = 0
best_epoch       = 0
t0 = time.time()

print(f"\nPocetak treninga V3 ({MAX_EPOCHS} epoha, patience={PATIENCE})")
print("-" * 80)

for epoch in range(1, MAX_EPOCHS + 1):
    t_ep = time.time()

    # Trening sa augmentacijom suma
    model.train()
    t_loss = t_correct = t_total = 0
    t_pred_list: list[np.ndarray] = []
    t_true_list: list[np.ndarray] = []
    t_aug_count = 0

    for X_batch, y_batch in train_loader:
        # Augmentacija suma na CPU, prije premjestanja na uredaj
        if aug_rng.random() < AUG_PROB:
            noise_type, snr_db = sample_noise_params(aug_rng, AUG_NOISE_TYPES, AUG_SNR_RANGE)
            X_batch = augment_batch(X_batch, noise_type, snr_db)
            t_aug_count += 1

        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch).squeeze(1)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        preds     = (torch.sigmoid(logits) >= 0.5).long()
        t_loss    += loss.item() * len(y_batch)
        t_correct += (preds == y_batch.long()).sum().item()
        t_total   += len(y_batch)
        t_pred_list.append(preds.cpu().numpy())
        t_true_list.append(y_batch.long().cpu().numpy())

    # Validacija (bez augmentacije)
    model.eval()
    v_loss = v_correct = v_total = 0
    v_pred_list: list[np.ndarray] = []
    v_true_list: list[np.ndarray] = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch).squeeze(1)
            loss   = criterion(logits, y_batch)
            preds  = (torch.sigmoid(logits) >= 0.5).long()
            v_loss    += loss.item() * len(y_batch)
            v_correct += (preds == y_batch.long()).sum().item()
            v_total   += len(y_batch)
            v_pred_list.append(preds.cpu().numpy())
            v_true_list.append(y_batch.long().cpu().numpy())

    tl  = t_loss / t_total
    vl  = v_loss / v_total
    ta  = t_correct / t_total
    va  = v_correct / v_total
    tf1 = f1_score(np.concatenate(t_true_list), np.concatenate(t_pred_list), zero_division=0)
    vf1 = f1_score(np.concatenate(v_true_list), np.concatenate(v_pred_list), zero_division=0)
    cur_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(tl)
    history["val_loss"].append(vl)
    history["train_acc"].append(ta)
    history["val_acc"].append(va)
    history["train_f1"].append(tf1)
    history["val_f1"].append(vf1)
    history["lr"].append(cur_lr)
    history["aug_count"].append(t_aug_count)

    scheduler.step(vl)

    marker  = " *" if vl < best_val_loss else ""
    ep_time = time.time() - t_ep
    n_batches = len(train_loader)
    print(
        f"Ep {epoch:3d}/{MAX_EPOCHS} | "
        f"train: L={tl:.4f} A={ta:.3f} F1={tf1:.3f} | "
        f"val: L={vl:.4f} A={va:.3f} F1={vf1:.3f} | "
        f"aug={t_aug_count}/{n_batches} | "
        f"lr={cur_lr:.2e} ({ep_time:.0f}s){marker}"
    )

    if vl < best_val_loss:
        best_val_loss    = vl
        patience_counter = 0
        best_epoch       = epoch
        torch.save(model.state_dict(), MODELS_PATH / "best_model_v3.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Rano zaustavljanje na epohi {epoch} (najbolja epoha: {best_epoch})")
            break

# Spremi finalni model
torch.save(model.state_dict(), MODELS_PATH / "final_model_v3.pt")

# Spremi historiju
history["best_epoch"]    = best_epoch
history["best_val_loss"] = best_val_loss
history_path = LOGS_PATH / "training_history_v3.json"
with open(history_path, "w") as f:
    json.dump(history, f, indent=2)

elapsed = time.time() - t0
print(f"\nTrening V3 zavrsen za {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"  Najbolja epoha    : {best_epoch}")
print(f"  Najbolji val loss : {best_val_loss:.4f}")
print(f"  Najbolji val F1   : {max(history['val_f1']):.4f}")
print(f"  Model spremen    -> {MODELS_PATH / 'best_model_v3.pt'}")
print(f"  Historija spr.   -> {history_path}")

# Krive treninga
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

epochs_range = list(range(1, len(history["train_loss"]) + 1))
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(epochs_range, history["train_loss"], label="trening")
axes[0].plot(epochs_range, history["val_loss"],   label="validacija")
axes[0].set_title("Funkcija gubitka")
axes[0].set_xlabel("Epoha")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history["train_acc"], label="trening")
axes[1].plot(epochs_range, history["val_acc"],   label="validacija")
axes[1].set_title("Tacnost")
axes[1].set_xlabel("Epoha")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs_range, history["train_f1"], label="trening")
axes[2].plot(epochs_range, history["val_f1"],   label="validacija")
axes[2].set_title("F1 mjera")
axes[2].set_xlabel("Epoha")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

fig.suptitle("V3 - trening sa augmentacijom suma", fontweight="bold")
plt.tight_layout()
fig.savefig(FIGURES_PATH / "training_curves_v3.png", dpi=300)
fig.savefig(FIGURES_PATH / "training_curves_v3.pdf")
plt.close(fig)
print(f"  Krive spr.       -> {FIGURES_PATH / 'training_curves_v3.png'}")

if max(history["val_f1"]) < 0.70:
    print("\n  UPOZORENJE: val F1 < 0.70 - rezultati mogu biti lose. Provjeri podatke.")
