"""
Trening petlja sa early stopping i cuvanjem najboljeg modela.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam


def train(
    model:        torch.nn.Module,
    train_loader,
    val_loader,
    config:       dict,
) -> dict:
    """
    Train model with early stopping. Saves best weights and history JSON.

    Parameters
    ----------
    model        : nn.Module (already on target device)
    train_loader : DataLoader
    val_loader   : DataLoader
    config       : full project config dict - uses config['training'] section

    Returns
    -------
    history : dict with lists train_loss, val_loss, train_acc, val_acc
    """
    cfg        = config["training"]
    lr         = float(cfg.get("lr", 1e-3))
    max_epochs = int(cfg.get("max_epochs", 50))
    patience   = int(cfg.get("early_stopping_patience", 7))

    save_dir = Path("outputs/models")
    save_dir.mkdir(parents=True, exist_ok=True)

    device    = next(model.parameters()).device
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }
    best_val_loss    = float("inf")
    patience_counter = 0
    best_epoch       = 0

    for epoch in range(1, max_epochs + 1):
        # --- trening faza ---
        model.train()
        t_loss = t_correct = t_total = 0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(signals)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * len(labels)
            t_correct += (logits.argmax(1) == labels).sum().item()
            t_total   += len(labels)

        # --- validacijska faza ---
        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                logits = model(signals)
                loss   = criterion(logits, labels)
                v_loss    += loss.item() * len(labels)
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_total   += len(labels)

        tl = t_loss / t_total
        vl = v_loss / v_total
        ta = t_correct / t_total
        va = v_correct / v_total

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)

        print(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"train loss={tl:.4f} acc={ta:.3f} | "
            f"val loss={vl:.4f} acc={va:.3f}"
        )

        # Cuvanje najboljeg modela i early stopping
        if vl < best_val_loss:
            best_val_loss    = vl
            patience_counter = 0
            best_epoch       = epoch
            torch.save(model.state_dict(), save_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best: {best_epoch})")
                break

    # Sacuvaj historiju treninga u JSON
    history_path = save_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved -> {history_path}")

    return history
