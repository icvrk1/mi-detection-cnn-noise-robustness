from __future__ import annotations

import numpy as np
import torch

from .metrics import compute_metrics


def evaluate(model: torch.nn.Module, dataloader, threshold: float = 0.5) -> dict:
    """
    Pokretanje inferencije i racunanje metrika.

    Parameters
    ----------
    model     : trained PyTorch model
    dataloader: DataLoader yielding (signals, labels) batches
    threshold : decision threshold applied to sigmoid probabilities (default 0.5)

    Returns
    -------
    dict sa skalarnim kljucevima: accuracy, precision, recall, specificity, f1, auc
         i nizovnim kljucevima:   y_true, y_pred, y_scores (numpy nizovi)
    """
    model.eval()
    device = next(model.parameters()).device

    all_true:   list[np.ndarray] = []
    all_pred:   list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    with torch.no_grad():
        for signals, labels in dataloader:
            signals = signals.to(device)
            logits  = model(signals).squeeze(1)     # (B,)
            probs   = torch.sigmoid(logits)
            preds   = (probs >= threshold).long()

            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_scores.append(probs.cpu().numpy())

    y_true   = np.concatenate(all_true)
    y_pred   = np.concatenate(all_pred)
    y_scores = np.concatenate(all_scores)

    metrics = compute_metrics(y_true, y_pred, y_scores)
    metrics["y_true"]   = y_true
    metrics["y_pred"]   = y_pred
    metrics["y_scores"] = y_scores
    return metrics
