from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    y_scores: np.ndarray,
) -> dict[str, float]:
    """
    Racunanje binarnih metrika klasifikacije.

    Parameters
    ----------
    y_true   : int niz  (0 = normalan, 1 = MI)
    y_pred   : int niz  (predvidena klasa)
    y_scores : float niz (vjerovatnoca klase 1)

    Returns
    -------
    dict sa kljucevima: accuracy, precision, recall, specificity, f1, auc_roc, auc_pr
    """
    y_true   = np.asarray(y_true)
    y_pred   = np.asarray(y_pred)
    y_scores = np.asarray(y_scores)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc":     float(roc_auc_score(y_true, y_scores)),
        "auc_pr":      float(average_precision_score(y_true, y_scores)),
    }
