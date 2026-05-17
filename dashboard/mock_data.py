"""
Shared mock data for all dashboard pages.
All generators are cached so every page sees the same data.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import streamlit as st

from ekg_mi.data.mock_dataset import generate_mock_dataset
from ekg_mi.noise.injection import add_noise


SEED = 42
N_SIGNALS = 20
SIGNAL_LENGTH = 1000
FS = 100

NOISE_TYPES = [
    "gaussian",
    "baseline_wander",
    "muscle_artifact",
    "electrode_motion",
    "powerline",
]

NOISE_LABELS_BS = {
    "gaussian":          "Gaussov bijeli sum",
    "baseline_wander":   "Bazno lutanje",
    "muscle_artifact":   "Misicni artefakt",
    "electrode_motion":  "Pomicanje elektrode",
    "powerline":         "Smetnja napajanja (50 Hz)",
}

SNR_VALUES = [-6, 0, 6, 12, 18, 24]


@st.cache_data(show_spinner=False)
def get_mock_signals() -> tuple[np.ndarray, np.ndarray]:
    """Return (signals, labels) arrays — 20 samples, shape (N, 1, 1000)."""
    signals, labels = generate_mock_dataset(
        n_normal=N_SIGNALS // 2,
        n_mi=N_SIGNALS // 2,
        length=SIGNAL_LENGTH,
        sampling_rate=FS,
        seed=SEED,
    )
    return signals, labels


@st.cache_data(show_spinner=False)
def get_noisy_signal(signal_1d: np.ndarray, noise_type: str, snr_db: float) -> np.ndarray:
    """Add noise to a 1D signal and return noisy 1D array."""
    return add_noise(signal_1d, noise_type=noise_type, snr_db=snr_db)


@st.cache_data(show_spinner=False)
def get_mock_training_history() -> dict:
    """Simulated training history over 40 epochs."""
    rng = np.random.default_rng(SEED)
    epochs = 40
    base_tr = 1 / (1 + np.exp(-0.2 * (np.arange(epochs) - 10)))
    base_vl = 1 / (1 + np.exp(-0.18 * (np.arange(epochs) - 12)))
    return {
        "train_loss": (0.7 - 0.6 * base_tr + rng.normal(0, 0.01, epochs)).tolist(),
        "val_loss":   (0.75 - 0.58 * base_vl + rng.normal(0, 0.015, epochs)).tolist(),
        "train_acc":  (0.5 + 0.48 * base_tr + rng.normal(0, 0.008, epochs)).clip(0, 1).tolist(),
        "val_acc":    (0.5 + 0.45 * base_vl + rng.normal(0, 0.012, epochs)).clip(0, 1).tolist(),
    }


@st.cache_data(show_spinner=False)
def get_mock_metrics() -> dict[str, float]:
    """Simulated evaluation metrics on the test set."""
    return {
        "accuracy":    0.9467,
        "precision":   0.9512,
        "recall":      0.9389,
        "specificity": 0.9545,
        "f1":          0.9450,
        "auc":         0.9821,
    }


@st.cache_data(show_spinner=False)
def get_robustness_matrix() -> np.ndarray:
    """
    Accuracy matrix: rows = noise types (5), cols = SNR levels (6).
    Sigmoid curve per noise type to simulate degradation at low SNR.
    """
    rng = np.random.default_rng(SEED + 1)
    snr = np.array(SNR_VALUES, dtype=float)
    # different degradation rates per noise type
    midpoints = [5, 3, 6, 4, 2]
    slopes    = [0.25, 0.30, 0.22, 0.28, 0.32]
    matrix = np.zeros((len(NOISE_TYPES), len(SNR_VALUES)))
    for i, (mid, slp) in enumerate(zip(midpoints, slopes)):
        base = 1 / (1 + np.exp(-slp * (snr - mid)))
        # scale so max ~0.94 and floor ~0.52
        matrix[i] = 0.52 + 0.42 * base + rng.normal(0, 0.008, len(SNR_VALUES))
    return matrix.clip(0, 1)
