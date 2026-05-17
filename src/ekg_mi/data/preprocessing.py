"""
EKG signal preprocesiranje: bandpass filtering i z score normalization.
Signali ocekivani da budu oblika (N, duzina, kanal).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(
    signal: np.ndarray,
    fs: int,
    low: float = 0.5,
    high: float = 40.0,
) -> np.ndarray:
    """
    Butterworth bandpass cetvrtog reda po kanalu

    Parametri
    ----------
    signal : oblik (length, channels)
    fs   : sampling rate u Hz
    low   : donji cutoff frekvenije in Hz
    high : gornji cutoff freq in Hz

    Vraca
    -------
    Filtrirani signal, isti shape kao input.
    """
    sos = butter(4, [low, high], btype="band", fs=fs, output="sos")
    filtered = np.empty_like(signal)
    for ch in range(signal.shape[1]):
        filtered[:, ch] = sosfiltfilt(sos, signal[:, ch])
    return filtered


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Z-score normalizacija po kanalu (mean 0, std 1).
    """
    mean = signal.mean(axis=0)
    std  = signal.std(axis=0)
    # avoid division nulom za ravne kanale
    std  = np.where(std == 0, 1.0, std)
    return (signal - mean) / std


def preprocess_dataset(signals: np.ndarray, fs: int) -> np.ndarray:
    """
    Bandpass filtering i z-score normalization na sve signale.

    Parametri
    ----------
    signals : shape (N, length, channels)
    fs   : sampling rate in Hz

    Vraca
    -------
    Preprocessed signali, isti shape kao input, float32.
    """
    # Predobrada in place: nema kopije cijelog niza zobg stednje memorije
    for i in range(len(signals)):
        filtered   = bandpass_filter(signals[i], fs)
        signals[i] = normalize_signal(filtered)
    return signals.astype(np.float32)
