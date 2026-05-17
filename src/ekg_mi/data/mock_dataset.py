"""
Generira sinteticki ECG dataset za testiranje pipeline-a.
MI klasa se razlikuje inverzijom T-talasa - nije medicinski realisticno,
ali je konzistentno razlicita od normalnih signala.
"""
from __future__ import annotations

import numpy as np
import neurokit2 as nk


def _invert_t_waves(ecg: np.ndarray, sampling_rate: int, heart_rate: int) -> np.ndarray:
    """Invertira T-talas u svakom srcanom ciklusu (aproksimacija periodicnim prozorom)."""
    ecg = ecg.copy()
    rr = int(sampling_rate * 60 / heart_rate)
    # T-talas je grubo 25-55% RR intervala - nema potrebe za detekcijom pikova
    t_start = int(0.25 * rr)
    t_end   = int(0.55 * rr)
    for i in range(0, len(ecg) - t_start, rr):
        s = i + t_start
        e = min(i + t_end, len(ecg))
        ecg[s:e] = -ecg[s:e] * 0.85  # inverzija sa blagim smanjenjem amplitude
    return ecg


def generate_mock_dataset(
    n_normal: int = 200,
    n_mi:     int = 200,
    length:   int = 1000,
    sampling_rate: int = 100,
    seed:     int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic ECG signals for pipeline testing.

    Normal class : standard NeuroKit ECG, heart rate 60-80 bpm
    MI class     : same simulation + T-wave inversion, heart rate 75-100 bpm

    Returns
    -------
    signals : np.ndarray, shape (N, 1, length), dtype float32
    labels  : np.ndarray, shape (N,),           dtype int64
    """
    rng      = np.random.default_rng(seed)
    duration = length / sampling_rate

    signals: list[np.ndarray] = []
    labels:  list[int]        = []

    # Normalni signali - klasa 0
    for _ in range(n_normal):
        hr  = int(rng.integers(60, 81))
        ecg = nk.ecg_simulate(
            duration=duration,
            sampling_rate=sampling_rate,
            heart_rate=hr,
            random_state=int(rng.integers(0, 100_000)),
            method="simple",
        )
        signals.append(np.asarray(ecg[:length], dtype=np.float32))
        labels.append(0)

    # MI signali - klasa 1, isti simulator ali sa inverzijom T-talasa
    for _ in range(n_mi):
        hr  = int(rng.integers(75, 101))
        ecg = nk.ecg_simulate(
            duration=duration,
            sampling_rate=sampling_rate,
            heart_rate=hr,
            random_state=int(rng.integers(0, 100_000)),
            method="simple",
        )
        ecg = _invert_t_waves(np.asarray(ecg[:length], dtype=np.float32), sampling_rate, hr)
        signals.append(ecg)
        labels.append(1)

    sig_arr = np.stack(signals, axis=0)[:, np.newaxis, :]  # (N, 1, L)
    lbl_arr = np.array(labels, dtype=np.int64)

    # Shuffle da ne budu sve 0 pa sve 1
    idx = rng.permutation(len(lbl_arr))
    return sig_arr[idx], lbl_arr[idx]
