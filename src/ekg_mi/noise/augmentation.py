"""
Augmentacija suma za trening: batch-nivo omotac oko add_noise.
"""
from __future__ import annotations

import numpy as np
import torch

from .injection import add_noise


def augment_batch(X: torch.Tensor, noise_type: str, snr_db: float) -> torch.Tensor:
    """
    Primijeni sum na svaki uzorak u batchu.

    Parametri
    ---------
    X          : (B, C, N) float32 tensor (mora biti na CPU pri pozivu)
    noise_type : tip suma koji se prosljedjuje add_noise
    snr_db     : ciljna vrijednost omjera signal-sum u dB

    Vraca
    -----
    Tensor istog oblika i dtype-a kao ulazni X.
    """
    device = X.device
    arr    = X.cpu().numpy()  # (B, C, N)
    out    = np.empty_like(arr)
    for i in range(arr.shape[0]):
        noisy  = add_noise(arr[i], noise_type=noise_type, snr_db=snr_db)
        out[i] = noisy.astype(arr.dtype)
    return torch.from_numpy(out).to(device)


def sample_noise_params(
    rng:       np.random.Generator,
    types:     list[str],
    snr_range: tuple[float, float],
) -> tuple[str, float]:
    """
    Odaberi slucajni tip suma i SNR nivo pomocu datog numpy Generator objekta.

    Parametri
    ---------
    rng       : numpy.random.Generator (npr. np.random.default_rng(42))
    types     : lista dostupnih tipova suma
    snr_range : (min_snr_db, max_snr_db)

    Vraca
    -----
    (noise_type, snr_db)
    """
    noise_type = str(rng.choice(types))
    snr_db     = float(rng.uniform(snr_range[0], snr_range[1]))
    return noise_type, snr_db
