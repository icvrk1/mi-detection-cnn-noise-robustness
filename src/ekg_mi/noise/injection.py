import numpy as np

from .nstdb import _RECORD_MAP, load_nstdb_noise
from .synthetic import generate_gaussian, generate_powerline

_NSTDB_TYPES: frozenset[str] = frozenset(_RECORD_MAP)
_SYNTHETIC_TYPES: frozenset[str] = frozenset({"powerline", "gaussian"})
_VALID_TYPES: frozenset[str] = _NSTDB_TYPES | _SYNTHETIC_TYPES
_FS = 100  # project sampling rate (Hz)


def add_noise(
    signal: np.ndarray,
    noise_type: str,
    snr_db: float,
    noise_source: np.ndarray | None = None,
) -> np.ndarray:
    """
    Add noise to an ECG signal at a precise target SNR.

    Parameters
    ----------
    signal      : shape (N,) or (C, N)
    noise_type  : 'baseline_wander' | 'muscle_artifact' | 'electrode_motion'
                  | 'powerline' | 'gaussian'
    snr_db      : target signal-to-noise ratio in dB
    noise_source: optional pre-loaded 1D noise array; bypasses internal loading /
                  generation.  Random-segment selection still applies when it is
                  longer than the signal.

    Returns
    -------
    Noisy signal with the same shape and dtype as the input.
    """
    if noise_type not in _VALID_TYPES:
        raise ValueError(
            f"Unknown noise type {noise_type!r}. Valid: {sorted(_VALID_TYPES)}"
        )

    signal = np.asarray(signal, dtype=np.float64)
    is_1d = signal.ndim == 1
    if is_1d:
        signal = signal[np.newaxis, :]  # treat as (1, N)

    _n_channels, n_samples = signal.shape

    # Load the full raw noise array once for NSTDB types.
    if noise_source is not None:
        raw_noise: np.ndarray | None = np.asarray(noise_source, dtype=np.float64).ravel()
    elif noise_type in _NSTDB_TYPES:
        raw_noise = load_nstdb_noise(noise_type)
    else:
        raw_noise = None  # generated fresh per channel

    out_channels: list[np.ndarray] = []
    for ch in signal:
        P_signal = float(np.mean(ch ** 2))

        if P_signal == 0.0:
            out_channels.append(ch.copy())
            continue

        # Obtain a noise segment of exactly n_samples.
        if raw_noise is not None:
            avail = len(raw_noise)
            if avail >= n_samples:
                start = np.random.randint(0, avail - n_samples + 1)
                segment = raw_noise[start : start + n_samples].copy()
            else:
                # Tile if noise array is shorter than the signal (rare edge case).
                reps = int(np.ceil(n_samples / avail))
                segment = np.tile(raw_noise, reps)[:n_samples].copy()
        elif noise_type == "powerline":
            segment = generate_powerline(n_samples, _FS)
        else:  # gaussian
            segment = generate_gaussian(n_samples)

        P_noise = float(np.mean(segment ** 2))
        if P_noise == 0.0:
            raise ValueError(
                f"Noise segment for type {noise_type!r} has zero power."
            )

        alpha = np.sqrt(P_signal / (P_noise * 10.0 ** (snr_db / 10.0)))
        out_channels.append(ch + alpha * segment)

    result = np.stack(out_channels, axis=0)
    return result[0] if is_1d else result
