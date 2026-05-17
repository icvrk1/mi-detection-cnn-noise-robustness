import numpy as np


def generate_powerline(
    length: int, fs: float, frequency: float = 50.0, phase: float = np.pi / 4
) -> np.ndarray:
    # Phase offset avoids degenerate all-zero samples when frequency == fs/2
    # (e.g. 50 Hz at 100 Hz sampling rate: sin(π·k) = 0 for every integer k).
    t = np.arange(length) / fs
    return np.sin(2 * np.pi * frequency * t + phase)


def generate_gaussian(length: int) -> np.ndarray:
    return np.random.randn(length)
