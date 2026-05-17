import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_ecg_segment(
    signal,
    noise=None,
    title: str | None = None,
    fs: int = 100,
    save_path=None,
):
    """
    Plot a single ECG segment with optional noise overlay.

    Parameters
    ----------
    signal    : 1D array-like (clean signal)
    noise     : 1D array-like, optional (added on top of signal for display)
    title     : plot title
    fs        : sampling rate in Hz
    save_path : str or Path, optional
    """
    signal = np.asarray(signal)
    t = np.arange(len(signal)) / fs

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, signal, lw=0.8, color="royalblue", label="signal")

    if noise is not None:
        noisy = signal + np.asarray(noise)
        ax.plot(t, noisy, lw=0.6, alpha=0.7, color="tomato", label="signal + noise")
        ax.legend()

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title or "ECG Segment")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return fig
