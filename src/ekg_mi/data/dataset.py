import numpy as np
import torch
from torch.utils.data import Dataset


class EKGDataset(Dataset):
    """
    Wraps NumPy signal and label arrays into a PyTorch Dataset.

    Parameters
    ----------
    signals : np.ndarray, shape (N, C, L) -- float32
    labels  : np.ndarray, shape (N,)      -- int64
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray) -> None:
        self.signals = torch.from_numpy(signals).float()
        self.labels  = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.signals[idx], self.labels[idx]
