import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Jedan konvolucijski blok: Conv1d -> BN -> ReLU -> MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    """
    Baseline 1D CNN za binarnu klasifikaciju EKG signala.

    Ulazni oblik : (batch, num_channels, signal_length)
    Izlazni oblik: (batch, 1) -- sirovi logit za BCEWithLogitsLoss
    """

    def __init__(self, num_channels: int = 12) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _ConvBlock(num_channels, 32,  kernel_size=16),
            _ConvBlock(32,           64,  kernel_size=5),
            _ConvBlock(64,           128, kernel_size=5),
            _ConvBlock(128,          256, kernel_size=5),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
