"""BASELINE heavy model: Conv1d feature extractor + LSTM temporal aggregator.

Input is the raw window series (window_days × steps_per_day), treated as one
1-D signal. Convolutions extract local consumption patterns; adaptive pooling
compresses to one vector per day; an LSTM aggregates across days.

This is a research baseline, not the proposed model. Its measured parameter
count, training time, and memory are reported honestly by the evaluation
module.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.base import TorchBinaryClassifier
from utilities.config import get


class CNNLSTM(nn.Module):
    def __init__(
        self,
        window_days: int,
        steps_per_day: int,
        conv_channels: int = 16,
        lstm_hidden: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        channels = conv_channels
        self.features = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # after two pools the length is ~ series_len / 4; pool to one vector/day
        per_day_len = 8
        self.pool = nn.AdaptiveAvgPool1d(window_days * per_day_len)
        self.lstm = nn.LSTM(input_size=channels * per_day_len, hidden_size=lstm_hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(lstm_hidden, 1))
        self.window_days = window_days

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, T) → treat as single channel signal (B, 1, W*T)
        b = x.shape[0]
        sig = x.reshape(b, 1, -1)
        feat = self.features(sig)
        pooled = self.pool(feat)                      # (B, C, W*per_day_len)
        seq = pooled.reshape(b, self.window_days, -1)  # (B, W, C*per_day_len)
        out, _ = self.lstm(seq)
        return self.head(out[:, -1, :])                # last time step


def build_cnn_lstm(cfg: dict, window_days: int, steps_per_day: int, seed: int = 42) -> TorchBinaryClassifier:
    torch.manual_seed(seed)
    model = CNNLSTM(
        window_days=window_days,
        steps_per_day=steps_per_day,
        conv_channels=int(get(cfg, "model.cnn_lstm.conv_channels", 16)),
        lstm_hidden=int(get(cfg, "model.cnn_lstm.lstm_hidden", 32)),
        kernel_size=int(get(cfg, "model.cnn_lstm.kernel_size", 3)),
        dropout=float(get(cfg, "model.cnn_lstm.dropout", 0.2)),
    )
    return TorchBinaryClassifier(
        model,
        name="cnn_lstm_baseline",
        learning_rate=float(get(cfg, "model.cnn_lstm.learning_rate", 1e-3)),
        weight_decay=float(get(cfg, "federated.weight_decay", 1e-4)),
    )
