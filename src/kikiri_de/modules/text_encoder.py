"""Acoustic text encoder: token IDs -> per-token features for the decoder.

Adapted from StyleTTS 2 (MIT) and Kokoro (Apache 2.0).
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from .layers import LayerNorm


def run_lstm(lstm: nn.LSTM, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    '''Run a batch-first LSTM over padded [B, T, C] input; output is zero-padded back to T.'''
    total = x.shape[1]
    packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
    lstm.flatten_parameters()
    out, _ = lstm(packed)
    out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=total)
    return out


class TextEncoder(nn.Module):
    def __init__(self, channels: int, kernel_size: int, depth: int, n_symbols: int):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                nn.LeakyReLU(0.2),
            )
            for _ in range(depth)
        ])
        self.lstm = nn.LSTM(channels, channels // 2, 1, batch_first=True, bidirectional=True)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''tokens [B, T], lengths [B], mask [B, T] (True = padding) -> [B, C, T].'''
        m = mask.unsqueeze(1)
        x = self.embedding(tokens).transpose(1, 2).masked_fill(m, 0.0)
        for conv in self.cnn:
            x = conv(x).masked_fill(m, 0.0)
        x = run_lstm(self.lstm, x.transpose(1, 2), lengths)
        return x.transpose(1, 2).masked_fill(m, 0.0)
