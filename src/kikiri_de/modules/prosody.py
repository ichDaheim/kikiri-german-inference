"""Prosody predictor: per-token durations, then frame-level F0 and energy.

Adapted from StyleTTS 2 (MIT) and Kokoro (Apache 2.0).
"""

from typing import Tuple

import torch
import torch.nn as nn

from .layers import AdaLayerNorm, AdainResBlk1d, LinearNorm
from .text_encoder import run_lstm


class DurationEncoder(nn.Module):
    '''Style-conditioned BiLSTM stack over PL-BERT features.'''

    def __init__(self, style_dim: int, d_model: int, nlayers: int):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(nn.LSTM(d_model + style_dim, d_model // 2, num_layers=1, batch_first=True, bidirectional=True))
            self.lstms.append(AdaLayerNorm(style_dim, d_model))

    def forward(self, x: torch.Tensor, style: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''x [B, C, T], style [B, S] -> [B, T, C + S].'''
        m = mask.unsqueeze(-1)
        s = style.unsqueeze(1).expand(-1, x.shape[-1], -1)
        x = torch.cat([x.transpose(1, 2), s], dim=-1).masked_fill(m, 0.0)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x, style)
                x = torch.cat([x, s], dim=-1).masked_fill(m, 0.0)
            else:
                x = run_lstm(block, x, lengths)
        return x


class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim: int, d_hid: int, nlayers: int, max_dur: int = 50):
        super().__init__()
        self.text_encoder = DurationEncoder(style_dim, d_hid, nlayers)
        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList([
            AdainResBlk1d(d_hid, d_hid, style_dim),
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True),
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim),
        ])
        self.N = nn.ModuleList([
            AdainResBlk1d(d_hid, d_hid, style_dim),
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True),
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim),
        ])
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def encode(self, texts: torch.Tensor, style: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''Style-conditioned token features d [B, T, C + S].'''
        return self.text_encoder(texts, style, lengths, mask)

    def durations(self, d: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        '''Duration logits [B, T, max_dur] (sigmoid-summed to frames).'''
        return self.duration_proj(run_lstm(self.lstm, d, lengths))

    def F0Ntrain(self, en: torch.Tensor, style: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Frame features [B, C + S, frames] -> F0 and energy curves [B, 2 * frames] (name as in Kokoro).'''
        x, _ = self.shared(en.transpose(1, 2))
        x = x.transpose(1, 2)
        F0 = x
        for block in self.F0:
            F0 = block(F0, style)
        N = x
        for block in self.N:
            N = block(N, style)
        return self.F0_proj(F0).squeeze(1), self.N_proj(N).squeeze(1)
