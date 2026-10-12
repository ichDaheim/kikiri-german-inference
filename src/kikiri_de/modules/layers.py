"""Building blocks shared by the text encoder, prosody predictor and decoder.

Adapted from StyleTTS 2 (MIT, yl4579/StyleTTS2) and Kokoro (Apache 2.0).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    '''Boolean mask [B, T] that is True on padding positions.'''
    return torch.arange(int(lengths.max()), device=lengths.device)[None, :] >= lengths[:, None]


class LinearNorm(nn.Module):
    '''nn.Linear under the parameter name `linear_layer`, as in StyleTTS 2.'''

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear_layer(x)


class LayerNorm(nn.Module):
    '''Layer norm over the channel dimension of a [B, C, T] tensor.'''

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class AdaIN1d(nn.Module):
    '''Adaptive instance norm: normalize over time, then scale/shift from the style vector.'''

    def __init__(self, style_dim: int, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        gamma, beta = self.fc(s).unsqueeze(-1).chunk(2, dim=1)
        return (1 + gamma) * x + beta


class AdaLayerNorm(nn.Module):
    '''Layer norm over channels of a [B, T, C] tensor, scale/shift from the style vector.'''

    def __init__(self, style_dim: int, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        gamma, beta = self.fc(s).unsqueeze(1).chunk(2, dim=-1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        return (1 + gamma) * x + beta


class UpSample1d(nn.Module):
    def __init__(self, upsample: bool):
        super().__init__()
        self.upsample = upsample

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest') if self.upsample else x


class AdainResBlk1d(nn.Module):
    '''Residual 1D conv block with AdaIN, optionally upsampling time by 2.'''

    def __init__(self, dim_in: int, dim_out: int, style_dim: int, upsample: bool = False):
        super().__init__()
        self.actv = nn.LeakyReLU(0.2)
        self.upsample = upsample
        self.upsample_shortcut = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))
        if upsample:
            self.pool = weight_norm(nn.ConvTranspose1d(
                dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1,
            ))
        else:
            self.pool = nn.Identity()

    def _shortcut(self, x):
        x = self.upsample_shortcut(x)
        return self.conv1x1(x) if self.learned_sc else x

    def _residual(self, x, s):
        x = self.actv(self.norm1(x, s))
        x = self.pool(x)
        x = self.conv1(x)
        x = self.actv(self.norm2(x, s))
        return self.conv2(x)

    def forward(self, x, s):
        return (self._residual(x, s) + self._shortcut(x)) / math.sqrt(2)
