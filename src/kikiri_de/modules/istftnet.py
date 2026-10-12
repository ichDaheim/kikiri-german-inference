"""ISTFTNet decoder: aligned text features + F0 + energy + style -> 24 kHz waveform.

Adapted from StyleTTS 2 (MIT) and Kokoro (Apache 2.0).
"""

import math
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from .layers import AdaIN1d, AdainResBlk1d, get_padding

SAMPLE_RATE = 24000


@dataclass
class DecoderConfig:
    upsample_rates: List[int] = field(default_factory=lambda: [10, 6])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [20, 12])
    upsample_initial_channel: int = 512
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    gen_istft_n_fft: int = 20
    gen_istft_hop_size: int = 5


class AdaINResBlock1(nn.Module):
    '''HiFi-GAN style residual block with AdaIN and Snake activations.'''

    def __init__(self, channels: int, kernel_size: int, dilation: List[int], style_dim: int):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)))
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
            for _ in dilation
        ])
        self.adain1 = nn.ModuleList([AdaIN1d(style_dim, channels) for _ in dilation])
        self.adain2 = nn.ModuleList([AdaIN1d(style_dim, channels) for _ in dilation])
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for _ in dilation])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for _ in dilation])

    def forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
            xt = n1(x, s)
            xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)  # Snake
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)
            xt = c2(xt)
            x = xt + x
        return x


class TorchSTFT(nn.Module):
    def __init__(self, filter_length: int, hop_length: int, win_length: int):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length, periodic=True), persistent=False)

    def transform(self, x):
        spec = torch.stft(x, self.filter_length, self.hop_length, self.win_length, window=self.window, return_complex=True)
        return torch.abs(spec), torch.angle(spec)

    def inverse(self, magnitude, phase):
        wave = torch.istft(magnitude * torch.exp(phase * 1j), self.filter_length, self.hop_length, self.win_length, window=self.window)
        return wave.unsqueeze(-2)


class SineGen(nn.Module):
    '''Harmonic sine source from an F0 curve (neural source-filter excitation).'''

    def __init__(self, sample_rate: int, upsample_scale: int, harmonic_num: int = 0,
                 sine_amp: float = 0.1, noise_std: float = 0.003, voiced_threshold: float = 0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.sample_rate = sample_rate
        self.voiced_threshold = voiced_threshold
        self.upsample_scale = upsample_scale

    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sample_rate) % 1
        # Random initial phase for the overtones, none for the fundamental.
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        # Integrate phase at the low frame rate, then upsample (cheaper, avoids drift).
        rad_values = F.interpolate(rad_values.transpose(1, 2), scale_factor=1 / self.upsample_scale, mode='linear').transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * math.pi
        phase = F.interpolate(phase.transpose(1, 2) * self.upsample_scale, scale_factor=self.upsample_scale, mode='linear').transpose(1, 2)
        return torch.sin(phase)

    def forward(self, f0):
        harmonics = torch.arange(1, self.harmonic_num + 2, device=f0.device, dtype=f0.dtype)
        sine_waves = self._f02sine(f0 * harmonics) * self.sine_amp
        uv = (f0 > self.voiced_threshold).to(f0.dtype)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        return sine_waves * uv + noise


class SourceModuleHnNSF(nn.Module):
    def __init__(self, sample_rate: int, upsample_scale: int, harmonic_num: int = 8,
                 sine_amp: float = 0.1, add_noise_std: float = 0.003, voiced_threshold: float = 10):
        super().__init__()
        self.l_sin_gen = SineGen(sample_rate, upsample_scale, harmonic_num, sine_amp, add_noise_std, voiced_threshold)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x):
        return self.l_tanh(self.l_linear(self.l_sin_gen(x)))


class Generator(nn.Module):
    def __init__(self, style_dim: int, config: DecoderConfig):
        super().__init__()
        c = config
        self.num_kernels = len(c.resblock_kernel_sizes)
        self.num_upsamples = len(c.upsample_rates)
        upsample_scale = math.prod(c.upsample_rates) * c.gen_istft_hop_size
        self.m_source = SourceModuleHnNSF(SAMPLE_RATE, upsample_scale, harmonic_num=8, voiced_threshold=10)
        self.f0_upsamp = nn.Upsample(scale_factor=upsample_scale)
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(c.upsample_rates, c.upsample_kernel_sizes)):
            self.ups.append(weight_norm(nn.ConvTranspose1d(
                c.upsample_initial_channel // (2 ** i), c.upsample_initial_channel // (2 ** (i + 1)), k, u, padding=(k - u) // 2,
            )))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = c.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(c.resblock_kernel_sizes, c.resblock_dilation_sizes):
                self.resblocks.append(AdaINResBlock1(ch, k, d, style_dim))
            if i + 1 < len(c.upsample_rates):
                stride_f0 = math.prod(c.upsample_rates[i + 1:])
                self.noise_convs.append(nn.Conv1d(
                    c.gen_istft_n_fft + 2, ch, kernel_size=stride_f0 * 2, stride=stride_f0, padding=(stride_f0 + 1) // 2,
                ))
                self.noise_res.append(AdaINResBlock1(ch, 7, [1, 3, 5], style_dim))
            else:
                self.noise_convs.append(nn.Conv1d(c.gen_istft_n_fft + 2, ch, kernel_size=1))
                self.noise_res.append(AdaINResBlock1(ch, 11, [1, 3, 5], style_dim))
        self.post_n_fft = c.gen_istft_n_fft
        self.conv_post = weight_norm(nn.Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT(filter_length=c.gen_istft_n_fft, hop_length=c.gen_istft_hop_size, win_length=c.gen_istft_n_fft)

    def forward(self, x, s, f0):
        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        har_source = self.m_source(f0)
        har_spec, har_phase = self.stft.transform(har_source.transpose(1, 2).squeeze(1))
        har = torch.cat([har_spec, har_phase], dim=1)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            x_source = self.noise_res[i](self.noise_convs[i](har), s)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source
            xs = sum(self.resblocks[i * self.num_kernels + j](x, s) for j in range(self.num_kernels))
            x = xs / self.num_kernels
        x = self.conv_post(F.leaky_relu(x))
        spec = torch.exp(x[:, :self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        return self.stft.inverse(spec, phase)


class Decoder(nn.Module):
    def __init__(self, dim_in: int, style_dim: int, config: DecoderConfig):
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
        self.decode = nn.ModuleList([
            AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
            AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
            AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
            AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True),
        ])
        self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.asr_res = nn.Sequential(weight_norm(nn.Conv1d(512, 64, kernel_size=1)))
        self.generator = Generator(style_dim, config)

    def forward(self, asr: torch.Tensor, F0_curve: torch.Tensor, N: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        '''asr [B, C, T], F0/N [B, 2T], style [B, S] -> waveform [B, 1, samples].'''
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N = self.N_conv(N.unsqueeze(1))
        x = self.encode(torch.cat([asr, F0, N], dim=1), s)
        asr_res = self.asr_res(asr)
        # Only the last block upsamples, so every block sees the frame-rate residuals.
        for block in self.decode:
            x = block(torch.cat([x, asr_res, F0, N], dim=1), s)
        return self.generator(x, s, F0_curve)
