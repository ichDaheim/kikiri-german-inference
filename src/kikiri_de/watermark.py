"""Inaudible AudioSeal watermark that marks all synthesized speech as AI-generated (EU AI Act, Art. 50(2)).

AudioSeal (Meta, MIT, https://github.com/facebookresearch/audioseal) survives compression, resampling, noise and
cutting. Its models work at 16 kHz only and, since 0.2, no longer resample: the watermark is computed on a 16 kHz copy
and resampled back to the input rate. The generator and detector are downloaded on first use (`AUDIOSEAL_CACHE_DIR`,
default `~/.cache/audioseal`). The watermark carries a fixed 16-bit message, so `detect` can tell kikiri-de output
apart from other AudioSeal-marked audio.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from .audio import SAMPLE_RATE

logger = logging.getLogger(__name__)

GENERATOR = 'audioseal_wm_16bits'
DETECTOR = 'audioseal_detector_16bits'
WATERMARK_RATE = 16000
# 'kk' (0x6B6B) as 16 bits, most significant first.
MESSAGE = tuple(int(bit) for bit in format(0x6B6B, '016b'))
DETECTION_THRESHOLD = 0.5


@dataclass
class Detection:
    probability: float  # share of frames that carry a watermark
    message: tuple  # decoded 16 bits

    @property
    def watermarked(self) -> bool:
        return self.probability >= DETECTION_THRESHOLD

    @property
    def is_kikiri(self) -> bool:
        return self.watermarked and self.message == MESSAGE


def resample(x: torch.Tensor, old_rate: int, new_rate: int, zeros: int = 24, rolloff: float = 0.945) -> torch.Tensor:
    '''Windowed-sinc resampling along the last dimension (the method of julius.resample_frac).'''
    gcd = math.gcd(old_rate, new_rate)
    old, new = old_rate // gcd, new_rate // gcd
    if old == new:
        return x
    cutoff = min(old, new) * rolloff
    width = math.ceil(zeros * old / cutoff)
    idx = torch.arange(-width, width + old, dtype=torch.float32, device=x.device)
    t = (-torch.arange(new, dtype=torch.float32, device=x.device)[:, None] / new + idx[None, :] / old) * cutoff
    t = t.clamp(-zeros, zeros) * math.pi
    kernel = torch.where(t == 0, torch.ones_like(t), torch.sin(t) / t) * torch.cos(t / zeros / 2) ** 2
    kernel = (kernel / kernel.sum(dim=-1, keepdim=True)).view(new, 1, -1)
    shape, length = x.shape, x.shape[-1]
    y = F.conv1d(F.pad(x.reshape(-1, 1, length), (width, width + old), mode='replicate'), kernel, stride=old)
    y = y.transpose(1, 2).reshape(*shape[:-1], -1)
    return y[..., :math.ceil(new * length / old)]


def _as_batch(audio: Union[np.ndarray, torch.Tensor], device: Union[str, torch.device]) -> torch.Tensor:
    audio = torch.as_tensor(np.asarray(audio, dtype=np.float32)) if isinstance(audio, np.ndarray) else audio.float()
    if audio.dim() == 2:  # [samples, channels] from soundfile -> mono
        audio = audio.mean(dim=1)
    return audio.reshape(1, 1, -1).to(device)


def _load(card: str, kind: str, device: torch.device) -> torch.nn.Module:
    from audioseal import AudioSeal

    logger.info('loading watermark %s %s', kind, card)
    model = AudioSeal.load_generator(card) if kind == 'generator' else AudioSeal.load_detector(card)
    return model.to(device).eval()


class Watermarker:
    '''Lazily loaded AudioSeal generator and detector.'''

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        self.device = torch.device(device or 'cpu')
        self._generator = None
        self._detector = None

    @torch.no_grad()
    def apply(self, audio: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
        '''`audio` ([samples]) with the kikiri-de watermark added; same shape, dtype and device.'''
        if audio.numel() == 0:
            return audio
        # AudioSeal 0.2 wraps its convolutions in torch.compile, which needs a C++ compiler (cl.exe on Windows).
        from audioseal.libs.moshi.utils.compile import no_compile

        if self._generator is None:
            self._generator = _load(GENERATOR, 'generator', self.device)
        wave = _as_batch(audio, self.device)
        message = torch.tensor([MESSAGE], device=self.device)
        with no_compile():
            mark = self._generator.get_watermark(resample(wave, sample_rate, WATERMARK_RATE), message=message)
        mark = resample(mark, WATERMARK_RATE, sample_rate)[..., :wave.shape[-1]]
        mark = F.pad(mark, (0, wave.shape[-1] - mark.shape[-1]))
        return (wave + mark).reshape(audio.shape).to(audio.device, audio.dtype)

    @torch.no_grad()
    def detect(self, audio: Union[np.ndarray, torch.Tensor], sample_rate: int = SAMPLE_RATE) -> Detection:
        '''Check a waveform ([samples] or [samples, channels]) for the watermark.'''
        from audioseal.libs.moshi.utils.compile import no_compile

        if self._detector is None:
            self._detector = _load(DETECTOR, 'detector', self.device)
        wave = resample(_as_batch(audio, self.device), sample_rate, WATERMARK_RATE)
        # forward() instead of detect_watermark(): the latter is wrapped in @contextmanager in AudioSeal 0.2.0.
        with no_compile():
            result, message = self._detector(wave)  # [1, 2, frames], [1, 16] bit probabilities
        probability = (result[:, 1, :] > DETECTION_THRESHOLD).float().mean().item()
        bits = tuple(int(b) for b in (message.reshape(-1) > 0.5).tolist())
        return Detection(probability=probability, message=bits)
