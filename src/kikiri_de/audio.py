"""Audio output: the model produces 24 kHz mono waveforms."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 24000


def save_wav(path: Union[str, Path], wave: Union[np.ndarray, torch.Tensor], sample_rate: int = SAMPLE_RATE) -> None:
    '''Write a waveform as 16-bit PCM WAV.'''
    if isinstance(wave, torch.Tensor):
        wave = wave.detach().cpu().numpy()
    sf.write(str(path), np.asarray(wave, dtype=np.float32).squeeze(), sample_rate, subtype='PCM_16')
