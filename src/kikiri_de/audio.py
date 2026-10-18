"""Audio output: the model produces 24 kHz mono waveforms."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf
import torch

from . import __version__

SAMPLE_RATE = 24000
# Machine-readable AI label in the WAV INFO chunk (EU AI Act, Art. 50(2)); the robust marking is the watermark.
AI_LABEL = 'KI-generiert (synthetische Sprache, kikiri-de) / AI-generated (synthetic speech, kikiri-de)'


def save_wav(path: Union[str, Path], wave: Union[np.ndarray, torch.Tensor], sample_rate: int = SAMPLE_RATE) -> None:
    '''Write a waveform as 16-bit PCM WAV, labelled as AI-generated in its metadata.'''
    if isinstance(wave, torch.Tensor):
        wave = wave.detach().cpu().numpy()
    with sf.SoundFile(str(path), 'w', sample_rate, 1, subtype='PCM_16') as f:
        f.comment = AI_LABEL
        f.software = f'kikiri-de {__version__}'
        f.write(np.asarray(wave, dtype=np.float32).squeeze())
