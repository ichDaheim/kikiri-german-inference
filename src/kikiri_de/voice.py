"""Voices: a single style vector per speaker, stored as `.safetensors` with metadata.

Unlike Kokoro's length-indexed `[510, 1, 256]` voicepacks, a kikiri-de voice
is one `[2 * style_dim]` vector (acoustic style, then prosodic style).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence, Union

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_TENSOR_KEY = 'style'
_METADATA_KEY = 'kikiri'
# Metadata that binds a voice to the model export it was extracted with; kept by `mix` only if all voices agree.
BINDING_KEYS = ('source_model_sha256', 'model_vocab_sha256')


@dataclass
class Voice:
    style: torch.Tensor  # [2 * style_dim]
    metadata: Dict[str, object] = field(default_factory=dict)

    def save(self, path: Union[str, Path]) -> None:
        tensors = {_TENSOR_KEY: self.style.detach().float().cpu().contiguous()}
        save_file(tensors, str(path), metadata={_METADATA_KEY: json.dumps(self.metadata, ensure_ascii=False)})

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Voice':
        with safe_open(str(path), framework='pt') as f:
            style = f.get_tensor(_TENSOR_KEY)
            raw = (f.metadata() or {}).get(_METADATA_KEY, '{}')
        if style.dim() != 1:
            raise ValueError(f'{path}: expected a 1-D style vector, got shape {tuple(style.shape)}')
        return cls(style=style, metadata=json.loads(raw))

    @classmethod
    def mix(cls, voices: Sequence['Voice'], weights: Sequence[float] = None) -> 'Voice':
        '''Weighted average of several voices (weights default to equal).'''
        if not voices:
            raise ValueError('need at least one voice')
        weights = torch.tensor(weights if weights is not None else [1.0] * len(voices), dtype=torch.float32)
        if len(weights) != len(voices):
            raise ValueError('need one weight per voice')
        weights = weights / weights.sum()
        style = sum(w * v.style for w, v in zip(weights, voices))
        metadata = {'mix': [v.metadata.get('name', '?') for v in voices], 'weights': weights.tolist()}
        for key in BINDING_KEYS:
            values = {v.metadata.get(key) for v in voices}
            if len(values) > 1:
                raise ValueError(f'voices come from different model exports ({key} differs)')
            if None not in values:
                metadata[key] = values.pop()
        return cls(style=style, metadata=metadata)
