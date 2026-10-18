"""Model architecture configuration, read from a model folder's `config.json`."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Type, TypeVar, Union, get_args, get_origin, get_type_hints

from .modules import DecoderConfig, PLBertConfig

T = TypeVar('T')


def from_dict(cls: Type[T], data: Dict[str, Any], where: str = '') -> T:
    '''Build a (nested) dataclass from a dict, rejecting unknown keys.

    Unknown keys are an error on purpose: a misspelled or misplaced option must
    not be silently ignored.
    '''
    if not isinstance(data, dict):
        raise ValueError(f'{where or cls.__name__} must be a mapping, got {type(data).__name__}')
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f'unknown option(s) {unknown} in {where or cls.__name__}')
    kwargs = {}
    for name, value in data.items():
        field_type = hints[name]
        path = f'{where}.{name}' if where else name
        if dataclasses.is_dataclass(field_type):
            value = from_dict(field_type, value, path)
        elif get_origin(field_type) in (list, List) and get_args(field_type) and dataclasses.is_dataclass(get_args(field_type)[0]):
            if not isinstance(value, list):
                raise ValueError(f'{path} must be a list')
            value = [from_dict(get_args(field_type)[0], item, f'{path}[{i}]') for i, item in enumerate(value)]
        kwargs[name] = value
    return cls(**kwargs)


@dataclass
class ModelConfig:
    '''Architecture of the kikiri-de acoustic model (Kokoro-82M layout, German vocabulary).'''

    n_token: int = 106
    hidden_dim: int = 512
    style_dim: int = 128
    n_layer: int = 3
    max_dur: int = 50
    text_encoder_kernel_size: int = 5
    plbert: PLBertConfig = field(default_factory=PLBertConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    # SHA-256 of mideki's token_map_de.json the model was trained with.
    vocab_sha256: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        return from_dict(cls, data)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ModelConfig':
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))
