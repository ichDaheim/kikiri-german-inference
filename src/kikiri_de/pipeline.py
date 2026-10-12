"""Text -> speech: sentence chunking, G2P, tokenization and synthesis."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Union

import torch

from .audio import SAMPLE_RATE
from .g2p import G2P
from .hub import resolve_model, resolve_voice
from .model import KikiriModel
from .tokenizer import PhonemeTokenizer
from .voice import Voice

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r'(?<=[.!?…])\s+|\n+')
_CLAUSE_END = re.compile(r'(?<=[,;:—])\s+')


@dataclass
class Result:
    text: str
    phonemes: str
    audio: torch.Tensor  # [samples], 24 kHz
    durations: torch.Tensor  # per token, in 25 ms frames


class Pipeline:
    '''German TTS pipeline.

    Args:
        model: a loaded `KikiriModel`, the path of a model folder, or None for the pinned release from
            the Hugging Face Hub (`hub.REPO_ID` at `hub.REVISION`).
        device: defaults to CUDA, then MPS, then CPU.
        max_tokens: chunk size limit, including the two frame separators.
    '''

    def __init__(self, model: Union[KikiriModel, str, Path, None] = None, device: Optional[str] = None, max_tokens: Optional[int] = None):
        self.model = model if isinstance(model, KikiriModel) else KikiriModel.from_pretrained(resolve_model(model), device=device)
        self.g2p = G2P()
        self.tokenizer = PhonemeTokenizer()
        if self.model.config.vocab_sha256 and self.model.config.vocab_sha256 != self.tokenizer.vocab_sha256:
            raise ValueError('the model was trained with a different mideki token map; install the matching version')
        self.max_tokens = min(max_tokens or self.model.context_length, self.model.context_length)

    def load_voice(self, voice: Union[Voice, str, Path]) -> Voice:
        '''Load a voice (path or name from the Hub) and check that it belongs to this model export.'''
        voice = voice if isinstance(voice, Voice) else Voice.load(resolve_voice(voice))
        source = voice.metadata.get('source_model_sha256')
        if source is None:
            logger.warning('voice %s has no source_model_sha256; cannot check that it fits this model', voice.metadata.get('name', '?'))
        elif self.model.info.get('source_model_sha256') not in (None, source):
            raise ValueError(f"voice {voice.metadata.get('name', '?')} was extracted from a different model export")
        vocab = voice.metadata.get('model_vocab_sha256')
        if vocab and self.model.config.vocab_sha256 and vocab != self.model.config.vocab_sha256:
            raise ValueError(f"voice {voice.metadata.get('name', '?')} belongs to a model with a different token map")
        if voice.style.shape != (2 * self.model.config.style_dim,):
            raise ValueError(f'voice has shape {tuple(voice.style.shape)}, the model expects ({2 * self.model.config.style_dim},)')
        return voice

    def chunks(self, text: str) -> List[str]:
        '''Split text into pieces that each fit into `max_tokens` after G2P.'''
        pieces: List[str] = []
        for sentence in (s.strip() for s in _SENTENCE_END.split(text)):
            if sentence:
                pieces.extend(self._fit(sentence))
        merged: List[str] = []
        for piece in pieces:
            if merged and self._n_tokens(merged[-1] + ' ' + piece) <= self.max_tokens and len(merged[-1]) < 150:
                merged[-1] = merged[-1] + ' ' + piece
            else:
                merged.append(piece)
        return merged

    def _n_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(self.g2p(text), strict=False))

    def _fit(self, text: str) -> List[str]:
        if self._n_tokens(text) <= self.max_tokens:
            return [text]
        clauses = [c for c in _CLAUSE_END.split(text) if c.strip()]
        if len(clauses) > 1:
            return [part for clause in clauses for part in self._fit(clause)]
        words = text.split()
        if len(words) == 1:
            raise ValueError(f'a single word is longer than {self.max_tokens} tokens: {text[:50]!r}')
        mid = len(words) // 2
        return self._fit(' '.join(words[:mid])) + self._fit(' '.join(words[mid:]))

    def __call__(self, text: str, voice: Union[Voice, str, Path], speed: float = 1.0) -> Iterator[Result]:
        voice = self.load_voice(voice)
        for chunk in self.chunks(text):
            phonemes = self.g2p(chunk)
            tokens = torch.tensor(self.tokenizer.encode(phonemes, strict=False))
            out = self.model(tokens, voice.style, speed=speed)
            yield Result(text=chunk, phonemes=phonemes, audio=out.audio.cpu(), durations=out.durations.cpu())

    def synthesize(self, text: str, voice: Union[Voice, str, Path], speed: float = 1.0, pause: float = 0.15) -> torch.Tensor:
        '''All chunks joined into one waveform, with `pause` seconds of silence between them.'''
        silence = torch.zeros(int(pause * SAMPLE_RATE))
        parts: List[torch.Tensor] = []
        for result in self(text, voice, speed):
            if parts:
                parts.append(silence)
            parts.append(result.audio)
        return torch.cat(parts) if parts else torch.zeros(0)
