"""KikiriModel: the acoustic model for inference, from token IDs and a voice to a waveform.

The weights come from a model folder (`config.json` + `model.safetensors`, optionally `info.json`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from safetensors.torch import load_file

from .config import ModelConfig
from .modules import Decoder, PLBert, ProsodyPredictor, TextEncoder, length_to_mask

CONFIG_FILE = 'config.json'
WEIGHTS_FILE = 'model.safetensors'
INFO_FILE = 'info.json'  # export provenance; voices are checked against its source_model_sha256


class KikiriModel(nn.Module):
    '''Kokoro-architecture TTS model for German.

    Submodules:
        bert, bert_encoder: PL-BERT-de and its projection to `hidden_dim`.
        predictor: durations, F0 and energy (conditioned on the prosodic style).
        text_encoder: acoustic token features for the decoder.
        decoder: ISTFTNet vocoder (conditioned on the acoustic style).

    A voice is a [2 * style_dim] vector: acoustic style first, then prosodic style.
    '''

    @dataclass
    class Output:
        audio: torch.Tensor  # [samples]
        durations: torch.Tensor  # [tokens], in aligner frames of 2 mel frames (600 samples, 25 ms)

    def __init__(self, config: ModelConfig):
        super().__init__()
        c = config
        self.config = c
        self.info: Dict[str, Any] = {}
        self.bert = PLBert(c.plbert)
        self.bert_encoder = nn.Linear(c.plbert.hidden_size, c.hidden_dim)
        self.predictor = ProsodyPredictor(c.style_dim, c.hidden_dim, c.n_layer, c.max_dur)
        self.text_encoder = TextEncoder(c.hidden_dim, c.text_encoder_kernel_size, c.n_layer, c.n_token)
        self.decoder = Decoder(c.hidden_dim, c.style_dim, c.decoder)
        if c.plbert.vocab_size != c.n_token:
            raise ValueError(f'plbert.vocab_size ({c.plbert.vocab_size}) must equal n_token ({c.n_token})')

    @property
    def device(self) -> torch.device:
        return self.bert_encoder.weight.device

    @property
    def context_length(self) -> int:
        '''Maximum number of tokens per call, including the two frame separators.'''
        return self.config.plbert.max_position_embeddings

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, voice: torch.Tensor, speed: float = 1.0) -> 'KikiriModel.Output':
        '''Synthesize one utterance.

        Args:
            tokens: [T] or [1, T] framed token IDs from `PhonemeTokenizer.encode`.
            voice: [2 * style_dim] or [1, 2 * style_dim].
            speed: > 1 speaks faster.
        '''
        tokens = tokens.reshape(1, -1).to(self.device)
        voice = voice.reshape(1, -1).to(self.device)
        if tokens.shape[1] > self.context_length:
            raise ValueError(f'{tokens.shape[1]} tokens exceed the context length of {self.context_length}')
        sd = self.config.style_dim
        acoustic, prosodic = voice[:, :sd], voice[:, sd:]
        lengths = torch.tensor([tokens.shape[1]], device=self.device)
        mask = length_to_mask(lengths)

        bert = self.bert(tokens, attention_mask=(~mask).int())
        d_en = self.bert_encoder(bert).transpose(1, 2)
        d = self.predictor.encode(d_en, prosodic, lengths, mask)
        logits = self.predictor.durations(d, lengths)
        durations = (torch.sigmoid(logits).sum(dim=-1) / speed).round().clamp(min=1).long().squeeze(0)

        # Hard monotonic alignment [T, frames] from the predicted durations.
        frame_to_token = torch.repeat_interleave(torch.arange(tokens.shape[1], device=self.device), durations)
        alignment = torch.zeros(tokens.shape[1], frame_to_token.shape[0], device=self.device)
        alignment[frame_to_token, torch.arange(frame_to_token.shape[0], device=self.device)] = 1
        alignment = alignment.unsqueeze(0)

        en = d.transpose(1, 2) @ alignment
        F0, N = self.predictor.F0Ntrain(en, prosodic)
        t_en = self.text_encoder(tokens, lengths, mask)
        audio = self.decoder(t_en @ alignment, F0, N, acoustic)
        return KikiriModel.Output(audio=audio.squeeze(), durations=durations)

    @classmethod
    def from_pretrained(cls, directory: Union[str, Path], device: Optional[Union[str, torch.device]] = None) -> 'KikiriModel':
        '''Load `config.json` + `model.safetensors` strictly (every key must match), and `info.json` if present.'''
        directory = Path(directory)
        model = cls(ModelConfig.load(directory / CONFIG_FILE))
        model.load_state_dict(load_file(str(directory / WEIGHTS_FILE)), strict=True)
        info = directory / INFO_FILE
        if info.is_file():
            model.info = json.loads(info.read_text(encoding='utf-8'))
        model.eval()
        return model.to(device or default_device())


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
