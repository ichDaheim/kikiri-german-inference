"""PL-BERT-de: ALBERT phoneme encoder pretrained on German phoneme sequences."""

from dataclasses import asdict, dataclass

import torch
from transformers import AlbertConfig, AlbertModel


@dataclass
class PLBertConfig:
    '''ALBERT hyperparameters of PL-BERT-de (identical to the English PL-BERT release).'''

    vocab_size: int = 106
    embedding_size: int = 128
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 2048
    max_position_embeddings: int = 512
    num_hidden_layers: int = 12

    def to_albert(self) -> AlbertConfig:
        return AlbertConfig(**asdict(self))


class PLBert(AlbertModel):
    '''ALBERT that returns only the last hidden state [B, T, hidden_size].'''

    def __init__(self, config: PLBertConfig):
        super().__init__(config.to_albert(), add_pooling_layer=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
