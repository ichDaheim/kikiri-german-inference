"""Phoneme string -> token IDs for PL-BERT-de and the text encoder.

The ID contract:

- IDs 0-104 come from mideki's `token_map_de.json` ([PAD]=0, [UNK]=1,
  [MASK]=2, [BOS]=3, [EOS]=4, 100 symbols).
- ID 105 is the word separator. It also marks silence at both ends of every sequence.
- A sequence is framed as `[105, w1..., 105, w2..., ..., wn..., 105]`.

The model was trained with exactly this tokenization; any change here breaks it.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

from mideki import split_symbols, token_map, token_map_sha256
from mideki.symbols import ARTIFACT_SYMBOLS

logger = logging.getLogger(__name__)

WORD_SEPARATOR = ' '


class UnknownSymbolError(ValueError):
    pass


class PhonemeTokenizer:
    '''Longest-match tokenizer over the canonical German symbol set.'''

    def __init__(self):
        tm = token_map()
        specials: Dict[str, int] = tm['special_tokens']
        self.pad_id = specials['[PAD]']
        self.unk_id = specials['[UNK]']
        self.mask_id = specials['[MASK]']
        self.bos_id = specials['[BOS]']
        self.eos_id = specials['[EOS]']
        self.separator_id = tm['vocab_size']
        self.n_token = tm['vocab_size'] + 1
        self.vocab_sha256 = token_map_sha256()

        self._symbol_to_id: Dict[str, int] = {
            s: i for s, i in tm['symbol_to_id'].items() if s not in ARTIFACT_SYMBOLS
        }
        self._symbol_to_id[WORD_SEPARATOR] = self.separator_id
        self._id_to_symbol = {i: s for s, i in self._symbol_to_id.items()}

    @property
    def special_ids(self) -> Dict[str, int]:
        return {
            'pad': self.pad_id,
            'unk': self.unk_id,
            'mask': self.mask_id,
            'bos': self.bos_id,
            'eos': self.eos_id,
            'separator': self.separator_id,
        }

    def vocab(self) -> Dict[str, int]:
        '''Symbol -> ID for every encodable symbol, including the word separator.'''
        return dict(self._symbol_to_id)

    def encode(self, phonemes: str, strict: bool = True) -> List[int]:
        '''Encode a phoneme string and add the separator frame at both ends.

        Args:
            phonemes: canonical phoneme string as produced by `mideki.DEG2P`.
            strict: raise `UnknownSymbolError` on symbols outside the canonical
                set. With `strict=False` they become [UNK] and a warning is logged.
        '''
        ids = [self.separator_id]
        unknown = []
        position = 0
        for piece in split_symbols(phonemes):
            symbol_id = None if piece is None else self._symbol_to_id.get(piece)
            if symbol_id is None:
                bad = phonemes[position] if piece is None else piece
                unknown.append(bad)
                symbol_id = self.unk_id
            ids.append(symbol_id)
            position += 1 if piece is None else len(piece)
        ids.append(self.separator_id)
        if unknown:
            message = f'unknown phoneme symbols {sorted(set(unknown))!r} in {phonemes!r}'
            if strict:
                raise UnknownSymbolError(message)
            logger.warning('%s, encoded as [UNK]', message)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        '''Inverse of `encode` (frame removed, [UNK] rendered as U+FFFD).'''
        ids = [int(i) for i in ids if int(i) != self.pad_id]
        if len(ids) >= 2 and ids[0] == ids[-1] == self.separator_id:
            ids = ids[1:-1]
        return ''.join(self._id_to_symbol.get(i, '�') for i in ids)
