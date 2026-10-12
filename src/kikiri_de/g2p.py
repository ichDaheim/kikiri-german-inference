"""Text -> phonemes for kikiri-de: mideki in word mode.

Word mode is not a matter of taste: PL-BERT-de and the kikiri-de training
data were phonemized word by word, so inference must be too.
"""

from __future__ import annotations

import re
from typing import MutableMapping, Optional

from mideki import G2P as MidekiG2P
from mideki.overrides import override_for

# Punctuation that mideki passes through but the acoustic model (almost) never saw in training
# (… ; ¡ ¿ 0 times, ( ) 4, — 27). Mapped to trained marks before G2P.
_ELLIPSIS = re.compile(r'\s*(?:…|\.{2,})')
_SPANISH = re.compile(r'[¡¿]')
_PAUSE = re.compile(r'\s*[;—–()\[\]]\s*|\s+-\s+')
_COMMA_RUN = re.compile(r',(?:\s*,)+')
_COMMA_BEFORE_MARK = re.compile(r'\s*,\s*(?=[.!?:])')
_EDGE_COMMA = re.compile(r'^\s*,\s*|\s*,\s*$')


_HYPHENATED = re.compile(r'(?<![\w-])\w+(?:-\w+)+(?![\w-])')


def split_override_compounds(text: str) -> str:
    '''Split hyphenated words at the hyphens when the whole word has no override but a part has one.

    mideki looks overrides up for the whole token, so "JSON-Review" would go to espeak-ng as one
    word, although "json" and "review" are overridden. Hyphenated words without overridden parts
    ("Fahrrad-Laden", "E-Mails") and overridden wholes ("Start-ups") stay as they are.
    '''
    def split(match: re.Match) -> str:
        word = match.group(0)
        if override_for(word) is not None:
            return word
        parts = word.split('-')
        return ' '.join(parts) if any(override_for(p) is not None for p in parts) else word
    return _HYPHENATED.sub(split, text)


def clean_punctuation(text: str) -> str:
    '''Replace untrained punctuation: … → .  ; — – ( ) [ ] and spaced hyphens → ,  ¡ ¿ → removed.'''
    text = _ELLIPSIS.sub('.', text)
    text = _SPANISH.sub('', text)
    text = _PAUSE.sub(', ', text)
    text = _COMMA_RUN.sub(',', text)
    text = _COMMA_BEFORE_MARK.sub('', text)
    return _EDGE_COMMA.sub('', text)


class G2P:
    def __init__(self, cache: Optional[MutableMapping[str, str]] = None):
        self._g2p = MidekiG2P(mode='word', cache=cache)

    def __call__(self, text: str) -> str:
        return self._g2p(split_override_compounds(clean_punctuation(text)))[0]