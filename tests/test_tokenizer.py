import pytest

from kikiri_de.g2p import G2P
from kikiri_de.tokenizer import PhonemeTokenizer, UnknownSymbolError


@pytest.fixture(scope='module')
def tok():
    return PhonemeTokenizer()


@pytest.fixture(scope='module')
def g2p():
    return G2P()


def test_id_contract(tok):
    assert tok.n_token == 106
    assert tok.special_ids == {'pad': 0, 'unk': 1, 'mask': 2, 'bos': 3, 'eos': 4, 'separator': 105}


def test_frame_and_word_boundaries(tok):
    ids = tok.encode('a b')
    assert ids[0] == ids[-1] == tok.separator_id
    assert ids.count(tok.separator_id) == 3


def test_syllabic_consonant_is_one_token(tok):
    ids = tok.encode('rˈɛːʦl̩')
    assert len(ids) == 2 + 6
    assert ids[-2] == tok.vocab()['l̩']


@pytest.mark.parametrize('word', ['sakrale', 'Rätsel', 'Zeit', 'Pfote', 'Heu', 'Chance', 'PyTorch'])
def test_problem_words_encode_without_unknowns(tok, g2p, word):
    phonemes = g2p(word)
    ids = tok.encode(phonemes)
    assert tok.unk_id not in ids
    assert tok.decode(ids) == phonemes


def test_stress_markers_are_tokens(tok):
    vocab = tok.vocab()
    ids = tok.encode('ˈaˌb')
    assert vocab['ˈ'] in ids and vocab['ˌ'] in ids


def test_unknown_symbols(tok):
    with pytest.raises(UnknownSymbolError):
        tok.encode('a§')
    assert tok.encode('a§', strict=False)[2] == tok.unk_id


def test_artifact_symbol_is_not_encodable(tok):
    with pytest.raises(UnknownSymbolError):
        tok.encode('a1')
