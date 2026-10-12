import pytest

from mideki import token_map_sha256, unknown_symbols
from kikiri_de.g2p import G2P

# Token map the released model was trained with (models/base_de config.json vocab_sha256).
MODEL_VOCAB_SHA256 = '3af12687579c50956ab35be02ab38da1b8d1f1713362266dc76c8e1729e95724'


@pytest.fixture(scope='module')
def g2p():
    return G2P()


def test_mideki_token_map_matches_model():
    assert token_map_sha256() == MODEL_VOCAB_SHA256


@pytest.mark.parametrize('text, expected', [
    ('Zeit', 'ʦˈIt'),
    ('Pfote', 'ꭧˈoːtə'),
    ('Kopf', 'kˈɔꭧ'),
    ('Heu', 'hˈʋ'),
    ('Rätsel', 'rˈɛːʦl̩'),
    ('sakrale', 'zakɾˈɑːlə'),
])
def test_known_words(g2p, text, expected):
    assert g2p(text) == expected


def test_output_is_canonical(g2p):
    ps = g2p('Am 3.10.2026 kostete das 12,50 € (inkl. Versand) – oder?')
    assert unknown_symbols(ps.replace(' ', '')) == []


def test_overrides_are_applied(g2p):
    assert g2p('PyTorch') == 'pˈItɔːɾʧ'  # ɹ replaced by ɾ (ɹ: 45 times in the training data)


def test_word_cache_is_filled_and_reused():
    cache = {}
    g2p = G2P(cache=cache)
    first = g2p('Hallo Welt')
    assert set(cache) == {'Hallo', 'Welt'}
    cache['Welt'] = 'X'
    assert g2p('Hallo Welt') != first
