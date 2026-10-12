from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from kikiri_de.audio import SAMPLE_RATE, save_wav
from kikiri_de.config import ModelConfig
from kikiri_de.g2p import clean_punctuation, split_override_compounds
from kikiri_de.model import KikiriModel
from kikiri_de.pipeline import Pipeline
from kikiri_de.tokenizer import PhonemeTokenizer
from kikiri_de.voice import Voice

ROOT = Path(__file__).resolve().parents[1]
BUNDLED_MODEL = ROOT / 'models' / 'base_de'
HOP_LENGTH = 300  # samples per mel frame

needs_bundled_model = pytest.mark.skipif(not (BUNDLED_MODEL / 'model.safetensors').exists(), reason='bundled model missing')


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(0)
    return KikiriModel(ModelConfig()).eval()


def test_wav_output(tmp_path):
    wave = (0.1 * np.sin(np.linspace(0, 1000, SAMPLE_RATE))).astype(np.float32)
    save_wav(tmp_path / 'a.wav', torch.from_numpy(wave))
    loaded, sr = sf.read(str(tmp_path / 'a.wav'), dtype='float32')
    assert sr == SAMPLE_RATE and loaded.shape == wave.shape
    assert np.abs(loaded - wave).max() < 1e-3


def test_config_rejects_unknown_keys():
    data = ModelConfig().to_dict()
    data['decoder']['upsample_ratez'] = [1]
    with pytest.raises(ValueError, match='decoder'):
        ModelConfig.from_dict(data)


def test_synthesis_shapes(model):
    tokens = torch.tensor(PhonemeTokenizer().encode('hˈaloː vˈɛlt'))
    voice = torch.randn(2 * model.config.style_dim)
    out = model(tokens, voice)
    assert out.durations.shape == tokens.shape
    # Durations count aligner frames of two mel frames (600 samples, 25 ms) each.
    assert out.audio.shape == (int(out.durations.sum()) * 2 * HOP_LENGTH,)
    assert torch.isfinite(out.audio).all()


def test_context_length_is_enforced(model):
    with pytest.raises(ValueError, match='context length'):
        model(torch.full((model.context_length + 1,), 105), torch.zeros(256))


def test_voice_roundtrip_and_mix(tmp_path):
    a = Voice(torch.ones(256), {'name': 'a'})
    b = Voice(torch.zeros(256), {'name': 'b'})
    a.save(tmp_path / 'a.safetensors')
    loaded = Voice.load(tmp_path / 'a.safetensors')
    assert torch.equal(loaded.style, a.style) and loaded.metadata == {'name': 'a'}
    assert torch.allclose(Voice.mix([a, b]).style, torch.full((256,), 0.5))


def test_mix_keeps_binding_only_if_all_voices_agree():
    a = Voice(torch.ones(256), {'name': 'a', 'source_model_sha256': 'x'})
    b = Voice(torch.zeros(256), {'name': 'b', 'source_model_sha256': 'x'})
    assert Voice.mix([a, b]).metadata['source_model_sha256'] == 'x'
    c = Voice(torch.zeros(256), {'name': 'c', 'source_model_sha256': 'y'})
    with pytest.raises(ValueError, match='different model exports'):
        Voice.mix([a, c])


def test_pipeline_rejects_voice_of_other_export(model, monkeypatch):
    monkeypatch.setattr(model, 'info', {'source_model_sha256': 'x'})
    monkeypatch.setattr(model.config, 'vocab_sha256', PhonemeTokenizer().vocab_sha256)
    pipeline = Pipeline(model)
    assert pipeline.load_voice(Voice(torch.zeros(256), {'source_model_sha256': 'x'}))
    with pytest.raises(ValueError, match='different model export'):
        pipeline.load_voice(Voice(torch.zeros(256), {'source_model_sha256': 'y'}))
    with pytest.raises(ValueError, match='different token map'):
        pipeline.load_voice(Voice(torch.zeros(256), {'model_vocab_sha256': 'z'}))


@needs_bundled_model
def test_bundled_model_and_voices_match():
    loaded = KikiriModel.from_pretrained(BUNDLED_MODEL, device='cpu')
    assert loaded.config.vocab_sha256 == PhonemeTokenizer().vocab_sha256
    voices = sorted((ROOT / 'voices').glob('*.safetensors'))
    assert voices
    for path in voices:
        voice = Voice.load(path)
        assert voice.style.shape == (2 * loaded.config.style_dim,)
        assert voice.metadata['model_vocab_sha256'] == loaded.config.vocab_sha256
        assert voice.metadata['source_model_sha256'] == loaded.info['source_model_sha256']


@needs_bundled_model
def test_pipeline_end_to_end():
    pipeline = Pipeline(BUNDLED_MODEL, device='cpu')
    audio = pipeline.synthesize('Guten Morgen! Wie geht es dir?', ROOT / 'voices' / 'Heinrich.safetensors')
    assert 0.5 * SAMPLE_RATE < audio.numel() < 5 * SAMPLE_RATE
    assert torch.isfinite(audio).all() and audio.abs().max() > 0.01


@pytest.mark.parametrize('text, expected', [
    ('Na ja… gut; oder (vielleicht) so?', 'Na ja. gut, oder, vielleicht, so?'),
    ('Warte... ¡Hola! ¿Qué?', 'Warte. Hola! Qué?'),
    ('Er kam – und ging — sofort.', 'Er kam, und ging, sofort.'),
    ('(Einleitung) Es war einmal (oder nicht).', 'Einleitung, Es war einmal, oder nicht.'),
    ('Ein Fahrrad-Laden, zwei E-Mails.', 'Ein Fahrrad-Laden, zwei E-Mails.'),
])
def test_untrained_punctuation_is_replaced(text, expected):
    assert clean_punctuation(text) == expected


@pytest.mark.parametrize('text, expected', [
    ('Die JSON-Review kommt.', 'Die JSON Review kommt.'),  # both parts overridden
    ('Das KI-Modell ist neu.', 'Das KI Modell ist neu.'),  # one part overridden
    ('Ein Fahrrad-Laden und zwei E-Mails.', 'Ein Fahrrad-Laden und zwei E-Mails.'),  # no part overridden
    ('Viele Start-ups scheitern.', 'Viele Start-ups scheitern.'),  # the whole word is overridden
    ('COVID-19 und Baden-Württemberg.', 'COVID-19 und Baden-Württemberg.'),
])
def test_hyphenated_words_are_split_for_overrides(text, expected):
    assert split_override_compounds(text) == expected
