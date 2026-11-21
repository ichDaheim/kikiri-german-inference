import pytest

from kikiri_de.pipeline import split_sentences


@pytest.mark.parametrize('text, expected', [
    ('Satz eins. Satz zwei?  Drei…', ['Satz eins.', 'Satz zwei?', 'Drei…']),
    ('Zeile eins\n\nZeile zwei', ['Zeile eins', 'Zeile zwei']),
    ('Bis zum 1. Januar bleibt alles. Danach nicht.', ['Bis zum 1. Januar bleibt alles.', 'Danach nicht.']),
    ('Dr. Müller kommt z. B. am 3. Mai. Gut!', ['Dr. Müller kommt z. B. am 3. Mai.', 'Gut!']),
    ('Das war 2020. Dann kam 2021.', ['Das war 2020.', 'Dann kam 2021.']),
])
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected
