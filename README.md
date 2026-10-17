# kikiri-de Inferenz

Deutsche Sprachsynthese mit dem kikiri-de-Basismodell: TTS auf der Kokoro/StyleTTS-2-Architektur,
von null auf Deutsch trainiert.

| Teil | Wo |
|---|---|
| Inferenzcode (Modell, Pipeline, Tokenizer, Stimmen, CLI) | dieses Repository, `src/kikiri_de/` |
| deutscher Phonemizer (G2P) mit dem Token-Vertrag des Modells | [mideki](https://github.com/ichDaheim/mideki), wird bei der Installation mitinstalliert |
| Modellgewichte und Stimmen (`Heinrich`) | Hugging Face, [`junkstage/kikiri-de`](https://huggingface.co/junkstage/kikiri-de), werden beim ersten Aufruf geladen |

Es läuft unter Windows und Linux, auf der CPU oder mit einer NVIDIA-GPU (CUDA), mit Python 3.10 bis 3.13.

## Installation

Für die Installation muss `git` vorhanden sein (mideki wird von GitHub installiert).
Torch muss **vor** dem Projekt aus dem passenden Index installiert werden, sonst holt pip die Standardvariante.

**Windows** (PowerShell, im Projektordner):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu     # nur CPU
# oder: pip install torch --index-url https://download.pytorch.org/whl/cu128   # NVIDIA-GPU
pip install -e .
```

**Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu     # oder .../whl/cu128
pip install -e .
```

**Mit uv:** `uv sync --extra cpu` oder `uv sync --extra cu128` (für die Tests zusätzlich `--group dev`).

espeak-ng wird über mideki und `espeakng-loader` mitgeliefert (Version 1.52.0). Eine System-Installation von espeak-ng
ist nicht nötig. Andere Versionen sprechen manche Wörter anders aus, als das Modell es gelernt hat.

## Sprache erzeugen

```bash
kikiri say "Die sakrale Kirche steht mitten im Dorf." --voice Heinrich -o ausgabe.wav
kikiri say -i text.txt --voice Heinrich -o kapitel.wav --speed 1.1
kikiri phonemize "Zwölf Boxkämpfer"        # Phoneme und Token-IDs ansehen
```

Beim ersten Aufruf werden Modell (ca. 325 MB) und Stimme von Hugging Face geladen und im Hugging-Face-Cache
abgelegt (Ort über `HF_HOME` änderbar). Danach geht es auch offline, mit `HF_HUB_OFFLINE=1` wird nur der Cache
genutzt. Ein lokaler Modellordner lässt sich mit `-m` angeben, eine lokale Stimme als Pfad zur `.safetensors`-Datei.
Ohne `--device` wird automatisch CUDA genutzt, wenn vorhanden, sonst die CPU. `--device cpu` erzwingt die CPU.
Lange Texte werden automatisch in Sätze zerlegt. Zahlen, Datum und Uhrzeit schreibt der Phonemizer aus.

Aus Python:

```python
from kikiri_de.pipeline import Pipeline
from kikiri_de.audio import save_wav

tts = Pipeline()  # oder Pipeline('pfad/zum/modellordner')
save_wav('ausgabe.wav', tts.synthesize('Guten Morgen! Wie geht es dir?', 'Heinrich'))
```

Die Synthese ist leicht zufällig (Rauschanteil im Vocoder), zwei Läufe sind also nicht bitgleich.
Für reproduzierbare Ausgaben vorher `torch.manual_seed(...)` setzen.

## Stimmen mischen

```bash
kikiri voice mix Heinrich andere_stimme.safetensors --weights 0.7,0.3 -o mix.safetensors
```

Eine Stimme ist ein einzelner Vektor (Klangfarbe und Prosodie). Gemischt wird als gewichteter Mittelwert.
Nur Stimmen desselben Modell-Exports lassen sich mischen.

## Versionen

Code, Phonemizer, Modell und Stimmen müssen zusammenpassen. Deshalb ist jede Version dieses Pakets fest verdrahtet:

- **mideki** ist in `pyproject.toml` auf einen Git-Tag festgelegt. Beim Laden prüft die Pipeline, ob die
  Token-Tabelle von mideki zum Modell passt (`vocab_sha256` in `config.json`).
- **Modell und Stimmen** werden von einem festen Tag des Hugging-Face-Repos geladen (`REVISION` in
  `src/kikiri_de/hub.py`, aktuell `v2.0`). Ältere Versionen des Pakets laden weiterhin ihren eigenen Stand.
- **Stimmen** tragen in ihren Metadaten den Hash des Modell-Exports, aus dem sie extrahiert wurden
  (`source_model_sha256`). Passt er nicht zu `info.json` des Modells, bricht die Pipeline mit einer Fehlermeldung ab.

| kikiri-de-inference | mideki | Hugging Face `junkstage/kikiri-de` |
|---|---|---|
| 0.2.0 | v1.0.0 | v2.0 |
| 0.1.0 | v1.0.0 | v1.0 |

## Tests

```bash
pip install pytest     # oder: uv sync --extra cpu --group dev
pytest
```

## Herkunft und Lizenz

Apache 2.0 (`LICENSE`). Fremdcode und Herkunft: `NOTICE`.
