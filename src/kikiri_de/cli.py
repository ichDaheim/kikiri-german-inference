"""Command line interface: `kikiri <command>` (see `kikiri --help`)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional


def _say(args) -> int:
    from .audio import SAMPLE_RATE, save_wav
    from .pipeline import Pipeline

    text = Path(args.input).read_text(encoding='utf-8') if args.input else args.text
    if not text:
        print('error: give TEXT or --input FILE', file=sys.stderr)
        return 2
    pipeline = Pipeline(args.model, device=args.device)
    audio = pipeline.synthesize(text, args.voice, speed=args.speed)
    save_wav(args.output, audio)
    print(f'{args.output}: {audio.numel() / SAMPLE_RATE:.1f} s')
    return 0


def _detect(args) -> int:
    import soundfile as sf

    from .watermark import Watermarker

    watermarker = Watermarker(args.device)
    found = 0
    for path in args.files:
        with sf.SoundFile(str(path)) as f:
            comment = f.copy_metadata().get('comment', '')
            wave = f.read(dtype='float32')
            sample_rate = f.samplerate
        detection = watermarker.detect(wave, sample_rate)
        if detection.is_kikiri:
            verdict = 'kikiri-de-Wasserzeichen gefunden'
        elif detection.watermarked:
            verdict = 'AudioSeal-Wasserzeichen gefunden (andere Kennung)'
        else:
            verdict = 'kein Wasserzeichen'
        found += detection.watermarked
        print(f'{path}: {verdict} (Wahrscheinlichkeit {detection.probability:.2f})')
        if comment:
            print(f'  Metadaten: {comment}')
    return 0 if found == len(args.files) else 1


def _phonemize(args) -> int:
    from .g2p import G2P
    from .tokenizer import PhonemeTokenizer

    phonemes = G2P()(args.text)
    print(phonemes)
    print(PhonemeTokenizer().encode(phonemes, strict=False))
    return 0


def _voice_mix(args) -> int:
    from .hub import resolve_voice
    from .voice import Voice

    weights = [float(w) for w in args.weights.split(',')] if args.weights else None
    Voice.mix([Voice.load(resolve_voice(v)) for v in args.voices], weights).save(args.output)
    print(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='kikiri', description='kikiri-de: deutsche Sprachsynthese')
    parser.add_argument('-v', '--verbose', action='store_true')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('say', help='Text in Sprache umwandeln')
    p.add_argument('text', nargs='?')
    p.add_argument('-i', '--input', help='Textdatei statt TEXT')
    p.add_argument('-m', '--model', help='lokaler Modellordner (Standard: passende Version von Hugging Face)')
    p.add_argument('--voice', required=True, help='Stimmenname (z. B. Heinrich) oder .safetensors-Datei')
    p.add_argument('-o', '--output', default='ausgabe.wav')
    p.add_argument('--speed', type=float, default=1.0)
    p.add_argument('--device')
    p.set_defaults(func=_say)

    p = sub.add_parser('detect', help='WAV-Dateien auf das KI-Wasserzeichen prüfen')
    p.add_argument('files', nargs='+', type=Path)
    p.add_argument('--device')
    p.set_defaults(func=_detect)

    p = sub.add_parser('phonemize', help='Phoneme und Token-IDs eines Textes anzeigen')
    p.add_argument('text')
    p.set_defaults(func=_phonemize)

    voice = sub.add_parser('voice', help='Stimmen mischen').add_subparsers(dest='voice_command', required=True)
    p = voice.add_parser('mix', help='Stimmen gewichtet mitteln')
    p.add_argument('voices', nargs='+')
    p.add_argument('--weights', help='z. B. 0.7,0.3')
    p.add_argument('-o', '--output', required=True)
    p.set_defaults(func=_voice_mix)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')  # IPA output on Windows consoles
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
