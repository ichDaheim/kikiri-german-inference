"""Model and voices from the Hugging Face Hub, pinned to the release that matches this code.

Local paths always win: an existing folder or `.safetensors` file is used as is.
Downloads are cached by `huggingface_hub` (`HF_HOME`); `HF_HUB_OFFLINE=1` uses the cache only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import hf_hub_download, snapshot_download

from .model import CONFIG_FILE, INFO_FILE, WEIGHTS_FILE

logger = logging.getLogger(__name__)

REPO_ID = 'junkstage/kikiri-de'
# HF tag of the model export and voices this code release belongs to. Never moved once published.
REVISION = 'v1.0'
VOICE_DIR = 'voices'


def resolve_model(model: Optional[Union[str, Path]] = None) -> Path:
    '''Local model folder, or the pinned release from the Hub if `model` is None.'''
    if model is not None:
        path = Path(model)
        if not path.is_dir():
            raise FileNotFoundError(f'model folder not found: {path}')
        return path
    logger.info('loading model %s@%s', REPO_ID, REVISION)
    return Path(snapshot_download(REPO_ID, revision=REVISION, allow_patterns=[CONFIG_FILE, WEIGHTS_FILE, INFO_FILE]))


def resolve_voice(voice: Union[str, Path]) -> Path:
    '''Local `.safetensors` file, or a voice name (e.g. `Heinrich`) from the pinned release.'''
    path = Path(voice)
    if path.is_file():
        return path
    if path.suffix or len(path.parts) > 1:
        raise FileNotFoundError(f'voice file not found: {path}')
    logger.info('loading voice %s from %s@%s', path.name, REPO_ID, REVISION)
    return Path(hf_hub_download(REPO_ID, f'{VOICE_DIR}/{path.name}.safetensors', revision=REVISION))
