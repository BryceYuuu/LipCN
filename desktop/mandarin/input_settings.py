"""Only an explicit input-mode preference is persisted; never utterances."""
from __future__ import annotations
import json
from pathlib import Path
import os
import tempfile

INPUT_MODES = (('speech-first', '语音优先'), ('lip-only', '纯口型'))
DEFAULT_INPUT_MODE = 'speech-first'


def load_input_mode(profile: Path) -> str:
    try:
        mode = json.loads((profile / 'input-settings.json').read_text('utf-8')).get('input_mode')
        return mode if mode in dict(INPUT_MODES) else DEFAULT_INPUT_MODE
    except (OSError, ValueError, AttributeError):
        return DEFAULT_INPUT_MODE


def save_input_mode(profile: Path, mode: str) -> None:
    if mode not in dict(INPUT_MODES):
        raise ValueError('Unsupported input mode')
    profile.mkdir(parents=True, exist_ok=True)
    path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=profile,
                                         prefix='.input-settings-', delete=False) as output:
            path = Path(output.name)
            json.dump({'input_mode': mode}, output)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        path.replace(profile / 'input-settings.json')
    finally:
        if path is not None:
            path.unlink(missing_ok=True)
