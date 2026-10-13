"""Input modes are session-only; every launch starts with lip reading."""
from __future__ import annotations
from pathlib import Path

INPUT_MODES = (('speech-first', '语音优先'), ('lip-only', '纯口型'))
DEFAULT_INPUT_MODE = 'lip-only'


def load_input_mode(profile: Path) -> str:
    """Compatibility hook: ignore all old preferences without reading files."""
    return DEFAULT_INPUT_MODE


def save_input_mode(profile: Path, mode: str) -> None:
    """Compatibility hook: validate a session choice but never persist it."""
    if mode not in dict(INPUT_MODES):
        raise ValueError('Unsupported input mode')
