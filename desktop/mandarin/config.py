"""Local paths for the standalone Mandarin desktop app.

Importing this module does not create directories or download assets. Set any
override before starting Python; relative overrides resolve from the caller's
working directory, and ``~`` expands to the current user's home directory.
"""
from __future__ import annotations

import os
from pathlib import Path


def _path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = _path('LIPCN_CACHE_DIR', Path.home() / '.cache' / 'lipcn')
MODELS = _path('LIPCN_MANDARIN_MODELS', CACHE / 'models' / 'mandarin')
ENGLISH_MODELS = _path('LIPCN_ENGLISH_MODELS', CACHE / 'models' / 'english')
ASR_MODEL = _path('LIPCN_ASR_MODEL', MODELS / 'faster-whisper-large-v3-turbo')
FORMATTER_MODEL = _path('LIPCN_FORMATTER_MODEL', MODELS / 'Qwen3-1.7B-4bit')
CHECKPOINT = _path('LIPCN_CNVSRC_CHECKPOINT', MODELS / 'model_avg_cncvs_2_3_cnvsrc.pth')
SOURCE = _path('LIPCN_CNVSRC_SOURCE', CACHE / 'CNVSRC2025')
# A personal adapter (research/scripts/train_personal_adapter.sh) is installed only
# after it beats Round8 on the user's own dev clips, so prefer it when present.
PERSONAL_ADAPTER = MODELS / 'personal-encoder-adapter.pth'
ADAPTER = _path('LIPCN_LIP_ADAPTER', PERSONAL_ADAPTER if PERSONAL_ADAPTER.is_file()
                else MODELS / 'round8-encoder-adapter.pth')
# Optional CMLR character LM (model.pth + model.json); absent means no LM fusion.
CHINESE_LM = _path('LIPCN_CHINESE_LM', MODELS / 'cmlr-lm')
STATE = _path('LIPCN_LIVE_STATE', CACHE / 'runtime' / 'mandarin')
PROFILE = _path('LIPCN_PROFILE_DIR', Path.home() / 'Library' / 'Application Support' / 'LipCN')
