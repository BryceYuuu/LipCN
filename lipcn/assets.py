"""Pinned public asset inventory. Importing/planning is offline and read-only."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path


MODES = ('speech', 'mandarin', 'english', 'all')
GROUPS = {'speech': {'asr', 'wording', 'face'},
          'mandarin': {'asr', 'wording', 'chinese', 'face'},
          'english': {'english', 'wording', 'face'},
          'all': {'asr', 'wording', 'chinese', 'face', 'english'}}


@dataclass(frozen=True)
class Asset:
    key: str
    group: str
    path: Path
    url: str | None
    sha256: str | None
    size: int | None
    revision: str | None = None
    source: str | None = None
    research_license: bool = False


def paths(environment=None) -> dict[str, Path]:
    env = os.environ if environment is None else environment
    home = Path(env.get('HOME', str(Path.home()))).expanduser()
    def path(name, default):
        return Path(env.get('LIPCN_' + name) or default).expanduser()
    cache = path('CACHE_DIR', home / '.cache/lipcn')
    models = path('MANDARIN_MODELS', cache / 'models/mandarin')
    return {'ASR_MODEL': path('ASR_MODEL', models / 'faster-whisper-large-v3-turbo'),
            'FORMATTER_MODEL': path('FORMATTER_MODEL', models / 'Qwen3-1.7B-4bit'),
            'CHECKPOINT': path('CNVSRC_CHECKPOINT', models / 'model_avg_cncvs_2_3_cnvsrc.pth'),
            'SOURCE': path('CNVSRC_SOURCE', cache / 'CNVSRC2025'),
            'ADAPTER': path('LIP_ADAPTER', models / 'round8-encoder-adapter.pth'),
            'ENGLISH_MODELS': path('ENGLISH_MODELS', cache / 'models/english'),
            'FACE_MODEL': path('FACE_MODEL', cache / 'models/face_landmarker.task')}


def plan(mode='speech', *, environment=None) -> list[Asset]:
    if mode not in GROUPS:
        raise ValueError('Unknown setup mode: ' + mode)
    locations = paths(environment)
    data = json.loads(Path(__file__).with_name('assets_manifest.json').read_text())
    result = []
    for row in data['assets']:
        if row['group'] in GROUPS[mode]:
            path = locations[row['base']]
            if row['relative']:
                path = path / row['relative']
            result.append(Asset(row['key'], row['group'], path, row['url'], row['sha256'],
                                row['size'], row['revision'], row['source'], row['research_license']))
    if mode in ('mandarin', 'all'):
        result.append(Asset('chinese:local-adapter', 'chinese', locations['ADAPTER'],
                            None, None, None, research_license=True))
    return result
