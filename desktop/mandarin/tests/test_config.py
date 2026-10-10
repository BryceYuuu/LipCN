"""Portable path resolution without importing backends or touching devices."""
from pathlib import Path
import runpy

import pytest


CONFIG = Path(__file__).resolve().parents[1] / 'config.py'
OVERRIDES = {
    'CACHE': 'LIPCN_CACHE_DIR',
    'MODELS': 'LIPCN_MANDARIN_MODELS',
    'ENGLISH_MODELS': 'LIPCN_ENGLISH_MODELS',
    'ASR_MODEL': 'LIPCN_ASR_MODEL',
    'FORMATTER_MODEL': 'LIPCN_FORMATTER_MODEL',
    'CHECKPOINT': 'LIPCN_CNVSRC_CHECKPOINT',
    'SOURCE': 'LIPCN_CNVSRC_SOURCE',
    'ADAPTER': 'LIPCN_LIP_ADAPTER',
    'STATE': 'LIPCN_LIVE_STATE',
    'PROFILE': 'LIPCN_PROFILE_DIR',
}


@pytest.fixture(autouse=True)
def clean_overrides(monkeypatch):
    for variable in OVERRIDES.values():
        monkeypatch.delenv(variable, raising=False)


def test_defaults_follow_checkout_and_home_without_creating_files(monkeypatch, tmp_path):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))
    config = runpy.run_path(str(CONFIG))
    cache = tmp_path / '.cache' / 'lipcn'
    models = cache / 'models' / 'mandarin'
    assert config['ROOT'] == CONFIG.parents[2]
    assert config['CACHE'] == cache
    assert config['ENGLISH_MODELS'] == cache / 'models' / 'english'
    assert config['ASR_MODEL'] == models / 'faster-whisper-large-v3-turbo'
    assert config['FORMATTER_MODEL'] == models / 'Qwen3-1.7B-4bit'
    assert config['CHECKPOINT'] == models / 'model_avg_cncvs_2_3_cnvsrc.pth'
    assert config['ADAPTER'] == models / 'round8-encoder-adapter.pth'
    assert config['SOURCE'] == cache / 'CNVSRC2025'
    assert config['STATE'] == cache / 'runtime' / 'mandarin'
    assert config['PROFILE'] == tmp_path / 'Library' / 'Application Support' / 'LipCN'
    assert not cache.exists()


def test_cache_and_model_root_overrides_cascade(monkeypatch, tmp_path):
    cache, models = tmp_path / 'cache', tmp_path / 'models'
    monkeypatch.setenv('LIPCN_CACHE_DIR', str(cache))
    monkeypatch.setenv('LIPCN_MANDARIN_MODELS', str(models))
    config = runpy.run_path(str(CONFIG))
    assert config['ASR_MODEL'].parent == models
    assert config['ENGLISH_MODELS'] == cache / 'models' / 'english'
    assert config['SOURCE'] == cache / 'CNVSRC2025'
    assert config['STATE'] == cache / 'runtime' / 'mandarin'


@pytest.mark.parametrize('constant,variable', OVERRIDES.items())
def test_individual_paths_support_home_expansion(monkeypatch, tmp_path, constant, variable):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))
    monkeypatch.setenv(variable, '~/custom asset')
    config = runpy.run_path(str(CONFIG))
    assert config[constant] == tmp_path / 'custom asset'


def test_relative_overrides_remain_relative_to_launch_directory(monkeypatch):
    monkeypatch.setenv('LIPCN_ASR_MODEL', 'local-assets/whisper')
    config = runpy.run_path(str(CONFIG))
    assert config['ASR_MODEL'] == Path('local-assets/whisper')
