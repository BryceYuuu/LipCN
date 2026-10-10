"""English reader wiring without importing real model backends or devices."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from english_lips import load_english_reader


@pytest.fixture
def assets(tmp_path):
    for name in ('vsr/model.json', 'vsr/model.pth', 'lm/model.json', 'lm/model.pth'):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'local test asset')
    return tmp_path


@pytest.mark.parametrize('name', ['vsr/model.json', 'vsr/model.pth', 'lm/model.json', 'lm/model.pth'])
@pytest.mark.parametrize('invalid', ['missing', 'empty', 'directory'])
def test_incomplete_assets_fail_before_importing_reader(monkeypatch, assets, name, invalid):
    path = assets / name
    path.unlink()
    if invalid == 'empty':
        path.touch()
    elif invalid == 'directory':
        path.mkdir()
    monkeypatch.setitem(sys.modules, 'lipflow.vsr', None)
    with pytest.raises(FileNotFoundError, match=name):
        load_english_reader(assets)


def test_english_recipe_and_encoder_only_warmup(monkeypatch, assets):
    calls = []

    class Reader:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def encode(self, rois):
            assert rois.shape == (25, 96, 96)
            assert rois.dtype == np.uint8
            assert not rois.any()
            calls.append('encode')

    monkeypatch.setitem(sys.modules, 'lipflow.vsr', SimpleNamespace(LipReader=Reader))
    reader = load_english_reader(str(assets))
    assert isinstance(reader, Reader)
    assert calls == [dict(language='en', model_dir=str(assets), beam_size=4, personal=False), 'encode']


def test_loading_error_is_not_replaced_with_a_fake_reader(monkeypatch, assets):
    def fail(**kwargs):
        raise RuntimeError('incompatible weights')

    monkeypatch.setitem(sys.modules, 'lipflow.vsr', SimpleNamespace(LipReader=fail))
    with pytest.raises(RuntimeError, match='incompatible weights'):
        load_english_reader(assets)
