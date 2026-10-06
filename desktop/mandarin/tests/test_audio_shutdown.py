"""Lifecycle/ordering checks with fake backends; no native model or capture."""
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from hybrid_audio import LocalASR


def install_backends(monkeypatch, events):
    import hybrid_audio
    monkeypatch.setattr(hybrid_audio, '_use_cached_packages', lambda: None)
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        disable_telemetry_events=lambda: events.append('disable_telemetry')))

    def timestamps(audio, **kwargs):
        events.append('vad_session')
        return []

    class Model:
        def __init__(self, *args, **kwargs):
            events.append('whisper_model')

    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=Model))
    monkeypatch.setitem(sys.modules, 'faster_whisper.vad', SimpleNamespace(
        VadOptions=lambda **kwargs: kwargs, get_speech_timestamps=timestamps,
        get_vad_model=SimpleNamespace(cache_clear=lambda: events.append('clear_vad'))))


def test_vad_telemetry_precedes_first_session_and_clears_once(monkeypatch):
    events = []
    install_backends(monkeypatch, events)
    asr = LocalASR(model=object())
    asr._ensure_vad()
    asr._speech_detector(np.zeros(16000, dtype=np.float32))
    assert events == ['disable_telemetry', 'vad_session']
    asr.close()
    asr.close()
    assert events == ['disable_telemetry', 'vad_session', 'clear_vad']
    assert not asr.loaded


def test_direct_load_also_disables_telemetry_before_backend(monkeypatch, tmp_path):
    events = []
    install_backends(monkeypatch, events)
    monkeypatch.setitem(sys.modules, 'opencc', SimpleNamespace(OpenCC=lambda mode: object()))
    for name in ('model.bin', 'config.json', 'tokenizer.json'):
        (tmp_path / name).touch()
    asr = LocalASR(tmp_path)
    asr.load()
    assert events == ['disable_telemetry', 'whisper_model']
    asr.close()
    assert asr._converter is None


def test_close_unloaded_backend_is_terminal_without_import(monkeypatch):
    asr = LocalASR()
    monkeypatch.setattr(asr, '_prepare_ort', lambda: pytest.fail('closed instance imported a backend'))
    asr.close()
    assert not asr.warmup()
    with pytest.raises(RuntimeError, match='closed'):
        asr.load()
    result = asr.transcribe(np.full(16000, .01, dtype=np.float32))
    assert result.reason == 'closed'
    assert not result.accepted


def test_close_waits_until_inflight_owner_releases_lock():
    asr = LocalASR(model=object(), speech_detector=lambda audio: [])
    entered = threading.Event()
    finished = threading.Event()

    def closing():
        entered.set()
        asr.close()
        finished.set()

    with asr._lock:
        worker = threading.Thread(target=closing)
        worker.start()
        assert entered.wait(1)
        assert not finished.wait(.05)
        assert asr.loaded
    worker.join(1)
    assert finished.is_set()
    assert not asr.loaded
