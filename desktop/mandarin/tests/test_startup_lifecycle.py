"""Startup usability and owning-worker lifecycle without native models/devices."""
from concurrent.futures import Future, ThreadPoolExecutor
import sys
import threading
from types import SimpleNamespace

import pytest

from test_hybrid_ui import (
    Harness, configure_startup, shutdown_app, complete_cleanup, next_shutdown_tick,
)


@pytest.fixture
def h(monkeypatch):
    return Harness(monkeypatch)


def test_speech_readiness_enables_recording_before_visual_and_wording_finish(h, monkeypatch):
    def during_visual_load():
        assert h.ui.speech_ready and not h.ui.lip_ready and not h.ui.wording_ready
        assert h.ui.loading and not h.ui.startup_complete
        h.ui.tick_(None)
        assert h.ui.start_button.enabled
        assert '可以开始' in h.ui.state_label.text and '口型' in h.ui.state_label.text
        h.begin_ready()
        assert h.ui.audio.starts == 1
    configure_startup(h, monkeypatch, on_lips=during_visual_load)
    h.ui._load_model()
    assert h.ui.recording and h.ui.startup_complete
    h.ui.tick_(None)
    assert '采集中' in h.ui.state_label.text


def test_voice_without_microphone_waits_for_lips_then_can_start(h, monkeypatch):
    h.ui.microphone_allowed = False
    def during_visual_load():
        assert h.ui.speech_ready and not h.ui._can_begin()
        h.ui.begin_(None)
        assert not h.cameras and h.ui.audio.starts == 0
    def during_wording_load():
        assert h.ui.lip_ready and h.ui._can_begin() and h.ui.loading
        h.begin_ready()
        assert h.ui.audio.starts == 0
    configure_startup(h, monkeypatch, on_lips=during_visual_load, on_wording=during_wording_load)
    h.ui._load_model()
    assert h.ui.recording and h.ui.startup_complete


@pytest.mark.parametrize('speech,lips', [(False, False), (RuntimeError('ASR failed'), False)])
def test_both_recognizers_failing_never_claim_ready_or_open_devices(h, monkeypatch, speech, lips):
    configure_startup(h, monkeypatch, speech=speech, lips=lips)
    h.ui._load_model()
    assert h.ui.startup_complete and not h.ui.loading and h.ui.model_failed
    h.ui.tick_(None)
    h.ui.begin_(None)
    assert not h.ui._can_begin() and not h.ui.start_button.enabled
    assert '失败' in h.ui.state_label.text and '失败' in h.ui.result.text
    assert not h.cameras and h.ui.audio.starts == 0
    assert h.statuses[-1] == 'error'


def test_wording_failure_does_not_disable_available_recognition(h, monkeypatch):
    configure_startup(h, monkeypatch, wording=RuntimeError('wording unavailable'))
    h.ui._load_model()
    assert h.ui.speech_ready and h.ui.lip_ready and not h.ui.wording_ready
    assert h.ui.startup_complete and h.ui._can_begin() and not h.ui.model_failed
    assert h.ui.result.text == ''


def test_startup_text_reports_stage_and_elapsed_and_queued_inference(h, monkeypatch):
    configure_startup(h, monkeypatch)
    h.now += 3.2
    h.ui._startup_phase('加载语音模型', h.now)
    h.ui.tick_(None)
    assert '语音' in h.ui.state_label.text and '3.2' in h.ui.state_label.text
    assert not h.ui.start_button.enabled
    h.ui.processing = True
    h.ui.inference_started = h.now - 1.5
    h.ui.tick_(None)
    assert '等待模型就绪' in h.ui.state_label.text and '1.5' in h.ui.state_label.text
    h.ui.startup_complete = True
    h.ui.tick_(None)
    assert '识别中' in h.ui.state_label.text


@pytest.mark.parametrize('close_at', ['speech', 'lips'])
def test_closing_between_startup_stages_skips_remaining_warmups(h, monkeypatch, close_at):
    calls = configure_startup(h, monkeypatch)
    if close_at == 'speech':
        def close_during_speech():
            calls.append('speech')
            h.ui._shutdown()
            return True
        h.ui.asr.warmup = close_during_speech
        expected = ['speech']
    else:
        original = h.ui._model_loaded
        def close_after_lips(*args):
            original(*args)
            h.ui._shutdown()
        h.ui._model_loaded = close_after_lips
        expected = ['speech', 'lips', 'lip_encode']
    h.ui._load_model()
    assert calls == expected and h.ui.closed
    assert not h.ui.startup_complete and not h.cameras and h.ui.audio.starts == 0
    assert len(h.executor.tasks) == 1 and h.executor.tasks[0]['target'] == h.ui._cleanup_models


def test_late_startup_callbacks_after_close_cannot_reenable_or_publish(h, monkeypatch):
    configure_startup(h, monkeypatch)
    h.ui._shutdown()
    states = list(h.statuses)
    h.ui._speech_loaded(True, 1)
    h.ui._model_loaded(object(), 'digest', 1)
    h.ui._startup_finished(True, 1, 3)
    h.ui._startup_phase('过期阶段', h.now)
    assert not h.ui._can_begin() and not h.ui.speech_ready and not h.ui.lip_ready
    assert not h.ui.startup_complete and h.ui.reader is None
    assert h.statuses == states


@pytest.mark.parametrize('field', ['model_worker', 'inference_worker', 'cleanup_future'])
def test_workers_alive_observes_real_pending_running_and_finished_futures(h, field):
    future = Future()
    setattr(h.ui, field, future)
    assert h.ui._workers_alive()
    assert future.set_running_or_notify_cancel()
    assert h.ui._workers_alive()
    future.set_result(None)
    assert not h.ui._workers_alive()


def test_repeated_quit_cancels_queued_inference_and_schedules_cleanup_once(h, monkeypatch):
    app, events = shutdown_app(h, monkeypatch)
    h.ui.inference_worker = h.executor.submit(lambda: pytest.fail('cancelled inference ran'))
    h.ui.quit_(None)
    h.ui.quit_(None)
    assert h.ui.inference_worker.cancelled()
    assert len(h.executor.tasks) == 2 and len(h.later) == 1
    assert not events and h.hotkey_stops == 1
    complete_cleanup(h)
    next_shutdown_tick(h)
    assert events[0] == 'stop' and h.statuses[-1] == 'closed'


def test_startup_inference_and_resource_cleanup_use_one_persistent_worker(h, monkeypatch):
    """Real lightweight executor scheduling, with every model and device a fake."""
    entered, release = threading.Event(), threading.Event()
    work = []
    def visual_load():
        work.append(('lips', threading.get_ident()))
        entered.set()
        assert release.wait(3), 'test did not release the synthetic startup task'
    configure_startup(h, monkeypatch, on_lips=visual_load)
    speech = h.ui.asr.warmup
    wording = h.ui.formatter.warmup
    h.ui.asr.warmup = lambda: (work.append(('speech', threading.get_ident())), speech())[1]
    h.ui.formatter.warmup = lambda: (work.append(('wording', threading.get_ident())), wording())[1]
    h.ui.asr.close = lambda: work.append(('cleanup', threading.get_ident()))
    h.ui._infer = lambda *args: work.append(('inference', threading.get_ident()))
    h.globals['ThreadPoolExecutor'] = ThreadPoolExecutor
    h.globals['NSTimer'] = SimpleNamespace(
        scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_=lambda *args: h.ui.timer)
    h.globals['CommandToggle'] = lambda *args, **kwargs: SimpleNamespace(
        start=lambda: True, stop=h.stop_hotkey, is_listening=True)
    h.ui._request_microphone = lambda: None
    monkeypatch.setitem(sys.modules, 'mlx.core', None)
    sys.modules['torch'].mps = SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None)
    h.ui.launch()
    executor = h.ui.model_executor
    try:
        assert entered.wait(3)
        assert h.ui.speech_ready and not h.ui.startup_complete
        h.begin_ready()
        h.ui.finish_(None)
        _, finish, args = h.later[-1]
        finish(*args)
        assert not h.ui.inference_worker.done()
        assert [name for name, _ in work] == ['speech', 'lips']
        release.set()
        h.ui.model_worker.result(timeout=3)
        h.ui.inference_worker.result(timeout=3)
        h.ui._shutdown()
        h.ui.cleanup_future.result(timeout=3)
        assert [name for name, _ in work] == ['speech', 'lips', 'wording', 'inference', 'cleanup']
        assert len({ident for _, ident in work}) == 1
        assert work[0][1] != threading.get_ident()
        assert h.ui.reader is None and h.ui.formatter._model is None
        assert h.ui.cleanup_errors == {} and not h.ui.audio.capturing and h.ui.camera is None
    finally:
        release.set()
        executor.shutdown(wait=True, cancel_futures=True)
