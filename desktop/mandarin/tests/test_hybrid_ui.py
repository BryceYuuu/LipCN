"""Headless UI-state regressions; AST extraction never imports AppKit or cameras.

All device, permissions, timers and worker threads are test doubles. No real
camera, microphone, keyboard monitor, filesystem telemetry or model is opened.
"""
from __future__ import annotations

import ast
import copy
import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'live_test.py'


class Widget:
    def __init__(self):
        self.enabled = True
        self.text = ''
        self.images = []
        self.calls = []

    def setEnabled_(self, value):
        self.enabled = bool(value)

    def setString_(self, value):
        self.text = value

    def setStringValue_(self, value):
        self.text = value

    def string(self):
        return self.text

    def setImage_(self, value):
        self.images.append(value)

    def deminiaturize_(self, sender):
        self.calls.append('deminiaturize')

    def makeKeyAndOrderFront_(self, sender):
        self.calls.append('show')

    def isMiniaturized(self):
        return False

    def isVisible(self):
        return True

    def orderOut_(self, sender):
        self.calls.append('hide')


class ManualExecutor:
    """Queue real Future objects without running threads or native model work."""
    def __init__(self):
        self.tasks = []
        self.shutdown_calls = []

    def submit(self, target, *args):
        future = Future()
        self.tasks.append({'target': target, 'args': args, 'future': future})
        return future

    def shutdown(self, **kwargs):
        self.shutdown_calls.append(kwargs)


class FakeAudio:
    def __init__(self):
        self.starts = self.stops = self.cancels = 0
        self.start_ok = True
        self.last_error = None
        self.capturing = False
        self.data = [0.1] * 16

    def start(self):
        self.starts += 1
        self.capturing = self.start_ok
        if not self.start_ok:
            self.last_error = 'microphone_unavailable:PermissionError'
        return self.start_ok

    def stop(self):
        self.stops += 1
        self.capturing = False
        return self.data

    def cancel(self):
        self.cancels += 1
        self.capturing = False


class FakeCamera:
    def __init__(self, owner):
        self.owner = owner
        self.is_open = False
        self.ready = SimpleNamespace(is_set=lambda: self.owner.camera_ready)
        self.error = owner.camera_error
        self.start_count = self.stop_count = self.close_count = 0
        self.recording = None
        self.track_always = False

    def ensure_open(self):
        self.is_open = True

    def start_recording(self):
        self.start_count += 1
        self.recording = SimpleNamespace(
            started=self.owner.now, ts=[], grays=[], anchors=[], mouth_open=[], mouth_pixels=[])
        return self.recording

    def stop_recording(self):
        self.stop_count += 1
        rec, self.recording = self.recording, None
        return rec

    def close(self):
        self.is_open = False
        self.close_count += 1


class Harness:
    def __init__(self, monkeypatch):
        self.now = 10.0
        self.later = []
        self.after = []
        self.threads = []
        self.executor = ManualExecutor()
        self.cameras = []
        self.permission_requests = []
        self.camera_permission = 3
        self.microphone_permission = 3
        self.camera_ready = True
        self.camera_error = None
        self.activations = []
        self.captures = 0
        self.hotkey_stops = 0
        self.timer_invalidations = 0

        def camera_factory(*args, **kwargs):
            camera = FakeCamera(self)
            self.cameras.append(camera)
            return camera

        def thread_factory(**kwargs):
            return SimpleNamespace(start=lambda: self.threads.append(kwargs), is_alive=lambda: False)

        def authorization(media):
            return self.camera_permission if media == 'video' else self.microphone_permission

        def request_permission(media, callback):
            self.permission_requests.append((media, callback))

        av = SimpleNamespace(
            AVMediaTypeVideo='video', AVMediaTypeAudio='audio',
            AVCaptureDevice=SimpleNamespace(
                authorizationStatusForMediaType_=authorization,
                requestAccessForMediaType_completionHandler_=request_permission))
        monkeypatch.setitem(sys.modules, 'AVFoundation', av)
        self.globals = {
            '__name__': 'headless_hybrid_ui', 'MAX_SECONDS': 20.0, 'TAIL_SECONDS': .4,
            'time': SimpleNamespace(monotonic=lambda: self.now),
            'AppHelper': SimpleNamespace(
                callLater=lambda delay, callback, *args: self.later.append((delay, callback, args)),
                callAfter=lambda callback, *args: self.after.append((callback, args))),
            'threading': SimpleNamespace(Thread=thread_factory), 'Camera': camera_factory,
            'sys': sys,
            'NSApplication': SimpleNamespace(sharedApplication=lambda: SimpleNamespace(
                activateIgnoringOtherApps_=lambda force: self.activations.append(force))),
        }
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LiveTest')
        methods = []
        for node in original.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node = copy.deepcopy(node)
                node.decorator_list = []
                methods.append(node)
        cls = ast.ClassDef(name='HeadlessUI', bases=[], keywords=[], body=methods, decorator_list=[])
        unit = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
        exec(compile(unit, str(SOURCE), 'exec'), self.globals)
        self.ui = self.globals['HeadlessUI']()
        ui = self.ui
        defaults = dict(
            reader=object(), camera=None, closed=False, suspended=False, loading=False,
            processing=False, recording=False, pending_stop=None, pending_begin=False,
            camera_permission_pending=False,
            capture_guidance='', token=0, frames=0, face_present=False, last_frame_at=0,
            last_status_write=0, completed=0, error=None, model_failed=False,
            microphone_allowed=True, microphone_note='语音优先已启用', hotkey_note='',
            result_variants=[], record_started=0, inference_started=0,
            audio=FakeAudio(), window=Widget(), result=Widget(), state_label=Widget(),
            start_button=Widget(), stop_button=Widget(), cancel_button=Widget(),
            access_label=Widget(), video=Widget(), mouth=Widget(), choice_buttons=[],
            hotkey=SimpleNamespace(stop=self.stop_hotkey),
            output_target=SimpleNamespace(capture=self.capture),
            timer=SimpleNamespace(invalidate=self.invalidate_timer),
            model_executor=self.executor,
        )
        for name, value in defaults.items():
            setattr(ui, name, value)
        self.statuses = []
        ui._status = self.statuses.append
        ui._camera_frame = lambda *args: None

    def capture(self):
        self.captures += 1

    def stop_hotkey(self):
        self.hotkey_stops += 1

    def invalidate_timer(self):
        self.timer_invalidations += 1

    def flush_after(self):
        while self.after:
            callback, args = self.after.pop(0)
            callback(*args)

    def grant_camera(self):
        self.camera_permission = 3
        callbacks = [callback for media, callback in self.permission_requests if media == 'video']
        assert callbacks, 'test expected a camera permission request'
        for callback in callbacks:
            callback(True)
        self.flush_after()

    def begin_ready(self):
        self.ui.begin_(None)
        # Prepared camera readiness may be checked by the next normal UI tick.
        if self.ui.pending_begin:
            self.ui.tick_(None)
        assert self.ui.recording


@pytest.fixture
def h(monkeypatch):
    return Harness(monkeypatch)


@pytest.mark.parametrize('blocked', ['closed', 'suspended', 'loading', 'processing', 'recording'])
def test_begin_does_not_touch_devices_when_unavailable(h, blocked):
    setattr(h.ui, blocked, True)
    h.ui.begin_(None)
    assert not h.cameras
    assert h.ui.audio.starts == 0


def test_normal_begin_starts_once_and_disables_old_choices(h):
    choice = Widget()
    h.ui.choice_buttons = [choice]
    h.ui.result_variants = ['old output']
    h.begin_ready()
    h.ui.begin_(None)
    assert len(h.cameras) == 1
    assert h.cameras[0].start_count == 1
    assert h.ui.audio.starts == 1
    assert not choice.enabled and not h.ui.result_variants


def test_denied_microphone_still_allows_lips_and_does_not_open_microphone(h):
    h.ui._microphone_permission(False)
    h.begin_ready()
    assert h.ui.audio.starts == 0
    assert h.cameras[0].start_count == 1
    assert '仅唇语' in h.ui.microphone_note


def test_microphone_stream_failure_is_visual_fallback(h):
    h.ui.audio.start_ok = False
    h.begin_ready()
    assert h.ui.audio.starts == 1
    assert h.cameras[0].start_count == 1
    assert '麦克风' in h.ui.microphone_note


def test_finish_keeps_one_tail_and_ignores_repeat_finish(h):
    h.begin_ready()
    h.now += 2
    before = len(h.later)
    h.ui.finish_(None)
    h.ui.finish_(None)
    assert len(h.later) == before + 1
    delay, _, args = h.later[-1]
    assert delay == pytest.approx(.4)
    assert args == (h.ui.token,)
    assert h.ui.recording and h.ui.pending_stop == h.ui.token
    assert not h.ui.stop_button.enabled


def test_finish_tail_respects_twenty_second_cap(h):
    h.begin_ready()
    h.now = h.ui.record_started + 19.85
    h.ui.finish_(None)
    assert h.later[-1][0] == pytest.approx(.15)


def test_cancel_invalidates_pending_tail_and_discards_audio(h):
    h.begin_ready()
    h.ui.finish_(None)
    _, finish, args = h.later[-1]
    h.ui.cancel_(None)
    finish(*args)
    assert not h.ui.recording and h.ui.pending_stop is None
    assert h.ui.audio.cancels == 1
    assert h.ui.camera is None
    assert not h.executor.tasks


def test_stale_auto_finish_does_not_stop_new_sentence(h):
    h.begin_ready()
    old_token = h.ui.token
    h.ui.cancel_(None)
    h.begin_ready()
    h.ui._auto_finish(old_token)
    assert h.ui.recording and h.ui.pending_stop is None


def test_tail_completion_closes_devices_and_queues_one_worker(h):
    h.begin_ready()
    rec = h.cameras[0].recording
    rec.ts = [rec.started + 1, rec.started + 19.9, rec.started + 20.2]
    for name in ['grays', 'anchors', 'mouth_open', 'mouth_pixels']:
        setattr(rec, name, [1, 2, 3])
    h.ui.finish_(None)
    _, finish, args = h.later[-1]
    finish(*args)
    finish(*args)
    assert h.ui.camera is None and not h.ui.recording
    assert h.ui.audio.stops == 1 and h.ui.processing
    assert len(h.executor.tasks) == 1 and not h.threads
    assert h.executor.tasks[0]['target'] == h.ui._infer
    assert h.executor.tasks[0]['args'][1] is rec
    assert h.ui.inference_worker is h.executor.tasks[0]['future']
    assert len(rec.ts) == 2 and len(rec.grays) == 2


def test_shortcut_captures_output_target_before_activating_window(h):
    sequence = []
    h.ui.output_target.capture = lambda: sequence.append('capture')
    h.ui.window.makeKeyAndOrderFront_ = lambda sender: sequence.append('window')
    h.ui.begin_ = lambda sender: sequence.append('begin')
    h.ui._toggle_command()
    assert sequence == ['capture', 'window', 'begin']


def test_shortcut_finishes_active_sentence_without_changing_output_target(h):
    h.begin_ready()
    h.ui._toggle_command()
    assert h.ui.pending_stop == h.ui.token
    assert h.captures == 0


@pytest.mark.parametrize('blocked', ['closed', 'processing'])
def test_shortcut_does_nothing_while_closed_or_processing(h, blocked):
    setattr(h.ui, blocked, True)
    h.ui._toggle_command()
    assert h.captures == 0 and not h.cameras


def test_suspend_cancels_input_and_closes_devices_but_keeps_hotkey(h):
    h.begin_ready()
    h.ui._suspend()
    assert h.ui.suspended and not h.ui.recording and h.ui.camera is None
    assert not h.ui.audio.capturing
    assert h.hotkey_stops == 0
    assert h.ui.video.images[-1] is None


def test_shutdown_is_idempotent_stops_input_monitor_and_invalidates_tail(h):
    h.begin_ready()
    h.ui.finish_(None)
    _, callback, args = h.later[-1]
    h.ui._shutdown()
    h.ui._shutdown()
    callback(*args)
    assert h.ui.closed and h.ui.camera is None and not h.ui.audio.capturing
    assert h.hotkey_stops == 1 and h.timer_invalidations == 1
    assert len(h.executor.tasks) == 1
    assert h.executor.tasks[0]['target'] == h.ui._cleanup_models
    assert not h.threads


def test_cold_camera_waits_before_capture_and_twenty_second_timer(h):
    h.camera_ready = False
    h.ui.begin_(None)
    assert h.ui.pending_begin and not h.ui.recording
    assert h.ui.audio.starts == 0 and h.cameras[0].start_count == 0
    assert all(callback.__name__ != '_auto_finish' for _, callback, _ in h.later)
    h.ui.tick_(None)
    assert '准备' in h.ui.state_label.text
    delay, callback, args = h.later[-1]
    h.now += delay
    h.camera_ready = True
    callback(*args)
    assert h.ui.recording and not h.ui.pending_begin
    assert h.ui.record_started == h.now
    assert h.ui.audio.starts == 1 and h.cameras[0].start_count == 1


def test_second_command_during_preparation_cancels_without_another_open(h):
    h.camera_ready = False
    h.ui._toggle_command()
    assert h.ui.pending_begin
    h.ui._toggle_command()
    assert not h.ui.pending_begin and not h.ui.recording
    assert len(h.cameras) == 1 and h.ui.camera is None
    for _, callback, args in list(h.later):
        callback(*args)
    assert h.ui.audio.starts == 0


@pytest.mark.parametrize('cancel_action', ['cancel_', '_suspend', '_shutdown'])
def test_late_camera_permission_after_cancellation_cannot_reopen_camera(h, cancel_action):
    h.camera_permission = 0
    h.ui.begin_(None)
    assert h.ui.pending_begin and h.permission_requests
    if cancel_action == 'cancel_':
        h.ui.cancel_(None)
    else:
        getattr(h.ui, cancel_action)()
    h.grant_camera()
    assert not h.cameras and not h.ui.recording and h.ui.camera is None
    for _, callback, args in list(h.later):
        callback(*args)
    assert h.ui.audio.starts == 0


def test_permission_denied_camera_can_record_authorized_voice(h):
    h.camera_permission = 2
    h.begin_ready()
    assert h.ui.camera is None and not h.cameras
    assert h.ui.audio.starts == 1


def test_idle_escape_preserves_results_and_does_not_cancel_devices(h):
    h.ui.result_variants = ['保留这句。']
    h.ui.result.text = '原来的识别结果'
    token = h.ui.token
    h.ui._escape()
    assert h.ui.result_variants == ['保留这句。']
    assert h.ui.result.text == '原来的识别结果'
    assert h.ui.token == token and h.ui.audio.cancels == 0


def test_escape_during_preparation_cancels_delayed_begin(h):
    h.camera_ready = False
    h.ui.begin_(None)
    h.ui._escape()
    assert not h.ui.pending_begin
    for _, callback, args in list(h.later):
        callback(*args)
    assert not h.ui.recording and h.ui.audio.starts == 0


def test_escape_during_capture_closes_devices(h):
    h.begin_ready()
    h.ui._escape()
    assert not h.ui.recording and not h.ui.audio.capturing and h.ui.camera is None


def configure_startup(h, monkeypatch, *, speech=True, lips=True, wording=True,
                      on_lips=None, on_wording=None):
    """Exercise production startup methods with only in-memory model doubles."""
    calls = []
    for name, value in dict(reader=None, loading=True, speech_ready=False,
            lip_ready=False, wording_ready=False, startup_complete=False,
            startup_stage='加载语音模型', startup_started=h.now,
            startup_stage_started=h.now, warmup_seconds={}).items():
        setattr(h.ui, name, value)

    def outcome(name, value):
        calls.append(name)
        h.now += 1
        if isinstance(value, Exception):
            raise value
        return value

    reader = SimpleNamespace(enc_device=SimpleNamespace(type='cpu'),
                             encode=lambda value: outcome('lip_encode', None))
    def load_reader(*args):
        if on_lips is not None:
            on_lips()
        if not outcome('lips', lips):
            raise FileNotFoundError('visual checkpoint missing')
        return reader

    def load_wording():
        if on_wording is not None:
            on_wording()
        return outcome('wording', wording)

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(set_num_threads=lambda count: None))
    monkeypatch.setitem(sys.modules, 'evaluate_cnvsrc', SimpleNamespace(
        _reader=load_reader, _load_adapter=lambda *args: {'file_sha256': '0' * 64}))
    h.globals.update(CHECKPOINT='unused', SOURCE='unused', ADAPTER='unused',
                     np=SimpleNamespace(zeros=lambda *args, **kwargs: None, uint8='uint8'))
    # Simulate the main loop consuming each startup notification between stages.
    h.globals['AppHelper'].callAfter = lambda fn, *args: fn(*args)
    h.ui.asr = SimpleNamespace(warmup=lambda: outcome('speech', speech))
    h.ui.formatter = SimpleNamespace(warmup=load_wording)
    return calls


def test_component_warmup_does_not_capture_and_tolerates_unavailable_asr(h, monkeypatch):
    calls = configure_startup(h, monkeypatch, speech=FileNotFoundError('missing ASR'))
    h.ui._load_model()
    assert calls == ['speech', 'lips', 'lip_encode', 'wording']
    assert set(h.ui.warmup_seconds) == {'speech', 'lips', 'wording'}
    assert not h.ui.speech_ready and h.ui.lip_ready and h.ui._can_begin()
    assert not h.cameras and h.ui.audio.starts == 0


def test_shutdown_prevents_unused_component_warmup(h, monkeypatch):
    calls = configure_startup(h, monkeypatch)
    h.ui.closed = True
    h.ui._load_model()
    assert calls == [] and not h.statuses
    assert h.ui.warmup_seconds == {}


def test_voice_warmup_still_finishes_when_visual_model_cannot_load(h, monkeypatch):
    calls = configure_startup(h, monkeypatch, lips=False)
    h.ui._load_model()
    assert calls == ['speech', 'lips', 'wording']
    assert h.ui.speech_ready and h.ui.wording_ready and not h.ui.lip_ready
    assert h.ui._can_begin() and not h.ui.loading and not h.ui.model_failed
    assert h.ui.result.text == ''


def test_hotkey_error_keeps_manual_buttons_usable(h):
    h.ui._hotkey_error('permission denied')
    assert '输入监控' in h.ui.hotkey_note
    assert h.ui._can_begin()
    h.begin_ready()
    assert h.ui.recording


def test_stale_camera_permission_cannot_clear_a_new_waiting_session(h):
    h.camera_permission = 0
    h.ui.begin_(None)
    old_token = h.ui.token
    h.ui.cancel_(None)
    h.ui.begin_(None)
    assert h.ui.camera_permission_pending
    h.ui._camera_permission(True, old_token)
    assert h.ui.camera_permission_pending
    assert h.ui.pending_begin and not h.ui.recording


def test_microphone_granted_mid_sentence_does_not_claim_it_is_recording(h):
    h.ui.microphone_allowed = False
    h.begin_ready()
    h.ui._microphone_permission(True)
    h.ui.tick_(None)
    assert h.ui.audio.starts == 0 and not h.ui.audio.capturing
    assert '语音＋口型' not in h.ui.state_label.text


def test_no_authorized_camera_or_microphone_does_not_record_empty_input(h):
    h.camera_permission = 2
    h.ui.microphone_allowed = False
    h.ui.begin_(None)
    assert not h.ui.pending_begin and not h.ui.recording
    assert h.ui.audio.starts == 0 and not h.cameras
    assert h.ui.error or '权限' in h.ui.result.text or '不可用' in h.ui.result.text


def test_cancel_clears_permission_wait_before_a_new_authorized_session(h):
    h.camera_permission = 0
    h.ui.begin_(None)
    assert h.ui.camera_permission_pending
    h.ui.cancel_(None)
    assert not h.ui.camera_permission_pending
    h.camera_permission = 3
    h.begin_ready()
    assert not h.ui.pending_begin and h.cameras[0].start_count == 1


def test_cancelled_processing_blocks_overlap_until_worker_returns(h):
    h.begin_ready()
    h.ui.finish_(None)
    _, finish, args = h.later[-1]
    finish(*args)
    old_token = h.ui.token
    h.ui.cancel_(None)
    assert h.ui.processing and not h.ui._can_begin()
    old_result = h.ui.result.text
    # Worker completion for an invalid token must unlock but not replace the UI.
    h.ui._infer_error(old_token, 'private cancelled text')
    assert not h.ui.processing and h.ui.result.text == old_result
    assert h.ui._can_begin()


class NativeWorkerDouble:
    """A native-work lifecycle flag without a thread or native runtime."""
    def __init__(self, alive):
        self.alive = alive

    def is_alive(self):
        return self.alive


def shutdown_app(h, monkeypatch):
    events = []
    def forbidden(*args):
        pytest.fail('Cocoa terminate/stopEventLoop would bypass normal Python teardown')
    application = SimpleNamespace(
        replyToApplicationShouldTerminate_=forbidden, terminate_=forbidden,
        stop_=lambda sender: events.append('stop'),
        postEvent_atStart_=lambda event, at_start: events.append(('wake', event, at_start)))
    h.globals['NSApplication'] = SimpleNamespace(sharedApplication=lambda: application)
    h.globals['AppHelper'].stopEventLoop = forbidden
    monkeypatch.setitem(sys.modules, 'AppKit', SimpleNamespace(
        NSEventTypeApplicationDefined=15,
        NSEvent=SimpleNamespace(
            otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_=
            lambda *args: ('application-defined', args[0]))))
    return application, events


def complete_cleanup(h):
    tasks = [task for task in h.executor.tasks if task['target'] == h.ui._cleanup_models]
    assert len(tasks) == 1, 'shutdown must enqueue exactly one owning-worker cleanup'
    tasks[0]['future'].set_result(None)


def next_shutdown_tick(h):
    delay, callback, args = h.later.pop(0)
    assert delay == pytest.approx(.1)
    callback(*args)


@pytest.mark.parametrize('model,inference,expected', [
    (None, None, False), (False, False, False), (True, None, True),
    (None, True, True), (True, False, True), (False, True, True), (True, True, True),
])
def test_workers_alive_tracks_both_native_workers(h, model, inference, expected):
    h.ui.model_worker = NativeWorkerDouble(model) if model is not None else None
    h.ui.inference_worker = NativeWorkerDouble(inference) if inference is not None else None
    assert h.ui._workers_alive() is expected


def test_workers_alive_handles_not_yet_created_workers(h):
    assert not hasattr(h.ui, 'model_worker') and not hasattr(h.ui, 'inference_worker')
    assert not h.ui._workers_alive()


def test_terminate_idle_closes_devices_then_stops_run_loop_after_cleanup(h, monkeypatch):
    h.begin_ready()
    app, events = shutdown_app(h, monkeypatch)
    token = h.ui.token
    h.later.clear()
    assert h.ui.applicationShouldTerminate_(app) == 0  # Cancel native process exit.
    assert h.ui.closed and not h.ui.recording and h.ui.camera is None
    assert not h.ui.audio.capturing and h.ui.token > token
    assert h.hotkey_stops == 1 and h.timer_invalidations == 1
    assert not events and len(h.later) == 1 and h.statuses[-1] == 'closing'
    next_shutdown_tick(h)
    assert not events and h.later and not h.executor.shutdown_calls
    complete_cleanup(h)
    next_shutdown_tick(h)
    assert events == ['stop', ('wake', ('application-defined', 15), False)]
    assert h.executor.shutdown_calls == [{'wait': True, 'cancel_futures': True}]
    assert h.ui.window.calls[-1] == 'hide'
    assert not h.later and h.statuses[-1] == 'closed'


@pytest.mark.parametrize('worker_name', ['model_worker', 'inference_worker'])
def test_terminate_waits_for_native_worker_and_cleanup_before_stopping_loop(h, monkeypatch, worker_name):
    h.begin_ready()
    app, events = shutdown_app(h, monkeypatch)
    worker = NativeWorkerDouble(True)
    setattr(h.ui, worker_name, worker)
    h.later.clear()
    assert h.ui.applicationShouldTerminate_(app) == 0
    assert h.ui.closed and h.ui.camera is None and not h.ui.audio.capturing
    assert h.hotkey_stops == 1 and not events
    assert len(h.later) == 1
    next_shutdown_tick(h)
    assert not events and len(h.later) == 1
    worker.alive = False
    next_shutdown_tick(h)
    assert not events and len(h.later) == 1  # Native work ended, cleanup still pending.
    complete_cleanup(h)
    next_shutdown_tick(h)
    assert events[0] == 'stop' and events[1][0] == 'wake'
    assert not h.later and h.statuses[-1] == 'closed'


def test_shutdown_waits_until_model_inference_and_cleanup_have_all_exited(h, monkeypatch):
    app, events = shutdown_app(h, monkeypatch)
    h.ui.model_worker = NativeWorkerDouble(True)
    h.ui.inference_worker = NativeWorkerDouble(True)
    assert h.ui.applicationShouldTerminate_(app) == 0
    h.ui.model_worker.alive = False
    next_shutdown_tick(h)
    assert not events and h.later
    h.ui.inference_worker.alive = False
    next_shutdown_tick(h)
    assert not events and h.later
    complete_cleanup(h)
    next_shutdown_tick(h)
    assert events[0] == 'stop' and not h.later


def test_last_window_closing_does_not_bypass_native_worker_shutdown(h):
    h.ui.model_worker = NativeWorkerDouble(True)
    assert h.ui.applicationShouldTerminateAfterLastWindowClosed_(None) is False
    h.ui.model_worker.alive = False
    assert h.ui.applicationShouldTerminateAfterLastWindowClosed_(None) is False
