"""App-lifetime preview with simulated devices, permissions and queued workers."""
from types import SimpleNamespace

import pytest

from test_hybrid_ui import complete_cleanup, h


def finish_recording(h):
    h.begin_ready()
    h.ui.finish_(None)
    _, finish, args = h.later[-1]
    finish(*args)
    h.globals['_nsimage'] = lambda pixels: pixels
    return h.cameras[0]


def enable_frame_callback(h):
    """Exercise the real callback/queue logic; replace only pixel conversion."""
    h.ui._camera_frame = type(h.ui)._camera_frame.__get__(h.ui)
    h.globals['cv2'] = SimpleNamespace(
        INTER_AREA=0, resize=lambda frame, *args, **kwargs: frame,
        flip=lambda frame, *args: frame)
    h.globals['mouth_view'] = lambda frame, *args: frame + ' mouth'
    h.globals['_nsimage'] = lambda pixels: pixels


def open_idle_preview(h):
    h.ui.input_mode = 'lip-only'
    h.ui.lip_ready = True
    h.ui.preview_active = True
    enable_frame_callback(h)
    h.ui._preview_tick()
    assert h.ui.camera is h.cameras[0] and h.ui.camera.is_open
    return h.ui.camera


def deliver_frame(h, camera, text):
    h.now += .05
    camera.on_frame(text, None, camera.recording is not None)
    h.flush_after()


def test_launch_opens_preview_before_model_ready_without_recording_or_microphone(h):
    h.ui.input_mode = 'lip-only'
    h.ui.loading = True
    h.ui.lip_ready = False
    h.microphone_permission = 0
    h.globals['NSTimer'] = SimpleNamespace(
        scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_=
        lambda *args: SimpleNamespace(invalidate=h.invalidate_timer))
    h.globals['CommandToggle'] = lambda *args, **kwargs: SimpleNamespace(
        start=lambda: True, stop=h.stop_hotkey)
    h.globals['ThreadPoolExecutor'] = lambda **kwargs: h.executor
    enable_frame_callback(h)
    h.ui.launch()
    camera = h.ui.camera
    assert h.ui.preview_active and camera.is_open and camera.track_always
    assert camera.start_count == 0 and camera.recording is None
    assert h.ui.audio.starts == 0 and not h.permission_requests
    assert not h.ui._can_begin()
    assert len(h.executor.tasks) == 1
    assert h.executor.tasks[0]['target'] == h.ui._load_model
    deliver_frame(h, camera, 'preview while loading')
    assert h.ui.video.images[-1] == 'preview while loading'


def test_idle_preview_is_continuous_and_does_not_start_a_sentence(h):
    camera = open_idle_preview(h)
    token = h.ui.camera_token
    for number in range(4):
        h.now += 10
        h.ui.tick_(None)
        deliver_frame(h, camera, f'idle {number}')
    assert h.ui.video.images == [f'idle {number}' for number in range(4)]
    assert len(h.cameras) == 1 and h.ui.camera_token == token
    assert camera.preview_tracking_interval == pytest.approx(1 / 12)
    assert camera.recording is None and camera.start_count == camera.close_count == 0
    assert not h.ui.recording and not h.ui.processing and h.ui.audio.starts == 0


def test_preview_continues_while_worker_waits_without_extending_recording(h):
    camera = finish_recording(h)
    rec = h.executor.tasks[0]['args'][1]
    for frame in ('frame one', 'frame two'):
        h.ui._draw_frame(frame, frame + ' mouth', h.ui.camera_token)
    assert h.ui.video.images[-2:] == ['frame one', 'frame two']
    assert camera.recording is None and camera.is_open
    assert rec.ts == [] and not h.ui.audio.capturing
    assert not h.ui.inference_worker.done()
    h.ui.tick_(None)
    assert h.ui.progress.animating
    assert '实时预览' in h.ui.state_label.text


def test_processing_frames_update_preview_without_modifying_the_stopped_clip(h):
    h.begin_ready()
    camera = h.ui.camera
    rec = camera.recording
    rec.ts = [rec.started + n / 25 for n in range(3)]
    rec.grays = ['recorded 1', 'recorded 2', 'recorded 3']
    rec.anchors = [None] * 3
    rec.mouth_open = [.1] * 3
    rec.mouth_pixels = [40.] * 3
    before = {name: list(getattr(rec, name)) for name in
              ('ts', 'grays', 'anchors', 'mouth_open', 'mouth_pixels')}
    h.ui.finish_(None)
    _, finish, args = h.later[-1]
    finish(*args)
    enable_frame_callback(h)
    for frame in ('live 1', 'live 2', 'live 3'):
        deliver_frame(h, camera, frame)
    assert h.ui.video.images[-3:] == ['live 1', 'live 2', 'live 3']
    assert camera.recording is None and camera.is_open and h.ui.processing
    assert camera.preview_tracking_interval == pytest.approx(1 / 12)
    assert not h.ui.audio.capturing
    assert {name: getattr(rec, name) for name in before} == before
    assert h.executor.tasks[0]['args'][1] is rec


def test_same_camera_callback_survives_two_sentences_and_idle_results(h):
    camera = open_idle_preview(h)
    camera_token = h.ui.camera_token
    records = []
    for number in range(2):
        h.begin_ready()
        records.append(camera.recording)
        assert camera.preview_tracking_interval == 0
        deliver_frame(h, camera, f'recording {number}')
        h.ui.finish_(None)
        _, finish, args = h.later[-1]
        finish(*args)
        result = SimpleNamespace(variants=[SimpleNamespace(text=f'结果 {number}')])
        h.ui.inference_worker.set_result(None)
        h.ui._inferred(h.ui.token, result, 2, 1, 25, {'route': 'lips'})
        deliver_frame(h, camera, f'result {number}')
        assert h.ui.result_variants == [f'结果 {number}']
        assert h.ui._can_begin() and not h.ui.recording and camera.recording is None
    assert h.ui.video.images == ['recording 0', 'result 0', 'recording 1', 'result 1']
    assert records[0] is not records[1]
    assert camera.start_count == 2 and camera.close_count == 0 and len(h.cameras) == 1
    assert h.ui.camera_token == camera_token and h.ui.token == 2
    assert h.ui.audio.starts == 0


def test_language_token_changes_do_not_freeze_existing_camera_callback(h):
    camera = open_idle_preview(h)
    camera_token = h.ui.camera_token
    h.ui.language_popup.selectItemAtIndex_(1)
    h.ui.changeLanguage_(h.ui.language_popup)
    assert h.ui.language == 'zh-Hant' and h.ui.token == 1
    assert not h.executor.tasks  # Simplified/traditional share the loaded reader.
    deliver_frame(h, camera, 'traditional preview')
    h.ui.language_popup.selectItemAtIndex_(2)
    h.ui.changeLanguage_(h.ui.language_popup)
    assert h.ui.language == 'en' and h.ui.language_switching and h.ui.token == 2
    assert len(h.executor.tasks) == 1  # Do not execute a real model switch.
    deliver_frame(h, camera, 'preview during model switch')
    assert h.ui.video.images == ['traditional preview', 'preview during model switch']
    assert h.ui.camera_token == camera_token and h.ui.camera is camera
    assert len(h.cameras) == 1 and camera.close_count == 0 and h.ui.audio.starts == 0


def test_stale_frame_cannot_clear_a_new_camera_generations_pending_slot(h):
    old_camera = open_idle_preview(h)
    old_token = h.ui.camera_token
    h.ui._close_camera()
    h.ui._preview_tick()
    current_camera = h.ui.camera
    current_token = h.ui.camera_token
    current_camera.on_frame('new frame', None, False)
    assert h.ui.ui_frame_pending and h.ui.ui_frame_token == current_token
    queued = len(h.after)
    h.ui._draw_frame('old frame', 'old mouth', old_token)
    old_camera.on_frame('old callback', None, False)
    assert h.ui.ui_frame_pending and h.ui.ui_frame_token == current_token
    assert len(h.after) == queued
    h.now += .1
    current_camera.on_frame('extra new frame', None, False)
    assert len(h.after) == queued  # The one-frame queue bound remains effective.
    h.flush_after()
    assert h.ui.video.images[-1] == 'new frame'
    assert not h.ui.ui_frame_pending and h.ui.ui_frame_token is None


def test_early_and_final_results_keep_live_preview(h):
    camera = finish_recording(h)
    enable_frame_callback(h)
    result = SimpleNamespace(variants=[SimpleNamespace(text='识别结果。')])
    h.ui._first_inferred(h.ui.token, result, 1, {'route': 'lips'})
    assert camera.is_open and h.ui.processing
    h.ui._inferred(h.ui.token, result, 2, 1, 25, {'route': 'lips'})
    deliver_frame(h, camera, 'after final result')
    assert camera.is_open and h.ui.camera is camera and camera.close_count == 0
    assert h.ui.video.images[-1] == 'after final result'
    assert h.ui.result_variants == ['识别结果。']
    h.ui.tick_(None)
    assert not h.ui.progress.animating


@pytest.mark.parametrize('exit_path', ['cancel', 'error'])
def test_cancel_or_error_ends_sentence_without_closing_preview(h, exit_path):
    camera = finish_recording(h)
    enable_frame_callback(h)
    token = h.ui.token
    if exit_path == 'cancel':
        h.ui.cancel_(None)
        h.ui._discarded(token)
    else:
        h.ui._infer_error(token, '识别失败')
    deliver_frame(h, camera, 'still live')
    assert h.ui.camera is camera and camera.is_open and camera.close_count == 0
    assert h.ui.video.images[-1] == 'still live'
    assert camera.recording is None and not h.ui.audio.capturing and not h.ui.processing


@pytest.mark.parametrize('suspend_callback,resume_callback', [
    ('windowDidMiniaturize_', 'windowDidDeminiaturize_'),
    ('applicationDidHide_', 'applicationDidUnhide_'),
])
def test_hidden_window_keeps_camera_but_stops_sentence_and_rendering(h, suspend_callback, resume_callback):
    camera = open_idle_preview(h)
    h.begin_ready()
    deliver_frame(h, camera, 'visible frame')
    h.now += .05
    camera.on_frame('queued before hiding', None, True)
    assert h.ui.ui_frame_pending
    getattr(h.ui, suspend_callback)(None)
    h.flush_after()
    deliver_frame(h, camera, 'hidden frame')
    assert h.ui.video.images == ['visible frame'] and not h.ui.ui_frame_pending
    assert h.ui.suspended and camera.is_open and camera.close_count == 0
    assert camera.recording is None and not h.ui.recording and not h.ui.audio.capturing
    assert camera.preview_tracking_interval == 1.0
    h.ui.tick_(None)
    assert len(h.cameras) == 1
    getattr(h.ui, resume_callback)(None)
    deliver_frame(h, camera, 'visible again')
    assert not h.ui.suspended and h.ui.video.images[-1] == 'visible again'
    assert camera.preview_tracking_interval == pytest.approx(1 / 12)
    assert camera.start_count == 1 and camera.recording is None


def test_cancelled_result_cannot_replace_new_sentence_or_stop_its_preview(h):
    camera = finish_recording(h)
    enable_frame_callback(h)
    old_token = h.ui.token
    camera_token = h.ui.camera_token
    h.ui.cancel_(None)
    h.ui._discarded(old_token)
    h.begin_ready()
    deliver_frame(h, camera, 'new sentence')
    result = SimpleNamespace(variants=[SimpleNamespace(text='cancelled result')])
    h.ui._inferred(old_token, result, 2, 1, 25, {'route': 'lips'})
    h.ui._infer_error(old_token, 'old error')
    assert h.ui.video.images[-1] == 'new sentence' and not h.ui.result_variants
    assert h.ui.camera_token == camera_token
    assert h.ui.camera is camera and camera.is_open and h.ui.recording


def test_failed_worker_submission_keeps_preview(h):
    h.begin_ready()
    camera = h.ui.camera
    h.ui.model_executor = None
    h.ui.finish_(None)
    _, finish, args = h.later[-1]
    finish(*args)
    finish(*args)
    assert camera.is_open and h.ui.camera is camera and not h.ui.processing
    assert camera.recording is None and camera.close_count == 0
    assert not h.ui.audio.capturing


def test_cancelled_worker_does_not_look_like_active_recognition_but_preview_is_live(h):
    camera = finish_recording(h)
    h.ui.cancel_(None)
    h.ui.tick_(None)
    assert h.ui.processing  # Native work is still draining in the background.
    assert '正在取消' in h.ui.state_label.text and '识别中' not in h.ui.state_label.text
    assert '实时预览' in h.ui.state_label.text and camera.is_open
    assert not h.ui.cancel_button.enabled


@pytest.mark.parametrize('suspended', [False, True])
def test_permission_callback_follows_camera_lifetime_not_utterance_or_visibility(h, suspended):
    h.ui.input_mode = 'lip-only'
    h.ui.lip_ready = True
    h.ui.preview_active = True
    h.camera_permission = 0
    enable_frame_callback(h)
    h.ui._preview_tick()
    token = h.ui.camera_token
    h.ui.begin_(None)
    h.ui.cancel_(None)
    h.ui.language_popup.selectItemAtIndex_(1)
    h.ui.changeLanguage_(h.ui.language_popup)
    if suspended:
        h.ui._suspend()
    assert h.ui.token != token and h.ui.camera_token == token
    h.grant_camera()
    camera = h.ui.camera
    assert camera.is_open and camera.start_count == 0 and not h.ui.recording
    assert len(h.permission_requests) == 1 and h.ui.audio.starts == 0
    assert camera.preview_tracking_interval == (1.0 if suspended else pytest.approx(1 / 12))
    deliver_frame(h, camera, 'authorized preview')
    assert h.ui.video.images == ([] if suspended else ['authorized preview'])


def test_device_failure_cancels_clip_and_retries_without_resuming_recording(h):
    camera = open_idle_preview(h)
    h.begin_ready()
    old_token = h.ui.token
    camera.error = 'device disconnected'
    h.ui._preview_tick()
    assert not h.ui.recording and h.ui.token > old_token
    assert camera.close_count == 1 and h.ui.camera is None
    assert '重连' in h.ui.preview_error
    h.now += 1.9
    h.ui._preview_tick()
    assert len(h.cameras) == 1
    h.now += .2
    h.ui._preview_tick()
    replacement = h.ui.camera
    assert replacement is not camera and replacement.is_open
    assert replacement.start_count == 0 and not h.ui.recording and h.ui.audio.starts == 0
    deliver_frame(h, replacement, 'reconnected preview')
    assert h.ui.video.images[-1] == 'reconnected preview'


def test_camera_reconnect_does_not_erase_editable_result(h):
    camera = open_idle_preview(h)
    h.ui.result.text = '用户修改过的方案'
    h.ui.result_variants = ['识别方案']
    camera.error = 'device disconnected'
    h.ui._preview_tick()
    h.now += 2.1
    h.ui._preview_tick()
    assert h.ui.result.text == '用户修改过的方案' and h.ui.result_variants == ['识别方案']


def test_shutdown_closes_camera_and_rejects_pending_and_future_preview_frames(h):
    camera = open_idle_preview(h)
    deliver_frame(h, camera, 'visible frame')
    h.now += .05
    camera.on_frame('queued before shutdown', None, False)
    h.ui._shutdown()
    h.flush_after()
    deliver_frame(h, camera, 'frame after shutdown')
    h.ui.tick_(None)
    h.ui._preview_tick()
    assert h.ui.closed and not h.ui.preview_active and h.ui.camera is None
    assert not camera.is_open and camera.close_count == 1 and len(h.cameras) == 1
    assert h.ui.video.images[-1] is None and h.ui.mouth.images[-1] is None
    assert not h.ui.ui_frame_pending and h.ui.ui_frame_token is None


def test_inflight_preview_close_is_nonblocking_and_keeps_shutdown_waiting(h):
    preview = open_idle_preview(h)
    waits = []
    preview.close = lambda *, wait=True: waits.append(wait)
    # Simulate a capture thread still inside detection after the stop request.
    preview._tracker = SimpleNamespace(close=lambda: pytest.fail('UI closed an in-use tracker'))
    h.ui.ui_frame_pending = True
    h.ui.ui_frame_token = h.ui.camera_token
    h.ui._shutdown()
    assert waits == [False]
    assert h.ui.camera is None and preview.is_open
    assert h.ui.video.images[-1] is None and h.ui.mouth.images[-1] is None
    assert not h.ui.ui_frame_pending and h.ui.ui_frame_token is None
    complete_cleanup(h)
    assert h.ui._workers_alive()  # Camera cleanup prevents final process exit.
    assert not h.ui._can_begin() and h.ui.closing_cameras == [preview]
    preview.is_open = False
    assert not h.ui._workers_alive() and h.ui.closing_cameras == []
    assert not h.ui._can_begin()  # Shutdown cannot restart the app's camera.
