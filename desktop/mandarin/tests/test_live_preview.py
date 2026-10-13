"""Preview lifecycle while inference is delayed, without opening real devices."""
from types import SimpleNamespace

import pytest

from test_hybrid_ui import h


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
    h.ui.ui_frame_pending = False
    h.ui.ui_frame_token = None
    h.ui.last_ui_frame = 0.0
    h.globals['cv2'] = SimpleNamespace(
        INTER_AREA=0, resize=lambda frame, *args, **kwargs: frame,
        flip=lambda frame, *args: frame)
    h.globals['mouth_view'] = lambda frame, *args: frame + ' mouth'
    h.globals['_nsimage'] = lambda pixels: pixels


def test_preview_continues_while_worker_waits_without_extending_recording(h):
    camera = finish_recording(h)
    rec = h.executor.tasks[0]['args'][1]
    for frame in ('frame one', 'frame two'):
        h.ui._draw_frame(frame, frame + ' mouth', h.ui.token)
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
        h.now += .05
        h.ui._camera_frame(frame, None, False, h.ui.token)
        callback, args = h.after.pop()
        callback(*args)
    assert h.ui.video.images[-3:] == ['live 1', 'live 2', 'live 3']
    assert camera.recording is None and camera.is_open and h.ui.processing
    assert camera.preview_tracking_interval == pytest.approx(1 / 12)
    assert not h.ui.audio.capturing
    assert {name: getattr(rec, name) for name in before} == before
    assert h.executor.tasks[0]['args'][1] is rec


def test_stale_frame_cannot_clear_a_new_generations_pending_slot(h):
    finish_recording(h)
    old_token = h.ui.token
    h.ui.cancel_(None)
    h.ui._discarded(old_token)
    h.begin_ready()
    enable_frame_callback(h)
    current_token = h.ui.token
    h.ui._camera_frame('new frame', None, True, current_token)
    assert h.ui.ui_frame_pending and h.ui.ui_frame_token == current_token
    queued = len(h.after)
    h.ui._draw_frame('old frame', 'old mouth', old_token)
    h.ui._camera_frame('old callback', None, False, old_token)
    assert h.ui.ui_frame_pending and h.ui.ui_frame_token == current_token
    assert len(h.after) == queued
    h.now += .1
    h.ui._camera_frame('extra new frame', None, True, current_token)
    assert len(h.after) == queued  # The one-frame queue bound is still effective.
    callback, args = h.after.pop()
    callback(*args)
    assert h.ui.video.images[-1] == 'new frame'
    assert not h.ui.ui_frame_pending and h.ui.ui_frame_token is None


def test_early_result_keeps_preview_until_final_result(h):
    camera = finish_recording(h)
    result = SimpleNamespace(variants=[SimpleNamespace(text='识别结果。')])
    h.ui._first_inferred(h.ui.token, result, 1, {'route': 'lips'})
    assert camera.is_open and h.ui.processing
    h.ui._inferred(h.ui.token, result, 2, 1, 25, {'route': 'lips'})
    assert not camera.is_open and h.ui.camera is None
    assert h.ui.video.images[-1] is None and h.ui.mouth.images[-1] is None
    assert h.ui.result_variants == ['识别结果。']
    h.ui.tick_(None)
    assert not h.ui.progress.animating


@pytest.mark.parametrize('exit_path', ['cancel', 'error', 'suspend', 'shutdown'])
def test_preview_released_on_each_exit_and_queued_frame_cannot_reappear(h, exit_path):
    camera = finish_recording(h)
    token = h.ui.token
    if exit_path == 'cancel':
        h.ui.cancel_(None)
    elif exit_path == 'error':
        h.ui._infer_error(token, '识别失败')
    elif exit_path == 'suspend':
        h.ui._suspend()
    else:
        h.ui._shutdown()
    assert h.ui.camera is None and not camera.is_open
    h.ui._draw_frame('late frame', 'late mouth', token)
    assert h.ui.video.images[-1] is None and h.ui.mouth.images[-1] is None


def test_old_frame_or_result_cannot_replace_or_close_new_preview(h):
    finish_recording(h)
    old_token = h.ui.token
    h.ui.cancel_(None)
    h.ui._discarded(old_token)
    h.begin_ready()
    camera = h.ui.camera
    h.ui._draw_frame('current frame', 'current mouth', h.ui.token)
    h.ui._draw_frame('old frame', 'old mouth', old_token)
    h.ui._infer_error(old_token, 'old error')
    assert h.ui.video.images[-1] == 'current frame'
    assert h.ui.camera is camera and camera.is_open and h.ui.recording


def test_failed_worker_submission_releases_preview(h):
    h.begin_ready()
    camera = h.ui.camera
    h.ui.model_executor = None
    h.ui.finish_(None)
    _, finish, args = h.later[-1]
    finish(*args)
    assert not camera.is_open and h.ui.camera is None and not h.ui.processing


def test_inflight_preview_close_is_nonblocking_and_remains_visible_to_shutdown(h):
    preview = finish_recording(h)
    token = h.ui.token
    waits = []
    preview.close = lambda *, wait=True: waits.append(wait)
    # Simulate a real capture thread still inside detection after the stop request.
    preview._tracker = SimpleNamespace(close=lambda: pytest.fail('UI closed an in-use tracker'))
    h.ui.ui_frame_pending = True
    h.ui.ui_frame_token = h.ui.token
    h.ui.cancel_(None)
    assert waits == [False]
    assert h.ui.camera is None and preview.is_open
    assert h.ui.video.images[-1] is None and h.ui.mouth.images[-1] is None
    assert not h.ui.ui_frame_pending and h.ui.ui_frame_token is None
    h.executor.tasks[0]['future'].set_result(None)
    h.ui._discarded(token)
    assert h.ui._workers_alive()  # Camera cleanup still prevents final process exit.
    assert not h.ui._can_begin()  # A new camera cannot race the old device's teardown.
    assert h.ui.closing_cameras == [preview]
    preview.is_open = False
    assert not h.ui._workers_alive()
    assert h.ui.closing_cameras == []
    assert h.ui._can_begin()
