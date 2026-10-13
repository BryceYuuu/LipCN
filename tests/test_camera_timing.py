"""Capture-clock regressions, with no camera or microphone hardware access."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from lipflow import camera, mic
from lipflow.vsr import LipReader


class SimulatedClock:
    def __init__(self):
        self.now = 10.0
        self.wall_calls = 0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def time(self):
        self.wall_calls += 1
        # Emulate an OS clock correction in the middle of capture.
        return 1_720_000_000.0 + (self.now if self.now < 10.12 else -3600.0)

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeCapture:
    def __init__(self, clock, *, playback):
        self.clock, self.playback = clock, playback
        self.read_count = 0
        self.released = False

    def read(self):
        self.read_count += 1
        if not self.playback:
            self.clock.now += 1 / 25
        return True, np.full((8, 8, 3), self.read_count, np.uint8)

    def release(self):
        self.released = True


@pytest.mark.parametrize('failure', ['tracker', 'read', 'detect'])
def test_camera_failure_releases_capture_and_publishes_retryable_error(monkeypatch, failure):
    clock = SimulatedClock()
    cap = FakeCapture(clock, playback=False)
    cam = camera.Camera(index=0)
    cam.track_always = True
    cam._file_fps = None
    cam.ready.set()
    monkeypatch.setattr(cam, '_open', lambda: cap)
    def fail(*args):
        raise RuntimeError('simulated component failure')
    monkeypatch.setattr(camera, 'FaceTracker', fail if failure == 'tracker' else
                        lambda: SimpleNamespace(detect=fail if failure == 'detect' else lambda *args: None))
    if failure == 'read':
        cap.read = fail
    cam._run()
    assert cap.released and cam._cap is None and not cam.ready.is_set()
    assert cam.error == 'simulated component failure'
    # A subsequent start can acquire a fresh capture rather than retain a leaked handle.
    replacement = FakeCapture(clock, playback=False)
    cam._tracker = SimpleNamespace(detect=lambda *args: None)
    monkeypatch.setattr(cam, '_open', lambda: replacement)
    cam.on_frame = lambda *args: cam._stop.set()
    cam._run()
    assert replacement.released and cam.error is None


@pytest.mark.parametrize("playback", [False, True])
def test_camera_and_microphone_share_monotonic_clock_during_wall_clock_step(monkeypatch, playback):
    clock = SimulatedClock()
    monkeypatch.setattr(camera, "time", clock)
    monkeypatch.setattr(mic, "time", clock)
    cap = FakeCapture(clock, playback=playback)
    detector_timestamps = []
    tracker = SimpleNamespace(detect=lambda frame, timestamp: detector_timestamps.append(timestamp))
    cam = camera.Camera(index=0)
    microphone = mic.Mic(device=0)
    microphone._stream = SimpleNamespace(latency=0.0)
    cam._rec = camera.Recording(started=clock.monotonic())
    recording = cam._rec
    cam._tracker = tracker
    cam._file_fps = 25 if playback else None
    monkeypatch.setattr(cam, "_open", lambda: cap)

    def frame_received(frame, obs, recording_active):
        assert recording_active
        microphone._cb(np.ones((640, 1), np.float32), 640, None, None)
        if cap.read_count == 5:
            cam._stop.set()

    cam.on_frame = frame_received
    cam._run()

    assert cam.error is None and cap.released
    assert clock.wall_calls == 0
    assert recording.ts == pytest.approx([10.12, 10.16, 10.20])
    assert recording.duration == pytest.approx(0.08)
    assert detector_timestamps == sorted(detector_timestamps)
    assert LipReader.resample(recording.ts, len(recording.ts)) == [0, 1, 2]
    assert [chunk[0] for chunk in microphone._chunks] == pytest.approx([10.0 + i / 25 for i in range(5)])
    audio = mic.segment(microphone._chunks, recording.ts[0] + mic.CAMERA_LATENCY, 2)
    np.testing.assert_array_equal(audio, np.ones(1280, np.float32))
    if playback:
        assert len(clock.sleeps) == 5
        assert clock.sleeps == pytest.approx([1 / 25] * 5)


def test_camera_recording_lifecycle_uses_monotonic_clock(monkeypatch):
    clock = SimulatedClock()
    monkeypatch.setattr(camera, "time", clock)
    cam = camera.Camera(index=0)
    cam._thread = SimpleNamespace(is_alive=lambda: True)
    clock.now = 12.5
    rec = cam.start_recording()
    assert cam._last_used == rec.started == 12.5
    clock.now = 13.75
    assert cam.stop_recording() is rec
    assert cam._last_used == 13.75
    assert clock.wall_calls == 0


def test_preview_tracking_is_throttled_but_each_new_recorded_frame_is_detected(monkeypatch):
    clock = SimulatedClock()
    monkeypatch.setattr(camera, 'time', clock)
    cap = FakeCapture(clock, playback=False)
    cam = camera.Camera(index=0)
    cam._file_fps = None
    cam._thread = SimpleNamespace(is_alive=lambda: True)
    cam.track_always = True
    cam.preview_tracking_interval = 1 / 12
    detected, delivered, recordings = [], [], []
    closed = []

    def detect(frame, timestamp):
        detected.append(cap.read_count)
        return SimpleNamespace(
            detected_at=cap.read_count, anchors=np.zeros((4, 2)), mouth_open=.1,
            outer_lips=np.array([[1., 1.], [4., 1.]]))

    cam._tracker = SimpleNamespace(detect=detect, close=lambda: closed.append('tracker'))
    monkeypatch.setattr(cam, '_open', lambda: cap)
    monkeypatch.setattr(camera, 'face_crop', lambda gray, obs, rec: gray)
    first = cam.start_recording()
    recordings.append(first)

    def received(frame, obs, recording_active):
        delivered.append((cap.read_count, obs.detected_at, recording_active))
        if cap.read_count == 6:
            assert cam.stop_recording() is first
        elif cap.read_count == 18:
            recordings.append(cam.start_recording())
        elif cap.read_count == 24:
            assert cam.stop_recording() is recordings[1]
        elif cap.read_count == 28:
            cam._stop.set()

    cam.on_frame = received
    cam._run()
    assert cam.error is None and cap.released
    assert detected[:6] == list(range(1, 7))
    assert [n for n in detected if 19 <= n <= 24] == list(range(19, 25))
    preview_detections = [n for n in detected if 7 <= n <= 18]
    assert 1 <= len(preview_detections) < 12
    assert len(delivered) == 28  # Tracking skips do not drop live preview frames.
    assert any(frame != tracked for frame, tracked, active in delivered if not active)
    assert all(frame == tracked for frame, tracked, active in delivered if active)
    assert len(first.ts) == len(first.grays) == len(first.anchors) == 4  # Warm-up removes two frames.
    second = recordings[1]
    assert len(second.ts) == len(second.grays) == len(second.anchors) == 6
    assert len(first.mouth_open) == len(first.mouth_pixels) == 4
    assert len(second.mouth_open) == len(second.mouth_pixels) == 6
    assert cam.recording is None and cam._tracker is None and closed == ['tracker']


def test_nonblocking_close_leaves_inflight_detection_to_release_its_own_resources(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    ownership, joins = [], []

    class BlockingTracker:
        def detect(self, frame, timestamp):
            entered.set()
            if not release.wait(5):
                raise AssertionError('test did not release simulated detection')
            return None

        def close(self):
            ownership.append(('tracker', threading.get_ident()))

    class Capture:
        def read(self):
            return True, np.zeros((8, 8, 3), np.uint8)

        def release(self):
            ownership.append(('capture', threading.get_ident()))

    cam = camera.Camera(index=0)
    cam.track_always = True
    cam._file_fps = None
    monkeypatch.setattr(cam, '_open', Capture)
    monkeypatch.setattr(camera, 'FaceTracker', BlockingTracker)
    cam.ensure_open()
    join = cam._thread.join
    try:
        assert entered.wait(2)
        worker_id = cam._thread.ident
        monkeypatch.setattr(cam._thread, 'join', lambda *args, **kwargs: joins.append((args, kwargs)))
        cam.close(wait=False)
        assert not joins  # The UI request must not wait even for the old two-second timeout.
        assert cam.is_open and cam._stop.is_set()
        assert not ownership  # Never close the tracker while detect() is executing.
    finally:
        release.set()
        join(timeout=2)
    assert not cam.is_open and cam._cap is None and cam._tracker is None
    assert sorted(ownership) == [('capture', worker_id), ('tracker', worker_id)]
    assert not cam.ready.is_set()


def test_idle_camera_closes_after_elapsed_time_even_if_wall_clock_steps(monkeypatch):
    clock = SimulatedClock()
    monkeypatch.setattr(camera, "time", clock)
    cap = FakeCapture(clock, playback=False)
    cam = camera.Camera(index=0, idle_close=0.1)
    cam._tracker = SimpleNamespace(detect=lambda *args: pytest.fail("idle capture must not track"))
    cam._file_fps = None
    monkeypatch.setattr(cam, "_open", lambda: cap)
    cam._run()
    assert cap.read_count == 3 and cap.released
    assert clock.wall_calls == 0


class ScriptedCapture:
    def __init__(self, clock, steps):
        self.clock, self.steps = clock, iter(steps)
        self.read_count, self.released = 0, False

    def read(self):
        try:
            elapsed, ok, frame = next(self.steps)
        except StopIteration:
            pytest.fail('Capture should have stopped before another read')
        self.clock.now += elapsed
        self.read_count += 1
        return ok, frame

    def release(self):
        self.released = True


def scripted_camera(monkeypatch, clock, steps, *, playback=False):
    monkeypatch.setattr(camera, 'time', clock)
    cap = ScriptedCapture(clock, steps)
    cam = camera.Camera(index=0)
    cam.track_always = True
    cam._file_fps = 25 if playback else None
    closed = []
    cam._tracker = SimpleNamespace(detect=lambda *args: None, close=lambda: closed.append('tracker'))
    monkeypatch.setattr(cam, '_open', lambda: cap)
    return cam, cap, closed


@pytest.mark.parametrize('ok,frame', [(False, None), (True, None), (True, np.zeros((0, 0, 3), np.uint8))])
def test_initial_invalid_camera_reads_stop_after_finite_elapsed_time(monkeypatch, ok, frame):
    clock = SimulatedClock()
    cam, cap, closed = scripted_camera(monkeypatch, clock, [(1., ok, frame)] * 3)
    cam.ready.set()  # Even stale readiness must be cleared on failure.
    cam.on_frame = lambda *args: pytest.fail('Invalid images must not enter preview or recording')
    cam._run()
    assert cap.read_count == 3 and cap.released
    assert 'no valid frames for 3 seconds' in cam.error
    assert cam._cap is None and cam._tracker is None and not cam.ready.is_set()
    assert closed == ['tracker'] and clock.wall_calls == 0


def test_valid_camera_read_resets_timeout_and_preserves_recorded_frames(monkeypatch):
    clock = SimulatedClock()
    frame = np.zeros((8, 8, 3), np.uint8)
    good = (.04, True, frame)
    steps = [good] * 3 + [(1.5, False, None), good, (1.5, False, None), good,
                          (1.6, False, None), (1.6, False, None)]
    cam, cap, closed = scripted_camera(monkeypatch, clock, steps)
    cam._rec = camera.Recording(started=clock.monotonic())
    rec = cam.recording
    delivered = []
    cam.on_frame = lambda *args: delivered.append(cap.read_count)
    cam._run()
    assert delivered == [1, 2, 3, 5, 7]
    assert rec.ts == pytest.approx([10.12, 11.67, 13.22])
    assert len(rec.grays) == len(rec.anchors) == len(rec.mouth_open) == len(rec.mouth_pixels) == 3
    assert cap.read_count == 9 and cap.released and closed == ['tracker']
    assert 'no valid frames' in cam.error and not cam.ready.is_set()


def test_file_eof_stops_immediately_without_camera_retry_error(monkeypatch):
    clock = SimulatedClock()
    frame = np.zeros((8, 8, 3), np.uint8)
    cam, cap, closed = scripted_camera(monkeypatch, clock,
        [(0., True, frame)] * 3 + [(0., False, None)], playback=True)
    delivered = []
    cam.on_frame = lambda *args: delivered.append(cap.read_count)
    cam._run()
    assert delivered == [1, 2, 3] and cap.read_count == 4
    assert cam.error is None and cap.released and closed == ['tracker']
    assert not cam.ready.is_set() and clock.now < 10.2


def test_reported_recognition_duration_uses_elapsed_time(monkeypatch):
    from lipflow import vsr

    clock = SimulatedClock()
    monkeypatch.setattr(vsr, "time", clock)
    reader = LipReader.__new__(LipReader)

    def encode(rois):
        clock.now += 0.25
        return None

    reader.encode = encode
    reader.greedy = lambda encoded: "TEST"
    text, seconds = reader.read(np.zeros((25, 96, 96), np.uint8), fast=True)
    assert text == "TEST" and seconds == 0.25
    assert clock.wall_calls == 0


def test_reported_audiovisual_recognition_duration_uses_elapsed_time(monkeypatch):
    from lipflow import av

    clock = SimulatedClock()
    monkeypatch.setattr(av, "time", clock)
    reader = av.AVReader.__new__(av.AVReader)

    def encode_av(rois, wave):
        clock.now += 0.25
        return None

    reader.encode_av = encode_av
    reader.beam_search = lambda encoded: "TEST"
    text, seconds = reader.read_av(np.zeros((25, 96, 96), np.uint8), np.ones(16000, np.float32))
    assert text == "TEST" and seconds == 0.25
    assert clock.wall_calls == 0
