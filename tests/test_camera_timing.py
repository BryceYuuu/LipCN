"""Capture-clock regressions, with no camera or microphone hardware access."""
from types import SimpleNamespace

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
