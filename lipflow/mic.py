"""Microphone for whisper mode: on only while you hold the key, timestamped on the same clock as
the camera frames, so the audio can be cut to exactly the recorded video.

Timing matters: on test clips, audio 0.4 s off the video doubled the word error rate.
"""
from __future__ import annotations

import threading
import time

import numpy as np

RATE = 16000
CAMERA_LATENCY = 0.06  # a frame is read ~2 frames after it was exposed


class Mic:
    def __init__(self, device=None):
        self.device = device if device is not None else self._builtin()
        self._stream = None
        self._chunks: list[tuple[float, np.ndarray]] = []
        self._lock = threading.Lock()
        self.error: str | None = None

    @staticmethod
    def _builtin():
        """The Mac's own microphone, not a Continuity iPhone (same trap as the camera)."""
        try:
            import sounddevice as sd
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0 and "iphone" not in d["name"].lower():
                    if "macbook" in d["name"].lower() or "built-in" in d["name"].lower():
                        return i
        except Exception:
            pass
        return None  # system default

    def start(self):
        import sounddevice as sd
        with self._lock:
            self._chunks = []
        if self._stream is not None:
            return
        try:
            self._stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32", device=self.device,
                                          blocksize=0, callback=self._cb)
            self._stream.start()
            self.error = None
        except Exception as e:
            self.error = str(e)
            self._stream = None
            print(f"[lipflow] microphone unavailable: {e}")

    def _cb(self, data, frames, t, status):
        # First sample in this block, on the camera's monotonic clock.
        latency = self._stream.latency if self._stream is not None else 0.0
        t0 = time.monotonic() - frames / RATE - float(latency or 0.0)
        with self._lock:
            self._chunks.append((t0, data[:, 0].copy()))

    def stop(self) -> list[tuple[float, np.ndarray]]:
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        with self._lock:
            chunks, self._chunks = self._chunks, []
        return chunks


def segment(chunks: list[tuple[float, np.ndarray]], t_start: float, n_frames: int, fps: int = 25) -> "np.ndarray | None":
    """Audio for n_frames video frames starting at t_start (video clock), 640 samples per frame."""
    if not chunks:
        return None
    t_start -= CAMERA_LATENCY
    want = n_frames * RATE // fps
    out = np.zeros(want, np.float32)
    got = 0
    for t0, a in chunks:
        off = int(round((t0 - t_start) * RATE))  # where this chunk lands in the output
        lo, hi = max(off, 0), min(off + len(a), want)
        if hi > lo:
            out[lo:hi] = a[lo - off:hi - off]
            got += hi - lo
    return out if got > want * 0.5 else None
