"""Webcam capture with live face tracking.

A background thread reads frames. While a recording is active every frame is run
through FaceTracker and kept (grayscale + timestamp + mouth anchors), so when you stop
talking the clip is already aligned and only needs the model.

The camera is opened lazily and released after `idle_close` seconds without a
recording, so the green light isn't on all day.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .face import FaceObs, FaceTracker


@dataclass
class Recording:
    started: float
    ts: list[float] = field(default_factory=list)
    grays: list[np.ndarray] = field(default_factory=list)
    anchors: list["np.ndarray | None"] = field(default_factory=list)
    mouth_open: list[float] = field(default_factory=list)

    def snapshot(self):
        # The capture thread appends to these one after another; take a length all three have.
        n = min(len(self.ts), len(self.grays), len(self.anchors))
        return self.ts[:n], self.grays[:n], self.anchors[:n]

    @property
    def face_ratio(self) -> float:
        return sum(a is not None for a in self.anchors) / max(len(self.anchors), 1)

    @property
    def duration(self) -> float:
        return self.ts[-1] - self.ts[0] if len(self.ts) > 1 else 0.0


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480, idle_close: float = 45.0,
                 on_frame=None):
        self.index, self.width, self.height = index, width, height
        self.idle_close = idle_close
        self.on_frame = on_frame  # (bgr, FaceObs | None, recording: bool) -> None
        self._cap = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._rec: Recording | None = None
        self._last_used = time.time()
        self._tracker: FaceTracker | None = None
        self.error: str | None = None
        self.ready = threading.Event()

    # -- lifecycle -----------------------------------------------------------------
    def ensure_open(self):
        self._last_used = time.time()
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.ready.clear()
        self._thread = threading.Thread(target=self._run, name="lipflow-camera", daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def is_open(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # -- recording -----------------------------------------------------------------
    def start_recording(self) -> Recording:
        self.ensure_open()
        with self._lock:
            self._rec = Recording(started=time.time())
            return self._rec

    def stop_recording(self) -> "Recording | None":
        with self._lock:
            rec, self._rec = self._rec, None
        self._last_used = time.time()
        return rec

    @property
    def recording(self) -> "Recording | None":
        return self._rec

    # -- thread --------------------------------------------------------------------
    def _open(self):
        if isinstance(self.index, str):  # a video file standing in for the webcam (testing / demos)
            cap = cv2.VideoCapture(self.index)
            self._file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if not cap.isOpened():
                raise RuntimeError(f"Could not open {self.index}")
            return cap
        self._file_fps = None
        cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            raise RuntimeError("Could not open the camera. Allow your terminal in "
                               "Settings → Privacy & Security → Camera")
        return cap

    def _run(self):
        try:
            self._cap = self._open()
            if self._tracker is None:
                self._tracker = FaceTracker()
            self.error = None
        except Exception as e:
            self.error = str(e)
            print(f"[camera] {e}")
            return
        t0 = time.time()
        warm = n_read = 0
        try:
            while not self._stop.is_set():
                ok, frame = self._cap.read()
                if self._file_fps:  # play the file back in real time
                    n_read += 1
                    time.sleep(max(0.0, t0 + n_read / self._file_fps - time.time()))
                now = time.time()
                if not ok:
                    time.sleep(0.01)
                    continue
                warm += 1
                if warm == 3:  # first frames are often black while exposure settles
                    self.ready.set()
                rec = self._rec
                obs: FaceObs | None = None
                if rec is not None:
                    obs = self._tracker.detect(frame, int((now - t0) * 1000))
                if rec is not None and warm >= 3:
                    with self._lock:
                        if self._rec is rec:
                            rec.ts.append(now)
                            rec.grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                            rec.anchors.append(obs.anchors if obs else None)
                            rec.mouth_open.append(obs.mouth_open if obs else 0.0)
                if self.on_frame is not None:
                    try:
                        self.on_frame(frame, obs, rec is not None)
                    except Exception as e:  # UI errors must not kill capture
                        print(f"[camera] on_frame: {e}")
                if rec is None and now - self._last_used > self.idle_close:
                    break
        finally:
            self._cap.release()
            self._cap = None
            self.ready.clear()


def mouth_thumbnail(frame_bgr: np.ndarray, obs: "FaceObs | None", size: int = 112) -> "np.ndarray | None":
    """A square, mirrored, colour crop around the lips for the HUD."""
    if obs is None:
        return None
    lips = obs.outer_lips
    cx, cy = lips.mean(0)
    half = max(np.ptp(lips[:, 0]), 1) * 0.95
    h, w = frame_bgr.shape[:2]
    x0, x1 = int(max(cx - half, 0)), int(min(cx + half, w))
    y0, y1 = int(max(cy - half, 0)), int(min(cy + half, h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = cv2.resize(frame_bgr[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
    return cv2.flip(crop, 1)


PINK = (115, 92, 250)      # BGR
PINK_SOFT = (170, 150, 255)


def mouth_view(frame_bgr: np.ndarray, obs: "FaceObs | None", w: int = 240, h: int = 150) -> np.ndarray:
    """Mirrored close-up of the lips with the tracked contour and points drawn on, for the HUD.

    Without a face it shows the whole (dimmed) frame so you can see how to line yourself up.
    """
    fh, fw = frame_bgr.shape[:2]
    if obs is None:
        view = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
        view = cv2.flip((view * 0.45).astype(np.uint8), 1)
        cv2.putText(view, "looking for your face...", (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (235, 235, 235), 1, cv2.LINE_AA)
        return view
    lips = obs.outer_lips
    cx, cy = lips.mean(0)
    half_w = max(np.ptp(lips[:, 0]), 10) * 0.9
    half_h = half_w * h / w
    x0, y0 = cx - half_w, cy - half_h
    scale = w / (2 * half_w)
    M = np.float32([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]])
    view = cv2.warpAffine(frame_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    view = cv2.flip(view, 1)

    def to_view(pts):
        p = (pts - (x0, y0)) * scale
        p[:, 0] = w - 1 - p[:, 0]
        return p

    overlay = view.copy()
    for contour in (obs.outer_lips, obs.inner_lips):
        cv2.polylines(overlay, [np.round(to_view(contour) * 4).astype(np.int32)], True, PINK_SOFT, 1,
                      cv2.LINE_AA, shift=2)
    view = cv2.addWeighted(overlay, 0.7, view, 0.3, 0)
    for x, y in to_view(obs.lip_points):
        cv2.circle(view, (int(round(x * 4)), int(round(y * 4))), 8, PINK, -1, cv2.LINE_AA, shift=2)
    return view
