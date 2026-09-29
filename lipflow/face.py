"""Face tracking and mouth-ROI extraction.

The VSR model was trained on 96x96 grayscale mouth crops taken from faces that were
first aligned to a mean face (eyes, nose base, mouth centre). We reproduce that exact
preprocessing, but compute the four anchor points live with MediaPipe FaceLandmarker
so a recording is ready for inference the moment you stop talking.
"""
from __future__ import annotations

import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

HERE = os.path.dirname(__file__)
DEFAULT_MODEL = os.path.join(HERE, "..", "models", "face_landmarker.task")

# Subject's right eye is on the image left in an unmirrored camera frame, matching
# dlib points 36-41 of the reference face.
_RIGHT_EYE = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]
_LEFT_EYE = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]
_NOSE_BASE = [97, 98, 2, 326, 327]  # ~ dlib 31-35
_LIPS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181,
         185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402,
         405, 409, 415]
_OUTER_LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84,
               181, 91, 146]
_UPPER_INNER, _LOWER_INNER = 13, 14


def _stable_reference(size=256):
    ref = np.load(os.path.join(HERE, "mean_face.npy"))
    pts = np.vstack([ref[36:42].mean(0), ref[42:48].mean(0), ref[31:36].mean(0), ref[48:68].mean(0)])
    return pts - (256 - size) / 2.0


STABLE_REFERENCE = _stable_reference()


class FaceTracker:
    """Per-frame landmarks. Use VIDEO mode so MediaPipe tracks between frames."""

    def __init__(self, model_path: str = DEFAULT_MODEL):
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.abspath(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, ts_ms: int) -> "FaceObs | None":
        ts_ms = max(int(ts_ms), self._last_ts + 1)  # MediaPipe needs strictly increasing time
        self._last_ts = ts_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
        if not res.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        pts = np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]], dtype=np.float32)
        return FaceObs(pts)

    def close(self):
        self._lm.close()


class FaceObs:
    __slots__ = ("pts",)

    def __init__(self, pts: np.ndarray):
        self.pts = pts

    @property
    def anchors(self) -> np.ndarray:
        """4x2: right eye, left eye, nose base, mouth centre (image coords)."""
        p = self.pts
        return np.vstack([p[_RIGHT_EYE].mean(0), p[_LEFT_EYE].mean(0), p[_NOSE_BASE].mean(0), p[_LIPS].mean(0)])

    @property
    def mouth_open(self) -> float:
        """Inner-lip gap normalised by mouth width — a cheap 'is the mouth moving' signal."""
        p = self.pts
        width = np.linalg.norm(p[61] - p[291]) + 1e-6
        return float(np.linalg.norm(p[_UPPER_INNER] - p[_LOWER_INNER]) / width)

    @property
    def outer_lips(self) -> np.ndarray:
        return self.pts[_OUTER_LIPS]


def _interpolate(anchors: list["np.ndarray | None"]) -> "list[np.ndarray] | None":
    valid = [i for i, a in enumerate(anchors) if a is not None]
    if not valid:
        return None
    out = list(anchors)
    for a, b in zip(valid, valid[1:]):
        for k in range(1, b - a):
            out[a + k] = out[a] + (out[b] - out[a]) * (k / (b - a))
    for i in range(valid[0]):
        out[i] = out[valid[0]]
    for i in range(valid[-1] + 1, len(out)):
        out[i] = out[valid[-1]]
    return out


def mouth_rois(gray_frames: list[np.ndarray], anchors: list["np.ndarray | None"],
               crop: int = 96, window_margin: int = 12) -> "np.ndarray | None":
    """Align each frame to the mean face and cut a crop x crop patch around the mouth.

    Mirrors Auto-AVSR's VideoProcess: temporally smoothed landmarks -> similarity
    transform onto the reference -> fixed-size patch centred on the mouth.
    """
    lms = _interpolate(anchors)
    if lms is None:
        return None
    n = len(lms)
    half = crop // 2
    patches = []
    for i, frame in enumerate(gray_frames):
        m = min(window_margin // 2, i, n - 1 - i)
        smoothed = np.mean(lms[i - m:i + m + 1], axis=0)
        smoothed += lms[i].mean(axis=0) - smoothed.mean(axis=0)
        tf, _ = cv2.estimateAffinePartial2D(smoothed.astype(np.float32), STABLE_REFERENCE.astype(np.float32),
                                            method=cv2.LMEDS)
        if tf is None:
            tf = cv2.estimateAffinePartial2D(lms[i].astype(np.float32), STABLE_REFERENCE.astype(np.float32))[0]
        warped = cv2.warpAffine(frame, tf, (256, 256), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mouth = smoothed[3] @ tf[:, :2].T + tf[:, 2]
        cx = int(round(np.clip(mouth[0], half, 256 - half)))
        cy = int(round(np.clip(mouth[1], half, 256 - half)))
        patches.append(warped[cy - half:cy + half, cx - half:cx + half])
    return np.stack(patches)
