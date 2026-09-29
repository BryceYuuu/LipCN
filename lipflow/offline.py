"""Transcribe a video file — the same pipeline the live app uses, for testing."""
from __future__ import annotations

import cv2

from .face import FaceTracker, mouth_rois
from .vsr import LipReader


def load_clip(path: str, tracker: FaceTracker, start: float = 0.0, end: float | None = None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if start:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    ts, grays, anchors = [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = start + i / fps
        if end is not None and t > end:
            break
        obs = tracker.detect(frame, int(t * 1000))
        ts.append(t)
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        anchors.append(obs.anchors if obs else None)
        i += 1
    cap.release()
    return ts, grays, anchors


def transcribe_file(path: str, reader: LipReader | None = None, start: float = 0.0,
                    end: float | None = None, save_rois: str | None = None) -> str:
    tracker = FaceTracker()
    ts, grays, anchors = load_clip(path, tracker, start, end)
    tracker.close()
    idx = LipReader.resample(ts, len(ts))
    rois = mouth_rois([grays[i] for i in idx], [anchors[i] for i in idx])
    if rois is None:
        raise RuntimeError("no face found in the video")
    if save_rois:
        h, w = rois.shape[1:]
        out = cv2.VideoWriter(save_rois, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h), False)
        for r in rois:
            out.write(r)
        out.release()
    reader = reader or LipReader()
    text, secs = reader.read(rois)
    found = sum(a is not None for a in anchors)
    print(f"[{len(rois)} frames @25fps, face in {found}/{len(anchors)}, decode {secs:.2f}s]")
    return text
