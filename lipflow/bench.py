"""Sentence-level benchmark clips from the public-domain test videos + their subtitles."""
from __future__ import annotations

import os
import re

import numpy as np

from .face import FaceTracker, mouth_rois
from .offline import load_clip
from .personal import words_of
from .vsr import LipReader

HERE = os.path.dirname(__file__)
SAMPLES = os.path.join(HERE, "..", "samples")
CACHE = os.path.join(SAMPLES, "bench.npz")


def _t(s):
    h, m, rest = s.strip().split(":")
    sec, ms = (rest.split(",") + ["0"])[:2]
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms.ljust(3, "0")[:3]) / 1000


def sentences(srt_path: str) -> list[tuple[float, float, str]]:
    """Merge subtitle cues into sentences: (start, end, TEXT)."""
    cues = []
    for block in open(srt_path, encoding="utf-8", errors="ignore").read().split("\n\n"):
        lines = [l for l in block.strip().splitlines() if l.strip()]
        if len(lines) >= 3 and "-->" in lines[1]:
            a, b = lines[1].split("-->")
            cues.append((_t(a), _t(b), " ".join(lines[2:])))
    out, cur, start = [], "", None
    for a, b, text in cues:
        text = re.sub(r"^The President:\s*", "", text)
        if start is None:
            start = a
        cur += " " + text
        if re.search(r"[.!?]\s*$", text):
            words = " ".join(w.upper() for w in words_of(cur) if not w.isdigit())
            if 4 <= len(words.split()) <= 30 and b - start < 12:
                out.append((start, b, words))
            cur, start = "", None
    return out


def build(force: bool = False) -> list[dict]:
    """[{video, start, end, text, rois}] cached in samples/bench.npz."""
    if os.path.exists(CACHE) and not force:
        d = np.load(CACHE, allow_pickle=True)
        return list(d["items"])
    items = []
    for name in ("2016-03-12", "2017-01-07"):
        video, srt = os.path.join(SAMPLES, f"{name}.mov"), os.path.join(SAMPLES, f"{name}.srt")
        if not (os.path.exists(video) and os.path.exists(srt)):
            continue
        for a, b, text in sentences(srt):
            tr = FaceTracker()
            ts, grays, anchors = load_clip(video, tr, a, b)
            tr.close()
            if not ts or sum(x is not None for x in anchors) < 0.8 * len(anchors):
                continue
            idx = LipReader.resample(ts, len(ts))
            rois = mouth_rois([grays[i] for i in idx], [anchors[i] for i in idx])
            if rois is not None:
                items.append({"video": name, "start": a, "end": b, "text": text, "rois": rois})
    np.savez_compressed(CACHE, items=np.array(items, dtype=object))
    return items


def wer(hyp: str, ref: str) -> tuple[int, int]:
    h, r = words_of(hyp), words_of(ref)
    d = list(range(len(r) + 1))
    for i, a in enumerate(h, 1):
        prev, d[0] = d[0], i
        for j, b in enumerate(r, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a != b))
    return d[-1], len(r)


def evaluate(reader: LipReader, items: list[dict]) -> float:
    errs = n = 0
    for it in items:
        hyp = reader.beam_search(reader.encode(it["rois"]))
        e, m = wer(hyp, it["text"])
        errs, n = errs + e, n + m
    return errs / max(n, 1)
