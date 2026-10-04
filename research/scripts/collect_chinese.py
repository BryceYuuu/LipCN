"""Manually collect independently labelled silent webcam clips for research.

Run only when the operator is ready to record:
    python research/scripts/collect_chinese.py --output samples/private/train \
        --speaker speaker01 --session day01 --split train
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lipflow_research_path import REPOSITORY_ROOT


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Manually record labelled silent webcam clips locally.")
    parser.add_argument("--output", required=True, help="local dataset directory")
    parser.add_argument("--speaker", required=True, help="consistent speaker ID, e.g. speaker01")
    parser.add_argument("--session", required=True, help="real recording session ID, e.g. day01")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="train")
    parser.add_argument("--sentences", help="UTF-8 file, one independently authored reference per line")
    parser.add_argument("--camera", default="auto", help="physical webcam number or Mac camera name")
    parser.add_argument("--max-seconds", type=float, default=15.0)
    parser.add_argument("--max-megabytes", type=float, default=512.0)
    args = parser.parse_args(argv)

    # These settings precede the optional camera imports. Parsing --help neither
    # loads collection dependencies nor requests permissions or opens devices.
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if stream is not None and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
    from research.collection import collect

    camera = int(args.camera) if args.camera.isdigit() else args.camera
    try:
        manifest = collect(args.output, args.speaker, args.session, args.split,
                           prompts=args.sentences, camera=camera,
                           max_seconds=args.max_seconds, max_megabytes=args.max_megabytes)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    if manifest.exists():
        print("Local dataset manifest:", manifest)
    else:
        print("No confirmed clips saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
