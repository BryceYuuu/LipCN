"""lipflow [run] | lipflow file VIDEO | lipflow doctor"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("GLOG_minloglevel", "2")  # quiet MediaPipe
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# OpenCV would ask for camera access from its capture thread, which silently fails; the app
# asks on the main thread instead (app.request_camera).
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")


def main(argv=None):
    from .hotkey import KEYS

    p = argparse.ArgumentParser(prog="lipflow", description="Silent dictation by lip reading.")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="start the menu-bar dictation app (default)")
    r.add_argument("--key", default="right_option", choices=list(KEYS), help="push-to-talk key")
    r.add_argument("--beam", type=int, default=10, help="beam size (higher = slower, slightly better)")
    r.add_argument("--cleanup", default="auto", choices=["auto", "claude", "ollama", "basic"])
    r.add_argument("--camera", default="0", help="camera index, or a video file to use instead")
    r.add_argument("--copy-only", action="store_true", help="copy to the clipboard instead of pasting")
    r.add_argument("--no-preview", action="store_true", help="don't show live words while you talk")

    f = sub.add_parser("file", help="lip-read a video file")
    f.add_argument("video")
    f.add_argument("--start", type=float, default=0.0)
    f.add_argument("--end", type=float, default=None)
    f.add_argument("--beam", type=int, default=10)
    f.add_argument("--cleanup", default="auto", choices=["auto", "claude", "ollama", "basic", "none"])

    sub.add_parser("doctor", help="check permissions, camera and model files")

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"run", "file", "doctor", "-h", "--help"}:
        argv.insert(0, "run")
    args = p.parse_args(argv)
    cmd = args.cmd

    if cmd == "file":
        from .offline import transcribe_file
        from .vsr import LipReader
        raw = transcribe_file(args.video, LipReader(beam_size=args.beam), args.start, args.end)
        print("raw:  ", raw)
        if args.cleanup != "none":
            from .cleanup import Cleaner
            c = Cleaner(args.cleanup)
            print(f"text:  {c([raw])}   [{c.describe()}]")
    elif cmd == "doctor":
        from .doctor import doctor
        sys.exit(doctor())
    else:
        from .app import Options, run
        run(Options(key=args.key, beam=args.beam, backend=args.cleanup, camera=int(args.camera) if args.camera.isdigit() else args.camera,
                    paste=not args.copy_only, live_preview=not args.no_preview))


if __name__ == "__main__":
    main()
