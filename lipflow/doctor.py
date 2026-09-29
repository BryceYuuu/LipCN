"""`lipflow doctor`: check everything the app needs before you hit the hotkey."""
from __future__ import annotations

import os

from .vsr import MODELS
from .paths import WHO


def doctor() -> int:
    ok = True

    def line(good, what, fix=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'✓' if good else '✗'} {what}" + ("" if good else f"\n      → {fix}"))

    print("Lipflow doctor\n")
    for rel, size in [("vsr/model.pth", 900e6), ("lm/model.pth", 200e6), ("face_landmarker.task", 3e6)]:
        path = os.path.join(MODELS, rel)
        line(os.path.exists(path) and os.path.getsize(path) > size, f"model file {rel}", "run ./setup.sh")

    import torch
    line(True, f"torch {torch.__version__}, encoder on {'mps (Apple GPU)' if torch.backends.mps.is_available() else 'cpu'}")

    import Quartz
    line(Quartz.CGPreflightListenEventAccess(), "Input Monitoring (for the push-to-talk key)",
         f"System Settings → Privacy & Security → Input Monitoring → enable {WHO}, then restart it")
    line(Quartz.CGPreflightPostEventAccess(), "Accessibility (to paste at your cursor)",
         f"System Settings → Privacy & Security → Accessibility → enable {WHO}")

    from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
    status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
    names = {0: "not asked yet (you'll be prompted on first use)", 1: "restricted", 2: "denied", 3: "granted"}
    line(status in (0, 3), f"Camera: {names.get(status, status)}",
         f"System Settings → Privacy & Security → Camera → enable {WHO}")

    from .cleanup import Cleaner
    c = Cleaner()
    line(True, f"cleanup backend: {c.describe()}"
         + ("  (set ANTHROPIC_API_KEY or run Ollama for much better accuracy)" if c.backend == "basic" else ""))
    print("\nAll good." if ok else "\nFix the ✗ items above, then run `lipflow`.")
    return 0 if ok else 1
