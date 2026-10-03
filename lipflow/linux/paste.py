"""Insert text at the cursor on Linux: clipboard + Ctrl+V.

Wayland copies with wl-copy. The keystroke is wtype on wlroots (Hyprland, Sway), or ydotool /
dotool where the compositor refuses the virtual-keyboard protocol (GNOME, KDE). X11 uses xclip
and xdotool.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time


def wayland_session() -> bool:
    """True when this session is Wayland, including when only WAYLAND_DISPLAY is set."""
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"


def get_text() -> str | None:
    if wayland_session() and shutil.which("wl-paste"):
        r = subprocess.run(["wl-paste", "-n"], capture_output=True, text=True, timeout=2)
        return r.stdout if r.returncode == 0 else None
    if shutil.which("xclip"):
        r = subprocess.run(["xclip", "-selection", "clipboard", "-o"], capture_output=True, text=True, timeout=2)
        return r.stdout if r.returncode == 0 else None
    return None


def set_text(text: str) -> None:
    if wayland_session() and shutil.which("wl-copy"):
        subprocess.run(["wl-copy"], input=text, text=True, check=True)  # stdin: text may start with "-"
        return
    if shutil.which("xclip"):
        subprocess.run(["xclip", "-selection", "clipboard"], input=text, text=True, check=True)
        return
    raise OSError("install wl-clipboard (Wayland) or xclip (X11)")


def _run(args: list[str], **kwargs) -> bool:
    if shutil.which(args[0]) is None:
        return False
    try:
        subprocess.run(args, check=True, timeout=2, capture_output=True, text=True, **kwargs)
        return True
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def _press_ctrl_v() -> None:
    if wayland_session():
        if _run(["wtype", "-M", "ctrl", "-k", "v"]):
            return
        # KEY_LEFTCTRL=29, KEY_V=47. ydotool injects through the kernel, so GNOME and KDE accept it.
        if _run(["ydotool", "key", "29:1", "47:1", "47:0", "29:0"]):
            return
        if _run(["dotool"], input="key ctrl+v\n"):
            return
    if _run(["xdotool", "key", "ctrl+v"]):
        return
    raise OSError(
        "install wtype (Hyprland, Sway) or ydotool with ydotoold running (GNOME, KDE, other Wayland), "
        "or xdotool on X11. Or run with --copy-only"
    )


def paste_text(text: str, restore_after: float = 0.6) -> None:
    if not text:
        return
    try:
        saved = get_text()
    except (OSError, subprocess.TimeoutExpired):
        saved = None
    set_text(text)
    _press_ctrl_v()

    def restore():
        time.sleep(restore_after)
        if saved is not None:
            try:
                set_text(saved)
            except OSError:
                pass

    threading.Thread(target=restore, daemon=True).start()


def copy_text(text: str) -> None:
    set_text(text)
