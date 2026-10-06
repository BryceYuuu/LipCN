"""Linux paths, paste tool choice, and window-title fallback.
Paste and context tests mock subprocess; nothing here presses keys.
Runs on Linux only (CI: .github/workflows/linux.yml)."""
import importlib
import json
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux only")


def test_data_lives_in_xdg(monkeypatch):
    import lipflow.paths as paths
    original_home = paths.HOME
    try:
        with monkeypatch.context() as env:
            env.delenv("LIPFLOW_HOME", raising=False)
            env.setenv("XDG_DATA_HOME", "/tmp/lipflow-xdg")
            assert importlib.reload(paths).HOME == "/tmp/lipflow-xdg/Lipflow"
    finally:
        importlib.reload(paths)
    assert paths.HOME == original_home


def test_xdg_default_is_local_share(monkeypatch):
    import lipflow.paths as paths
    original_home = paths.HOME
    try:
        with monkeypatch.context() as env:
            env.delenv("LIPFLOW_HOME", raising=False)
            env.delenv("XDG_DATA_HOME", raising=False)
            env.setenv("HOME", "/home/lipflow")
            assert importlib.reload(paths).HOME == "/home/lipflow/.local/share/Lipflow"
    finally:
        importlib.reload(paths)
    assert paths.HOME == original_home


def _stub_tools(monkeypatch, paste, present):
    calls = []

    def which(name):
        return f"/usr/bin/{name}" if name in present else None

    def run(args, **kwargs):
        calls.append((list(args), kwargs))
        return subprocess.CompletedProcess(args, 0, "saved", "")

    monkeypatch.setattr(paste.shutil, "which", which)
    monkeypatch.setattr(paste.subprocess, "run", run)
    return calls


def test_wayland_clipboard_and_paste_use_wl_tools(monkeypatch):
    from lipflow.linux import paste
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    calls = _stub_tools(monkeypatch, paste, {"wl-copy", "wl-paste", "wtype", "xclip", "xdotool"})
    paste.set_text("-5 degrees")
    paste._press_ctrl_v()
    assert calls[0][0] == ["wl-copy"] and calls[0][1]["input"] == "-5 degrees"
    assert calls[1][0] == ["wtype", "-M", "ctrl", "-k", "v"]


def test_wayland_display_without_session_type(monkeypatch):
    from lipflow.linux import paste
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    calls = _stub_tools(monkeypatch, paste, {"wl-copy", "wtype"})
    assert paste.wayland_session()
    paste._press_ctrl_v()
    assert calls[0][0] == ["wtype", "-M", "ctrl", "-k", "v"]


def test_wayland_without_wtype_uses_ydotool(monkeypatch):
    from lipflow.linux import paste
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    calls = _stub_tools(monkeypatch, paste, {"ydotool"})
    paste._press_ctrl_v()
    assert calls[0][0] == ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"]


def test_wtype_failure_falls_through_to_ydotool(monkeypatch):
    from lipflow.linux import paste
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    calls = []

    def which(name):
        return f"/usr/bin/{name}" if name in ("wtype", "ydotool") else None

    def run(args, **kwargs):
        calls.append(list(args))
        if args[0] == "wtype":
            raise subprocess.CalledProcessError(1, args)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(paste.shutil, "which", which)
    monkeypatch.setattr(paste.subprocess, "run", run)
    paste._press_ctrl_v()
    assert [c[0] for c in calls] == ["wtype", "ydotool"]


def test_x11_uses_xclip_and_xdotool(monkeypatch):
    from lipflow.linux import paste
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    calls = _stub_tools(monkeypatch, paste, {"xclip", "xdotool", "wl-copy", "wtype"})
    paste.set_text("hi")
    paste._press_ctrl_v()
    assert calls[0][0][0] == "xclip"
    assert calls[0][1]["input"] == "hi"
    assert calls[1][0] == ["xdotool", "key", "ctrl+v"]


def test_hyprctl_success_skips_xdotool(monkeypatch):
    from lipflow.context import Context, _capture_linux
    calls = []

    def which(name):
        return f"/usr/bin/{name}" if name in ("hyprctl", "xdotool") else None

    def run(args, **kwargs):
        calls.append(list(args))
        payload = json.dumps({"title": "Alice", "class": "Alacritty"})
        return subprocess.CompletedProcess(args, 0, payload, "")

    monkeypatch.setattr(shutil, "which", which)
    monkeypatch.setattr(subprocess, "run", run)
    ctx = _capture_linux(Context())
    assert calls == [["hyprctl", "activewindow", "-j"]]
    assert ctx.title == "Alice" and ctx.app == "Alacritty"


def test_hyprctl_failure_falls_through_to_xdotool(monkeypatch):
    from lipflow.context import Context, _capture_linux
    calls = []

    def which(name):
        return f"/usr/bin/{name}" if name in ("hyprctl", "xdotool") else None

    def run(args, **kwargs):
        calls.append(list(args))
        if args[0] == "hyprctl":
            return subprocess.CompletedProcess(args, 1, "", "")
        return subprocess.CompletedProcess(args, 0, "Notes\n", "")

    monkeypatch.setattr(shutil, "which", which)
    monkeypatch.setattr(subprocess, "run", run)
    ctx = _capture_linux(Context())
    assert [c[0] for c in calls] == ["hyprctl", "xdotool"]
    assert ctx.title == "Notes"


def test_skips_virtual_paste_device():
    from lipflow.linux.hotkey import KEY_A, KEY_ENTER, KEY_RIGHTCTRL, is_keyboard, key_role
    keys = {KEY_A, KEY_ENTER, KEY_RIGHTCTRL}
    assert is_keyboard("AT Translated Set 2 keyboard", keys)
    assert not is_keyboard("ydotool virtual device", keys)
    assert not is_keyboard("python-uinput", keys)
    assert key_role(KEY_RIGHTCTRL) == "ctrl_r"
    assert key_role(1) == "esc"


def test_sway_title_when_hyprctl_missing(monkeypatch):
    from lipflow.context import Context, _capture_linux
    tree = {"nodes": [{"name": "Alacritty", "app_id": "Alacritty", "focused": True}]}

    def which(name):
        return "/usr/bin/swaymsg" if name == "swaymsg" else None

    def run(args, **kwargs):
        assert args[0] == "swaymsg"
        return subprocess.CompletedProcess(args, 0, json.dumps(tree), "")

    monkeypatch.setattr(shutil, "which", which)
    monkeypatch.setattr(subprocess, "run", run)
    ctx = _capture_linux(Context())
    assert ctx.title == "Alacritty" and ctx.app == "Alacritty"
