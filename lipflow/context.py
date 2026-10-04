"""What you're typing into: app, window title, and the text around the cursor.

Read once, locally, through the Accessibility API (on Windows, just the window title) when you
press the push-to-talk key (the app you're dictating into is frontmost at that moment). Only names
and terms are pulled out, to help with the words lip reading gets wrong most often — who you're
writing to, the thread you're in.
Nothing is saved to disk or sent anywhere.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field

_STOP = set("""a an the and or but if of to in on at for with from by as is are was were be been am i
you he she it we they me my your our their this that these those do does did have has had not no so
just can will would could should there here what when where who how why all about up out then than
too very really also like get got go going new re fwd inbox search compose sent drafts home today
monday tuesday wednesday thursday friday saturday sunday january february march april may june july
august september october november december untitled message messages chat thread channel reply""".split())


@dataclass
class Context:
    app: str = ""
    title: str = ""
    near_text: str = ""
    names: list[str] = field(default_factory=list)
    element: object = None   # the focused text field (for learning from corrections); memory only
    target: int = 0          # PID on macOS / HWND on Windows
    value: str = ""          # its full contents before the paste; memory only

    def describe(self) -> str:
        bits = [self.app] + ([f'"{self.title[:60]}"'] if self.title else [])
        return " · ".join(b for b in bits if b)


def _ax(el, attr):
    from ApplicationServices import AXUIElementCopyAttributeValue
    err, val = AXUIElementCopyAttributeValue(el, attr, None)
    return val if err == 0 else None


def extract_names(*texts: str, limit: int = 30) -> list[str]:
    """Capitalised words that aren't sentence starts or common words: names, products, places."""
    seen, out = set(), []
    from .text import HAN
    for t in texts:
        # Only explicit recipient/title markers, never arbitrary runs of Chinese prose.
        patterns = [rf"(?:收件人|发送给|联系人)\s*[:：]\s*([{HAN}]{{2,12}})(?=$|[\s，,；;<>])",
                    rf"([{HAN}]{{2,12}})\s*[（(](?:私聊|DM)[）)]"]
        for pattern in patterns:
            for w in re.findall(pattern, t or ""):
                if w not in seen:
                    seen.add(w)
                    out.append(w)
        for sent in re.split(r"[.!?\n|•·—\-–:]+", t or ""):
            toks = re.findall(r"[A-Za-z][A-Za-z'\-]+", sent)
            for i, w in enumerate(toks):
                if not w[0].isupper() or w.isupper() and len(w) > 4:
                    continue
                if w.lower() in _STOP or len(w) < 3:
                    continue
                if i == 0 and w.lower() in _STOP:
                    continue
                if w.lower() not in seen:
                    seen.add(w.lower())
                    out.append(w)
    return out[:limit]


def _focused_node(node: dict):
    """The focused window in a swaymsg / i3 tree."""
    if not isinstance(node, dict):
        return None
    if node.get("focused"):
        return node
    for key in ("nodes", "floating_nodes"):
        for child in node.get(key) or []:
            found = _focused_node(child)
            if found is not None:
                return found
    return None


def _run_json(args: list[str]):
    import json
    import subprocess
    r = subprocess.run(args, capture_output=True, text=True, timeout=1, check=False)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return json.loads(r.stdout)


def _capture_linux(ctx: Context) -> Context:
    """Active window title only. Hyprland, Sway, KDE, then X11.

    Each tool is skipped when it is missing or fails, so a Hyprland binary on an X11 session
    still falls through to xdotool. GNOME has no stable title API.
    """
    import shutil
    import subprocess

    got = False
    if shutil.which("hyprctl"):
        try:
            data = _run_json(["hyprctl", "activewindow", "-j"])
            if isinstance(data, dict):
                ctx.title = str(data.get("title") or "")
                ctx.app = str(data.get("class") or "")
                got = bool(ctx.title or ctx.app)
        except Exception:
            pass
    if not got and shutil.which("swaymsg"):
        try:
            tree = _run_json(["swaymsg", "-t", "get_tree"])
            node = _focused_node(tree) if isinstance(tree, dict) else None
            if node is not None:
                ctx.title = str(node.get("name") or "")
                props = node.get("window_properties") or {}
                ctx.app = str(node.get("app_id") or props.get("class") or "")
                got = bool(ctx.title or ctx.app)
        except Exception:
            pass
    if not got and shutil.which("kdotool"):
        try:
            r = subprocess.run(
                ["kdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=1, check=False,
            )
            if r.returncode == 0 and r.stdout.strip():
                ctx.title = r.stdout.strip()
                got = True
        except Exception:
            pass
    if not got and shutil.which("xdotool"):
        try:
            r = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=1, check=False,
            )
            if r.returncode == 0:
                ctx.title = r.stdout.strip()
        except Exception:
            pass
    ctx.names = extract_names(ctx.title)
    return ctx


def _capture_windows(ctx: Context) -> Context:
    """Windows: the foreground window's title and program name. Text near the cursor would need
    UI Automation, so names come from the title only (e.g. a chat or document name)."""
    import ctypes
    from ctypes import wintypes
    user32, kernel32 = ctypes.WinDLL("user32"), ctypes.WinDLL("kernel32")
    user32.GetForegroundWindow.restype = wintypes.HWND
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return ctx
    ctx.target = int(hwnd)
    buf = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, buf, 512)
    ctx.title = buf.value
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    kernel32.OpenProcess.restype = wintypes.HANDLE
    proc = kernel32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
    if proc:
        size = wintypes.DWORD(1024)
        path = ctypes.create_unicode_buffer(1024)
        if kernel32.QueryFullProcessImageNameW(proc, 0, path, ctypes.byref(size)):
            ctx.app = os.path.splitext(os.path.basename(path.value))[0]
        kernel32.CloseHandle(proc)
    ctx.names = extract_names(ctx.title)
    return ctx


def capture(max_chars: int = 600) -> Context:
    """Snapshot of the frontmost app. Never raises: context is a bonus, not a requirement."""
    ctx = Context()
    try:
        if sys.platform == "win32":
            return _capture_windows(ctx)
        if sys.platform == "linux":
            return _capture_linux(ctx)
        from AppKit import NSWorkspace
        from ApplicationServices import AXUIElementCreateApplication
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return ctx
        ctx.target = int(app.processIdentifier())
        ctx.app = str(app.localizedName() or "")
        ax_app = AXUIElementCreateApplication(app.processIdentifier())
        win = _ax(ax_app, "AXFocusedWindow")
        if win is not None:
            ctx.title = str(_ax(win, "AXTitle") or "")
        focused = _ax(ax_app, "AXFocusedUIElement")
        if focused is not None:
            val = _ax(focused, "AXValue")
            if isinstance(val, str):
                ctx.element, ctx.value = focused, val
                selection = _ax(focused, "AXSelectedTextRange")
                pos = int(selection.location) if selection is not None and hasattr(selection, "location") else len(val)
                ctx.near_text = val[max(0, pos-max_chars//2):pos+max_chars//2]
            if not ctx.near_text:
                ph = _ax(focused, "AXPlaceholderValue")  # e.g. Slack's "Message Miguel"
                if isinstance(ph, str):
                    ctx.near_text = ph
        ctx.names = extract_names(ctx.title, ctx.near_text)
    except Exception as e:  # permissions, sandboxed apps, odd elements
        print(f"[lipflow] context unavailable: {e.__class__.__name__}")
    return ctx
