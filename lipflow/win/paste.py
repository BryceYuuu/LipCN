"""Insert text at the cursor on Windows: clipboard + Ctrl+V, then put the old clipboard back."""
from __future__ import annotations

import ctypes
import threading
import time
from ctypes import wintypes

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
user32.OpenClipboard.argtypes = [wintypes.HWND]
user32.GetClipboardData.restype = wintypes.HANDLE
user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
user32.SetClipboardData.restype = wintypes.HANDLE
user32.GetClipboardSequenceNumber.restype = wintypes.DWORD
kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalLock.restype = ctypes.c_void_p
kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]


class _Clipboard:
    def __enter__(self):
        for _ in range(20):  # another app may hold it for a moment
            if user32.OpenClipboard(None):
                return self
            time.sleep(0.02)
        raise OSError("the clipboard is busy")

    def __exit__(self, *exc):
        user32.CloseClipboard()


def get_text() -> "str | None":
    with _Clipboard():
        h = user32.GetClipboardData(CF_UNICODETEXT)
        if not h:
            return None
        p = kernel32.GlobalLock(h)
        try:
            return ctypes.wstring_at(p) if p else None
        finally:
            kernel32.GlobalUnlock(h)


def set_text(text: str):
    data = ctypes.create_unicode_buffer(text)
    size = ctypes.sizeof(data)
    h = kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
    if not h:
        raise MemoryError("GlobalAlloc failed")
    p = kernel32.GlobalLock(h)
    ctypes.memmove(p, data, size)
    kernel32.GlobalUnlock(h)
    with _Clipboard():
        user32.EmptyClipboard()
        if not user32.SetClipboardData(CF_UNICODETEXT, h):
            kernel32.GlobalFree(h)  # only freed by us if the clipboard didn't take it
            raise OSError("SetClipboardData failed")


def _press_ctrl_v():
    from pynput.keyboard import Controller, Key, KeyCode
    v = KeyCode.from_vk(0x56)  # the V key itself: on a Russian or Greek layout there's no "v" character
    kb = Controller()
    with kb.pressed(Key.ctrl):
        kb.press(v)
        kb.release(v)


def paste_text(text: str, restore_after: float = 0.6):
    if not text:
        return
    try:
        saved = get_text()
    except OSError:
        saved = None
    set_text(text)
    change = user32.GetClipboardSequenceNumber()
    _press_ctrl_v()

    def restore():
        time.sleep(restore_after)
        if user32.GetClipboardSequenceNumber() == change and saved is not None:  # don't clobber a newer copy
            try:
                set_text(saved)
            except OSError:
                pass

    threading.Thread(target=restore, daemon=True).start()


def copy_text(text: str):
    set_text(text)
