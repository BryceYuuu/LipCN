"""Insert text at the cursor of whatever app is focused: clipboard + ⌘V, then restore."""
from __future__ import annotations

import threading
import time

import Quartz
from AppKit import NSPasteboard, NSPasteboardTypeString

_V = 9  # kVK_ANSI_V
MARK = 0x11FF10  # kCGEventSourceUserData on every event Lipflow posts, so its own hotkey tap ignores them


def _post(ev, flags=0):
    Quartz.CGEventSetFlags(ev, flags)  # never inherit the held push-to-talk modifier
    Quartz.CGEventSetIntegerValueField(ev, Quartz.kCGEventSourceUserData, MARK)
    Quartz.CGEventPost(Quartz.kCGAnnotatedSessionEventTap, ev)


def _press_cmd_v():
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    for down in (True, False):
        _post(Quartz.CGEventCreateKeyboardEvent(src, _V, down), Quartz.kCGEventFlagMaskCommand)
        time.sleep(0.01)


def paste_text(text: str, restore_after: float = 0.6):
    if not text:
        return
    pb = NSPasteboard.generalPasteboard()
    saved = pb.stringForType_(NSPasteboardTypeString)
    pb.clearContents()
    pb.setString_forType_(text, NSPasteboardTypeString)
    change = pb.changeCount()
    _press_cmd_v()

    def restore():
        time.sleep(restore_after)
        if pb.changeCount() == change and saved is not None:  # don't clobber a newer copy
            pb.clearContents()
            pb.setString_forType_(saved, NSPasteboardTypeString)

    threading.Thread(target=restore, daemon=True).start()


def copy_text(text: str):
    pb = NSPasteboard.generalPasteboard()
    pb.clearContents()
    pb.setString_forType_(text, NSPasteboardTypeString)

