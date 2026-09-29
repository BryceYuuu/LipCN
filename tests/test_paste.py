"""Opt-in: this really presses ⌘V in whatever app is focused. Focus an empty text field, then
LIPFLOW_TEST_PASTE=1 uv run pytest tests/test_paste.py"""
import os

import pytest
import Quartz
from AppKit import NSPasteboard, NSPasteboardTypeString

import lipflow.paste as P

pytestmark = pytest.mark.skipif(os.environ.get("LIPFLOW_TEST_PASTE") != "1",
                                reason="presses ⌘V for real; set LIPFLOW_TEST_PASTE=1")


def test_paste_posts_cmd_v_and_restores_clipboard(monkeypatch):
    seen = []

    def cb(proxy, etype, event, refcon):
        code = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
        seen.append((code, bool(Quartz.CGEventGetFlags(event) & Quartz.kCGEventFlagMaskCommand)))
        return event

    tap = Quartz.CGEventTapCreate(Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
                                  Quartz.kCGEventTapOptionListenOnly,
                                  Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown), cb, None)
    assert tap is not None, "needs Input Monitoring"
    Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), Quartz.CFMachPortCreateRunLoopSource(None, tap, 0),
                              Quartz.kCFRunLoopCommonModes)
    Quartz.CGEventTapEnable(tap, True)

    pb = NSPasteboard.generalPasteboard()
    before = pb.stringForType_(NSPasteboardTypeString)
    on_clipboard = []
    real = P._press_cmd_v
    monkeypatch.setattr(P, "_press_cmd_v", lambda: (on_clipboard.append(pb.stringForType_(NSPasteboardTypeString)), real()))

    P.paste_text("lipflow test", restore_after=0.2)
    Quartz.CFRunLoopRunInMode(Quartz.kCFRunLoopDefaultMode, 0.6, False)

    assert on_clipboard == ["lipflow test"]
    assert (9, True) in seen  # V with Command
    assert pb.stringForType_(NSPasteboardTypeString) == before
