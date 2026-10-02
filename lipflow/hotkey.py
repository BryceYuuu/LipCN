"""Global push-to-talk key on macOS via a Quartz event tap (listen-only). Timing lives in ptt.py.

Needs "Input Monitoring" (and for pasting, "Accessibility") permission for whatever app
launched lipflow: Lipflow.app, or your terminal.
"""
from __future__ import annotations

import Quartz
from .paths import WHO
from .ptt import DOUBLE_TAP, TAP_MAX, PushToTalkState  # noqa: F401  (re-exported for tests)

KEYS = {
    # name: (keycode, device-specific modifier mask)
    "right_option": (61, 0x00000040),
    "left_option": (58, 0x00000020),
    "right_command": (54, 0x00000010),
    "right_control": (62, 0x00002000),
    "fn": (63, 0x00800000),
}
ESC = 53


class PushToTalk(PushToTalkState):
    def __init__(self, key: str, on_start, on_stop, on_cancel):
        if key not in KEYS:
            raise ValueError(f"unknown key {key!r}; choose from {', '.join(KEYS)}")
        super().__init__(on_start, on_stop, on_cancel)
        self.keycode, self.mask = KEYS[key]
        self._tap = None

    def install(self):
        mask = Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged) | Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
        self._tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap, Quartz.kCGEventTapOptionListenOnly,
            mask, self._callback, None)
        if self._tap is None:
            raise PermissionError(
                f"Couldn't listen for the hotkey. Allow {WHO} in System Settings → Privacy & Security → "
                "Input Monitoring (and Accessibility), then restart it.")
        src = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetMain(), src, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(self._tap, True)

    def _callback(self, proxy, etype, event, refcon):
        if etype in (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput):
            Quartz.CGEventTapEnable(self._tap, True)
            return event
        if Quartz.CGEventGetIntegerValueField(event, Quartz.kCGEventSourceUserData) == 0x11FF10:
            return event  # our own typing / paste
        code = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
        if etype == Quartz.kCGEventKeyDown:
            # Option+letter is a real shortcut (e.g. typing special characters), not dictation.
            self.other_key(code == ESC)
            return event
        if code != self.keycode:
            return event
        if Quartz.CGEventGetFlags(event) & self.mask:
            self.key_down()
        else:
            self.key_up()
        return event
