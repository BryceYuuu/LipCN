"""Global push-to-talk key via a Quartz event tap (listen-only).

Hold the key to dictate, release to type. Double-tap it to start hands-free mode; tap
again to stop. Esc cancels the current recording.

Needs "Input Monitoring" (and for pasting, "Accessibility") permission for whatever app
launched lipflow — usually your terminal.
"""
from __future__ import annotations

import time

import Quartz

KEYS = {
    # name: (keycode, device-specific modifier mask)
    "right_option": (61, 0x00000040),
    "left_option": (58, 0x00000020),
    "right_command": (54, 0x00000010),
    "right_control": (62, 0x00002000),
    "fn": (63, 0x00800000),
}
ESC = 53
DOUBLE_TAP = 0.35  # seconds between taps
TAP_MAX = 0.25     # a press shorter than this is a tap, not a hold


class PushToTalk:
    def __init__(self, key: str, on_start, on_stop, on_cancel):
        if key not in KEYS:
            raise ValueError(f"unknown key {key!r}; choose from {', '.join(KEYS)}")
        self.keycode, self.mask = KEYS[key]
        self.on_start, self.on_stop, self.on_cancel = on_start, on_stop, on_cancel
        self.down = False
        self.down_at = 0.0
        self.last_tap = 0.0
        self.hands_free = False
        self.active = False
        self._tap = None

    def install(self):
        mask = Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged) | Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
        self._tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap, Quartz.kCGEventTapOptionListenOnly,
            mask, self._callback, None)
        if self._tap is None:
            raise PermissionError(
                "Couldn't listen for the hotkey. Allow your terminal in System Settings → Privacy & Security → "
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
            if code == ESC and self.active:
                self.active = self.hands_free = False
                self.on_cancel()
            elif self.down and not self.hands_free:
                # Option+letter is a real shortcut (e.g. typing special characters) — not dictation.
                self.down = False
                if self.active:
                    self.active = False
                    self.on_cancel()
            return event
        if code != self.keycode:
            return event
        pressed = bool(Quartz.CGEventGetFlags(event) & self.mask)
        now = time.time()
        if pressed and not self.down:
            self.down, self.down_at = True, now
            if self.hands_free:
                return event  # stop happens on release
            if not self.active:
                self.active = True
                self.on_start(hands_free=False)
        elif not pressed and self.down:
            self.down = False
            held = now - self.down_at
            if self.hands_free:
                self.hands_free = self.active = False
                self.on_stop()
            elif held < TAP_MAX:
                if now - self.last_tap < DOUBLE_TAP:
                    self.hands_free = True  # second tap: keep listening until the next tap
                    self.last_tap = 0.0
                    self.on_start(hands_free=True)
                else:
                    self.last_tap = now
                    self.active = False
                    self.on_cancel(silent=True)
            elif self.active:
                self.active = False
                self.on_stop()
        return event
