"""Push-to-talk timing, shared by the macOS (Quartz) and Windows (pynput) key listeners.

Hold the key to dictate, release to type. Double-tap it to start hands-free mode; tap
again to stop. Esc cancels the current recording. Any other key pressed while the
push-to-talk key is held makes it a shortcut (Option+letter, Ctrl+C…), so that cancels too.
"""
from __future__ import annotations

import time

DOUBLE_TAP = 0.35  # seconds between taps
TAP_MAX = 0.25     # a press shorter than this is a tap, not a hold


class PushToTalkState:
    def __init__(self, on_start, on_stop, on_cancel):
        self.on_start, self.on_stop, self.on_cancel = on_start, on_stop, on_cancel
        self.down = False
        self.down_at = 0.0
        self.last_tap = 0.0
        self.hands_free = False
        self.active = False

    def other_key(self, is_esc: bool):
        """A key other than the push-to-talk key went down."""
        if is_esc and self.active:
            self.active = self.hands_free = False
            self.on_cancel()
        elif self.down and not self.hands_free:
            self.down = False
            if self.active:
                self.active = False
                self.on_cancel()

    def key_down(self, now: "float | None" = None):
        if self.down:
            return  # key repeat
        self.down, self.down_at = True, time.time() if now is None else now
        if self.hands_free:
            return  # stop happens on release
        if not self.active:
            self.active = True
            self.on_start(hands_free=False)

    def key_up(self, now: "float | None" = None):
        if not self.down:
            return
        now = time.time() if now is None else now
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
