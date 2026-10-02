"""Global push-to-talk key via a pynput keyboard hook. Timing lives in ptt.py.

Windows needs no permission for this. Keystrokes Lipflow injects itself (the Ctrl+V paste) are
skipped, so they can't be mistaken for a shortcut. That injected-key filter is Windows-only.
"""
from __future__ import annotations

import sys

from pynput import keyboard

from ..ptt import PushToTalkState

K = keyboard.Key
KEYS = {
    "right_control": (K.ctrl_r,),
    "right_alt": (K.alt_r, K.alt_gr),  # AltGr on many European layouts
    "left_alt": (K.alt_l,),
    "right_shift": (K.shift_r,),
}
# Like macOS, only a non-modifier key turns a held push-to-talk key into a shortcut. AltGr also sends
# a Left Ctrl of its own, which must not cancel the dictation.
MODIFIERS = {K.ctrl, K.ctrl_l, K.ctrl_r, K.alt, K.alt_l, K.alt_r, K.alt_gr, K.shift, K.shift_l, K.shift_r,
             K.cmd, K.cmd_l, K.cmd_r}
DEFAULT_KEY = "right_control"
LLKHF_INJECTED = 0x10
MASK_VK = 0xE8  # unassigned: tapping it while Alt is held stops Alt's release from opening app menus


class PushToTalk(PushToTalkState):
    def __init__(self, key: str, on_start, on_stop, on_cancel):
        if key not in KEYS:
            raise ValueError(f"unknown key {key!r}; choose from {', '.join(KEYS)}")
        super().__init__(on_start, on_stop, on_cancel)
        self.keys = KEYS[key]
        self._listener = None

    def install(self):
        kwargs = {"on_press": self.press, "on_release": self.release}
        if sys.platform == "win32":
            def filt(msg, data):
                # Runs before on_press/on_release; returning False hides the event from them only.
                return not (data.flags & LLKHF_INJECTED)
            kwargs["win32_event_filter"] = filt
        self._listener = keyboard.Listener(**kwargs)
        self._listener.daemon = True
        self._listener.start()

    def stop(self):
        if self._listener is not None:
            self._listener.stop()

    # Called on the listener thread with pynput Key / KeyCode objects.
    def press(self, key):
        if key in self.keys:
            if not self.down and any(k in (K.alt_l, K.alt_r, K.alt_gr) for k in self.keys):
                self._mask_alt()
            self.key_down()
        elif key not in MODIFIERS:
            self.other_key(key == K.esc)

    @staticmethod
    def _mask_alt():
        kc = keyboard.KeyCode.from_vk(MASK_VK)
        c = keyboard.Controller()
        c.press(kc)
        c.release(kc)

    def release(self, key):
        if key in self.keys:
            self.key_up()
