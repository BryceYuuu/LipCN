"""Global push-to-talk on Wayland by reading the keyboard devices directly.

pynput's Linux backend is the X11 record extension, so it misses keys while a native Wayland
window is focused. /dev/input still sees them. The user needs to be in the input group.
Synthetic devices (ydotool, wtype) are skipped so the paste keystroke is not a second press.
"""
from __future__ import annotations

import threading

# Linux input event codes. Named here so the listener does not import evdev until a device opens.
KEY_ESC = 1
KEY_ENTER = 28
KEY_A = 30
KEY_LEFTCTRL = 29
KEY_LEFTSHIFT = 42
KEY_RIGHTSHIFT = 54
KEY_LEFTALT = 56
KEY_RIGHTCTRL = 97
KEY_RIGHTALT = 100
KEY_LEFTMETA = 125
KEY_RIGHTMETA = 126

_SKIP_NAME = ("ydotool", "ydotoold", "uinput", "virtual keyboard", "wtype", "dotool")
_ROLE = {
    KEY_RIGHTCTRL: "ctrl_r",
    KEY_LEFTCTRL: "ctrl_l",
    KEY_RIGHTALT: "alt_r",
    KEY_LEFTALT: "alt_l",
    KEY_RIGHTSHIFT: "shift_r",
    KEY_LEFTSHIFT: "shift_l",
    KEY_LEFTMETA: "cmd_l",
    KEY_RIGHTMETA: "cmd_r",
    KEY_ESC: "esc",
}


def is_keyboard(name: str, key_codes: set[int]) -> bool:
    """A real keyboard, not the virtual device our own paste keystroke comes from."""
    lowered = (name or "").lower()
    if any(part in lowered for part in _SKIP_NAME):
        return False
    return KEY_A in key_codes and KEY_ENTER in key_codes


def key_role(code: int) -> "str | None":
    return _ROLE.get(code)


def _key_codes(raw) -> set[int]:
    codes: set[int] = set()
    for item in raw or []:
        if isinstance(item, int):
            codes.add(item)
        elif isinstance(item, (list, tuple)) and item and isinstance(item[0], int):
            codes.add(item[0])
    return codes


def open_keyboards() -> list:
    """Keyboard devices we can read. Empty when evdev is missing or /dev/input is closed to us."""
    try:
        import evdev
        from evdev import ecodes
    except ImportError:
        return []
    found = []
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
        except OSError:
            continue
        keys = _key_codes(dev.capabilities().get(ecodes.EV_KEY, []))
        if is_keyboard(dev.name, keys):
            found.append(dev)
        else:
            try:
                dev.close()
            except OSError:
                pass
    return found


def keyboards_readable() -> bool:
    devices = open_keyboards()
    for dev in devices:
        try:
            dev.close()
        except OSError:
            pass
    return bool(devices)


def _pynput_key(code: int):
    from pynput import keyboard
    role = key_role(code)
    if role is None:
        return object()  # any other key: cancels a shortcut, is not Esc
    return getattr(keyboard.Key, role)


class EvdevHook:
    """Background reader. start() is false when no keyboard could be opened."""

    def __init__(self, on_press, on_release):
        self._on_press = on_press
        self._on_release = on_release
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._devices: list = []

    def start(self) -> bool:
        self._devices = open_keyboards()
        if not self._devices:
            return False
        self._thread = threading.Thread(target=self._loop, name="lipflow-keys", daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        for dev in self._devices:
            try:
                dev.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1)

    def _loop(self):
        import select

        from evdev import ecodes

        fds = {dev.fd: dev for dev in self._devices}
        while not self._stop.is_set() and fds:
            try:
                ready, _, _ = select.select(list(fds), [], [], 0.2)
            except (OSError, ValueError):
                break
            for fd in ready:
                dev = fds.get(fd)
                if dev is None:
                    continue
                try:
                    events = dev.read()
                except OSError:
                    fds.pop(fd, None)
                    continue
                for event in events:
                    if event.type != ecodes.EV_KEY or event.value == 2:  # 2 is key repeat
                        continue
                    key = _pynput_key(event.code)
                    if event.value == 1:
                        self._on_press(key)
                    elif event.value == 0:
                        self._on_release(key)
