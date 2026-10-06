"""Single-tap right Command for the isolated Mandarin test window.

The upstream application uses hold-to-talk / double-tap (lipflow.hotkey and ptt).
This optional listener changes only this window's gesture: tap and release the
RIGHT Command key to toggle recording. It never suppresses keyboard or mouse
input, never requests permission, and does not install anything on import.
Call start()/stop() with the window's lifecycle. All application callbacks are
posted to the AppKit main loop. A failed start leaves normal UI buttons usable.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable

RIGHT_COMMAND = 54
LEFT_COMMAND = 55
ESCAPE = 53
RIGHT_COMMAND_MASK = 0x00000010
LEFT_COMMAND_MASK = 0x00000008
# Shift, Control, Option, Fn. Caps Lock is a setting, not a shortcut chord.
OTHER_MODIFIERS = (1 << 17) | (1 << 18) | (1 << 19) | (1 << 23)
CHORD_MODIFIER_KEYS = {LEFT_COMMAND, 56, 58, 59, 60, 61, 62, 63}
SYNTHETIC_LIPFLOW_EVENT = 0x11FF10
MIN_TAP_SECONDS = 0.025
MAX_TAP_SECONDS = 0.5
PERMISSION_MESSAGE = (
    "右 Command 快捷键未启用：请在“系统设置 → 隐私与安全性 → 输入监控”中允许 "
    "LipCN（若从终端启动，则允许该终端），然后重新打开应用。"
    "现在仍可使用窗口中的开始、结束按钮。"
)


class CommandGesture:
    """Pure gesture state; event names/flags mirror a listen-only Quartz tap.

    A toggle is emitted only after a clean short press/release. Any other key,
    modifier, mouse button or scroll invalidates that press, even if released
    before Command. Keeping held keys/buttons also catches reverse-order chords.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.right_down = False
        self.down_at = 0.0
        self.blocked = False
        self.held_keys: set[int] = set()
        self.held_buttons: set[int] = set()

    def feed(
        self, kind: str, *, keycode: int = 0, flags: int = 0,
        now: float | None = None, repeat: bool = False,
    ) -> str | None:
        now = time.monotonic() if now is None else now
        if kind == "key_down":
            previously_held = keycode in self.held_keys
            self.held_keys.add(keycode)
            if self.right_down:
                self.blocked = True
            if keycode == ESCAPE and not repeat and not previously_held:
                return "cancel"
        elif kind == "key_up":
            self.held_keys.discard(keycode)
        elif kind == "mouse_down":
            self.held_buttons.add(keycode)
            if self.right_down:
                self.blocked = True
        elif kind == "mouse_up":
            self.held_buttons.discard(keycode)
        elif kind == "scroll":
            if self.right_down:
                self.blocked = True
        elif kind == "flags":
            forbidden = bool(flags & (OTHER_MODIFIERS | LEFT_COMMAND_MASK))
            if self.right_down and (forbidden or keycode in CHORD_MODIFIER_KEYS):
                self.blocked = True
            if keycode != RIGHT_COMMAND:
                return None
            pressed = bool(flags & RIGHT_COMMAND_MASK)
            if pressed:
                if not self.right_down:  # Repeated flags events must not reset timing.
                    self.right_down = True
                    self.down_at = now
                    self.blocked = forbidden or bool(self.held_keys or self.held_buttons)
            elif self.right_down:
                duration = now - self.down_at
                self.right_down = False
                toggle = not self.blocked and MIN_TAP_SECONDS <= duration <= MAX_TAP_SECONDS
                self.blocked = False
                if toggle:
                    return "toggle"
        return None


def _load_quartz():
    import Quartz
    return Quartz


def _post_main(callback: Callable[[], None]) -> None:
    from PyObjCTools import AppHelper
    AppHelper.callAfter(callback)


class CommandToggle:
    """Right Command listener with explicit lifecycle and inspectable status.

    ``CommandToggle(on_toggle, on_error, on_cancel=None)`` accepts no-argument
    toggle/cancel callbacks and an error callback accepting one Chinese message.
    ``start()`` returns whether the listener was installed; it never opens a
    privacy prompt. ``status`` is stopped/listening/permission_denied/error;
    ``last_error`` contains the reason when unavailable. ``stop()`` also discards
    callbacks queued before shutdown, so they cannot reopen a closed camera.
    Escape invokes optional on_cancel; the owner decides whether it is recording.
    """

    def __init__(
        self, on_toggle: Callable[[], None], on_error: Callable[[str], None],
        *, on_cancel: Callable[[], None] | None = None,
    ) -> None:
        self.on_toggle = on_toggle
        self.on_error = on_error
        self.on_cancel = on_cancel
        self.status = "stopped"
        self.last_error: str | None = None
        self.permission_available: bool | None = None
        self.gesture = CommandGesture()
        self._quartz = None
        self._tap = None
        self._source = None
        self._generation = 0

    @property
    def is_listening(self) -> bool:
        return self.status == "listening"

    def _queue(self, callback, *args, require_listening: bool = False) -> None:
        generation = self._generation

        def deliver():
            if generation != self._generation:
                return
            if require_listening and not self.is_listening:
                return
            try:
                callback(*args)
            except Exception as exc:
                # A UI callback cannot escape into the event-tap callback.
                self.last_error = f"快捷键回调失败：{type(exc).__name__}: {exc}"

        try:
            _post_main(deliver)
        except Exception as exc:
            self.last_error = f"无法投递快捷键回调：{type(exc).__name__}: {exc}"
            # This only handles missing AppKit during a failed/non-Mac start.
            # Never invoke application callbacks on a background thread.
            if threading.current_thread() is threading.main_thread():
                deliver()

    def _fail(self, message: str, *, permission: bool = False) -> bool:
        self._release()
        self.gesture.reset()
        self.status = "permission_denied" if permission else "error"
        self.last_error = message
        if permission:
            self.permission_available = False
        self._queue(self.on_error, message)
        return False

    def start(self) -> bool:
        if self.is_listening:
            return True
        self._generation += 1
        self.gesture.reset()
        self.last_error = None
        try:
            q = self._quartz = _load_quartz()
            preflight = getattr(q, "CGPreflightListenEventAccess", None)
            if preflight is not None:
                self.permission_available = bool(preflight())
                if not self.permission_available:
                    return self._fail(PERMISSION_MESSAGE, permission=True)
            event_types = (
                q.kCGEventFlagsChanged, q.kCGEventKeyDown, q.kCGEventKeyUp,
                q.kCGEventLeftMouseDown, q.kCGEventLeftMouseUp,
                q.kCGEventRightMouseDown, q.kCGEventRightMouseUp,
                q.kCGEventOtherMouseDown, q.kCGEventOtherMouseUp, q.kCGEventScrollWheel,
            )
            mask = 0
            for event_type in event_types:
                mask |= q.CGEventMaskBit(event_type)
            self._tap = q.CGEventTapCreate(
                q.kCGSessionEventTap, q.kCGHeadInsertEventTap,
                q.kCGEventTapOptionListenOnly, mask, self._callback, None,
            )
            if self._tap is None:
                return self._fail(PERMISSION_MESSAGE, permission=True)
            self._source = q.CFMachPortCreateRunLoopSource(None, self._tap, 0)
            if self._source is None:
                return self._fail("无法安装右 Command 快捷键；请使用窗口按钮并重新打开应用。")
            q.CFRunLoopAddSource(q.CFRunLoopGetMain(), self._source, q.kCFRunLoopCommonModes)
            q.CGEventTapEnable(self._tap, True)
            if hasattr(q, "CGEventTapIsEnabled") and not q.CGEventTapIsEnabled(self._tap):
                return self._fail(PERMISSION_MESSAGE, permission=True)
            self.permission_available = True
            self.status = "listening"
            return True
        except Exception as exc:
            return self._fail(f"右 Command 快捷键不可用：{type(exc).__name__}: {exc}。请使用窗口按钮。")

    def _release(self) -> None:
        q, source, tap = self._quartz, self._source, self._tap
        self._source = self._tap = None
        if q is None:
            return
        if source is not None:
            try:
                q.CFRunLoopRemoveSource(q.CFRunLoopGetMain(), source, q.kCFRunLoopCommonModes)
            except Exception:
                pass
        if tap is not None:
            try:
                q.CGEventTapEnable(tap, False)
            except Exception:
                pass
            try:
                q.CFMachPortInvalidate(tap)
            except Exception:
                pass

    def stop(self) -> None:
        self._generation += 1
        self._release()
        self.gesture.reset()
        self.status = "stopped"

    def _callback(self, proxy, event_type, event, refcon):
        # Returning the original event is mandatory even on failures/chords.
        if not self.is_listening:
            return event
        q = self._quartz
        try:
            if event_type in (q.kCGEventTapDisabledByTimeout, q.kCGEventTapDisabledByUserInput):
                self.gesture.reset()
                q.CGEventTapEnable(self._tap, True)
                if hasattr(q, "CGEventTapIsEnabled") and not q.CGEventTapIsEnabled(self._tap):
                    self._fail(PERMISSION_MESSAGE, permission=True)
                return event
            if q.CGEventGetIntegerValueField(event, q.kCGEventSourceUserData) == SYNTHETIC_LIPFLOW_EVENT:
                return event
            keycode = 0
            flags = q.CGEventGetFlags(event)
            repeat = False
            if event_type in (q.kCGEventFlagsChanged, q.kCGEventKeyDown, q.kCGEventKeyUp):
                keycode = q.CGEventGetIntegerValueField(event, q.kCGKeyboardEventKeycode)
                kind = {
                    q.kCGEventFlagsChanged: "flags", q.kCGEventKeyDown: "key_down",
                    q.kCGEventKeyUp: "key_up",
                }[event_type]
                repeat = bool(q.CGEventGetIntegerValueField(event, q.kCGKeyboardEventAutorepeat))
            elif event_type in (q.kCGEventLeftMouseDown, q.kCGEventRightMouseDown, q.kCGEventOtherMouseDown):
                kind = "mouse_down"
                keycode = q.CGEventGetIntegerValueField(event, q.kCGMouseEventButtonNumber)
            elif event_type in (q.kCGEventLeftMouseUp, q.kCGEventRightMouseUp, q.kCGEventOtherMouseUp):
                kind = "mouse_up"
                keycode = q.CGEventGetIntegerValueField(event, q.kCGMouseEventButtonNumber)
            elif event_type == q.kCGEventScrollWheel:
                kind = "scroll"
            else:
                return event
            # Use physical event timing rather than callback delivery timing. A
            # busy UI can deliver press/release together after a long delay.
            timestamp = getattr(q, "CGEventGetTimestamp", lambda _: 0)(event)
            event_time = timestamp / 1_000_000_000 if timestamp > 0 else None
            action = self.gesture.feed(
                kind, keycode=keycode, flags=flags, repeat=repeat, now=event_time,
            )
            if action == "toggle":
                self._queue(self.on_toggle, require_listening=True)
            elif action == "cancel" and self.on_cancel is not None:
                self._queue(self.on_cancel, require_listening=True)
        except Exception as exc:
            self._fail(f"右 Command 快捷键已停止：{type(exc).__name__}: {exc}。请使用窗口按钮。")
        return event
