"""Synthetic events only: these tests never install a real input listener."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hybrid_hotkey as hk


COMMAND = (1 << 20) | hk.RIGHT_COMMAND_MASK


def down(gesture, now=10.0, flags=COMMAND):
    return gesture.feed("flags", keycode=hk.RIGHT_COMMAND, flags=flags, now=now)


def up(gesture, now=10.1, flags=0):
    return gesture.feed("flags", keycode=hk.RIGHT_COMMAND, flags=flags, now=now)


def test_toggle_requires_release_and_repeated_flags_do_not_repeat():
    gesture = hk.CommandGesture()
    assert down(gesture) is None
    assert down(gesture, 10.05) is None
    assert up(gesture) == "toggle"
    assert up(gesture, 10.11) is None
    assert down(gesture, 11.0) is None
    assert up(gesture, 11.1) == "toggle"


@pytest.mark.parametrize("duration", [-1.0, 0, 0.001, 0.51, 5.0])
def test_bounce_long_press_and_clock_reversal_never_toggle(duration):
    gesture = hk.CommandGesture()
    down(gesture)
    assert up(gesture, 10 + duration) is None


@pytest.mark.parametrize("keycode", [0, 8, 9, 12, 48, 49])
def test_command_letter_and_command_tab_chords_never_toggle(keycode):
    gesture = hk.CommandGesture()
    down(gesture)
    gesture.feed("key_down", keycode=keycode)
    gesture.feed("key_up", keycode=keycode)
    assert up(gesture) is None
    down(gesture, 11)
    assert up(gesture, 11.1) == "toggle"


def test_key_held_before_command_is_a_chord_too():
    gesture = hk.CommandGesture()
    gesture.feed("key_down", keycode=8)
    down(gesture)
    gesture.feed("key_up", keycode=8)
    assert up(gesture) is None


@pytest.mark.parametrize("mask", [hk.LEFT_COMMAND_MASK, 1 << 17, 1 << 18, 1 << 19, 1 << 23])
def test_other_modifier_already_held_blocks_press(mask):
    gesture = hk.CommandGesture()
    down(gesture, flags=COMMAND | mask)
    assert up(gesture) is None


@pytest.mark.parametrize("modifier", sorted(hk.CHORD_MODIFIER_KEYS))
def test_modifier_pressed_during_command_blocks_even_if_released_first(modifier):
    gesture = hk.CommandGesture()
    down(gesture)
    gesture.feed("flags", keycode=modifier, flags=COMMAND, now=10.03)
    assert up(gesture) is None


def test_left_command_alone_and_unmatched_release_never_toggle():
    gesture = hk.CommandGesture()
    assert gesture.feed("flags", keycode=hk.LEFT_COMMAND, flags=(1 << 20) | hk.LEFT_COMMAND_MASK) is None
    assert gesture.feed("flags", keycode=hk.LEFT_COMMAND, flags=0) is None
    assert up(gesture) is None


def test_caps_lock_setting_does_not_disable_shortcut():
    gesture = hk.CommandGesture()
    down(gesture, flags=COMMAND | (1 << 16))
    assert up(gesture, flags=1 << 16) == "toggle"


@pytest.mark.parametrize("kind", ["mouse_down", "scroll"])
def test_command_mouse_or_scroll_never_toggle(kind):
    gesture = hk.CommandGesture()
    down(gesture)
    gesture.feed(kind, keycode=0)
    gesture.feed("mouse_up", keycode=0)
    assert up(gesture) is None


def test_mouse_already_held_blocks_press():
    gesture = hk.CommandGesture()
    gesture.feed("mouse_down", keycode=1)
    down(gesture)
    gesture.feed("mouse_up", keycode=1)
    assert up(gesture) is None


def test_escape_cancels_once_per_press_and_blocks_pending_toggle():
    gesture = hk.CommandGesture()
    down(gesture)
    assert gesture.feed("key_down", keycode=hk.ESCAPE) == "cancel"
    assert gesture.feed("key_down", keycode=hk.ESCAPE, repeat=True) is None
    assert gesture.feed("key_down", keycode=hk.ESCAPE) is None
    gesture.feed("key_up", keycode=hk.ESCAPE)
    assert up(gesture) is None
    assert gesture.feed("key_down", keycode=hk.ESCAPE) == "cancel"


def test_reset_discards_pending_and_held_keys():
    gesture = hk.CommandGesture()
    down(gesture)
    gesture.feed("key_down", keycode=8)
    gesture.reset()
    assert up(gesture) is None
    down(gesture, 11)
    assert up(gesture, 11.1) == "toggle"


class FakeQuartz:
    kCGEventFlagsChanged = 12
    kCGEventKeyDown = 10
    kCGEventKeyUp = 11
    kCGEventLeftMouseDown = 1
    kCGEventLeftMouseUp = 2
    kCGEventRightMouseDown = 3
    kCGEventRightMouseUp = 4
    kCGEventOtherMouseDown = 25
    kCGEventOtherMouseUp = 26
    kCGEventScrollWheel = 22
    kCGEventTapDisabledByTimeout = -1
    kCGEventTapDisabledByUserInput = -2
    kCGSessionEventTap = 1
    kCGHeadInsertEventTap = 0
    kCGEventTapOptionListenOnly = 1
    kCGEventSourceUserData = "userdata"
    kCGKeyboardEventKeycode = "keycode"
    kCGKeyboardEventAutorepeat = "repeat"
    kCGMouseEventButtonNumber = "button"
    kCFRunLoopCommonModes = "common"

    def __init__(self):
        self.permission = True
        self.creates = []
        self.enables = []
        self.removes = []
        self.invalidates = []
        self.null_tap = False
        self.null_source = False
        self.can_enable = True
        self.enabled = False

    def CGPreflightListenEventAccess(self):
        return self.permission

    @staticmethod
    def CGEventMaskBit(event_type):
        return 1 << event_type

    def CGEventTapCreate(self, *args):
        self.creates.append(args)
        return None if self.null_tap else "tap"

    def CFMachPortCreateRunLoopSource(self, *args):
        return None if self.null_source else "source"

    @staticmethod
    def CFRunLoopGetMain():
        return "main"

    def CFRunLoopAddSource(self, *args):
        self.added = args

    def CFRunLoopRemoveSource(self, *args):
        self.removes.append(args)

    def CGEventTapEnable(self, tap, enabled):
        self.enables.append((tap, enabled))
        self.enabled = enabled and self.can_enable

    def CGEventTapIsEnabled(self, tap):
        return self.enabled

    def CFMachPortInvalidate(self, tap):
        self.invalidates.append(tap)

    @staticmethod
    def CGEventGetIntegerValueField(event, field):
        return event.get(field, 0)

    @staticmethod
    def CGEventGetTimestamp(event):
        return event.get("timestamp", 0)

    @staticmethod
    def CGEventGetFlags(event):
        return event.get("flags", 0)


@pytest.fixture
def listener(monkeypatch):
    quartz = FakeQuartz()
    queued = []
    delivered = []
    clock = [10.0]
    monkeypatch.setattr(hk, "_load_quartz", lambda: quartz)
    monkeypatch.setattr(hk, "_post_main", queued.append)
    monkeypatch.setattr(hk.time, "monotonic", lambda: clock[0])
    toggle = hk.CommandToggle(
        lambda: delivered.append("toggle"),
        lambda error: delivered.append(("error", error)),
        on_cancel=lambda: delivered.append("cancel"),
    )
    return toggle, quartz, queued, delivered, clock


def tap(toggle, quartz, clock):
    event = {"keycode": hk.RIGHT_COMMAND, "flags": COMMAND}
    assert toggle._callback(None, quartz.kCGEventFlagsChanged, event, None) is event
    clock[0] += 0.1
    event = {"keycode": hk.RIGHT_COMMAND, "flags": 0}
    assert toggle._callback(None, quartz.kCGEventFlagsChanged, event, None) is event


def flush(queued):
    while queued:
        queued.pop(0)()


def test_no_install_on_construction_and_read_only_main_loop_install(listener):
    toggle, q, queued, delivered, _ = listener
    assert toggle.status == "stopped"
    assert not q.creates
    assert toggle.start()
    assert toggle.is_listening
    assert q.creates[0][2] == q.kCGEventTapOptionListenOnly
    assert q.added == ("main", "source", "common")
    assert toggle.start()
    assert len(q.creates) == 1
    assert not delivered and not queued


def test_permission_failure_never_requests_access_or_attempts_tap(listener):
    toggle, q, queued, delivered, _ = listener
    q.permission = False
    assert not toggle.start()
    assert toggle.status == "permission_denied"
    assert toggle.permission_available is False
    assert not q.creates
    assert not delivered
    flush(queued)
    assert delivered == [("error", hk.PERMISSION_MESSAGE)]


def test_failed_tap_creation_reports_permission_help(listener):
    toggle, q, queued, delivered, _ = listener
    q.null_tap = True
    assert not toggle.start()
    flush(queued)
    assert delivered == [("error", hk.PERMISSION_MESSAGE)]


def test_failed_runloop_source_releases_tap(listener):
    toggle, q, queued, delivered, _ = listener
    q.null_source = True
    assert not toggle.start()
    assert q.invalidates == ["tap"]
    flush(queued)
    assert delivered[0][0] == "error"


def test_toggle_and_cancel_callbacks_are_queued_and_events_preserved(listener):
    toggle, q, queued, delivered, clock = listener
    assert toggle.start()
    tap(toggle, q, clock)
    assert not delivered
    flush(queued)
    assert delivered == ["toggle"]
    event = {"keycode": hk.ESCAPE}
    assert toggle._callback(None, q.kCGEventKeyDown, event, None) is event
    assert delivered == ["toggle"]
    flush(queued)
    assert delivered == ["toggle", "cancel"]


def test_stop_invalidates_queued_toggle_and_releases_everything_once(listener):
    toggle, q, queued, delivered, clock = listener
    toggle.start()
    tap(toggle, q, clock)
    toggle.stop()
    toggle.stop()
    flush(queued)
    assert not delivered
    assert toggle.status == "stopped"
    assert q.removes == [("main", "source", "common")]
    assert q.invalidates == ["tap"]


def test_restart_invalidates_old_errors_and_old_presses(listener):
    toggle, q, queued, delivered, clock = listener
    q.permission = False
    toggle.start()
    q.permission = True
    assert toggle.start()
    flush(queued)
    assert not delivered
    tap(toggle, q, clock)
    flush(queued)
    assert delivered == ["toggle"]


def test_command_copy_keeps_system_events_and_does_not_toggle(listener):
    toggle, q, queued, delivered, clock = listener
    toggle.start()
    events = [
        (q.kCGEventFlagsChanged, {"keycode": hk.RIGHT_COMMAND, "flags": COMMAND}),
        (q.kCGEventKeyDown, {"keycode": 8, "flags": COMMAND}),
        (q.kCGEventKeyUp, {"keycode": 8, "flags": COMMAND}),
        (q.kCGEventFlagsChanged, {"keycode": hk.RIGHT_COMMAND, "flags": 0}),
    ]
    for event_type, event in events:
        assert toggle._callback(None, event_type, event, None) is event
        clock[0] += 0.03
    flush(queued)
    assert not delivered


def test_synthetic_paste_does_not_trigger_cancel(listener):
    toggle, q, queued, delivered, clock = listener
    toggle.start()
    event = {"keycode": hk.ESCAPE, "userdata": hk.SYNTHETIC_LIPFLOW_EVENT}
    assert toggle._callback(None, q.kCGEventKeyDown, event, None) is event
    flush(queued)
    assert not delivered


@pytest.mark.parametrize("disabled_type", [-1, -2])
def test_tap_interruption_discards_partial_gesture_and_reenables(listener, disabled_type):
    toggle, q, queued, delivered, clock = listener
    toggle.start()
    event = {"keycode": hk.RIGHT_COMMAND, "flags": COMMAND}
    toggle._callback(None, q.kCGEventFlagsChanged, event, None)
    assert toggle._callback(None, disabled_type, None, None) is None
    clock[0] += 0.1
    toggle._callback(None, q.kCGEventFlagsChanged, {"keycode": hk.RIGHT_COMMAND}, None)
    flush(queued)
    assert toggle.is_listening
    assert not delivered
    assert q.enables[-1] == ("tap", True)


def test_tap_cannot_reenable_reports_failure(listener):
    toggle, q, queued, delivered, clock = listener
    toggle.start()
    q.can_enable = False
    toggle._callback(None, q.kCGEventTapDisabledByTimeout, None, None)
    flush(queued)
    assert toggle.status == "permission_denied"
    assert delivered == [("error", hk.PERMISSION_MESSAGE)]


def test_delayed_main_loop_uses_event_timestamps_not_callback_delivery_time(listener):
    toggle, q, queued, delivered, clock = listener
    toggle.start()
    # Both callbacks arrive at the same clock time, but physical press was 100 ms.
    for flags, timestamp in [(COMMAND, 20_000_000_000), (0, 20_100_000_000)]:
        event = {"keycode": hk.RIGHT_COMMAND, "flags": flags, "timestamp": timestamp}
        toggle._callback(None, q.kCGEventFlagsChanged, event, None)
    flush(queued)
    assert delivered == ["toggle"]
