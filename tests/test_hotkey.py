import time

import Quartz

from lipflow import hotkey
from lipflow.hotkey import KEYS, PushToTalk


def _flags(ptt, pressed):
    code, mask = KEYS["right_option"]
    ev = Quartz.CGEventCreateKeyboardEvent(None, code, pressed)
    Quartz.CGEventSetType(ev, Quartz.kCGEventFlagsChanged)
    Quartz.CGEventSetFlags(ev, (Quartz.kCGEventFlagMaskAlternate | mask) if pressed else 0)
    ptt._callback(None, Quartz.kCGEventFlagsChanged, ev, None)


def _key(ptt, code):
    ev = Quartz.CGEventCreateKeyboardEvent(None, code, True)
    ptt._callback(None, Quartz.kCGEventKeyDown, ev, None)


def make():
    log = []
    ptt = PushToTalk("right_option", lambda hands_free: log.append(("start", hands_free)),
                     lambda: log.append(("stop",)), lambda silent=False: log.append(("cancel", silent)))
    return ptt, log


def test_hold_to_talk():
    ptt, log = make()
    _flags(ptt, True)
    time.sleep(hotkey.TAP_MAX + 0.05)
    _flags(ptt, False)
    assert log == [("start", False), ("stop",)]


def test_single_tap_is_silently_ignored():
    ptt, log = make()
    _flags(ptt, True); _flags(ptt, False)
    assert log == [("start", False), ("cancel", True)]


def test_double_tap_enters_hands_free_until_next_tap():
    ptt, log = make()
    _flags(ptt, True); _flags(ptt, False)
    _flags(ptt, True); _flags(ptt, False)
    assert log[-1] == ("start", True) and ptt.hands_free
    time.sleep(0.5)
    _flags(ptt, True); _flags(ptt, False)
    assert log[-1] == ("stop",) and not ptt.hands_free


def test_option_shortcut_cancels_instead_of_dictating():
    ptt, log = make()
    _flags(ptt, True)
    _key(ptt, 0)  # Option+A
    _flags(ptt, False)
    assert log == [("start", False), ("cancel", False)]


def test_escape_cancels():
    ptt, log = make()
    _flags(ptt, True)
    _key(ptt, hotkey.ESC)
    assert log[-1] == ("cancel", False)


def test_ignores_lipflows_own_keystrokes():
    from lipflow.paste import MARK
    ptt, log = make()
    _flags(ptt, True)
    ev = Quartz.CGEventCreateKeyboardEvent(None, 0, True)
    Quartz.CGEventSetIntegerValueField(ev, Quartz.kCGEventSourceUserData, MARK)
    ptt._callback(None, Quartz.kCGEventKeyDown, ev, None)  # live typing while the key is held
    assert log == [("start", False)]
