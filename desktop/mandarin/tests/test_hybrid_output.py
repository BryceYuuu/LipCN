"""Delivery policy tests: no live GUI, permission prompts, or clipboard access."""
from collections import deque
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybrid_output import AppIdentity, OutputError, OutputTarget


ORIGIN = AppIdentity(101, 'example.editor')
LIPFLOW = AppIdentity(202, 'local.lipflow')
OTHER = AppIdentity(303, 'example.other')


class FakeDesktop:
    def __init__(self):
        self.front = ORIGIN
        self.front_reads = deque()
        self.allow = True
        self.can_activate = True
        self.auto_focus = True
        self.calls = []

    def frontmost(self):
        self.calls.append(('frontmost',))
        return self.front_reads.popleft() if self.front_reads else self.front

    def trusted(self):
        self.calls.append(('trusted',))
        return self.allow

    def activate(self, target):
        self.calls.append(('activate', target))
        if self.can_activate and self.auto_focus:
            self.front = target
        return self.can_activate

    def copy(self, text):
        self.calls.append(('copy', text))

    def paste(self, text):
        self.calls.append(('paste', text))


@pytest.fixture
def rig():
    desktop = FakeDesktop()
    now = [0.0]
    def advance(seconds):
        now[0] += seconds
    out = OutputTarget(desktop, own_pid=LIPFLOW.pid, activation_timeout=0.08,
                       clock=lambda: now[0], sleep=advance)
    return desktop, out


def test_capture_records_only_application_and_no_output(rig):
    desktop, out = rig
    assert out.capture() is out
    assert out.target == ORIGIN
    assert out.has_target
    assert desktop.calls == [('frontmost',)]
    assert vars(out.target) == {'pid': 101, 'bundle_id': 'example.editor'}


@pytest.mark.parametrize('front', [LIPFLOW, None, AppIdentity(0, None)])
def test_capture_in_self_or_without_application_clears_old_target(rig, front):
    desktop, out = rig
    out.capture()
    desktop.front = front
    out.capture()
    assert not out.has_target


def test_copy_requires_no_target_or_accessibility(rig):
    desktop, out = rig
    out.copy('这是确认的句子。')
    assert desktop.calls == [('copy', '这是确认的句子。')]


def test_confirmed_insert_activates_and_checks_destination_twice(rig):
    desktop, out = rig
    out.capture()
    desktop.front = LIPFLOW
    desktop.calls.clear()
    out.insert('确认后插入。')
    assert desktop.calls == [('trusted',), ('activate', ORIGIN), ('frontmost',),
                             ('frontmost',), ('paste', '确认后插入。')]


def test_missing_target_does_not_touch_system(rig):
    desktop, out = rig
    with pytest.raises(OutputError, match='没有原输入应用'):
        out.insert('你好。')
    assert desktop.calls == []


def test_permission_denied_does_not_activate_or_touch_clipboard(rig):
    desktop, out = rig
    out.capture()
    desktop.calls.clear()
    desktop.allow = False
    with pytest.raises(OutputError, match='辅助功能'):
        out.insert('你好。')
    assert desktop.calls == [('trusted',)]


def test_closed_target_does_not_paste(rig):
    desktop, out = rig
    out.capture()
    desktop.can_activate = False
    with pytest.raises(OutputError, match='原应用已关闭'):
        out.insert('你好。')
    assert not any(call[0] in ('paste', 'copy') for call in desktop.calls)


def test_wrong_frontmost_times_out_without_pasting(rig):
    desktop, out = rig
    out.capture()
    desktop.front = OTHER
    desktop.auto_focus = False
    with pytest.raises(OutputError, match='未能回到原应用'):
        out.insert('你好。')
    assert not any(call[0] in ('paste', 'copy') for call in desktop.calls)


def test_focus_stolen_between_activation_and_paste_is_rejected(rig):
    desktop, out = rig
    out.capture()
    desktop.front_reads.extend([ORIGIN, OTHER])
    with pytest.raises(OutputError, match='前台应用已改变'):
        out.insert('你好。')
    assert not any(call[0] in ('paste', 'copy') for call in desktop.calls)


def test_same_pid_with_different_bundle_is_not_original_target(rig):
    desktop, out = rig
    out.capture()
    desktop.auto_focus = False
    desktop.front = AppIdentity(ORIGIN.pid, 'example.replacement')
    with pytest.raises(OutputError, match='未能回到原应用'):
        out.insert('你好。')
    assert not any(call[0] in ('paste', 'copy') for call in desktop.calls)


@pytest.mark.parametrize('text', ['', '  \n', None])
@pytest.mark.parametrize('action', ['copy', 'insert'])
def test_empty_candidate_rejected_before_system_access(rig, text, action):
    desktop, out = rig
    with pytest.raises(OutputError, match='非空句子'):
        getattr(out, action)(text)
    assert desktop.calls == []


def test_unexpected_native_error_is_actionable_and_does_not_paste(rig):
    desktop, out = rig
    out.capture()
    def broken(_target):
        raise RuntimeError('native failure')
    desktop.activate = broken
    with pytest.raises(OutputError, match='无法安全插入'):
        out.insert('你好。')
    assert not any(call[0] in ('paste', 'copy') for call in desktop.calls)
