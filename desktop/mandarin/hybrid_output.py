"""Explicit candidate delivery, without reading focused fields or private text.

Call ``capture`` before showing LipCN in response to the user's shortcut.
Call ``copy`` or ``insert`` only after the user selects/confirms a candidate.
The module never emits Return and never requests Accessibility permission.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import time


class OutputError(RuntimeError):
    """An actionable message suitable for the candidate-selection UI."""


@dataclass(frozen=True)
class AppIdentity:
    pid: int
    bundle_id: str | None


class _MacDesktop:
    """Lazy imports let the policy tests run without macOS or a clipboard."""

    def frontmost(self) -> AppIdentity | None:
        from AppKit import NSWorkspace

        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return None
        return AppIdentity(int(app.processIdentifier()), app.bundleIdentifier())

    def trusted(self) -> bool:
        from ApplicationServices import AXIsProcessTrusted

        return bool(AXIsProcessTrusted())  # No prompt option, no focused-field read.

    def activate(self, target: AppIdentity) -> bool:
        from AppKit import NSApplicationActivateIgnoringOtherApps, NSRunningApplication

        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(target.pid)
        if app is None or app.isTerminated():
            return False
        if app.bundleIdentifier() != target.bundle_id:
            return False
        return bool(app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps))

    def copy(self, text: str) -> None:
        from lipflow.paste import copy_text

        copy_text(text)

    def paste(self, text: str) -> None:
        from lipflow.paste import paste_text

        # Maintainer implementation emits only Command-V and restores clipboard.
        paste_text(text)


class OutputTarget:
    """Remember only the source application; do not infer a focused text field.

    ``insert`` may wait up to ``activation_timeout`` seconds for macOS activation.
    Failure keeps the candidate available to the UI for a manual Copy action.
    All system effects are injectable, so tests never use the user's clipboard.
    """

    def __init__(self, desktop=None, *, own_pid=None, activation_timeout=0.6,
                 clock=None, sleep=None):
        self._desktop = desktop if desktop is not None else _MacDesktop()
        self._own_pid = os.getpid() if own_pid is None else own_pid
        self._timeout = max(0.0, float(activation_timeout))
        self._clock = time.monotonic if clock is None else clock
        self._sleep = time.sleep if sleep is None else sleep
        self.target: AppIdentity | None = None

    @property
    def has_target(self) -> bool:
        return self.target is not None

    def capture(self) -> OutputTarget:
        # Clear stale destinations, including when a shortcut is pressed in LipCN.
        self.target = None
        try:
            app = self._desktop.frontmost()
        except Exception as exc:
            raise OutputError('无法确认原应用，请使用“复制”，再手动粘贴。') from exc
        if app is not None and app.pid > 0 and app.pid != self._own_pid:
            self.target = AppIdentity(app.pid, app.bundle_id)
        return self

    @staticmethod
    def _text(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise OutputError('请先选择一个非空句子。')
        return text

    def copy(self, text: str) -> None:
        text = self._text(text)
        try:
            self._desktop.copy(text)
        except Exception as exc:
            raise OutputError('复制失败，请重试。') from exc

    def insert(self, text: str) -> None:
        text = self._text(text)
        target = self.target
        if target is None:
            raise OutputError('没有原输入应用。请先在其他应用的输入框中按右 Command，或使用“复制”。')
        try:
            if not self._desktop.trusted():
                raise OutputError('需要辅助功能权限才能插入。请到系统设置 → 隐私与安全性 → 辅助功能启用 LipCN，或使用“复制”。')
            if not self._desktop.activate(target):
                raise OutputError('原应用已关闭或无法激活，请使用“复制”后手动粘贴。')
            deadline = self._clock() + self._timeout
            while self._desktop.frontmost() != target:
                if self._clock() >= deadline:
                    raise OutputError('未能回到原应用，已取消插入。请使用“复制”后手动粘贴。')
                self._sleep(min(0.02, max(0.0, deadline - self._clock())))
            # A second read catches a focus change between activation and delivery.
            # Never inspect AX focused fields, values, selections, or app content.
            if self.target != target or self._desktop.frontmost() != target:
                raise OutputError('前台应用已改变，已取消插入。请使用“复制”后手动粘贴。')
            self._desktop.paste(text)
        except OutputError:
            raise
        except Exception as exc:
            raise OutputError('无法安全插入原应用，请使用“复制”后手动粘贴。') from exc
