"""Exercise both front ends with fake OS/UI/model boundaries; no camera or paste.

Windows control flow runs on every platform, but this does not replace native
Windows clipboard/hotkey tests in test_windows.py.
"""
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from lipflow.cleanup import Cleaner
from lipflow.confidence import Hypothesis as H, Quality
from lipflow.delivery import choose_result
from lipflow.dictation import configure_options
from lipflow.personal import Personal


@pytest.fixture(params=["macos", "windows"])
def frontend(request, monkeypatch):
    if request.param == "macos":
        pytest.importorskip("AppKit")
        import lipflow.app as module
    else:
        # Load the actual Windows app code with only OS-facing imports stubbed.
        for name, attrs in {
            "tkinter": {"Tk": lambda: NS(withdraw=lambda: None)},
            "lipflow.win.hotkey": {"DEFAULT_KEY": "right_control", "KEYS": {"right_control": ()},
                                   "PushToTalk": object},
            "lipflow.win.hud": {"HUD": object, "tray_image": lambda *a: None},
            "lipflow.win.paste": {"copy_text": lambda *a: None, "paste_text": lambda *a: None},
        }.items():
            fake = types.ModuleType(name)
            fake.__dict__.update(attrs)
            monkeypatch.setitem(sys.modules, name, fake)
        name = "lipflow.win._runtime_policy_test"
        path = Path(__file__).resolve().parents[1] / "lipflow/win/app.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module, request.param


@pytest.mark.parametrize("settings, overrides, expected", [
    ({}, {}, "off"),
    ({"confidence_policy": "review"}, {}, "review"),
    ({"confidence_policy_en": "auto"}, {}, "auto"),
    ({"confidence_policy_en": "review"}, {"confidence_policy": "off"}, "off"),
    ({"confidence_policy_en": "off"}, {"confidence_policy": "review"}, "review"),
    ({"language": "zh", "confidence_policy": "review"}, {"language": "en"}, "off"),
    ({"language": "zh", "confidence_policy_en": "auto"}, {"language": "en"}, "auto"),
    ({"language": "zh"}, {}, "review"),
    ({}, {"language": "zh", "confidence_policy": "off"}, "review"),
    ({}, {"language": "zh", "confidence_policy": "auto"}, "review"),
    ({}, {"cleanup_mode": "polish", "confidence_policy": "off"}, "review"),
    ({"cleanup_mode": "polish"}, {"confidence_policy": "auto"}, "review"),
])
def test_policy_language_defaults_saved_optin_and_cli_override(settings, overrides, expected):
    opts = NS(language=None, cleanup_mode=None, confidence_policy=None, input_mode=None)
    vars(opts).update(overrides)
    configure_options(opts, settings)
    assert opts.confidence_policy == expected


@pytest.mark.parametrize("language, policy, effective, guarded", [
    (None, None, "off", False),
    ("en", "review", "review", True),
    ("en", "auto", "auto", True),
    ("zh", "off", "review", True),
])
def test_frontend_constructor_applies_same_policy_and_whisper(frontend, monkeypatch, language, policy, effective, guarded):
    module, platform = frontend
    monkeypatch.setattr(module, "load_settings", lambda language="en": {})
    monkeypatch.setattr(module, "Camera", lambda *a, **k: NS())
    monkeypatch.setattr("lipflow.mic.Mic", lambda: NS())
    opts = module.Options(backend="basic", language=language, confidence_policy=policy, input_mode="whisper")
    app = (module.Lipflow.alloc().initWithOptions_(opts) if platform == "macos" else module.Lipflow(opts))
    assert app.opts.confidence_policy == effective
    assert app.cleaner.guard_sensitive is guarded
    assert app.settings["whisper"] is True


@pytest.mark.parametrize("policy", ["off", "review", "auto"])
def test_context_disabled_never_reads_focused_field_on_either_frontend(frontend, monkeypatch, policy):
    import lipflow.delivery as delivery
    import lipflow.context as context
    module, platform = frontend
    captured = []
    monkeypatch.setattr(context, "capture", lambda: pytest.fail("Focused context must not be read"))
    monkeypatch.setattr(delivery, "capture_target", lambda: captured.append(True) or context.Context(target=123))
    monkeypatch.setattr(module.threading, "Thread", lambda *a, **k: NS(start=lambda: None))
    rec = NS()
    app = NS(review_pending=False, loading=False, pending_stop=None, hands_free=False,
             session=0, opts=NS(confidence_policy=policy), settings={"use_context": False},
             camera=NS(recording=None, start_recording=lambda: rec, ready=NS(is_set=lambda: True)),
             whisper_on=False, hud=NS(show=lambda *a, **k: None),
             _preview_loop=lambda *a: None, _set_status_icon=lambda *a: None, _set_icon=lambda *a: None)
    module.Lipflow.on_start(app, False)
    assert rec.ctx.names == [] and rec.ctx.element is None and rec.ctx.value == ""
    assert rec.ctx.title == "" and rec.ctx.near_text == ""
    assert captured == ([] if policy == "off" else [True])


def test_enabled_context_retains_names_and_field_snapshot(monkeypatch):
    import lipflow.context as context
    from lipflow.delivery import context_for_recording
    captured = context.Context(names=["Amy"], near_text="Message Amy", element=object(), value="existing")
    monkeypatch.setattr(context, "capture", lambda: captured)
    assert context_for_recording({"use_context": True}, "off") is captured


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_target_only_capture_reads_identity_without_title_or_text(monkeypatch, platform):
    from lipflow.delivery import capture_target
    monkeypatch.setattr(sys, "platform", platform)
    calls = []
    if platform == "darwin":
        fake = types.ModuleType("AppKit")
        app = NS(processIdentifier=lambda: calls.append("pid") or 123)
        fake.NSWorkspace = NS(sharedWorkspace=lambda: NS(frontmostApplication=lambda: app))
        monkeypatch.setitem(sys.modules, "AppKit", fake)
    else:
        import ctypes
        def foreground():
            calls.append("hwnd")
            return 123
        dll = NS(GetForegroundWindow=foreground)
        monkeypatch.setattr(ctypes, "WinDLL", lambda *a: dll, raising=False)
    ctx = capture_target()
    assert ctx.target == 123 and calls == (["pid"] if platform == "darwin" else ["hwnd"])
    assert ctx.names == [] and ctx.element is None and ctx.value == ""
    assert ctx.app == "" and ctx.title == "" and ctx.near_text == ""


@pytest.mark.parametrize("policy", ["off", "review", "auto"])
def test_mandarin_cannot_disable_confirmation(policy):
    cleaner = Cleaner("basic", language="zh")
    cleaner.personal = Personal("/nonexistent")
    decision, result, choices = choose_result([H("你好", -1, 2), H("您好", -5, 2)], "你好",
        Quality(), cleaner, None, "", policy=policy)
    assert decision.action == "review" and result.text == "你好。"
    assert choices


@pytest.mark.parametrize("policy", ["off", "auto"])
def test_polish_cannot_disable_confirmation(policy, monkeypatch):
    cleaner = Cleaner("basic", mode="polish", guard_sensitive=False)
    cleaner.personal = Personal("/nonexistent")
    cleaner.backend = "ollama"
    monkeypatch.setattr(cleaner, "_ollama", lambda *a: "Could you send it tomorrow?")
    decision, result, choices = choose_result([H("SEND IT TOMORROW", -1, 3), H("SEND IT TODAY", -8, 3)],
        "SEND IT TOMORROW", Quality(), cleaner, None, "", policy=policy)
    assert decision.action == "review" and choices and result.needs_review


def test_default_english_runs_cleanup_without_new_quality_or_score_gates(monkeypatch):
    cleaner = Cleaner("basic")
    cleaner.personal = Personal("/nonexistent")
    cleaner.backend = "ollama"
    monkeypatch.setattr(cleaner, "_ollama", lambda *a: "Park 5.")
    monkeypatch.setattr("lipflow.vocab.load", lambda language="en": [])
    decision, result, choices = choose_result([H("BARK FIVE", float("nan"), 2)], "",
        Quality(face_ratio=.5, brightness=20, contrast=0), cleaner, None, "")
    assert decision.action == "auto" and choices == []
    assert result.text == "Park 5." and not result.needs_review


@pytest.mark.parametrize("av_hypotheses", [None, [H("", -1, 0)]])
def test_default_english_frontend_final_pastes_cleaned_result_without_picker(frontend, monkeypatch, av_hypotheses):
    import lipflow.delivery as delivery
    from lipflow.context import Context
    module, platform = frontend
    pasted = []
    cleaner = Cleaner("basic")
    cleaner.personal = Personal("/nonexistent")
    cleaner.backend = "ollama"
    monkeypatch.setattr(cleaner, "_ollama", lambda *a: "Park 5.")
    monkeypatch.setattr("lipflow.vocab.load", lambda language="en": [])
    monkeypatch.setattr(module, "paste_text", pasted.append)
    monkeypatch.setattr(module, "copy_text", lambda *a: pytest.fail("Default English must paste"))
    monkeypatch.setattr(module, "log_history", lambda *a, **k: None)
    monkeypatch.setattr(module, "keep_clip", lambda *a, **k: None)
    monkeypatch.setattr(delivery, "target_is_current", lambda *a: pytest.fail("No new default English focus gate"))
    rois = np.full((30, 96, 96), 20, np.uint8)
    if platform == "macos":
        monkeypatch.setattr(module, "ui", lambda fn, *a, **k: fn(*a, **k))
    else:
        monkeypatch.setattr(module, "rois_for", lambda rec: rois)
    rec = NS(duration=1.2, ts=list(range(30)), face_ratio=.5, mouth_open=[.02, .08] * 15,
             mouth_pixels=[80] * 30, ctx=Context(), session=1)
    app = NS(session=1, opts=NS(language="en", input_mode="silent", confidence_policy="off",
             paste=True, min_margin=.5), cleaner=cleaner, onboarding=None, context=[], settings={},
             last_paste_at=0, _rois=lambda rec: rois, _av_candidates=lambda *a: av_hypotheses,
             reader=NS(encode=lambda r: object(), hypotheses=lambda *a, **k: [H("BARK FIVE", -1, 2)],
                       greedy=lambda *a: pytest.fail("Default English must not add greedy decoding")),
             hud=NS(show=lambda *a, **k: None),
             review=NS(show=lambda *a: pytest.fail("Default English must not open candidate picker")),
             ui=lambda fn, *a, **k: fn(*a, **k))
    app._deliver = lambda *a: module.Lipflow._deliver(app, *a)
    app._publish = lambda *a: module.Lipflow._publish(app, *a)
    module.Lipflow._final(app, rec)
    assert pasted == ["Park 5."] and app.last_output == "Park 5."


@pytest.mark.parametrize("policy", ["review", "auto"])
def test_explicit_english_optin_retries_bad_quality_before_cleanup(policy):
    cleaner = NS(language="en", process=lambda *a, **k: pytest.fail("Bad quality should skip cleanup"))
    decision, result, choices = choose_result([H("hello", -1, 1)], "hello", Quality(face_ratio=.1),
        cleaner, None, "", policy=policy)
    assert decision.action == "retry" and result is None and choices == []


@pytest.mark.parametrize("argv, expected", [
    ([], "review"),
    (["run", "--confidence-policy", "off"], "off"),
    (["run", "--confidence-policy", "auto"], "auto"),
    (["run", "--cleanup-mode", "polish", "--confidence-policy", "off"], "review"),
])
def test_cli_unset_keeps_saved_optin_but_explicit_off_overrides(monkeypatch, argv, expected):
    import lipflow.__main__ as cli
    settings = {"confidence_policy_en": "review"}
    seen = []
    monkeypatch.setattr(cli, "_app", lambda: (NS, lambda opts: seen.append(configure_options(opts, settings))))
    cli.main(argv)
    assert seen[0].confidence_policy == expected
