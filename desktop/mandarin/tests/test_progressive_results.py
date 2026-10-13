"""Editing, cancellation and device isolation of the first-result workflow."""
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from test_hybrid_ui import h, Widget
from test_hybrid_routing import controller, recording
from input_settings import DEFAULT_INPUT_MODE, load_input_mode, save_input_mode


def result(*texts):
    return SimpleNamespace(variants=[SimpleNamespace(text=t) for t in texts])


def editors(ui):
    ui.result_editors = [Widget() for _ in range(3)]
    ui.result.show_choices = lambda: None
    ui.choice_buttons = []
    for i in range(3):
        button = Widget()
        button.tag = lambda i=i: i
        ui.choice_buttons.append(button)


def test_first_result_can_be_copied_and_edited_before_formatter_returns(h):
    ui = h.ui
    editors(ui)
    ui.processing = True
    ui.output_target.copy = Mock()
    ui.output_target.insert = Mock()
    ui._first_inferred(0, result('原句。'), .2, {'route': 'lips'})
    assert ui.processing and ui.choice_buttons[0].enabled
    assert not any(button.enabled for button in ui.choice_buttons[1:])
    ui.result_editors[0].setString_('我修正的句子。')
    ui.copyChoice_(ui.choice_buttons[0])
    ui.insertChoice_(ui.choice_buttons[0])
    ui.output_target.copy.assert_called_once_with('我修正的句子。')
    ui.output_target.insert.assert_called_once_with('我修正的句子。')
    ui._inferred(0, result('原句。', '自然表达。', '简洁表达。'), 3, 1, 25, {'route': 'lips'})
    assert ui.result_editors[0].string() == '我修正的句子。'
    assert [e.string() for e in ui.result_editors[1:]] == ['自然表达。', '简洁表达。']
    assert not ui.processing and all(b.enabled for b in ui.choice_buttons)


def test_deleting_early_candidate_is_preserved_and_disables_output(h):
    ui = h.ui
    editors(ui)
    ui._first_inferred(0, result('原句。'), .2, {'route': 'speech'})
    ui.result_editors[0].setString_('')
    ui.textDidChange_(None)
    assert not ui.choice_buttons[0].enabled
    ui._inferred(0, result('原句。', '二。', '三。'), 2, 1, 25, {'route': 'speech'})
    assert ui.result_editors[0].string() == ''
    assert not ui.choice_buttons[0].enabled


def test_stale_result_never_clears_new_processing_or_replaces_edits(h):
    ui = h.ui
    editors(ui)
    ui.token = ui.processing_token = 2
    ui.processing = True
    ui._first_inferred(1, result('旧内容'), .2, {'route': 'lips'})
    ui._inferred(1, result('旧内容'), 2, 1, 25, {'route': 'lips'})
    ui._infer_error(1, '旧错误')
    assert ui.processing and ui.result_variants == [] and not ui.error


def test_native_escape_direct_cancel_selector_preserves_idle_edits(h):
    ui = h.ui
    editors(ui)
    ui._inferred(0, result('原句。', '二。', '三。'), 2, 1, 25, {'route': 'speech'})
    ui.result_editors[0].setString_('已经修正。')
    ui.cancel_(None)  # Cocoa responder chain, not the guarded global hotkey.
    assert ui.token == 0 and ui.result_editors[0].string() == '已经修正。'
    assert ui.result_variants == ['原句。', '二。', '三。']
    assert ui.audio.cancels == 0


def test_cancelled_early_result_cannot_reappear(h):
    ui = h.ui
    ui.processing = True
    ui.processing_token = 0
    ui._first_inferred(0, result('原句。'), .2, {'route': 'speech'})
    ui.cancel_(None)
    ui._inferred(0, result('原句。', '二。'), 2, 1, 25, {'route': 'speech'})
    assert not ui.result_variants and not ui.processing
    assert '取消' in ui.result.string()


def test_production_pipeline_publishes_first_before_running_wording(controller):
    ui = controller
    ui.asr.transcribe.return_value = SimpleNamespace(accepted=True, text='原句', reason='accepted')
    ui.formatter.initial.return_value = result('原句。')
    def format_text(*args, **kwargs):
        assert ui.result_variants == ['原句。']
        assert ui.processing
        assert ui.first_result_seconds is not None
        return result('原句。', '二。', '三。')
    ui.formatter.format.side_effect = format_text
    ui._infer(3, None, np.zeros(16000, np.float32))
    assert ui.result_variants == ['原句。', '二。', '三。']
    assert ui.completed == 1


def test_wording_failure_keeps_first_candidate_available(controller):
    ui = controller
    ui.asr.transcribe.return_value = SimpleNamespace(accepted=True, text='原句', reason='accepted')
    ui.formatter.initial.return_value = result('原句。')
    ui.formatter.format.side_effect = RuntimeError('model unavailable')
    ui._infer(3, None, np.zeros(16000, np.float32))
    assert ui.result_variants == ['原句。'] and not ui.processing


def test_default_fast_pipeline_never_calls_slow_formatter(controller):
    ui = controller
    del ui.wording_mode  # The shipped default also applies to older controllers.
    ui.asr.transcribe.return_value = SimpleNamespace(accepted=True, text='原句', reason='accepted')
    ui.formatter.initial.return_value = result('原句。')
    ui.formatter.format_fast = Mock(return_value=result('原句。', '二。', '三。'))
    ui._infer(3, None, np.zeros(16000, np.float32))
    ui.formatter.format.assert_not_called()
    ui.formatter.format_fast.assert_called_once_with(['原句'], source='speech', language='zh-Hans')
    assert ui.result_variants == ['原句。', '二。', '三。']
    assert not ui.processing and ui.completed == 1


def test_visual_load_time_is_reported_separately_from_decoding(controller):
    ui = controller
    ui._infer(3, recording(), np.zeros(16000, np.float32))
    assert ui.last_stage_seconds['lip_load'] >= 0
    assert ui.last_stage_seconds['lip_encoder'] >= 0
    assert ui.last_stage_seconds['lip_decoder'] >= 0


def test_old_inference_phase_cannot_replace_current_status(controller):
    ui = controller
    ui.inference_stage = '识别语音'
    ui._inference_phase(2, '准备口型模型')
    assert ui.inference_stage == '识别语音'


def test_pure_lips_never_requests_microphone_or_starts_audio(h):
    ui = h.ui
    ui.input_mode = 'lip-only'
    ui.lip_ready = True
    ui._request_microphone()
    ui.requestMicrophone_(None)
    h.begin_ready()
    assert not h.permission_requests
    assert ui.audio.starts == 0 and not ui.audio.capturing
    assert ui.recording and h.cameras[0].start_count == 1


def test_pure_lips_never_calls_asr_even_if_audio_supplied(controller):
    ui = controller
    ui.input_mode = 'lip-only'
    ui._infer(3, recording(), np.ones(16000, np.float32))
    ui.asr.transcribe.assert_not_called()
    ui.reader.encode.assert_called_once()
    assert ui.last_route == 'lips'


def test_input_mode_ignores_old_speech_preference_and_session_choices_never_persist(tmp_path):
    assert DEFAULT_INPUT_MODE == load_input_mode(tmp_path) == 'lip-only'
    previous = '{"input_mode":"speech-first"}\n'
    path = tmp_path / 'input-settings.json'
    path.write_text(previous)
    for mode in ('speech-first', 'lip-only'):
        assert load_input_mode(tmp_path) == 'lip-only'
        save_input_mode(tmp_path, mode)
        assert path.read_text() == previous
    path.write_text('corrupt')
    assert load_input_mode(tmp_path) == 'lip-only'
    with pytest.raises(ValueError):
        save_input_mode(tmp_path, 'automatic')


def test_compatibility_mode_hooks_do_not_access_profile_or_create_files(tmp_path):
    profile = Mock()
    profile.__truediv__ = Mock(side_effect=AssertionError('preferences must not be read'))
    profile.mkdir.side_effect = AssertionError('preferences must not be written')
    assert load_input_mode(profile) == 'lip-only'
    save_input_mode(profile, 'speech-first')
    missing = tmp_path / 'missing-profile'
    save_input_mode(missing, 'speech-first')
    assert not missing.exists()
