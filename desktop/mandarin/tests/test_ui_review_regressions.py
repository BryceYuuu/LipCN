"""Concrete mode/panel regressions: no Cocoa, models, devices or file writes."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from candidate_panel import CandidatePanel
from test_hybrid_ui import h, Widget


def panel(ui):
    editors = [Widget() for _ in range(3)]
    managers = [SimpleNamespace(removeAllActions=Mock()) for _ in editors]
    for editor, manager in zip(editors, managers):
        editor.undoManager = lambda manager=manager: manager
    cards = [Widget() for _ in range(6)]
    ui.result = CandidatePanel(Widget(), Widget(), cards, editors)
    ui.result_editors = editors
    ui.choice_buttons = [Widget() for _ in editors]
    for index, button in enumerate(ui.choice_buttons):
        button.tag = lambda index=index: index
    ui.output_target.copy = Mock()
    ui.output_target.insert = Mock()
    return managers


def variants(*texts):
    return SimpleNamespace(variants=[SimpleNamespace(text=text) for text in texts])


def select_mode(ui, mode):
    ui.mode_popup.selectItemAtIndex_(0 if mode == 'speech-first' else 1)
    ui.changeInputMode_(ui.mode_popup)


def test_ready_speech_mode_clears_prior_visual_failure_without_loading_or_recording(h):
    ui = h.ui
    ui.input_mode = 'lip-only'
    ui.speech_ready = True
    ui.lip_ready = False
    ui.reader = None
    ui.model_failed = True
    ui.error = '识别模型加载失败'
    ui.result.setString_(ui.error)
    select_mode(ui, 'speech-first')
    assert not ui.model_failed and ui.error is None and ui.result.string() == ''
    assert ui._can_begin() and ui.start_button.enabled
    ui.tick_(None)
    assert ui.state_label.string() == '就绪'
    assert not h.executor.tasks and not h.cameras and ui.audio.starts == 0


def test_ready_mode_switch_preserves_existing_candidate_edits(h):
    ui = h.ui
    panel(ui)
    ui.speech_ready = ui.lip_ready = True
    ui._first_inferred(0, variants('原句'), .1, {'route': 'speech'})
    ui.result_editors[0].setString_('用户改过的句子')
    ui.error = '其他方案暂不可用'
    select_mode(ui, 'lip-only')
    assert ui.error is None and ui._can_begin()
    assert ui.result_variants == ['原句']
    assert ui.result_editors[0].string() == '用户改过的句子'
    ui.copyChoice_(ui.choice_buttons[0])
    ui.output_target.copy.assert_called_once_with('用户改过的句子')


def test_ready_speech_without_microphone_cannot_mask_missing_visual_model(h):
    ui = h.ui
    ui.input_mode = 'lip-only'
    ui.speech_ready = True
    ui.lip_ready = False
    ui.reader = None
    h.microphone_permission = 2
    select_mode(ui, 'speech-first')
    assert not ui.microphone_allowed and not ui._can_begin()
    assert ui.loading and h.executor.tasks[-1]['target'] == ui._load_model
    assert ui.audio.starts == 0


def test_switch_to_missing_visual_model_cannot_keep_speech_mode_available(h):
    ui = h.ui
    ui.speech_ready = True
    ui.lip_ready = False
    ui.reader = None
    select_mode(ui, 'lip-only')
    assert not ui._uses_speech() and not ui._can_begin()
    assert ui.loading and h.executor.tasks[-1]['target'] == ui._load_model
    assert not h.permission_requests and ui.audio.starts == 0


@pytest.mark.parametrize('language,label', [('zh-Hans', '纯口型模式'), ('zh-Hant', '纯口型模式'),
                                           ('en', 'English 模式')])
def test_visual_capture_error_names_the_selected_mode(h, language, label):
    ui = h.ui
    ui.language = language
    ui.reader_language = 'en' if language == 'en' else 'zh'
    ui.input_mode = 'lip-only'
    ui.lip_ready = True
    h.camera_permission = 2
    ui.begin_(None)
    assert not ui.recording and label in ui.result.string()
    assert ui.audio.starts == 0 and not h.permission_requests


def test_cancel_clears_real_panel_editors_undo_and_blocks_stale_copy_and_callback(h):
    ui = h.ui
    managers = panel(ui)
    ui.processing = True
    ui.processing_token = 0
    ui._first_inferred(0, variants('旧句'), .1, {'route': 'speech'})
    ui.result_editors[0].setString_('旧句的修改')
    ui.cancel_(None)
    assert all(editor.string() == '' and not editor.editable for editor in ui.result_editors)
    assert all(card.hidden for card in ui.result.cards)
    assert all(manager.removeAllActions.call_count == 1 for manager in managers)
    for button in ui.choice_buttons:
        assert not button.enabled
        ui.copyChoice_(button)
        ui.insertChoice_(button)
    ui._first_inferred(0, variants('过期'), .2, {'route': 'speech'})
    ui._inferred(0, variants('过期', '二', '三'), 2, 1, 25, {'route': 'speech'})
    ui.output_target.copy.assert_not_called()
    ui.output_target.insert.assert_not_called()
    assert ui.result_variants == [] and '取消' in ui.result.string()
    assert all(editor.string() == '' for editor in ui.result_editors)


def test_starting_next_utterance_clears_real_editors_and_disables_previous_output(h):
    ui = h.ui
    panel(ui)
    ui._inferred(0, variants('旧一', '旧二', '旧三'), 2, 1, 25, {'route': 'speech'})
    assert all(button.enabled for button in ui.choice_buttons)
    ui.begin_(None)
    assert ui.result_variants == []
    assert all(editor.string() == '' and not editor.editable for editor in ui.result_editors)
    assert not any(button.enabled for button in ui.choice_buttons)
    for button in ui.choice_buttons:
        ui.copyChoice_(button)
    ui.output_target.copy.assert_not_called()
