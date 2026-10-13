"""Decoder-to-editor regressions, using real candidate presentation and fake models."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hybrid_candidates import CandidateFormatter
from languages import recognition_language
from test_hybrid_routing import controller, recording
from test_hybrid_ui import h
from test_ui_review_regressions import panel, variants


def visual_controller(ui, language, beams, *, wording_mode='model'):
    ui.language = language
    ui.reader_language = recognition_language(language)
    ui.input_mode = 'lip-only'
    ui.wording_mode = wording_mode
    hypotheses = [SimpleNamespace(text=text) for text in beams]
    ui.reader.hypotheses = Mock(return_value=hypotheses)
    ui.backend.configured_hypotheses.return_value = (hypotheses, None)
    ui.formatter = CandidateFormatter()
    ui.formatter.recognition_choices = Mock(wraps=ui.formatter.recognition_choices)
    for name in ('initial', 'format', 'format_fast'):
        setattr(ui.formatter, name, Mock(side_effect=AssertionError(
            'visual hypotheses must bypass all single-recognition wording paths')))
    ui.output_target = SimpleNamespace(copy=Mock(), insert=Mock())
    panel(ui)
    ui._refresh_choices()  # Match begin_: no output is enabled before any result.
    return ui


def run_visual(ui):
    ui._infer(ui.token, recording(), np.zeros(16000, np.float32))


@pytest.mark.parametrize('wording_mode', ['fast', 'model'])
@pytest.mark.parametrize('language,beams,expected', [
    ('zh-Hans', ['我明天開會', '我明天不開會', '我後天開會', '我昨天開會'],
     ['我明天开会。', '我明天不开会。', '我后天开会。']),
    ('zh-Hant', ['我明天开会', '我明天不开会', '我后天开会', '我昨天开会'],
     ['我明天開會。', '我明天不開會。', '我後天開會。']),
    ('en', ['I CAN PAY 100', 'I CANNOT PAY 100', 'I CAN PAY 200', 'I CAN PAY 300'],
     ['I CAN PAY 100', 'I CANNOT PAY 100', 'I CAN PAY 200']),
])
def test_visual_decoder_candidates_reach_three_editors_without_rewriting(
        controller, wording_mode, language, beams, expected):
    ui = visual_controller(controller, language, beams, wording_mode=wording_mode)
    run_visual(ui)
    assert ui.result_variants == expected
    assert [editor.string() for editor in ui.result_editors] == expected
    assert all(editor.editable for editor in ui.result_editors)
    assert all(button.enabled for button in ui.choice_buttons)
    assert ui.completed == 1 and not ui.processing and ui.last_route == 'lips'
    ui.asr.transcribe.assert_not_called()
    ui.reader.encode.assert_called_once()
    ui.formatter.recognition_choices.assert_called_once_with(beams, language=language)
    for method in (ui.formatter.initial, ui.formatter.format, ui.formatter.format_fast):
        method.assert_not_called()
    if language == 'en':
        ui.reader.hypotheses.assert_called_once_with('encoded', nbest=4)
        ui.backend.configured_hypotheses.assert_not_called()
    else:
        ui.backend.configured_hypotheses.assert_called_once_with(
            ui.reader, 'encoded', beam_size=40, ctc_weight=.1, nbest=10,
            reverse_weight=.3, reverse_scoring='batched', candidate_pool_size=10,
            pre_beam_ratio=1.5)
        ui.reader.hypotheses.assert_not_called()


def test_chinese_existing_pool_supplies_distinct_candidates_after_duplicate_top_beams(controller):
    beams = ['我明天去', '我明天去。', '我 明 天 去', '我后天去', '我后天去。',
             '我昨天去', '我昨天去！', '我今年去', '我明年去', '我不去']
    ui = visual_controller(controller, 'zh-Hans', beams)
    run_visual(ui)
    assert ui.result_variants == ['我明天去。', '我后天去。', '我昨天去。']
    assert [editor.string() for editor in ui.result_editors] == ui.result_variants
    assert ui.backend.configured_hypotheses.call_count == ui.reader.encode.call_count == 1


@pytest.mark.parametrize('beams,expected', [
    (['我不去', '我不去。', '我 不 去'], ['我不去。']),
    (['我不去', '我不去。', '我去'], ['我不去。', '我去。']),
])
def test_duplicate_or_short_decoder_output_leaves_empty_disabled_slots(controller, beams, expected):
    ui = visual_controller(controller, 'zh-Hans', beams)
    # Recycled view state must not leak an old alternative into a missing slot.
    ui._present_variants(variants('旧一', '旧二', '旧三').variants)
    for editor in ui.result_editors:
        editor.setString_(editor.string() + '旧修改')
    ui.result_variants = []  # A new utterance resets the candidate identities.
    run_visual(ui)
    count = len(expected)
    assert ui.result_variants == expected
    assert [editor.string() for editor in ui.result_editors] == expected + [''] * (3 - count)
    assert [button.enabled for button in ui.choice_buttons] == [True] * count + [False] * (3 - count)
    assert [editor.editable for editor in ui.result_editors] == [True] * count + [False] * (3 - count)
    assert [ui._choice_text(index) for index in range(count, 3)] == [''] * (3 - count)


@pytest.mark.parametrize('language,beams,expected', [
    ('zh-Hans', ['。！？', '軟件不會改', '软件会改'], ['软件不会改。', '软件会改。']),
    ('zh-Hant', ['。！？', '软件不会改', '軟件會改'], ['軟件不會改。', '軟件會改。']),
    ('en', ['...', 'ALICE WILL NOT PAY', 'ALICE WILL PAY'], ['ALICE WILL NOT PAY', 'ALICE WILL PAY']),
])
def test_validated_first_candidate_is_queued_before_final_and_user_edit_survives(
        controller, language, beams, expected):
    ui = visual_controller(controller, language, beams)
    notifications = []
    ui.env['AppHelper'] = SimpleNamespace(callAfter=lambda callback, *args:
                                         notifications.append((callback, args)))
    run_visual(ui)
    names = [callback.__name__ for callback, _ in notifications]
    first_index, final_index = names.index('_first_inferred'), names.index('_inferred')
    assert first_index < final_index
    assert ui.result_variants == []  # Queued callbacks have not touched the editors.
    callback, args = notifications[first_index]
    assert [v.text for v in args[1].variants] == expected[:1]
    assert args[1].variants[0].label == '方案一'
    callback(*args)
    assert ui.result_variants == expected[:1] and ui.processing
    assert [b.enabled for b in ui.choice_buttons] == [True, False, False]
    ui.result_editors[0].setString_('user corrected the first candidate')
    callback, args = notifications[final_index]
    callback(*args)
    assert ui.result_variants == expected and not ui.processing
    assert ui.result_editors[0].string() == 'user corrected the first candidate'
    assert ui._choice_text(0) == 'user corrected the first candidate'
    assert [e.string() for e in ui.result_editors[1:]] == [expected[1], '']
    assert [b.enabled for b in ui.choice_buttons] == [True, True, False]


@pytest.mark.parametrize('beams', [[], ['', ' '], ['。！？', '...']])
def test_no_meaningful_beam_does_not_publish_a_placeholder_or_duplicate(controller, beams):
    ui = visual_controller(controller, 'zh-Hans', beams)
    ui._first_inferred = Mock(wraps=ui._first_inferred)
    ui._inferred = Mock(wraps=ui._inferred)
    run_visual(ui)
    ui._first_inferred.assert_not_called()
    ui._inferred.assert_not_called()
    assert not ui.result_variants and ui.completed == 0 and not ui.processing
    assert '没有识别出可用文字' in ui.result.string()
    assert not any(button.enabled for button in ui.choice_buttons)


@pytest.mark.parametrize('count', [0, 1, 2])
def test_fewer_final_candidates_clear_old_editors_and_disable_copy_insert(h, count):
    ui = h.ui
    panel(ui)
    ui._present_variants(variants('旧一', '旧二', '旧三').variants)
    ui.result_editors[2].setString_('旧的第三项修改')
    current = ['新一', '新二'][:count]
    ui._inferred(ui.token, variants(*current), 1, 1, 25, {'route': 'lips'})
    assert ui.result_variants == current
    for index in range(count, 3):
        assert ui.result_editors[index].string() == '' and not ui.result_editors[index].editable
        assert not ui.choice_buttons[index].enabled
        ui.copyChoice_(ui.choice_buttons[index])
        ui.insertChoice_(ui.choice_buttons[index])
    ui.output_target.copy.assert_not_called()
    ui.output_target.insert.assert_not_called()


def test_cancel_after_first_visual_candidate_blocks_queued_beams_and_old_copy(h):
    ui = h.ui
    panel(ui)
    formatted = CandidateFormatter().recognition_choices(['我不去', '我去', '你去'])
    ui.processing = True
    ui.processing_token = ui.token
    old_token = ui.token
    ui._first_inferred(old_token, variants(formatted.variants[0].text), .1, {'route': 'lips'})
    ui.result_editors[0].setString_('取消前的修改')
    ui.cancel_(None)
    ui._first_inferred(old_token, formatted, .2, {'route': 'lips'})
    ui._inferred(old_token, formatted, 1, 1, 25, {'route': 'lips'})
    assert not ui.result_variants and not ui.processing and '取消' in ui.result.string()
    assert all(editor.string() == '' and not editor.editable for editor in ui.result_editors)
    for button in ui.choice_buttons:
        assert not button.enabled
        ui.copyChoice_(button)
        ui.insertChoice_(button)
    ui.output_target.copy.assert_not_called()
    ui.output_target.insert.assert_not_called()


def test_cancellation_during_candidate_preparation_cannot_publish_visual_results(controller):
    ui = visual_controller(controller, 'en', ['I WILL GO', 'I WILL NOT GO'])
    prepare = ui.formatter.recognition_choices
    def cancel_then_prepare(*args, **kwargs):
        ui.token += 1
        return prepare(*args, **kwargs)
    ui.formatter.recognition_choices = Mock(side_effect=cancel_then_prepare)
    ui._infer(3, recording(), np.zeros(16000, np.float32))
    assert ui.result_variants == [] and ui.completed == 0 and not ui.processing
    assert not ui.status_path.exists()
    assert not any(button.enabled for button in ui.choice_buttons)
