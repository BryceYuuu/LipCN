"""Language changes exercise shipped controller code without devices or models."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from test_hybrid_ui import Harness, Widget
from test_hybrid_routing import controller, recording
from languages import DEFAULT_LANGUAGE, LANGUAGE_CHOICES


@pytest.fixture
def h(monkeypatch):
    harness = Harness(monkeypatch)
    ui = harness.ui
    ui.lip_ready = ui.speech_ready = ui.startup_complete = True
    ui.warmup_seconds = {}
    ui.reader = SimpleNamespace(enc_device=SimpleNamespace(type='cpu'))
    ui.asr = SimpleNamespace()
    ui._preflight_visual = lambda language: None
    harness.globals['ENGLISH_MODELS'] = 'local-english-models'
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        set_num_threads=lambda value: None,
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        mps=SimpleNamespace(empty_cache=lambda: None, synchronize=lambda: None)))
    return harness


def choose(h, language):
    h.ui.language_popup.selectItemAtIndex_(
        next(i for i, (code, _) in enumerate(LANGUAGE_CHOICES) if code == language))
    h.ui.changeLanguage_(h.ui.language_popup)


def finish_load(h):
    task = h.executor.tasks[-1]
    task['target'](*task['args'])
    task['future'].set_result(None)
    h.flush_after()


def english_ready(h):
    reader = SimpleNamespace(enc_device=SimpleNamespace(type='cpu'))
    h.globals['load_english_reader'] = lambda path: reader
    choose(h, 'en')
    finish_load(h)
    return reader


def test_default_and_script_change_reuse_visual_model_and_clear_old_choices(h):
    assert h.ui.language == DEFAULT_LANGUAGE == 'zh-Hans'
    original = h.ui.reader
    h.ui.result_variants = ['旧的简体内容']
    h.ui.choice_buttons = [Widget()]
    choose(h, 'zh-Hant')
    assert h.ui.language == 'zh-Hant' and h.ui.reader is original
    assert not h.executor.tasks and h.ui._can_begin()
    assert not h.ui.result_variants and not h.ui.choice_buttons[0].enabled
    choose(h, 'zh-Hans')
    assert h.ui.reader is original and not h.executor.tasks
    assert h.ui.audio.starts == 0 and not h.cameras


@pytest.mark.parametrize('busy', ['loading', 'pending_begin', 'recording', 'processing', 'language_switching'])
def test_language_cannot_change_during_capture_inference_or_model_load(h, busy):
    setattr(h.ui, busy, True)
    token = h.ui.token
    choose(h, 'en')
    assert h.ui.language == 'zh-Hans' and h.ui.language_popup.selected_index == 0
    assert h.ui.token == token and not h.executor.tasks
    h.ui._refresh_language_control()
    assert not h.ui.language_popup.enabled


def test_english_load_is_queued_on_model_executor_and_releases_chinese_first(h):
    reader = SimpleNamespace(enc_device=SimpleNamespace(type='cpu'))
    original = h.ui.reader
    checked = []
    def preflight(language):
        assert h.ui.reader is original
        checked.append(language)
    h.ui._preflight_visual = preflight
    def load(path):
        assert path == 'local-english-models'
        assert h.ui.reader is None and checked == ['en']
        return reader
    h.globals['load_english_reader'] = load
    choose(h, 'en')
    assert len(h.executor.tasks) == 1
    assert h.executor.tasks[0]['target'] == h.ui._load_language_model
    assert h.ui.language_switching and not h.ui._can_begin()
    assert not h.ui.language_popup.enabled and not h.ui.start_button.enabled
    finish_load(h)
    assert h.ui.reader is reader and h.ui.reader_language == 'en'
    assert h.ui._can_begin() and h.ui.language_popup.enabled
    assert not h.cameras and h.ui.audio.starts == 0


def test_english_capture_uses_camera_without_opening_microphone(h):
    english_ready(h)
    assert h.ui.microphone_allowed
    h.begin_ready()
    assert h.ui.recording and len(h.cameras) == 1
    assert h.ui.audio.starts == 0 and not h.ui.audio.capturing


def test_english_does_not_fall_back_to_chinese_speech_without_camera(h):
    english_ready(h)
    h.camera_permission = 2
    h.ui.begin_(None)
    assert not h.ui.recording and h.ui.audio.starts == 0
    assert 'English' in h.ui.result.text and '摄像头' in h.ui.result.text


def test_missing_target_assets_keep_original_reader_and_roll_back_language(h):
    original = h.ui.reader
    h.ui._preflight_visual = Mock(side_effect=FileNotFoundError('/private/model/path'))
    load = h.globals['load_english_reader'] = Mock()
    choose(h, 'en')
    finish_load(h)
    load.assert_not_called()
    assert h.ui.reader is original and h.ui.lip_ready
    assert h.ui.language == 'zh-Hans' and h.ui.language_popup.selected_index == 0
    assert not h.ui.model_failed and h.ui._can_begin()
    assert h.ui.language_popup.enabled and not h.ui.language_switching
    assert '英文口型模型' in h.ui.result.text and '/private' not in h.ui.result.text


def test_native_target_failure_reloads_previous_reader_and_rolls_back(h):
    h.ui.input_mode = 'lip-only'
    original = h.ui.reader
    restored = SimpleNamespace(enc_device=SimpleNamespace(type='cpu'))
    calls = []
    def create(language):
        assert h.ui.reader is None
        calls.append(language)
        if language == 'en':
            raise RuntimeError('/private/model/path')
        return restored, 'restored-digest'
    h.ui._create_visual = create
    choose(h, 'en')
    finish_load(h)
    assert calls == ['en', 'zh']
    assert h.ui.reader is restored and h.ui.reader is not original
    assert h.ui.reader_language == 'zh' and h.ui.language == 'zh-Hans'
    assert h.ui.lip_ready and not h.ui.model_failed and h.ui._can_begin()
    assert '/private' not in h.ui.result.text


def test_failed_target_and_restore_leave_previous_selection_unavailable_and_retryable(h):
    h.ui.input_mode = 'lip-only'
    h.ui._create_visual = Mock(side_effect=RuntimeError('model load failed'))
    choose(h, 'en')
    finish_load(h)
    assert h.ui._create_visual.call_args_list == [(('en',),), (('zh',),)]
    assert h.ui.reader is None and not h.ui.lip_ready
    assert h.ui.language == 'zh-Hans' and h.ui.model_failed and not h.ui._can_begin()
    replacement = SimpleNamespace(enc_device=SimpleNamespace(type='cpu'))
    h.ui._create_visual = Mock(return_value=(replacement, 'new-digest'))
    choose(h, 'zh-Hans')
    finish_load(h)
    assert h.ui.reader is replacement and h.ui._can_begin() and not h.ui.model_failed
    assert h.ui.error is None and h.ui.result.text == ''


def test_switch_back_to_chinese_uses_ready_speech_without_queuing_visual_work(h):
    english = english_ready(h)
    task_count = len(h.executor.tasks)
    choose(h, 'zh-Hant')
    assert len(h.executor.tasks) == task_count
    assert h.ui.reader is english and h.ui.reader_language == 'en'
    assert not h.ui.lip_ready and h.ui.speech_ready and h.ui._can_begin()
    assert h.ui.language == 'zh-Hant'


def test_lip_only_switch_back_to_chinese_loads_chinese_reader_and_adapter(h, monkeypatch):
    english_ready(h)
    h.ui.input_mode = 'lip-only'
    reader = SimpleNamespace(enc_device=SimpleNamespace(type='cpu'), encode=Mock())
    def load(*args):
        assert h.ui.reader is None
        return reader
    backend = SimpleNamespace(_reader=Mock(side_effect=load),
                              _load_adapter=Mock(return_value={'file_sha256': 'digest'}))
    monkeypatch.setitem(sys.modules, 'evaluate_cnvsrc', backend)
    h.globals.update(CHECKPOINT='checkpoint', SOURCE='source', ADAPTER='adapter', np=np)
    choose(h, 'zh-Hant')
    finish_load(h)
    assert h.ui.reader is reader and h.ui.reader_language == 'zh'
    assert h.ui.language == 'zh-Hant' and h.ui._can_begin()
    backend._reader.assert_called_once_with('checkpoint', 'source', 40, .1, 'mps')
    backend._load_adapter.assert_called_once_with(reader, 'adapter')
    assert reader.encode.call_args.args[0].shape == (25, 96, 96)


def test_matching_visual_reader_is_reused_without_preflight_or_native_load(h):
    original = h.ui.reader
    h.ui._preflight_visual = Mock(side_effect=AssertionError('no file scan for reuse'))
    h.ui._create_visual = Mock(side_effect=AssertionError('no repeated load'))
    assert h.ui._ensure_visual('zh') is original
    h.ui._preflight_visual.assert_not_called()
    h.ui._create_visual.assert_not_called()
    assert h.ui.lip_ready


def test_close_during_preflight_keeps_old_reader_for_owning_worker_cleanup(h):
    original = h.ui.reader
    h.ui._preflight_visual = lambda language: setattr(h.ui, 'closed', True)
    h.ui._create_visual = Mock()
    with pytest.raises(RuntimeError, match='closing'):
        h.ui._ensure_visual('en')
    assert h.ui.reader is original
    h.ui._create_visual.assert_not_called()


@pytest.mark.parametrize('nested', [False, True])
def test_chinese_preflight_checks_only_model_and_source_files(h, tmp_path, monkeypatch, nested):
    source = tmp_path / 'source'
    vsr = source / 'VSR' if nested else source
    files = [tmp_path / 'checkpoint', tmp_path / 'adapter',
             vsr / 'conf/train_cncvs_2_3_cnvsrc.yaml', vsr / 'datamodule/char_units.txt']
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('asset')
    h.globals.update(CHECKPOINT=files[0], ADAPTER=files[1], SOURCE=source)
    monkeypatch.setitem(sys.modules, 'torch', None)
    monkeypatch.setitem(sys.modules, 'evaluate_cnvsrc', None)
    del h.ui._preflight_visual
    h.ui._preflight_visual('zh')
    files[-1].write_text('')
    with pytest.raises(FileNotFoundError):
        h.ui._preflight_visual('zh')


def test_english_preflight_needs_only_target_files_and_no_native_imports(tmp_path, monkeypatch):
    from english_lips import preflight_english_assets
    monkeypatch.setitem(sys.modules, 'lipflow.vsr', None)
    monkeypatch.setitem(sys.modules, 'torch', None)
    for relative in ('vsr/model.json', 'vsr/model.pth', 'lm/model.json', 'lm/model.pth'):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('asset')
    assert preflight_english_assets(tmp_path) == tmp_path
    (tmp_path / 'lm/model.pth').unlink()
    with pytest.raises(FileNotFoundError, match='lm/model.pth'):
        preflight_english_assets(tmp_path)


def test_hiding_during_load_does_not_leave_selected_language_permanently_unavailable(h):
    reader = SimpleNamespace(enc_device=SimpleNamespace(type='cpu'))
    h.globals['load_english_reader'] = lambda path: reader
    choose(h, 'en')
    h.ui._suspend()
    finish_load(h)
    assert h.ui.suspended and h.ui.lip_ready and not h.ui._can_begin()
    h.ui._resume()
    assert h.ui._can_begin() and h.ui.reader is reader
    assert not h.cameras and h.ui.audio.starts == 0


def test_close_before_model_switch_runs_skips_native_load(h):
    load = h.globals['load_english_reader'] = Mock()
    choose(h, 'en')
    task = h.executor.tasks[-1]
    h.ui._shutdown()
    task['target'](*task['args'])
    h.flush_after()
    load.assert_not_called()
    assert h.ui.closed and not h.ui._can_begin()


def test_english_inference_uses_english_beams_and_skips_chinese_asr_and_decoder(controller):
    controller.language = controller.reader_language = 'en'
    controller.reader.hypotheses = Mock(return_value=[SimpleNamespace(text='HELLO ALICE')])
    controller._infer(3, recording(), np.ones(16000, np.float32))
    controller.asr.transcribe.assert_not_called()
    controller.backend.configured_hypotheses.assert_not_called()
    controller.reader.hypotheses.assert_called_once_with('encoded', nbest=3)
    controller.formatter.format.assert_called_once_with(['HELLO ALICE'], source='lips', language='en')
    assert controller.completed == 1 and controller.last_route == 'lips'


def test_traditional_inference_uses_same_chinese_reader_and_formatter_with_output_script(controller):
    controller.language = 'zh-Hant'
    controller._infer(3, recording(), np.zeros(16000, np.float32))
    controller.backend.configured_hypotheses.assert_called_once()
    controller.formatter.format.assert_called_once_with(['口型原始内容'], source='lips', language='zh-Hant')


def test_wrong_visual_model_is_never_used_for_selected_language(controller):
    controller.language = 'en'
    controller.reader_language = 'zh'
    controller._infer(3, recording(), np.zeros(16000, np.float32))
    controller.reader.encode.assert_not_called()
    controller.formatter.format.assert_not_called()
    assert controller.completed == 0 and '当前语言' in controller.result.text
