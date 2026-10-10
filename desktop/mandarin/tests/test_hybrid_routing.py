"""Exercise the shipped controller methods without importing AppKit or a model.

Methods are AST-extracted from live_test.py on every fixture creation, so these
tests cover the production routing bodies and actual status serializer.
"""
import ast
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from languages import DEFAULT_LANGUAGE, recognition_language

SOURCE = Path(__file__).resolve().parents[1] / 'live_test.py'
PRIVATE_TEXT = '这是只在内存中显示的私人测试句子'


class Widget:
    def __init__(self):
        self.text = ''
        self.enabled = False

    def setString_(self, value):
        self.text = value

    def setEnabled_(self, value):
        self.enabled = value


@pytest.fixture
def controller(tmp_path, monkeypatch):
    wanted = {'_infer', '_inferred', '_infer_error', '_discarded', '_status'}
    parsed = ast.parse(SOURCE.read_text(encoding='utf-8'))
    cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == 'LiveTest')
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {method.name for method in methods} == wanted
    for method in methods:
        method.decorator_list = []
    env = {'time': time, 'np': np, 'json': json, 'os': os,
           'DEFAULT_LANGUAGE': DEFAULT_LANGUAGE, 'recognition_language': recognition_language,
           'STATUS': tmp_path / 'status.json', 'TAIL_SECONDS': 0.4, 'MAX_SECONDS': 20,
           'AppHelper': SimpleNamespace(callAfter=lambda fn, *args: fn(*args)),
           'clip_problem': Mock(return_value=None),
           'rois_for': Mock(return_value=np.zeros((15, 88, 88), np.float32))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
                 str(SOURCE), 'exec'), env)
    harness = type('Harness', (), {name: env[name] for name in wanted})
    obj = harness()
    obj.token = 3
    obj.closed = False
    obj.loading = False
    obj.processing = True
    obj.recording = False
    obj.completed = 0
    obj.last_latency = None
    obj.last_route = None
    obj.last_status_write = None
    obj.error = None
    obj.result = Widget()
    obj.cancel_button = Widget()
    obj.choice_buttons = []
    obj.reader = SimpleNamespace(encode=Mock(return_value='encoded'))
    obj.model_seconds = 1.0
    obj.adapter_sha256 = '0' * 64
    obj.camera = None
    obj.frames = 0
    obj.face_present = False
    obj.microphone_allowed = True
    obj.audio = SimpleNamespace(capturing=False)
    obj.hotkey = None
    obj.window = SimpleNamespace(isVisible=lambda: True, isMiniaturized=lambda: False)
    obj._can_begin = lambda: False
    obj.asr = SimpleNamespace(transcribe=Mock(return_value=SimpleNamespace(
        accepted=False, text='', reason='low_confidence')))
    obj.formatter = SimpleNamespace(format=Mock(return_value=SimpleNamespace(
        raw=PRIVATE_TEXT, variants=[SimpleNamespace(label='原意', text=PRIVATE_TEXT + '。', note='')], warnings=[])))
    backend = ModuleType('evaluate_cnvsrc')
    backend.configured_hypotheses = Mock(return_value=([SimpleNamespace(text='口型原始内容')], None))
    monkeypatch.setitem(sys.modules, 'evaluate_cnvsrc', backend)
    obj.env = env
    obj.backend = backend
    obj.status_path = env['STATUS']
    return obj


def recording():
    return SimpleNamespace(duration=1.0, ts=list(np.linspace(0, 1, 15)),
                           mouth_pixels=[40] * 15, face_ratio=1.0)


@pytest.mark.parametrize('rec', [None, recording()])
def test_accepted_audio_skips_lip_quality_and_model_even_without_camera(controller, rec):
    controller.asr.transcribe.return_value = SimpleNamespace(accepted=True, text=PRIVATE_TEXT, reason='accepted')
    controller.env['clip_problem'].return_value = ("Can't see your face", '')
    controller._infer(3, rec, np.ones(16000, np.float32) * 0.1)
    controller.env['clip_problem'].assert_not_called()
    controller.env['rois_for'].assert_not_called()
    controller.reader.encode.assert_not_called()
    controller.backend.configured_hypotheses.assert_not_called()
    controller.formatter.format.assert_called_once_with([PRIVATE_TEXT], source='speech', language='zh-Hans')
    assert controller.completed == 1 and controller.last_route == 'speech'


@pytest.mark.parametrize('reason,audio', [('low_confidence', np.ones(16000, np.float32) * 0.1),
                                        ('no_audio', np.empty(0, np.float32))])
def test_rejected_or_empty_audio_falls_back_to_lips(controller, reason, audio):
    controller.asr.transcribe.return_value = SimpleNamespace(accepted=False, text='', reason=reason)
    controller._infer(3, recording(), audio)
    controller.env['clip_problem'].assert_called_once()
    controller.reader.encode.assert_called_once()
    controller.backend.configured_hypotheses.assert_called_once()
    controller.formatter.format.assert_called_once_with(['口型原始内容'], source='lips', language='zh-Hans')
    assert controller.completed == 1 and controller.last_route == 'lips'


@pytest.mark.parametrize('rec,problem', [(None, None), (recording(), ('No lip movement', ''))])
def test_rejected_audio_and_bad_lips_fail_without_formatting(controller, rec, problem):
    controller.env['clip_problem'].return_value = problem
    controller._infer(3, rec, np.empty(0, np.float32))
    controller.reader.encode.assert_not_called()
    controller.formatter.format.assert_not_called()
    assert controller.completed == 0 and controller.error
    assert '本句识别失败' in controller.result.text
    assert json.loads(controller.status_path.read_text(encoding='utf-8'))['state'] == 'inference_error'


def test_cancelled_before_start_skips_every_model(controller):
    controller.token = 4
    controller._infer(3, recording(), np.zeros(16000, np.float32))
    controller.asr.transcribe.assert_not_called()
    controller.reader.encode.assert_not_called()
    controller.formatter.format.assert_not_called()
    assert not controller.status_path.exists()


def test_cancelled_during_asr_never_starts_vsr_or_formatter(controller):
    def cancel(audio):
        controller.token += 1
        return SimpleNamespace(accepted=False, text='', reason='low_confidence')
    controller.asr.transcribe.side_effect = cancel
    controller._infer(3, recording(), np.zeros(16000, np.float32))
    controller.reader.encode.assert_not_called()
    controller.formatter.format.assert_not_called()
    assert controller.completed == 0 and not controller.status_path.exists()


def test_cancelled_while_building_rois_never_starts_vsr_or_formatter(controller):
    def cancel(rec):
        controller.token += 1
        return np.zeros((15, 88, 88), np.float32)
    controller.env['rois_for'].side_effect = cancel
    controller._infer(3, recording(), np.zeros(16000, np.float32))
    controller.reader.encode.assert_not_called()
    controller.backend.configured_hypotheses.assert_not_called()
    controller.formatter.format.assert_not_called()


def test_cancelled_after_vsr_does_not_format_or_publish(controller):
    def cancel(*args, **kwargs):
        controller.token += 1
        return [SimpleNamespace(text=PRIVATE_TEXT)], None
    controller.backend.configured_hypotheses.side_effect = cancel
    controller._infer(3, recording(), np.zeros(16000, np.float32))
    controller.formatter.format.assert_not_called()
    assert controller.completed == 0 and not controller.status_path.exists()


def test_status_on_success_contains_no_private_text_or_samples(controller, tmp_path):
    controller.asr.transcribe.return_value = SimpleNamespace(accepted=True, text=PRIVATE_TEXT, reason='accepted')
    controller._infer(3, None, np.ones(16000, np.float32) * 0.1)
    assert PRIVATE_TEXT in controller.result.text  # text is allowed in live UI memory
    files = list(tmp_path.iterdir())
    assert files == [controller.status_path]
    payload = controller.status_path.read_text(encoding='utf-8')
    assert PRIVATE_TEXT not in payload and '原始识别' not in payload
    metadata = json.loads(payload)
    assert not metadata['audio_persisted'] and not metadata['frames_persisted']
    assert not metadata['transcripts_persisted']
    assert metadata['last_route'] == 'speech'


def test_status_on_error_does_not_serialize_exception_text(controller):
    controller.asr.transcribe.side_effect = ValueError(PRIVATE_TEXT)
    controller._infer(3, None, np.empty(0, np.float32))
    assert PRIVATE_TEXT not in controller.status_path.read_text(encoding='utf-8')
    assert json.loads(controller.status_path.read_text(encoding='utf-8'))['error'] is True


def test_result_shows_only_three_choices_without_changing_candidate_text(controller):
    texts = ['你能听到我的声音吗？', '能听见我说话吗？', '听得见吗？']
    formatted = SimpleNamespace(
        raw='原始识别',
        variants=[SimpleNamespace(text=text, label='内部标签', note='内部说明') for text in texts],
        warnings=['内部警告'])
    controller._inferred(3, formatted, 2.0, 4.0, 100, {'route': 'lips', 'fps': 25})
    assert controller.result.text == '\n\n'.join(
        f'方案{number}：{text}' for number, text in zip('一二三', texts))
    assert controller.result_variants == texts
    assert controller.completed == 1
