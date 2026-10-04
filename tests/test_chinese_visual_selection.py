from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research' / 'scripts'))
import evaluate_chinese
from select_chinese_visual import select, main
from lipflow.confidence import Quality
from research.evaluation import Dataset, Prediction, Sample, evaluate


def report(preset, stress, raw, *, action='review', error=None):
    samples = tuple(Sample(str(i), '/local/file.mp4', '今天不要发送', f'speaker{i}', 'session',
                           'independent label', True, 'mouth_roi', True, sha256=str(i) * 64, split='dev')
                    for i in (1, 2))
    dataset = Dataset(samples, split='dev')
    result = evaluate(dataset, [Prediction(s.id, raw, 2, 1, action, error=error) for s in samples])
    result['model'] = {'name': 'CNVSRC2025', 'checkpoint_sha256': 'f' * 64,
                       'beam_size': 40, 'ctc_weight': .1, 'personal': False,
                       'visual_configuration': {'preprocessing': preset, 'stress': stress, 'seed': 0}}
    for row in result['samples']:
        row['visual_input'] = {'original_pixels_sha256': row['video_sha256'], 'shape': [50, 96, 96],
                               'input_pixels_sha256': row['video_sha256'],
                               'configuration': dict(result['model']['visual_configuration']),
                               'dtype': 'uint8',
                               'sample_seed': int.from_bytes(hashlib.sha256(row['sample_id'].encode()).digest()[:8], 'big') % (2 ** 32)}
    manifest = {'schema_version': 1, 'split': 'dev', 'samples': [
        {'id': s.id, 'reference': s.reference, 'speaker': s.speaker, 'sha256': s.sha256,
         'split': 'dev'} for s in samples]}
    return result, manifest


def arms():
    clean, manifest = report('identity', 'clean', '今天不要发送')
    stressed, _ = report('identity', 'synthetic_mild', '今天发送')
    candidate_clean, _ = report('stabilize_mild', 'clean', '今天不要发送')
    candidate_stressed, _ = report('stabilize_mild', 'synthetic_mild', '今天不要发送')
    return clean, stressed, candidate_clean, candidate_stressed, manifest


def test_selects_fewer_stressed_errors_when_clean_and_coverage_preserved():
    a, b, c, d, manifest = arms()
    result = select(a, b, [(c, d)], manifest)
    assert result['selected_preprocessing'] == 'stabilize_mild'
    assert result['selected_pooled_cer'] == 0
    assert not result['test_used_for_selection']
    assert not result['automatic_application_activation']


@pytest.mark.parametrize('partition', ['test', 'train', 'heldout'])
def test_no_test_or_training_selection(partition):
    a, b, c, d, manifest = arms()
    manifest['split'] = partition
    with pytest.raises(ValueError, match='Only complete dev'):
        select(a, b, [(c, d)], manifest)


def test_test_row_hidden_in_dev_is_rejected():
    a, b, c, d, manifest = arms()
    manifest['samples'][0]['split'] = 'test'
    with pytest.raises(ValueError, match='Only complete dev'):
        select(a, b, [(c, d)], manifest)


def test_clean_regression_cannot_be_hidden_by_stress_improvement():
    a, b, c, d, manifest = arms()
    c, _ = report('stabilize_mild', 'clean', '今天不要发')
    result = select(a, b, [(c, d)], manifest)
    assert result['selected_preprocessing'] == 'identity'
    assert not result['candidates'][0]['checks']['clean_raw_cer_not_worse']


@pytest.mark.parametrize('action,error', [('retry', None), ('review', 'model failed')])
def test_coverage_and_failures_cannot_be_hidden_by_better_raw_text(action, error):
    a, b, c, d, manifest = arms()
    d, _ = report('stabilize_mild', 'synthetic_mild', '今天不要发送', action=action, error=error)
    assert select(a, b, [(c, d)], manifest)['selected_preprocessing'] == 'identity'


def test_equal_error_count_keeps_identity():
    a, b, c, d, manifest = arms()
    d, _ = report('stabilize_mild', 'synthetic_mild', '今天发送')
    assert select(a, b, [(c, d)], manifest)['selected_preprocessing'] == 'identity'


@pytest.mark.parametrize('mutation', ['model', 'seed', 'pixels', 'stress', 'duplicate', 'reference'])
def test_comparison_requires_same_inputs_model_and_degradation(mutation):
    a, b, c, d, manifest = arms()
    if mutation == 'model':
        c['model']['ctc_weight'] = .5
    elif mutation == 'seed':
        c['model']['visual_configuration']['seed'] = 1
    elif mutation == 'pixels':
        c['samples'][0]['visual_input']['original_pixels_sha256'] = 'a' * 64
    elif mutation == 'stress':
        d['model']['visual_configuration']['stress'] = 'synthetic_moderate'
    elif mutation == 'reference':
        c['samples'][0]['reference'] = '别的内容'
    pairs = [(c, d), (deepcopy(c), deepcopy(d))] if mutation == 'duplicate' else [(c, d)]
    with pytest.raises(ValueError):
        select(a, b, pairs, manifest)


def test_visual_boundary_strips_reference_and_preserves_identity_pixels(monkeypatch):
    sample = Sample('independent-id', '/video', 'DO NOT READ THIS LABEL', 'speaker', 'session',
                    'author', True, 'mouth_roi', True)
    frames = np.tile(np.arange(96, dtype=np.uint8), (5, 96, 1))
    seen = []
    def visual(s, reader):
        seen.append(s)
        assert s.reference == ''
        return frames, .2, Quality(brightness=0, contrast=0, mouth_pixels=96)
    monkeypatch.setattr(evaluate_chinese, '_visual_input', visual)
    images, duration, quality, evidence = evaluate_chinese._prepared_visual_input(sample, object(), SimpleNamespace())
    np.testing.assert_array_equal(images, frames)
    assert sample.reference == 'DO NOT READ THIS LABEL'
    assert duration == .2 and quality.brightness == pytest.approx(47.5)
    assert evidence['original_pixels_sha256'] == evidence['input_pixels_sha256']
    assert not evidence['real_webcam_evidence']


def test_identical_model_inputs_receive_identical_synthetic_pixels(monkeypatch):
    frames = np.tile(np.arange(96, dtype=np.uint8), (5, 96, 1))
    sample = Sample('same-id', '/video', '你好', 'speaker', 'session', 'author', True, 'webcam')
    monkeypatch.setattr(evaluate_chinese, '_visual_input', lambda *args: (frames, .2, Quality()))
    args = SimpleNamespace(visual_preprocessing='identity', visual_stress='synthetic_mild', visual_seed=19)
    a = evaluate_chinese._prepared_visual_input(sample, object(), args)
    b = evaluate_chinese._prepared_visual_input(replace(sample, reference='不同标签'), object(), args)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[3] == b[3]
    assert not a[3]['real_webcam_evidence']


@pytest.mark.parametrize('mutation', ['missing', 'nonhex', 'configuration', 'seed', 'shape', 'dtype'])
def test_selection_rejects_incomplete_or_forged_successful_pixel_evidence(mutation):
    a, b, c, d, manifest = arms()
    evidence = c['samples'][0]['visual_input']
    if mutation == 'missing':
        c['samples'][0]['visual_input'] = None
    elif mutation == 'nonhex':
        evidence['original_pixels_sha256'] = 'z' * 64
    elif mutation == 'configuration':
        evidence['configuration']['preprocessing'] = 'identity'
    elif mutation == 'seed':
        evidence['sample_seed'] += 1
    elif mutation == 'shape':
        evidence['shape'] = [0, 96, 96]
    else:
        evidence['dtype'] = 'float32'
    with pytest.raises(ValueError):
        select(a, b, [(c, d)], manifest)


def test_selection_cli_cannot_overwrite_input_evidence(tmp_path):
    path = tmp_path / 'manifest.json'; path.write_text('preserve evidence')
    with pytest.raises(SystemExit) as exc:
        main(['--manifest', str(path), '--baseline-clean', 'a', '--baseline-stress', 'b',
              '--candidate-clean', 'c', '--candidate-stress', 'd', '--output', str(path)])
    assert exc.value.code == 2
    assert path.read_text() == 'preserve evidence'


def test_selector_cli_output_is_accepted_by_the_benchmark(tmp_path):
    from benchmark_chinese_models import selected_preset, selection_reports
    a, b, c, d, manifest = arms()
    files = {}
    for name, data in [('manifest', manifest), ('clean', a), ('stress', b),
                       ('candidate_clean', c), ('candidate_stress', d)]:
        files[name] = tmp_path / f'{name}.json'
        files[name].write_text(json.dumps(data), encoding='utf-8')
    output = tmp_path / 'nested/selection.json'
    assert main(['--manifest', str(files['manifest']), '--baseline-clean', str(files['clean']),
                 '--baseline-stress', str(files['stress']), '--candidate-clean', str(files['candidate_clean']),
                 '--candidate-stress', str(files['candidate_stress']), '--output', str(output)]) == 0
    preset, choice = selected_preset(output)
    assert preset == 'stabilize_mild'
    assert str(files['manifest']) not in choice['input_sha256']
    assert choice['manifest_sha256'] == hashlib.sha256(files['manifest'].read_bytes()).hexdigest()
    test = Dataset((Sample('test-data', '/test.mp4', '无关句子', 'new-person', 'test-session',
                           'verified', True, 'mouth_roi', True, sha256='a' * 64),))
    assert len(selection_reports(output, choice, test)) == 4


@pytest.mark.parametrize('seed', [-1, 2 ** 32, True])
def test_invalid_visual_seed_rejected_before_transform(seed):
    with pytest.raises(ValueError):
        evaluate_chinese.visual_configuration(SimpleNamespace(visual_seed=seed))


def test_synthetic_readiness_has_consistent_status_and_a_failed_gate():
    result = {'readiness': {'ready': True, 'status': 'ready_under_declared_protocol', 'gates': []}, 'limitations': []}
    evaluate_chinese.block_synthetic_readiness(result, SimpleNamespace(visual_stress='synthetic_mild'))
    assert result['readiness']['status'] == 'not_ready'
    assert not result['readiness']['ready']
    assert result['readiness']['gates'][0]['passed'] is False


@pytest.mark.parametrize('field,value', [('candidate_pool_size', 20), ('pre_beam_ratio', 2.0),
                                      ('reverse_scoring', 'batched'), ('reverse_tie_tolerance', .01)])
def test_visual_only_selection_cannot_silently_change_search_space(field, value):
    a, b, c, d, manifest = arms()
    for item in (a, b, c, d):
        item['model']['nbest'] = 10
    c['model'][field] = value
    d['model'][field] = value
    with pytest.raises(ValueError, match='identical models and decoder settings'):
        select(a, b, [(c, d)], manifest)


def test_visual_binding_accepts_explicit_defaults_from_old_implicit_search_reports():
    from select_chinese_visual import _binding
    a, b, c, d, manifest = arms()
    for item in (a, b, c, d):
        item['model']['nbest'] = 10
    old_binding = _binding(a)
    explicit = deepcopy(a)
    explicit['model'].update(candidate_pool_size=10, pre_beam_ratio=1.5,
                              reverse_scoring='sequential', reverse_tie_tolerance=.001)
    assert _binding(explicit) == old_binding
    for item in (c, d):
        item['model'].update(candidate_pool_size=10, pre_beam_ratio=1.5,
                              reverse_scoring='sequential', reverse_tie_tolerance=.001)
    result = select(a, b, [(c, d)], manifest)
    assert result['selected_preprocessing'] == 'stabilize_mild'
