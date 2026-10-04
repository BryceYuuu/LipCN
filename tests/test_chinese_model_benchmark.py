from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research' / 'scripts'))
from benchmark_chinese_models import prepare_shared, shared_loader, charge_preparation, selected_preset
from lipflow.confidence import Quality
from research.evaluation import Dataset, Prediction, Sample, evaluate


def dataset():
    return Dataset(tuple(Sample(str(i), f'/local/{i}.mp4', '不要发货', f'speaker{i}', 'session',
                                'verified label', True, 'mouth_roi', True) for i in (1, 2)))


def test_cache_prepares_each_video_once_without_reference_and_preserves_failed_rows():
    seen = []
    def loader(sample, reader):
        assert sample.reference == ''
        seen.append(sample.id)
        if sample.id == '2':
            raise ValueError('unreadable video')
        return np.ones((2, 2, 2), np.uint8), 1, Quality()
    cache, costs, used = prepare_shared(dataset(), loader, object(), max_bytes=8)
    assert seen == ['1', '2'] and set(cache) == set(costs) == {'1', '2'}
    assert used == 8 and isinstance(cache['2'], ValueError)
    assert cache['1'][0].flags.writeable is False
    assert dataset().samples[0].reference == '不要发货'


def test_memory_budget_does_not_drop_a_video_or_give_one_model_better_coverage():
    def loader(sample, reader):
        return np.ones((2, 2, 2), np.uint8), 1, Quality()
    cache, _, used = prepare_shared(dataset(), loader, object(), max_bytes=8)
    assert used == 8 and isinstance(cache['2'], ValueError)
    module = SimpleNamespace(_visual_input=loader)
    with shared_loader(module, cache):
        # All model arms receive the exact same cached failure.
        for _ in range(3):
            with pytest.raises(ValueError, match='budget exceeded'):
                module._visual_input(SimpleNamespace(id='2', reference=''), object())
    assert module._visual_input is loader


def test_shared_loader_restores_original_on_failure_and_forbids_labels():
    old = lambda *args: 'original'
    module = SimpleNamespace(_visual_input=old)
    images = np.ones((1, 2, 2), np.uint8)
    with pytest.raises(RuntimeError):
        with shared_loader(module, {'1': (images, 1, Quality())}):
            assert module._visual_input(SimpleNamespace(id='1', reference=''), object())[0] is images
            with pytest.raises(ValueError, match='reference leaked'):
                module._visual_input(SimpleNamespace(id='1', reference='不要发货'), object())
            raise RuntimeError('stop')
    assert module._visual_input is old


def test_latency_charges_shared_preparation_to_every_arm_and_recomputes_metrics():
    data = dataset()
    for _ in range(3):
        report = evaluate(data, [Prediction(s.id, '不要发货', 4, 1) for s in data.samples])
        report['model'] = {'visual_configuration': {'stress': 'clean'}}
        charge_preparation(report, data, {'1': 2, '2': 4})
        assert [r['processing_seconds'] for r in report['samples']] == [3, 5]
        assert [r['rtf'] for r in report['samples']] == [.75, 1.25]
        assert report['metrics']['warm_processing_seconds_p50'] == 4
        assert report['metrics']['samples'] == 2


def test_charging_joins_by_id_and_rejects_changed_identity():
    data = dataset()
    report = evaluate(data, [Prediction(s.id, s.reference, 4, 1) for s in data.samples])
    report['model'] = {'visual_configuration': {'stress': 'clean'}}
    report['samples'].reverse()
    charge_preparation(report, data, {'1': 2, '2': 4})
    assert [r['rtf'] for r in report['samples']] == [1.25, .75]
    report['samples'][0]['reference'] = 'changed'
    with pytest.raises(ValueError, match='identity/reference'):
        charge_preparation(report, data, {'1': 0, '2': 0})


def test_synthetic_status_and_gate_cannot_claim_real_camera_readiness():
    data = dataset()
    report = evaluate(data, [Prediction(s.id, s.reference, 4, 1) for s in data.samples])
    report['model'] = {'visual_configuration': {'stress': 'synthetic_mild'}}
    charge_preparation(report, data, {'1': 0, '2': 0})
    assert not report['readiness']['ready']
    assert report['readiness']['status'] == 'not_ready'
    assert any(g['criterion'] == 'real_camera_evidence' and not g['passed'] for g in report['readiness']['gates'])


@pytest.mark.parametrize('changes', [
    {'selection_partition': 'test'}, {'test_used_for_selection': True},
    {'selected_preprocessing': 'invented'}, {'provenance': {}},
])
def test_frozen_dev_choice_required(tmp_path, changes):
    data = {'schema_version': 1, 'selection_partition': 'dev', 'test_used_for_selection': False,
            'selected_preprocessing': 'stabilize_mild', 'provenance': {'dev_report_sha256': {'dev.json': 'a' * 64}}}
    data.update(changes)
    path = tmp_path / 'choice.json'; path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        selected_preset(path)


def test_no_choice_keeps_identity_and_valid_choice_is_read_once(tmp_path):
    assert selected_preset(None) == ('identity', None)
    data = {'schema_version': 1, 'selection_partition': 'dev', 'test_used_for_selection': False,
            'selected_preprocessing': 'stabilize_mild', 'input_sha256': {'dev.json': 'b' * 64}}
    path = tmp_path / 'choice.json'; path.write_text(json.dumps(data))
    assert selected_preset(path) == ('stabilize_mild', data)
