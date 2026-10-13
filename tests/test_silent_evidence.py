"""Metadata/error tests use tiny synthetic byte files, no actual human recording."""
import copy
import hashlib
import json

import pytest

from research.scripts import audit_silent_data as auditor
from research.scripts import report_silent_errors as errors


def dataset(tmp_path):
    paths = {}
    for split, size in [('train', 1), ('dev', 1), ('test', 50)]:
        directory = tmp_path / split
        directory.mkdir()
        rows = []
        for i in range(size):
            video = directory / f'{i}.mp4'
            payload = f'synthetic non-video {split}-{i}'.encode()
            video.write_bytes(payload)
            rows.append({'id': f'{split}-{i}', 'video': video.name, 'speaker': f'{split}-speaker-{i // 10}',
                         'session': f'day-{i % 2}', 'recording_date': f'2026-10-{13 + i % 2}',
                         'reference': f'{split}原句{i}', 'label_source': 'synthetic fixture', 'label_verified': True,
                         'domain': 'webcam', 'mouth_roi': False, 'articulation': 'silent',
                         'articulation_verified': True, 'sha256': hashlib.sha256(payload).hexdigest(), 'split': split})
        path = directory / 'manifest.json'
        path.write_text(json.dumps({'schema_version': 1, 'split': split, 'samples': rows,
                                    'training_overlap_checked': True}), encoding='utf-8')
        paths[split] = path
    return paths


def change(path, callback):
    data = json.loads(path.read_text(encoding='utf-8'))
    callback(data)
    path.write_text(json.dumps(data), encoding='utf-8')


def test_audit_eligible_is_not_model_acceptance_and_review_is_reproducible(tmp_path):
    paths = dataset(tmp_path)
    result = auditor.audit(paths)
    assert result['data_eligible'] and not result['model_acceptance']
    assert result['partitions']['test']['samples'] == 50
    assert len(result['manual_review_samples']['test']) == 10
    assert result == auditor.audit(paths)


@pytest.mark.parametrize('edit,expected', [
    (lambda d: d['samples'][0].update(articulation='voiced'), 'unconfirmed_silent'),
    (lambda d: d['samples'][0].update(sha256='f' * 64), 'declared_sha256_mismatch'),
    (lambda d: d['samples'][0].update(recording_date='yesterday'), 'missing_or_invalid_recording_date'),
    (lambda d: d.update(training_overlap_checked=False), 'overlap review is unconfirmed'),
    (lambda d: d['samples'][0].update(speaker='train-speaker-0'), 'overlapping speakers'),
    (lambda d: d['samples'][0].update(reference='train原句0'), 'overlapping normalized_references'),
])
def test_audit_detects_invalid_provenance_and_leakage(tmp_path, edit, expected):
    paths = dataset(tmp_path)
    change(paths['test'], edit)
    result = auditor.audit(paths)
    assert not result['data_eligible']
    assert any(expected in issue for issue in result['issues'])


def test_actual_video_tamper_is_detected(tmp_path):
    paths = dataset(tmp_path)
    (paths['test'].parent / '0.mp4').write_bytes(b'tampered')
    result = auditor.audit(paths)
    assert any('declared_sha256_mismatch' in issue for issue in result['issues'])


def predictions():
    manifest = {'samples': [
        {'id': 'one', 'reference': '小林九点到', 'speaker': 's1', 'recording_date': '2026-10-13',
         'entities_verified': True, 'entities': [{'kind': 'person_name', 'text': '小林'}, {'kind': 'number', 'text': '九'}]},
        {'id': 'two', 'reference': '请发文件', 'speaker': 's2', 'recording_date': '2026-10-14'},
    ]}
    before = {'samples': [{'sample_id': 'one', 'reference': '小林九点到', 'raw': '小李九点'},
                          {'sample_id': 'two', 'reference': '请发文件', 'raw': '', 'error': 'synthetic failure'}]}
    after = {'samples': [{'sample_id': 'one', 'reference': '小林九点到', 'raw': '小林九点到到'},
                         {'sample_id': 'two', 'reference': '请发文件', 'raw': '请发文件'}]}
    return manifest, before, after


def test_complete_raw_alignment_retains_failure_and_groups_missing_annotations():
    result = errors.report(*predictions())
    before, after = result['baseline']['metrics'], result['candidate']['metrics']
    assert (before['substitutions'], before['deletions'], before['insertions']) == (1, 5, 0)
    assert before['inference_failure'] == 1 and before['reference_characters'] == 9
    assert after['insertions'] == 1 and after['reference_characters'] == 9
    assert result['entity_annotation']['unannotated_samples'] == 1
    assert result['baseline']['per_entity_sentence_cohort']['person_name']['samples'] == 1
    assert result['baseline']['per_entity_sentence_cohort']['unannotated']['samples'] == 1
    assert set(result['baseline']['per_recording_date']) == {'2026-10-13', '2026-10-14'}
    assert result['paired_comparison']['delta']['character_errors'] == -5
    assert result['model_acceptance'] is False


@pytest.mark.parametrize('mutation', ['drop', 'duplicate', 'reference', 'speaker', 'counts', 'bad_entity', 'mode', 'aggregate'])
def test_report_rejects_incomplete_or_mismatched_predictions(mutation):
    manifest, before, after = predictions()
    if mutation == 'drop':
        after['samples'].pop()
    elif mutation == 'duplicate':
        after['samples'].append(copy.deepcopy(after['samples'][0]))
    elif mutation == 'reference':
        after['samples'][0]['reference'] = '别的原句'
    elif mutation == 'speaker':
        after['samples'][0]['speaker'] = 'different person'
    elif mutation == 'counts':
        after['samples'][0]['character_errors'] = 0
    elif mutation == 'bad_entity':
        manifest['samples'][0]['entities'][0]['text'] = '不存在'
    elif mutation == 'mode':
        before['mode'], after['mode'] = 'silent', 'hybrid'
    elif mutation == 'aggregate':
        after['metrics'] = {'character_errors': 0}
    with pytest.raises(ValueError):
        errors.report(manifest, before, after)


def test_tools_cannot_overwrite_input(tmp_path):
    path = tmp_path / 'input.json'
    path.write_text('unchanged', encoding='utf-8')
    with pytest.raises(SystemExit) as info:
        auditor.main(['--train', str(path), '--dev', str(path), '--test', str(path), '--output', str(path)])
    assert info.value.code == 2 and path.read_text(encoding='utf-8') == 'unchanged'
    with pytest.raises(SystemExit) as info:
        errors.main(['--manifest', str(path), '--baseline', str(path), '--candidate', str(path), '--output', str(path)])
    assert info.value.code == 2 and path.read_text(encoding='utf-8') == 'unchanged'
