"""Cohort comparisons cannot invent webcam, unseen-speaker or success evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research' / 'scripts'))
from compare_chinese_cohorts import compare_cohorts, main


def _sample(identity, reference='甲乙', speaker=None, **extras):
    return {'id': identity, 'reference': reference, 'speaker': speaker or identity,
            'session': f'{identity}-session', 'video': f'{identity}.mp4',
            'sha256': hashlib.sha256(identity.encode()).hexdigest(), 'split': 'test',
            'domain': 'mouth_roi', 'mouth_roi': True, 'articulation': 'voiced',
            'articulation_verified': True, 'label_source': 'independent reference',
            'label_verified': True, **extras}


def _manifest(*samples, split='test'):
    return {'schema_version': 1, 'split': split, 'samples': list(samples)}


def _report(manifest, predictions=None, failures=()):
    predictions = predictions or {}
    return {'schema_version': 1, 'mode': 'silent', 'samples': [
        {'sample_id': row['id'], 'raw': predictions.get(row['id'], row['reference']),
         'reference': row['reference'], 'speaker': row['speaker'],
         'video_sha256': row.get('sha256'),
         **{key: row[key] for key in ('session', 'split', 'domain', 'articulation', 'articulation_verified') if key in row},
         'error': 'decode failed' if row['id'] in failures else None,
         'action': 'retry' if row['id'] in failures else 'review'} for row in manifest['samples']]}


def _local(role, identity, speaker=None, **extras):
    return _manifest(_sample(identity, speaker=speaker, split=role, **extras), split=role)


def _sidecar(identity, **extras):
    return {'schema_version': 1, 'samples': {identity: {'provenance': 'independent capture log and recorder notes',
                                                     'metadata_verified': True, **extras}}}


def _with_pixels(report, *, preprocessing='identity', stress='clean', seed=0):
    configuration = {'preprocessing': preprocessing, 'stress': stress, 'seed': seed}
    report['model'] = {'visual_configuration': configuration.copy()}
    for row in report['samples']:
        identity = row['sample_id']
        sample_seed = (seed + int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'big')) % (2 ** 32)
        row['visual_input'] = {
            'configuration': configuration.copy(), 'sample_seed': sample_seed,
            'original_pixels_sha256': hashlib.sha256(f'original-{identity}'.encode()).hexdigest(),
            'input_pixels_sha256': hashlib.sha256(f'{identity}-{preprocessing}-{stress}-{seed}'.encode()).hexdigest(),
            'shape': [50, 96, 96], 'dtype': 'uint8', 'diagnostics': {'synthetic': stress != 'clean'},
            'real_webcam_evidence': stress == 'clean' and row.get('domain') == 'webcam'}
    return report


def _compare(manifest, *, visual_configuration=None, **kwargs):
    report = _report(manifest)
    if visual_configuration is not None:
        _with_pixels(report, **visual_configuration)
    return compare_cohorts(report, report, manifest, resamples=20, **kwargs)


def test_same_frozen_ids_required_even_if_both_models_omit_the_failure():
    manifest = _manifest(_sample('a'), _sample('b'))
    full = _report(manifest)
    partial = _report(_manifest(manifest['samples'][0]))
    with pytest.raises(ValueError, match='identical sample IDs'):
        compare_cohorts(full, partial, manifest)
    with pytest.raises(ValueError, match='every frozen manifest'):
        compare_cohorts(partial, partial, manifest)


def test_failure_remains_in_overall_and_each_shared_cohort_denominator():
    manifest = _manifest(_sample('a', speaker='same'), _sample('b', speaker='same'))
    baseline, candidate = _report(manifest), _report(manifest, {'a': ''}, failures=['a'])
    original = deepcopy((baseline, candidate, manifest))
    result = compare_cohorts(baseline, candidate, manifest, resamples=20)
    for item in (result['overall'], result['cohorts']['speaker']['same']['comparison'],
                 result['cohorts']['domain']['mouth_roi']['comparison']):
        assert item['baseline']['samples'] == item['candidate']['samples'] == 2
        assert item['candidate']['reference_characters'] == 4
        assert item['candidate']['character_errors'] == 2
        assert item['candidate']['inference_failures'] == 1
        assert item['candidate']['empty_outputs'] == 1
        assert item['candidate']['cer'] == .5
    assert (baseline, candidate, manifest) == original


def test_corpus_field_reference_and_hash_changes_reject_comparison():
    manifest = _manifest(_sample('a'))
    baseline = _report(manifest)
    for key, value, message in (('reference', '丙丁', 'reference'),
                                ('speaker', 'wrong', 'speaker'),
                                ('video_sha256', 'a' * 64, 'sha256'),
                                ('domain', 'full_face', 'domain')):
        candidate = deepcopy(baseline)
        candidate['samples'][0][key] = value
        with pytest.raises(ValueError, match=message):
            compare_cohorts(baseline, candidate, manifest)


def test_frozen_hash_required_even_when_predictions_and_labels_are_identical():
    manifest = _manifest(_sample('a'))
    manifest['samples'][0].pop('sha256')
    with pytest.raises(ValueError, match='sha256'):
        _compare(manifest)
    manifest['samples'][0]['sha256'] = 'not-a-file-hash'
    with pytest.raises(ValueError, match='sha256'):
        _compare(manifest)


def test_absent_metadata_is_unknown_not_inferred_from_ids_or_webcam_domain():
    sample = _sample('office-silent-webcam-daily-unseen', domain='webcam', mouth_roi=False,
                     articulation='silent', articulation_verified=True)
    result = _compare(_manifest(sample))
    assert result['sample_metadata'][0]['capture_type'] == 'unknown'
    assert result['sample_metadata'][0]['content_style'] == 'unknown'
    for target in ('verified_real_silent_webcam', 'verified_everyday_conversation',
                   'unseen_relative_to_local_adaptation'):
        assert result['target_coverage'][target]['available'] is False
        assert result['target_coverage'][target]['samples'] == 0
        assert result['target_coverage'][target]['comparison'] is None
    assert result['target_coverage']['full_face_input']['samples'] == 1
    assert result['local_adaptation_audit']['base_model_training_exposure'] == 'unknown'
    assert result['product_readiness_established'] is False


def test_public_voiced_precropped_footage_does_not_establish_webcam_or_everyday_coverage():
    manifest = _manifest(_sample('a'), _sample('b'))
    sidecar = _sidecar('a', capture_type='pre_cropped', content_style='read_speech', scene='studio')
    result = _compare(manifest, metadata=sidecar)
    assert result['cohorts']['capture_type']['pre_cropped']['samples'] == 1
    assert result['cohorts']['capture_type']['unknown']['samples'] == 1
    assert result['cohorts']['scene']['studio']['samples'] == 1
    for target in ('verified_real_silent_webcam', 'full_face_input', 'verified_everyday_conversation'):
        assert result['target_coverage'][target]['samples'] == 0


def test_verified_real_silent_everyday_webcam_target_requires_all_explicit_evidence():
    sample = _sample('a', domain='webcam', mouth_roi=False, articulation='silent')
    manifest = _manifest(sample)
    sidecar = _sidecar('a', capture_type='real_webcam', content_style='everyday_conversation', scene='office')
    options = {'metadata': sidecar, 'training_manifests': [_local('train', 't')],
               'development_manifests': [_local('dev', 'd')], 'visual_configuration': {}}
    result = _compare(manifest, **options)
    for target in ('verified_real_silent_webcam', 'full_face_input', 'verified_everyday_conversation',
                   'verified_real_silent_webcam_everyday_unseen_locally'):
        assert result['target_coverage'][target]['samples'] == 1
    assert result['sample_metadata'][0]['local_adaptation_exposure'] == 'unseen_relative_to_local_adaptation'
    # A silent declaration without independent verification cannot count.
    sample['articulation_verified'] = False
    assert _compare(manifest, **options)['target_coverage']['verified_real_silent_webcam']['samples'] == 0
    sample['articulation_verified'] = True
    sidecar['samples']['a']['metadata_verified'] = False
    result = _compare(manifest, **options)
    assert result['cohorts']['capture_type']['real_webcam']['samples'] == 1
    assert result['target_coverage']['verified_real_silent_webcam']['samples'] == 0
    assert result['target_coverage']['verified_everyday_conversation']['samples'] == 0


def test_unseen_local_scope_needs_both_supplied_splits_and_never_claims_base_unseen():
    manifest = _manifest(_sample('a', speaker='new-speaker'))
    train, dev = _local('train', 't'), _local('dev', 'd')
    for kwargs in ({}, {'training_manifests': [train]}, {'development_manifests': [dev]}):
        result = _compare(manifest, **kwargs)
        assert result['sample_metadata'][0]['local_adaptation_exposure'] == 'unknown'
        assert not result['target_coverage']['unseen_relative_to_local_adaptation']['available']
    result = _compare(manifest, training_manifests=[train], development_manifests=[dev])
    assert result['target_coverage']['unseen_relative_to_local_adaptation']['samples'] == 1
    audit = result['local_adaptation_audit']
    assert audit['base_model_training_exposure'] == 'unknown'
    assert audit['full_adaptation_manifest_inventory_verified'] is False
    assert result['test_based_configuration_selection'] is False


def test_shared_speaker_is_explicitly_seen_but_kept_for_honest_paired_comparison():
    manifest = _manifest(_sample('a', speaker='known'), _sample('b', speaker='other'))
    train, dev = _local('train', 't', speaker='known'), _local('dev', 'd')
    result = _compare(manifest, training_manifests=[train], development_manifests=[dev])
    audit = result['local_adaptation_audit']
    assert audit['speaker_overlaps'] == ['known']
    assert result['cohorts']['local_adaptation_exposure']['seen_relative_to_local_adaptation']['sample_ids'] == ['a']
    assert result['target_coverage']['unseen_relative_to_local_adaptation']['sample_ids'] == ['b']
    assert result['overall']['candidate']['samples'] == 2


@pytest.mark.parametrize('field', ['id', 'sha256'])
def test_local_adaptation_id_or_hash_leakage_rejects_test_comparison(field):
    sample = _sample('a')
    train_sample = _sample('t', split='train')
    train_sample[field] = sample[field]
    with pytest.raises(ValueError, match='local adaptation content overlaps'):
        _compare(_manifest(sample), training_manifests=[_manifest(train_sample, split='train')])


def test_embedded_exclusions_are_audited_without_inventing_complete_split_inventory():
    manifest = _manifest(_sample('a'))
    manifest['development_samples'] = [{'id': 'x', 'speaker': 'a', 'session': 'other'}]
    result = _compare(manifest)
    assert result['sample_metadata'][0]['local_adaptation_exposure'] == 'seen_relative_to_local_adaptation'
    assert result['local_adaptation_audit']['both_local_split_manifests_supplied'] is False
    manifest['development_samples'][0]['id'] = 'a'
    with pytest.raises(ValueError, match='local adaptation content overlaps'):
        _compare(manifest)


def test_same_video_overlapping_ranges_and_inconsistent_identity_are_rejected():
    first = _sample('a', speaker='same', start=0, end=2)
    second = _sample('b', speaker='same', session=first['session'], video=first['video'],
                     sha256=first['sha256'], start=1, end=3)
    with pytest.raises(ValueError, match='overlapping test video ranges'):
        _compare(_manifest(first, second))
    second['start'] = 2
    assert _compare(_manifest(first, second))['overall']['baseline']['samples'] == 2
    second['speaker'] = 'different'
    with pytest.raises(ValueError, match='inconsistent speaker'):
        _compare(_manifest(first, second))
    second['sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='inconsistent sha256'):
        _compare(_manifest(first, second))


def test_unknown_speaker_cannot_become_locally_unseen_by_absence_from_train_dev():
    result = _compare(_manifest(_sample('a', speaker='unknown')),
                      training_manifests=[_local('train', 't')],
                      development_manifests=[_local('dev', 'd')])
    assert result['sample_metadata'][0]['local_adaptation_exposure'] == 'unknown'
    assert result['local_adaptation_audit']['unknown_test_speaker_ids'] == ['a']
    assert result['target_coverage']['unseen_relative_to_local_adaptation']['samples'] == 0


def test_train_dev_duplicate_content_is_rejected_but_shared_speakers_are_audited():
    train, dev = _local('train', 't', speaker='same'), _local('dev', 'd', speaker='same')
    result = _compare(_manifest(_sample('a')), training_manifests=[train], development_manifests=[dev])
    assert result['local_adaptation_audit']['train_dev_speaker_overlaps'] == ['same']
    dev['samples'][0]['sha256'] = train['samples'][0]['sha256']
    with pytest.raises(ValueError, match='train/dev content overlaps'):
        _compare(_manifest(_sample('a')), training_manifests=[train], development_manifests=[dev])


def test_unverified_reference_provenance_is_visible_and_does_not_drop_samples():
    result = _compare(_manifest(_sample('a', label_verified=False)))
    assert result['local_adaptation_audit']['unverified_reference_ids'] == ['a']
    assert result['overall']['baseline']['samples'] == 1


@pytest.mark.parametrize('changes', [
    {'capture_type': 'simulated_webcam'}, {'content_style': 'daily-from-filename'},
    {'scene': ''}, {'metadata_verified': 1}, {'provenance': ''},
    {'model_success': True}, {'base_model_unseen': True},
])
def test_sidecar_fields_are_validated_and_cannot_override_outcomes(changes):
    metadata = _sidecar('a', **changes)
    with pytest.raises(ValueError):
        _compare(_manifest(_sample('a')), metadata=metadata)


def test_mismatched_sidecar_ids_and_realwebcam_claim_on_cropped_domain_rejected():
    with pytest.raises(ValueError, match='IDs are not in frozen manifest'):
        _compare(_manifest(_sample('a')), metadata=_sidecar('wrong'))
    with pytest.raises(ValueError, match='real_webcam metadata requires webcam'):
        _compare(_manifest(_sample('a')), metadata=_sidecar('a', capture_type='real_webcam'))


def test_grouped_speaker_bootstrap_remains_paired_and_failure_tolerant():
    manifest = _manifest(_sample('a', reference='甲乙', speaker='A'),
                         _sample('b', reference='丙丁', speaker='A'),
                         _sample('c', reference='甲乙', speaker='B'))
    baseline = _report(manifest, {'a': '甲', 'b': ''}, failures=['b'])
    candidate = _report(manifest, {'c': '甲丙'})
    result = compare_cohorts(baseline, candidate, manifest)
    group = result['cohorts']['domain']['mouth_roi']['comparison']
    assert group['cer_difference_95pct_interval']['resamples'] == 5000
    assert group['cer_difference_95pct_interval']['lower'] == -.75
    assert group['cer_difference_95pct_interval']['upper'] == .5
    assert group['baseline']['inference_failures'] == 1
    assert group['candidate']['reference_characters'] == 6


def test_cli_hashes_all_inputs_and_prevents_overwriting_any_evidence(tmp_path, capsys):
    manifest = _manifest(_sample('a'))
    files = {'baseline.json': _report(manifest), 'candidate.json': _report(manifest),
             'test.json': manifest, 'train.json': _local('train', 't'),
             'dev.json': _local('dev', 'd'), 'metadata.json': _sidecar('a', capture_type='pre_cropped')}
    for name, obj in files.items():
        (tmp_path / name).write_text(json.dumps(obj), encoding='utf-8')
    output = tmp_path / 'comparison.json'
    args = [str(tmp_path / 'baseline.json'), str(tmp_path / 'candidate.json'),
            '--manifest', str(tmp_path / 'test.json'), '--train-manifest', str(tmp_path / 'train.json'),
            '--dev-manifest', str(tmp_path / 'dev.json'), '--metadata', str(tmp_path / 'metadata.json'),
            '--resamples', '20']
    assert main([*args, '--output', str(output)]) == 0
    result = json.loads(output.read_text())
    assert len(result['provenance']['inputs']) == 6
    for record in result['provenance']['inputs']:
        assert record['sha256'] == hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()
    capsys.readouterr()
    for name in files:
        original = (tmp_path / name).read_bytes()
        with pytest.raises(SystemExit) as error:
            main([*args, '--output', str(tmp_path / name)])
        assert error.value.code == 2
        assert (tmp_path / name).read_bytes() == original
        capsys.readouterr()


def test_standalone_help_needs_neither_cwd_checkout_nor_camera_or_model_deps(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'research' / 'scripts' / 'compare_chinese_cohorts.py'
    # -S omits site-packages: only standard-library imports may be required.
    result = subprocess.run([sys.executable, '-S', str(script), '--help'], cwd=tmp_path,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert '--train-manifest' in result.stdout


def test_both_models_same_preprocessing_require_identical_prepared_pixels():
    manifest = _manifest(_sample('a'), _sample('b'))
    left, right = _with_pixels(_report(manifest)), _with_pixels(_report(manifest))
    result = compare_cohorts(left, right, manifest, resamples=20)
    audit = result['prepared_pixel_audit']
    assert audit['all_original_pixels_verified']
    assert audit['original_pixels_verified_samples'] == 2
    assert audit['prepared_same_preprocessing_pixels_verified_samples'] == 2
    assert audit['stress_and_seed_verified_equal']
    right['samples'][0]['visual_input']['input_pixels_sha256'] = 'c' * 64
    with pytest.raises(ValueError, match='same preprocessing requires identical prepared input pixels'):
        compare_cohorts(left, right, manifest, resamples=20)


def test_different_frozen_preprocessing_keeps_common_original_pixels_checked():
    manifest = _manifest(_sample('a'))
    left = _with_pixels(_report(manifest))
    right = _with_pixels(_report(manifest), preprocessing='stabilize_mild')
    result = compare_cohorts(left, right, manifest, resamples=20)
    audit = result['prepared_pixel_audit']
    assert audit['all_original_pixels_verified']
    assert audit['different_preprocessing_samples'] == 1
    assert audit['prepared_same_preprocessing_pixels_verified_samples'] == 0
    right['samples'][0]['visual_input']['original_pixels_sha256'] = 'c' * 64
    with pytest.raises(ValueError, match='original visual pixels original_pixels_sha256 mismatch'):
        compare_cohorts(left, right, manifest, resamples=20)


@pytest.mark.parametrize('mutation', ['stress', 'seed', 'shape', 'dtype', 'sample_seed', 'model_config', 'hash'])
def test_invalid_or_unpaired_pixel_evidence_is_rejected(mutation):
    manifest = _manifest(_sample('a'))
    left = _with_pixels(_report(manifest))
    right = _with_pixels(_report(manifest))
    evidence = right['samples'][0]['visual_input']
    if mutation == 'stress':
        right = _with_pixels(_report(manifest), stress='synthetic_mild')
    elif mutation == 'seed':
        right = _with_pixels(_report(manifest), seed=1)
    elif mutation == 'shape':
        evidence['shape'] = [49, 96, 96]
    elif mutation == 'dtype':
        evidence['dtype'] = 'float32'
    elif mutation == 'sample_seed':
        evidence['sample_seed'] += 1
    elif mutation == 'model_config':
        right['model']['visual_configuration']['preprocessing'] = 'stabilize_mild'
    else:
        evidence['original_pixels_sha256'] = 'z' * 64
    with pytest.raises(ValueError):
        compare_cohorts(left, right, manifest, resamples=20)


def test_legacy_missing_pixel_evidence_stays_unknown_and_failed_rows_are_retained():
    manifest = _manifest(_sample('a'), _sample('b'))
    legacy = _report(manifest)
    candidate = _with_pixels(_report(manifest, {'a': ''}, failures=['a']))
    candidate['samples'][0]['visual_input'] = None
    result = compare_cohorts(legacy, candidate, manifest, resamples=20)
    audit = result['prepared_pixel_audit']
    assert audit['original_pixels_verified_samples'] == 0
    assert not audit['all_original_pixels_verified']
    assert audit['comparison_input_condition'] == 'unknown'
    assert audit['stress_and_seed_verified_equal'] is None
    assert audit['unknown_pixel_evidence_sample_ids'] == ['a', 'b']
    assert result['overall']['candidate']['samples'] == 2
    assert result['overall']['candidate']['inference_failures'] == 1


def test_synthetic_model_inputs_never_count_as_genuine_silent_webcam_evaluation():
    manifest = _manifest(_sample('a', domain='webcam', mouth_roi=False, articulation='silent'))
    metadata = _sidecar('a', capture_type='real_webcam', content_style='everyday_conversation')
    result = _compare(manifest, metadata=metadata,
                      visual_configuration={'stress': 'synthetic_mild'},
                      training_manifests=[_local('train', 't')], development_manifests=[_local('dev', 'd')])
    assert result['verified_real_silent_webcam_source_coverage']['samples'] == 1
    assert result['target_coverage']['verified_real_silent_webcam']['samples'] == 0
    assert result['target_coverage']['verified_real_silent_webcam_everyday_unseen_locally']['samples'] == 0
    assert result['target_coverage']['explicitly_synthetic_visual_stress']['samples'] == 1
    assert result['prepared_pixel_audit']['comparison_input_condition'] == 'synthetic_stress'
    assert not result['prepared_pixel_audit']['real_webcam_measurement_established']
    assert not result['product_readiness_established']


def test_unknown_model_input_condition_separates_camera_source_from_verified_evaluation():
    manifest = _manifest(_sample('a', domain='webcam', mouth_roi=False, articulation='silent'))
    result = _compare(manifest, metadata=_sidecar('a', capture_type='real_webcam'))
    assert result['verified_real_silent_webcam_source_coverage']['samples'] == 1
    assert result['target_coverage']['verified_real_silent_webcam']['samples'] == 0
    assert result['prepared_pixel_audit']['comparison_input_condition'] == 'unknown'


@pytest.mark.parametrize('field,value', [('synthetic', True), ('text_used', True), ('audio_used', True),
                                       ('input_pixels_sha256', 'c' * 64), ('shape', [49, 96, 96])])
def test_contradictory_visual_diagnostics_do_not_verify_pixels(field, value):
    manifest = _manifest(_sample('a'))
    left, right = _with_pixels(_report(manifest)), _with_pixels(_report(manifest))
    right['samples'][0]['visual_input']['diagnostics'][field] = value
    with pytest.raises(ValueError):
        compare_cohorts(left, right, manifest, resamples=20)
