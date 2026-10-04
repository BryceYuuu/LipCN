"""Offline candidate oracles must not hide failures, ranks or missing evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research' / 'scripts'))
from diagnose_chinese_candidates import diagnose, main


def _sample(identity, reference='甲乙', speaker=None, split='test'):
    return {'id': identity, 'reference': reference, 'speaker': speaker or identity,
            'sha256': hashlib.sha256(identity.encode()).hexdigest(), 'split': split}


def _manifest(*samples, split='test'):
    return {'schema_version': 1, 'split': split, 'samples': list(samples)}


def _row(sample, raw, hypotheses=None, **extras):
    result = {'sample_id': sample['id'], 'reference': sample['reference'], 'speaker': sample['speaker'],
              'video_sha256': sample['sha256'], 'split': sample['split'], 'raw': raw, 'action': 'review', **extras}
    if hypotheses is not None:
        result['hypotheses'] = [item if isinstance(item, dict) else {'text': item} for item in hypotheses]
    return result


def _report(*rows):
    return {'schema_version': 1, 'mode': 'silent', 'samples': list(rows)}


def test_sdi_top1_oracle_and_all_failure_denominators_are_recomputed():
    a, b, c, d = [_sample(name, reference) for name, reference in
                   [('a', '甲乙'), ('b', '丙丁'), ('c', '戊'), ('d', 'Ａ2，')]]
    report = _report(_row(a, '甲丙', ['甲丙', '甲乙', '甲丙。'], duration_seconds=1),
                     _row(b, '丙', error='decoder failed', action='retry', duration_seconds=3),
                     _row(c, '己戊', ['己戊', ''], duration_seconds=None),
                     _row(d, 'a2', ['a2', 'A２。'], duration_seconds=17))
    frozen = deepcopy((report, _manifest(a, b, c, d)))
    result = diagnose(report, frozen[1], include_samples=True)
    assert (report, frozen[1]) == frozen
    metrics = result['metrics']
    assert metrics['samples'] == 4
    top1 = metrics['top1']
    assert top1['substitutions'] == top1['deletions'] == top1['insertions'] == 1
    assert top1['reference_characters'] == 7
    assert top1['character_errors'] == 3 and top1['cer'] == 3 / 7
    assert top1['exact_sentences'] == 1
    assert top1['hypothesis_reference_length_ratio'] == 1
    assert metrics['inference_failures'] == metrics['rejected_samples'] == 1
    assert metrics['candidate_evidence_samples'] == 3 and metrics['missing_candidate_samples'] == 1
    assert metrics['empty_candidate_texts'] == 1
    assert metrics['oracle_at_k']['1']['cer'] == top1['cer']
    assert metrics['oracle_at_k']['3']['cer'] == 2 / 7
    assert metrics['oracle_at_k']['3']['raw_fallback_samples'] == 1
    assert metrics['oracle_at_k']['3']['candidate_list_exact_hits'] == 2
    assert not metrics['oracle_at_k']['3']['candidate_evidence_available_for_every_sample']
    assert not metrics['oracle_at_k']['3']['available']
    assert result['per_duration']['unknown']['samples'] == 1
    assert result['per_duration']['(4,8]']['available'] is False
    assert result['per_duration']['(4,8]']['top1']['cer'] is None


def test_frozen_complete_identity_validation_catches_symmetric_omission_and_changes():
    a, b = _sample('a'), _sample('b')
    manifest, partial = _manifest(a, b), _report(_row(a, '甲乙'))
    with pytest.raises(ValueError, match='every frozen manifest sample'):
        diagnose(partial, manifest)
    complete = _report(_row(a, '甲乙'), _row(b, '甲乙'))
    for key, value in [('reference', '丙丁'), ('speaker', 'other'), ('video_sha256', 'e' * 64)]:
        altered = deepcopy(complete)
        altered['samples'][0][key] = value
        with pytest.raises(ValueError, match='frozen manifest'):
            diagnose(altered, manifest)
    manifest['samples'][0]['sha256'] = 'invalid'
    with pytest.raises(ValueError, match='64-hex sha256'):
        diagnose(complete, manifest)


def test_duplicate_samples_and_dishonest_recorded_metrics_cannot_be_hidden():
    sample = _sample('a')
    row = _row(sample, '甲丙', ['甲丙'])
    with pytest.raises(ValueError, match='unique'):
        diagnose(_report(row, row), _manifest(sample))
    report = _report(row)
    report['metrics'] = {'samples': 2}
    with pytest.raises(ValueError, match='full sample rows'):
        diagnose(report, _manifest(sample))
    report['metrics'] = {'samples': 1, 'character_errors': 0}
    with pytest.raises(ValueError, match='full sample rows'):
        diagnose(report, _manifest(sample))
    report.pop('metrics')
    row['character_errors'] = 0
    with pytest.raises(ValueError, match='recorded character_errors'):
        diagnose(report, _manifest(sample))


def test_first_hypothesis_must_equal_raw_exactly_not_merely_after_normalization():
    sample = _sample('a')
    for alternative in ('甲乙', '甲丙。'):
        report = _report(_row(sample, '甲丙', [alternative]))
        with pytest.raises(ValueError, match='first hypothesis text must agree exactly'):
            diagnose(report, _manifest(sample))


def test_duplicates_do_not_promote_a_fourth_rank_answer_into_oracle_at_three():
    sample = _sample('a')
    result = diagnose(_report(_row(sample, '甲丙', ['甲丙', '甲丙。', '甲丙', '甲乙'])),
                      _manifest(sample), include_samples=True)
    metrics = result['metrics']
    assert metrics['unique_canonical_candidates'] == 2
    assert metrics['unique_literal_candidates'] == 3
    assert metrics['oracle_at_k']['3']['character_errors'] == 1
    assert metrics['oracle_at_k']['3']['candidate_list_exact_hits'] == 0
    assert metrics['oracle_at_k']['5']['character_errors'] == 0
    assert metrics['oracle_at_k']['5']['candidate_list_exact_hits'] == 1
    assert metrics['oracle_at_k']['5']['samples_with_fewer_than_k_hypotheses'] == 1
    assert metrics['oracle_at_k']['5']['available']
    assert not metrics['oracle_at_k']['5']['complete_k_evidence']
    assert result['samples'][0]['oracle_at_k']['5']['selected_rank'] == 4


def test_raw_only_fallback_does_not_manufacture_candidate_oracle_evidence():
    a, b = _sample('a'), _sample('b')
    result = diagnose(_report(_row(a, '甲乙'), _row(b, '', [], error='failed', action='retry')),
                      _manifest(a, b), include_samples=True)
    for entry in result['metrics']['oracle_at_k'].values():
        assert entry['samples'] == 2 and entry['reference_characters'] == 4
        assert entry['character_errors'] == 2
        assert entry['exact_sentences'] == 1
        assert entry['candidate_list_exact_hits'] == 0
        assert entry['candidate_evidence_samples'] == 0 and entry['raw_fallback_samples'] == 2
        assert not entry['available']
        assert entry['candidate_list_exact_hit_rate_available_denominator'] is None
    assert result['metrics']['unique_canonical_candidates'] == 0
    assert result['samples'][1]['oracle_at_k']['10']['selected_rank'] is None


def test_an_explicit_empty_candidate_is_real_evidence_retained_in_denominators():
    sample = _sample('a')
    result = diagnose(_report(_row(sample, '', [''])), _manifest(sample))
    assert result['metrics']['candidate_evidence_samples'] == 1
    assert result['metrics']['empty_raw_samples'] == result['metrics']['empty_candidate_texts'] == 1
    assert result['metrics']['unique_canonical_candidates'] == 1
    assert result['metrics']['oracle_at_k']['10']['character_errors'] == 2
    assert result['metrics']['oracle_at_k']['10']['raw_fallback_samples'] == 0


def test_diversity_uses_canonical_edit_space_and_handles_empty_alternatives():
    sample = _sample('a')
    result = diagnose(_report(_row(sample, '甲乙', ['甲乙', '甲丙', '甲乙。'])), _manifest(sample))
    diversity = result['metrics']['diversity']
    assert diversity['candidate_pairs'] == 1
    assert diversity['character_edit_distance'] == 1
    assert diversity['mean_normalized_edit_distance'] == diversity['pooled_normalized_edit_distance'] == .5
    result = diagnose(_report(_row(sample, '甲乙', ['甲乙', ''])), _manifest(sample))
    assert result['metrics']['diversity']['mean_normalized_edit_distance'] == 1
    result = diagnose(_report(_row(sample, '甲乙', ['甲乙'])), _manifest(sample))
    assert not result['metrics']['diversity']['available']
    assert result['metrics']['diversity']['mean_normalized_edit_distance'] is None


def test_forced_eos_is_reported_only_from_explicit_flags_without_claiming_causes():
    sample = _sample('a')
    hypotheses = [{'text': '甲丙', 'forced_eos': False, 'reverse_rerank_skipped_reason': 'forced_eos_in_beam'},
                  {'text': '甲乙', 'forced_eos': True}, {'text': '甲乙。'}]
    result = diagnose(_report(_row(sample, '甲丙', hypotheses, action='retry')), _manifest(sample))
    forced = result['metrics']['forced_eos']
    assert forced['marked_candidates'] == 1 and forced['reported_candidates'] == 2
    assert forced['unreported_candidates'] == 1
    assert forced['samples_with_any_marked'] == forced['samples_reverse_skipped_for_forced_eos'] == 1
    assert forced['top1_marked_samples'] == 0
    assert result['failure_causes_established'] is False
    assert result['correctness_probabilities_produced'] is False


def test_edit_space_support_is_reference_free_and_does_not_promote_duplicate_beams():
    sample = _sample('a', reference='己己')
    report = _report(_row(sample, '甲甲', ['甲甲', '甲乙', '甲甲。', '乙乙']))
    result = diagnose(report, _manifest(sample), include_samples=True)
    support = result['samples'][0]['diversity']['edit_space_support']
    assert support['first_unique_candidate_ranks'] == [1, 2, 4]
    assert support['mean_normalized_distances'] == [.75, .5, .75]
    assert support['medoid_first_ranks'] == [2] and support['top1_is_medoid'] is False
    assert not support['references_used']
    sample['reference'] = report['samples'][0]['reference'] = '甲甲'
    other = diagnose(report, _manifest(sample), include_samples=True)
    assert other['samples'][0]['diversity']['edit_space_support'] == support
    assert other['metrics']['top1']['exact_sentences'] == 1


def test_reference_length_bins_duration_boundaries_and_speakers_preserve_every_row():
    lengths = [20, 21, 40, 41, 60, 61]
    durations = [2, 4, 8, 16, 17, None]
    samples = [_sample(str(index), reference='甲' * length, speaker='S' + str(index % 2))
               for index, length in enumerate(lengths)]
    report = _report(*[_row(sample, '', duration_seconds=duration) for sample, duration in zip(samples, durations)])
    result = diagnose(report, _manifest(*samples))
    assert [result['per_reference_length'][name]['samples'] for name in result['reference_length_bins']] == [1, 2, 2, 1]
    assert all(result['per_duration'][name]['samples'] == 1 for name in result['duration_bins_seconds'])
    assert all(value['samples'] == 3 for value in result['per_speaker'].values())
    assert result['metrics']['top1']['deletions'] == sum(lengths)
    assert result['metrics']['top1']['cer'] == 1


def test_nested_adapter_metrics_are_verified_and_require_explicit_stage_and_split():
    sample = _sample('a', split='dev')
    section = _report(_row(sample, '甲乙', ['甲乙']))
    nested = {'baseline': {'dev': section}, 'candidate': {'dev': deepcopy(section)}}
    manifest = _manifest(sample, split='dev')
    with pytest.raises(ValueError, match='explicit --stage and --split'):
        diagnose(nested, manifest)
    result = diagnose(nested, manifest, stage='candidate', split='dev')
    assert result['metrics']['samples'] == 1
    nested['candidate']['dev']['metrics'] = {'samples': 2}
    with pytest.raises(ValueError, match='full sample rows'):
        diagnose(nested, manifest, stage='candidate', split='dev')
    with pytest.raises(ValueError, match='different declared split'):
        diagnose(section, manifest, split='test')


@pytest.mark.parametrize('candidate', [{'text': None}, {'text': '甲乙', 'forced_eos': 1},
                                      {'text': '甲乙', 'score': float('nan')}, {'text': '甲乙', 'score': True},
                                      {'text': '甲乙', 'reverse_rerank_skipped_reason': 3}])
def test_malformed_candidate_evidence_is_rejected_not_dropped(candidate):
    sample = _sample('a')
    with pytest.raises(ValueError):
        diagnose(_report(_row(sample, '甲乙', [candidate])), _manifest(sample))


@pytest.mark.parametrize('duration', [0, -1, True, float('nan'), '2'])
def test_invalid_durations_cannot_reclassify_or_drop_samples(duration):
    sample = _sample('a')
    with pytest.raises(ValueError, match='duration_seconds'):
        diagnose(_report(_row(sample, '甲乙', duration_seconds=duration)), _manifest(sample))


def test_output_has_no_predicted_or_reference_text_even_with_sample_metadata():
    sample = _sample('safe-id', reference='SECRET_REFERENCE_TEXT')
    report = _report(_row(sample, 'SECRET_PREDICTION_TEXT', ['SECRET_PREDICTION_TEXT', 'SECRET_ALTERNATIVE_TEXT']))
    for include in (False, True):
        result = diagnose(report, _manifest(sample), include_samples=include)
        serialized = json.dumps(result)
        assert 'SECRET_REFERENCE_TEXT' not in serialized
        assert 'SECRET_PREDICTION_TEXT' not in serialized
        assert 'SECRET_ALTERNATIVE_TEXT' not in serialized
        assert ('samples' in result) == include
        assert not result['raw_text_included']
        assert result['references_used_for_offline_oracle']
        assert not result['references_used_for_training'] and not result['references_used_for_decoding']


def test_cli_requires_manifest_hashes_inputs_and_prevents_overwrite(tmp_path, capsys):
    sample = _sample('a')
    paths = {name: tmp_path / name for name in ('report.json', 'manifest.json', 'output.json')}
    paths['report.json'].write_text(json.dumps(_report(_row(sample, '甲乙', ['甲乙']))), encoding='utf-8')
    paths['manifest.json'].write_text(json.dumps(_manifest(sample)), encoding='utf-8')
    arguments = [str(paths['report.json']), '--manifest', str(paths['manifest.json'])]
    assert main([*arguments, '--output', str(paths['output.json'])]) == 0
    result = json.loads(paths['output.json'].read_text())
    assert result['provenance']['report_sha256'] == hashlib.sha256(paths['report.json'].read_bytes()).hexdigest()
    assert result['provenance']['manifest_sha256'] == hashlib.sha256(paths['manifest.json'].read_bytes()).hexdigest()
    capsys.readouterr()
    for name in ('report.json', 'manifest.json'):
        original = paths[name].read_bytes()
        with pytest.raises(SystemExit) as error:
            main([*arguments, '--output', str(paths[name])])
        assert error.value.code == 2
        assert paths[name].read_bytes() == original
        capsys.readouterr()
    with pytest.raises(SystemExit) as error:
        main([str(paths['report.json'])])
    assert error.value.code == 2


def test_standalone_help_has_no_model_camera_or_site_package_dependencies(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'research' / 'scripts' / 'diagnose_chinese_candidates.py'
    result = subprocess.run([sys.executable, '-S', str(script), '--help'], cwd=tmp_path,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert '--include-samples' in result.stdout
