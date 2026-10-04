"""Search choices must use frozen dev top-1 results on identical pixels."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research' / 'scripts'))
from select_chinese_search import select, settings, main
from research.evaluation import Dataset, Prediction, Sample, evaluate


def report(stress, raw='今天不要发送', *, pool=None, ratio=None, action='review', error=None):
    samples = tuple(Sample(str(index), f'/local/{index}.mp4', '今天不要发送', f'speaker{index}',
                           f'session{index}', 'independent label', True, 'mouth_roi', True,
                           sha256=hashlib.sha256(str(index).encode()).hexdigest(), split='dev') for index in (1, 2))
    data = Dataset(samples, split='dev')
    raws = raw if isinstance(raw, list) else [raw] * len(samples)
    result = evaluate(data, [Prediction(sample.id, text, 2, 1, action, error=error)
                             for sample, text in zip(samples, raws)])
    result['model'] = {'name': 'CNVSRC2025', 'language': 'zh', 'checkpoint_sha256': 'f' * 64,
                       'configuration_sha256': 'c' * 64, 'vocabulary_sha256': 'd' * 64,
                       'adapter': None, 'beam_size': 40, 'nbest': 10, 'ctc_weight': .1,
                       'reverse_weight': .3, 'length_bonus': 0.0, 'external_lm': False,
                       'personal': False, 'encoder_device': 'cpu', 'decoder_device': 'cpu',
                       'visual_configuration': {'preprocessing': 'identity', 'stress': stress, 'seed': 0}}
    if pool is not None:
        result['model']['candidate_pool_size'] = pool
    if ratio is not None:
        result['model']['pre_beam_ratio'] = ratio
    for row in result['samples']:
        identity = row['sample_id']
        row['visual_input'] = {
            'configuration': dict(result['model']['visual_configuration']),
            'original_pixels_sha256': hashlib.sha256(f'original-{identity}'.encode()).hexdigest(),
            'input_pixels_sha256': hashlib.sha256(f'{stress}-{identity}'.encode()).hexdigest(),
            'shape': [50, 96, 96], 'dtype': 'uint8',
            'sample_seed': int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'big') % (2 ** 32)}
    manifest = {'schema_version': 1, 'split': 'dev', 'samples': [
        {'id': sample.id, 'reference': sample.reference, 'speaker': sample.speaker,
         'session': sample.session, 'sha256': sample.sha256, 'split': 'dev',
         'domain': 'mouth_roi', 'articulation': 'unknown', 'articulation_verified': False} for sample in samples]}
    return result, manifest


def arms():
    clean, manifest = report('clean')
    stressed, _ = report('synthetic_mild', '今天发送')
    candidate_clean, _ = report('clean', pool=20)
    candidate_stressed, _ = report('synthetic_mild', pool=20)
    return clean, stressed, candidate_clean, candidate_stressed, manifest


def test_selects_strictly_better_raw_results_without_mutating_frozen_inputs():
    a, b, c, d, manifest = arms()
    frozen = deepcopy((a, b, c, d, manifest))
    result = select(a, b, [(c, d)], manifest)
    assert (a, b, c, d, manifest) == frozen
    assert result['baseline_settings'] == {'candidate_pool_size': 10, 'pre_beam_ratio': 1.5}
    assert result['selected_settings'] == {'candidate_pool_size': 20, 'pre_beam_ratio': 1.5}
    assert result['selected_pooled_cer'] == 0
    assert result['selection_partition'] == 'dev' and not result['test_used_for_selection']
    assert not result['automatic_application_activation']
    assert result['candidates'][0]['comparisons'][0]['paired_pixels']['all_original_pixels_verified']


def test_equal_raw_errors_keep_original_settings_even_if_candidates_have_perfect_oracles():
    a, b, c, d, manifest = arms()
    d, _ = report('synthetic_mild', '今天发送', pool=20)
    for report_data in (c, d):
        report_data['oracle_at_k'] = {'10': {'cer': 0, 'exact_sentence_rate': 1}}
        report_data['oracle_cer'] = 0
        for row in report_data['samples']:
            row['hypotheses'] = [{'text': row['raw']}, {'text': row['reference']}]
            row['oracle_character_errors'] = 0
    result = select(a, b, [(c, d)], manifest)
    assert result['selected_settings'] == result['baseline_settings']
    assert not result['candidates'][0]['checks']['pooled_errors_strictly_lower']


def test_clean_regression_cannot_be_hidden_by_larger_stress_improvement():
    a, manifest = report('clean', '今天发送')
    b, _ = report('synthetic_mild', '发送')
    c, _ = report('clean', '今天送', pool=20)
    d, _ = report('synthetic_mild', pool=20)
    result = select(a, b, [(c, d)], manifest)
    checks = result['candidates'][0]['checks']
    assert checks['pooled_errors_strictly_lower']
    assert not checks['raw_errors_not_worse_in_either_condition']
    assert result['selected_settings'] == result['baseline_settings']


def test_exact_sentence_regression_is_checked_even_when_clean_error_count_is_equal():
    a, manifest = report('clean', ['今天不要发送', '今天发送'])
    b, _ = report('synthetic_mild', '今天发送')
    c, _ = report('clean', '今天不要发', pool=20)
    d, _ = report('synthetic_mild', pool=20)
    result = select(a, b, [(c, d)], manifest)
    checks = result['candidates'][0]['checks']
    assert checks['pooled_errors_strictly_lower'] and checks['raw_errors_not_worse_in_either_condition']
    assert not checks['exact_sentences_not_lower']
    assert result['selected_settings'] == result['baseline_settings']


def test_rejected_outputs_and_decoder_failures_are_retained_in_full_denominators():
    a, b, c, d, manifest = arms()
    d, _ = report('synthetic_mild', ['', '今天不要发送'], pool=20, error='decode failed', action='retry')
    result = select(a, b, [(c, d)], manifest)
    candidate = result['candidates'][0]['comparisons'][1]['candidate']
    assert candidate['samples'] == 2 and candidate['reference_characters'] == 12
    assert candidate['empty_outputs'] == 1
    assert candidate['inference_failures'] == 2
    assert candidate['character_errors'] == 6 and candidate['cer'] == .5
    assert candidate['non_rejected_coverage'] == 0
    assert not result['candidates'][0]['checks']['no_inference_failures']
    assert result['selected_settings'] == result['baseline_settings']


def test_retry_with_correct_raw_text_cannot_pass_coverage_gate():
    a, b, c, d, manifest = arms()
    d, _ = report('synthetic_mild', pool=20, action='retry')
    result = select(a, b, [(c, d)], manifest)
    checks = result['candidates'][0]['checks']
    assert checks['pooled_errors_strictly_lower'] and checks['no_inference_failures']
    assert not checks['coverage_not_lower']
    assert result['candidates'][0]['comparisons'][1]['candidate']['samples'] == 2
    assert result['selected_settings'] == result['baseline_settings']


@pytest.mark.parametrize('partition', ['test', 'train', 'heldout'])
def test_only_complete_development_partition_can_select(partition):
    a, b, c, d, manifest = arms()
    manifest['split'] = partition
    with pytest.raises(ValueError, match='Only complete dev'):
        select(a, b, [(c, d)], manifest)


def test_test_sample_hidden_in_development_manifest_is_rejected():
    a, b, c, d, manifest = arms()
    manifest['samples'][0]['split'] = 'test'
    with pytest.raises(ValueError, match='Only complete dev'):
        select(a, b, [(c, d)], manifest)


def test_manifest_must_explicitly_declare_development_partition():
    a, b, c, d, manifest = arms()
    manifest.pop('split')
    with pytest.raises(ValueError, match='Only complete dev'):
        select(a, b, [(c, d)], manifest)


@pytest.mark.parametrize('reported_split', [None, 'test', 'train'])
def test_reports_cannot_hide_missing_or_non_dev_rows_behind_manifest_header(reported_split):
    a, b, c, d, manifest = arms()
    for sample in manifest['samples']:
        sample.pop('split')
    for item in (a, b, c, d):
        for row in item['samples']:
            if reported_split is None:
                row.pop('split')
            else:
                row['split'] = reported_split
    with pytest.raises(ValueError):
        select(a, b, [(c, d)], manifest)


def test_symmetric_sample_omission_cannot_shrink_the_frozen_corpus():
    a, b, c, d, manifest = arms()
    for item in (a, b, c, d):
        item['samples'].pop()
        item.pop('metrics')
    with pytest.raises(ValueError, match='every frozen manifest sample'):
        select(a, b, [(c, d)], manifest)


@pytest.mark.parametrize('field,value', [
    ('checkpoint_sha256', 'a' * 64), ('configuration_sha256', 'a' * 64),
    ('vocabulary_sha256', 'a' * 64),
    ('adapter', {'file_sha256': 'a' * 64, 'schema_version': 1,
                 'language': 'zh', 'base_checkpoint_sha256': 'f' * 64}),
    ('beam_size', 50), ('nbest', 5), ('ctc_weight', .5), ('reverse_weight', .6),
    ('length_bonus', 1), ('external_lm', True), ('personal', True),
    ('reverse_scoring', 'batched'), ('reverse_tie_tolerance', .01),
])
def test_model_adapter_and_other_decoder_changes_cannot_be_search_only_comparison(field, value):
    a, b, c, d, manifest = arms()
    c['model'][field] = value
    d['model'][field] = value
    with pytest.raises(ValueError, match='identical model, adapter and other decoder settings'):
        select(a, b, [(c, d)], manifest)


@pytest.mark.parametrize('mutation', ['original_pixels', 'input_pixels', 'missing_pixels', 'shape',
                                      'sample_seed', 'visual_configuration', 'reference', 'speaker', 'video_sha256'])
def test_identical_source_and_prepared_pixels_and_frozen_identities_are_required(mutation):
    a, b, c, d, manifest = arms()
    row = c['samples'][0]
    if mutation == 'original_pixels':
        row['visual_input']['original_pixels_sha256'] = 'a' * 64
    elif mutation == 'input_pixels':
        row['visual_input']['input_pixels_sha256'] = 'a' * 64
    elif mutation == 'missing_pixels':
        row['visual_input'] = None
    elif mutation == 'shape':
        row['visual_input']['shape'] = [49, 96, 96]
    elif mutation == 'sample_seed':
        row['visual_input']['sample_seed'] += 1
    elif mutation == 'visual_configuration':
        c['model']['visual_configuration']['preprocessing'] = 'stabilize_mild'
    elif mutation == 'reference':
        row['reference'] = '不要发送今天'
    elif mutation == 'speaker':
        row['speaker'] = 'another-person'
    else:
        row['video_sha256'] = 'a' * 64
    with pytest.raises(ValueError):
        select(a, b, [(c, d)], manifest)


def test_matched_distinct_candidate_pairs_and_equal_score_cost_ties():
    a, b, c, d, manifest = arms()
    with pytest.raises(ValueError, match='distinct matched pairs'):
        select(a, b, [(c, d), (deepcopy(c), deepcopy(d))], manifest)
    changed = deepcopy(d)
    changed['model']['candidate_pool_size'] = 30
    with pytest.raises(ValueError, match='distinct matched pairs'):
        select(a, b, [(c, changed)], manifest)
    expensive_clean, expensive_stress = deepcopy(c), deepcopy(d)
    for item in (expensive_clean, expensive_stress):
        item['model']['candidate_pool_size'] = 30
    result = select(a, b, [(c, d), (expensive_clean, expensive_stress)], manifest)
    assert result['selected_settings']['candidate_pool_size'] == 20


@pytest.mark.parametrize('key,value', [('candidate_pool_size', 9), ('candidate_pool_size', 41),
                                     ('candidate_pool_size', True), ('pre_beam_ratio', .9),
                                     ('pre_beam_ratio', 4.1), ('pre_beam_ratio', float('nan')),
                                     ('pre_beam_ratio', True)])
def test_malformed_search_space_fails_before_comparison(key, value):
    item, _ = report('clean')
    item['model'][key] = value
    with pytest.raises(ValueError):
        settings(item)


def test_old_implicit_defaults_equal_new_explicit_search_defaults():
    old, _ = report('clean')
    new = deepcopy(old)
    new['model'].update(candidate_pool_size=10, pre_beam_ratio=1.5)
    assert settings(old) == settings(new)


@pytest.mark.parametrize('field', ['name', 'language', 'checkpoint_sha256', 'configuration_sha256',
                                  'vocabulary_sha256', 'adapter', 'beam_size', 'nbest', 'ctc_weight',
                                  'reverse_weight', 'length_bonus', 'external_lm', 'personal',
                                  'encoder_device', 'decoder_device'])
def test_absence_in_every_arm_is_not_equal_valid_model_provenance(field):
    a, b, c, d, manifest = arms()
    for item in (a, b, c, d):
        item['model'].pop(field)
    with pytest.raises(ValueError):
        select(a, b, [(c, d)], manifest)


def test_cli_hashes_all_frozen_inputs_and_cannot_overwrite_any_evidence(tmp_path, capsys):
    a, b, c, d, manifest = arms()
    files = {}
    for name, value in [('manifest', manifest), ('clean', a), ('stress', b),
                         ('candidate_clean', c), ('candidate_stress', d)]:
        files[name] = tmp_path / f'{name}.json'
        files[name].write_text(json.dumps(value), encoding='utf-8')
    output = tmp_path / 'nested/selection.json'
    arguments = ['--manifest', str(files['manifest']), '--baseline-clean', str(files['clean']),
                 '--baseline-stress', str(files['stress']), '--candidate-clean', str(files['candidate_clean']),
                 '--candidate-stress', str(files['candidate_stress'])]
    assert main([*arguments, '--output', str(output)]) == 0
    result = json.loads(output.read_text())
    assert result['selected_settings']['candidate_pool_size'] == 20
    assert result['manifest_sha256'] == hashlib.sha256(files['manifest'].read_bytes()).hexdigest()
    assert len(result['input_sha256']) == 5
    for path, digest in result['input_sha256'].items():
        assert digest == hashlib.sha256(Path(path).read_bytes()).hexdigest()
    capsys.readouterr()
    for path in files.values():
        original = path.read_bytes()
        with pytest.raises(SystemExit) as exc:
            main([*arguments, '--output', str(path)])
        assert exc.value.code == 2 and path.read_bytes() == original
        capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        main([*arguments, '--candidate-clean', str(files['candidate_clean']), '--output', str(output)])
    assert exc.value.code == 2


def test_standalone_help_requires_no_model_camera_or_site_packages(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'research' / 'scripts' / 'select_chinese_search.py'
    result = subprocess.run([sys.executable, '-S', str(script), '--help'], cwd=tmp_path,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
