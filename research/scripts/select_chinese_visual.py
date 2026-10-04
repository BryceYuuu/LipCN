"""Select a fixed pixel preprocessing preset on clean AND stressed dev videos.

Raw text, failures, and every declared dev video remain in scoring. The model,
decoder, original pixels and synthetic degradation must be identical across
arms. A preset needs fewer pooled errors and no clean-CER, exact-match, coverage
or failure regression. References are used only by this offline scoring tool.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lipflow_research_path import REPOSITORY_ROOT

import argparse
import hashlib
import json
import re

from compare_chinese_reports import compare


def _binding(report):
    model = report.get('model')
    if not isinstance(model, dict):
        raise ValueError('model provenance is required')
    keys = ('name', 'language', 'checkpoint_sha256', 'configuration_sha256',
            'vocabulary_sha256', 'file_sha256', 'adapter', 'beam_size', 'ctc_weight',
            'nbest', 'reverse_weight', 'length_bonus', 'external_lm', 'personal',
            'decoder_weights', 'encoder_device', 'decoder_device')
    result = {key: model.get(key) for key in keys}
    # Older reports used nbest as the reverse-scoring pool and ESPnet's 1.5
    # shortlist ratio. A visual comparison must not silently change search too.
    result['candidate_pool_size'] = model.get('candidate_pool_size', model.get('nbest'))
    result['pre_beam_ratio'] = model.get('pre_beam_ratio', 1.5)
    result['reverse_scoring'] = model.get('reverse_scoring', 'sequential')
    result['reverse_tie_tolerance'] = model.get('reverse_tie_tolerance', 1e-3)
    if not (result['checkpoint_sha256'] or result['file_sha256']):
        raise ValueError('model weight fingerprints are required')
    return result


def _configuration(report):
    config = report.get('model', {}).get('visual_configuration')
    if not isinstance(config, dict) or set(config) != {'preprocessing', 'stress', 'seed'}:
        raise ValueError('complete visual configuration is required')
    if config['preprocessing'] not in ('identity', 'stabilize_mild', 'stabilize_moderate'):
        raise ValueError('unknown preprocessing preset')
    if config['stress'] not in ('clean', 'synthetic_mild', 'synthetic_moderate'):
        raise ValueError('unknown stress preset')
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2 ** 32:
        raise ValueError('invalid visual seed')
    return config


def _pair(baseline, candidate, manifest):
    if _binding(baseline) != _binding(candidate):
        raise ValueError('visual selection requires identical models and decoder settings')
    before, after = _configuration(baseline), _configuration(candidate)
    if any(before[key] != after[key] for key in ('stress', 'seed')):
        raise ValueError('visual selection requires identical stress and seed')
    paired = compare(baseline, candidate, manifest=manifest, resamples=1000)
    originals = {row['sample_id']: row for row in baseline['samples']}
    for row in candidate['samples']:
        original_row = originals[row['sample_id']]
        left, right = original_row.get('visual_input'), row.get('visual_input')
        for evidence, source_row, config in ((left, original_row, before), (right, row, after)):
            if evidence is None and source_row.get('error'):
                continue  # Failure remains scored and prevents promotion.
            if not isinstance(evidence, dict) or evidence.get('configuration') != config:
                raise ValueError('successful rows require matching visual configuration evidence')
            for key in ('original_pixels_sha256', 'input_pixels_sha256'):
                if not isinstance(evidence.get(key), str) or not re.fullmatch('[0-9a-f]{64}', evidence[key]):
                    raise ValueError('valid pixel SHA256 is required')
            shape = evidence.get('shape')
            if (not isinstance(shape, list) or len(shape) != 3 or any(type(x) is not int or x < 1 for x in shape)
                    or evidence.get('dtype') != 'uint8'):
                raise ValueError('valid visual shape/dtype evidence is required')
            expected_seed = (config['seed'] + int.from_bytes(hashlib.sha256(source_row['sample_id'].encode()).digest()[:8], 'big')) % (2 ** 32)
            if type(evidence.get('sample_seed')) is not int or evidence['sample_seed'] != expected_seed:
                raise ValueError('per-sample visual seed differs from its frozen configuration')
        if left is not None and right is not None:
            for key in ('original_pixels_sha256', 'shape', 'dtype', 'sample_seed'):
                if left[key] != right[key]:
                    raise ValueError('paired visual inputs differ before preprocessing')
    return paired


def select(baseline_clean, baseline_stress, candidate_pairs, manifest):
    if manifest.get('split') != 'dev' or not manifest.get('samples') or any(
            row.get('split', 'dev') != 'dev' for row in manifest['samples']):
        raise ValueError('Only complete dev partitions may select preprocessing')
    clean_config, stress_config = _configuration(baseline_clean), _configuration(baseline_stress)
    if clean_config['preprocessing'] != 'identity' or stress_config['preprocessing'] != 'identity':
        raise ValueError('baseline preprocessing must be identity')
    if clean_config['stress'] != 'clean' or stress_config['stress'] == 'clean':
        raise ValueError('one clean and one explicitly synthetic baseline are required')
    if clean_config['seed'] != stress_config['seed'] or _binding(baseline_clean) != _binding(baseline_stress):
        raise ValueError('clean and stress baselines require identical model and seed')
    self_clean = _pair(baseline_clean, baseline_clean, manifest)
    self_stress = _pair(baseline_stress, baseline_stress, manifest)
    before = [self_clean['baseline'], self_stress['baseline']]
    denominator = sum(row['reference_characters'] for row in before)
    best_errors = sum(row['character_errors'] for row in before)
    selected, evaluated, seen = 'identity', [], set()
    for clean, stressed in candidate_pairs:
        config, other = _configuration(clean), _configuration(stressed)
        preset = config['preprocessing']
        if preset == 'identity' or preset in seen or other['preprocessing'] != preset:
            raise ValueError('candidate presets must be distinct matched non-identity pairs')
        if config['stress'] != 'clean' or other['stress'] != stress_config['stress']:
            raise ValueError('candidate stress does not match the frozen baseline')
        seen.add(preset)
        pairs = [_pair(baseline_clean, clean, manifest), _pair(baseline_stress, stressed, manifest)]
        after = [pair['candidate'] for pair in pairs]
        errors = sum(row['character_errors'] for row in after)
        checks = {
            'pooled_raw_errors_strictly_lower': errors < sum(row['character_errors'] for row in before),
            'clean_raw_cer_not_worse': after[0]['cer'] <= before[0]['cer'],
            'exact_sentences_not_lower_in_either_condition': all(a['exact_sentences'] >= b['exact_sentences'] for a, b in zip(after, before)),
            'coverage_not_lower_in_either_condition': all(a['non_rejected_coverage'] >= b['non_rejected_coverage'] for a, b in zip(after, before)),
            'no_baseline_or_candidate_inference_failures': all(row['inference_failures'] == 0 for row in before + after),
        }
        eligible = all(checks.values())
        evaluated.append({'preprocessing': preset, 'eligible': eligible, 'checks': checks,
                          'clean': pairs[0], 'synthetic_stress': pairs[1],
                          'pooled_cer': errors / denominator})
        if eligible and errors < best_errors:
            selected, best_errors = preset, errors
    return {'schema_version': 1, 'selected_preprocessing': selected,
            'selection_partition': 'dev', 'test_used_for_selection': False,
            'baseline_clean': before[0], 'baseline_synthetic_stress': before[1],
            'selected_pooled_cer': best_errors / denominator,
            'stress_configuration': stress_config, 'model_binding': _binding(baseline_clean),
            'candidates': evaluated, 'automatic_application_activation': False,
            'limitations': ['Synthetic degradation does not establish real webcam performance.',
                            'Pixel stabilization does not resolve visually ambiguous Mandarin phonemes.',
                            'Published model weights and derived adapters retain their source licenses.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--baseline-clean', required=True)
    parser.add_argument('--baseline-stress', required=True)
    parser.add_argument('--candidate-clean', action='append', default=[])
    parser.add_argument('--candidate-stress', action='append', default=[])
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if len(args.candidate_clean) != len(args.candidate_stress) or not args.candidate_clean:
        parser.error('provide matched --candidate-clean and --candidate-stress reports')
    paths = [args.manifest, args.baseline_clean, args.baseline_stress, *args.candidate_clean, *args.candidate_stress]
    if Path(args.output).resolve() in {Path(path).resolve() for path in paths}:
        parser.error('selection output must not overwrite an input report or manifest')
    try:
        inputs = {path: Path(path).read_bytes() for path in paths}
        read = lambda path: json.loads(inputs[path])
        result = select(read(args.baseline_clean), read(args.baseline_stress),
                        [(read(a), read(b)) for a, b in zip(args.candidate_clean, args.candidate_stress)],
                        read(args.manifest))
        result['manifest_sha256'] = hashlib.sha256(inputs[args.manifest]).hexdigest()
        result['input_sha256'] = {str(Path(path).resolve()): hashlib.sha256(data).hexdigest()
                                  for path, data in inputs.items() if path != args.manifest}
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(result['selected_preprocessing'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
