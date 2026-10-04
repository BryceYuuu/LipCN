"""Select candidate-generation settings from complete clean/stressed DEV reports.

Only candidate_pool_size and pre_beam_ratio may change. No labels enter search;
this offline selector scores raw outputs after inference. A held-out report is
never accepted for selecting settings. Defaults stay unchanged in the GUI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lipflow_research_path import REPOSITORY_ROOT
from compare_chinese_reports import compare
from compare_chinese_cohorts import _paired_pixels
from select_chinese_visual import _binding, _configuration


def settings(report):
    model = report.get('model', {})
    beam, nbest = model.get('beam_size'), model.get('nbest')
    pool = model.get('candidate_pool_size', nbest)
    ratio = model.get('pre_beam_ratio', 1.5)
    if (any(type(v) is not int for v in (beam, nbest, pool))
            or not 1 <= nbest <= pool <= beam):
        raise ValueError('search requires integer beam_size >= candidate_pool_size >= nbest >= 1')
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 1 <= ratio <= 4:
        raise ValueError('pre_beam_ratio must be finite within [1,4]')
    return {'candidate_pool_size': pool, 'pre_beam_ratio': float(ratio)}


def binding(report):
    # Selection can promote a new search recipe, so missing provenance cannot
    # compare equal as two None values. Legacy visual-report compatibility is
    # intentionally left in _binding; this stricter boundary belongs here.
    if not isinstance(report, dict) or not isinstance(report.get('model'), dict):
        raise ValueError('search selection requires complete model provenance')
    model = report['model']

    def digest(value, key):
        if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
            raise ValueError(f'search model {key} requires a lowercase 64-hex sha256')
        return value

    for key in ('name', 'encoder_device', 'decoder_device'):
        if not isinstance(model.get(key), str) or not model[key].strip():
            raise ValueError(f'search model requires nonempty {key}')
    if model.get('language') != 'zh':
        raise ValueError('search model language must be zh')
    weights = []
    for key in ('checkpoint_sha256', 'file_sha256'):
        if model.get(key) is not None:
            weights.append(digest(model[key], key))
    if not weights:
        raise ValueError('search model weight fingerprints are required')
    for key in ('configuration_sha256', 'vocabulary_sha256'):
        digest(model.get(key), key)
    if 'adapter' not in model:
        raise ValueError('search model must explicitly declare adapter, including null when absent')
    adapter = model['adapter']
    if adapter is not None:
        if not isinstance(adapter, dict):
            raise ValueError('search model adapter must be an object or null')
        digest(adapter.get('file_sha256'), 'adapter.file_sha256')
        base = digest(adapter.get('base_checkpoint_sha256'), 'adapter.base_checkpoint_sha256')
        if (type(adapter.get('schema_version')) is not int or adapter['schema_version'] != 1
                or adapter.get('language') != 'zh'):
            raise ValueError('search adapter requires schema_version=1 and language=zh')
        if base != model.get('checkpoint_sha256'):
            raise ValueError('search adapter is bound to a different model checkpoint')
    # This also validates beam/nbest and the two search settings even when this
    # helper is used directly rather than through select().
    settings(report)
    for key in ('ctc_weight', 'reverse_weight', 'length_bonus'):
        value = model.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'search model {key} must be finite numeric evidence')
        if key != 'length_bonus' and not 0 <= value <= 1:
            raise ValueError(f'search model {key} must be within [0,1]')
    for key in ('external_lm', 'personal'):
        if type(model.get(key)) is not bool:
            raise ValueError(f'search model {key} must be explicitly boolean')
    if 'decoder_weights' in model:
        declared = model['decoder_weights']
        if (not isinstance(declared, dict) or not declared
                or any(not isinstance(key, str) or not key.strip()
                       or type(value) not in (int, float) or not math.isfinite(value)
                       for key, value in declared.items())):
            raise ValueError('search model decoder_weights must be nonempty finite named weights')
    reverse_mode = model.get('reverse_scoring', 'sequential')
    tolerance = model.get('reverse_tie_tolerance', 1e-3)
    if reverse_mode not in ('sequential', 'batched'):
        raise ValueError('search model reverse_scoring must be sequential or batched')
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('search model reverse_tie_tolerance must be finite and nonnegative')
    result = _binding(report)
    for key in ('candidate_pool_size', 'pre_beam_ratio'):
        result.pop(key)
    result['reverse_scoring'] = reverse_mode
    result['reverse_tie_tolerance'] = float(tolerance)
    return result


def paired(baseline, candidate, manifest):
    if any(row.get('split') != 'dev' for report in (baseline, candidate) for row in report.get('samples', [])):
        raise ValueError('Only dev report rows may select search settings')
    if binding(baseline) != binding(candidate):
        raise ValueError('search selection requires identical model, adapter and other decoder settings')
    if _configuration(baseline) != _configuration(candidate):
        raise ValueError('search selection requires identical visual configurations')
    result = compare(baseline, candidate, manifest=manifest, resamples=1000)
    pixels = _paired_pixels(baseline, candidate)
    if pixels['prepared_same_preprocessing_pixels_verified_samples'] != len(manifest['samples']):
        raise ValueError('search selection requires prepared pixel evidence for every video')
    result['paired_pixels'] = pixels
    return result


def select(baseline_clean, baseline_stress, candidate_pairs, manifest):
    if (not isinstance(manifest, dict) or type(manifest.get('schema_version')) is not int
            or manifest['schema_version'] != 1 or not isinstance(manifest.get('samples'), list)
            or any(not isinstance(row, dict) for row in manifest['samples'])):
        raise ValueError('frozen manifest requires schema_version=1 and sample objects')
    if (manifest.get('split') != 'dev' or not manifest.get('samples')
            or any(row.get('split', 'dev') != 'dev' for row in manifest['samples'])):
        raise ValueError('Only complete dev partitions may select search settings')
    if any(not isinstance(row.get('sha256'), str) or re.fullmatch(r'[0-9a-f]{64}', row['sha256']) is None
           for row in manifest['samples']):
        raise ValueError('every frozen manifest sample requires a lowercase 64-hex sha256')
    clean, stress = _configuration(baseline_clean), _configuration(baseline_stress)
    if (clean['stress'] != 'clean' or stress['stress'] == 'clean'
            or any(clean[k] != stress[k] for k in ('seed', 'preprocessing'))):
        raise ValueError('one clean and one matched explicitly synthetic condition are required')
    original = settings(baseline_clean)
    if settings(baseline_stress) != original or binding(baseline_clean) != binding(baseline_stress):
        raise ValueError('baseline conditions must use identical search and model settings')
    before = [paired(report, report, manifest)['baseline'] for report in (baseline_clean, baseline_stress)]
    baseline_errors = sum(r['character_errors'] for r in before)
    best_errors, selected = baseline_errors, original
    denominator = sum(r['reference_characters'] for r in before)
    seen, evaluated = {tuple(original.values())}, []
    for candidate_clean, candidate_stress in candidate_pairs:
        config = settings(candidate_clean)
        if settings(candidate_stress) != config or tuple(config.values()) in seen:
            raise ValueError('search candidates must be distinct matched pairs')
        seen.add(tuple(config.values()))
        comparisons = [paired(a, b, manifest) for a, b in zip(
            (baseline_clean, baseline_stress), (candidate_clean, candidate_stress))]
        after = [row['candidate'] for row in comparisons]
        errors = sum(r['character_errors'] for r in after)
        checks = {
            'pooled_errors_strictly_lower': errors < baseline_errors,
            'raw_errors_not_worse_in_either_condition': all(a['character_errors'] <= b['character_errors'] for a, b in zip(after, before)),
            'exact_sentences_not_lower': all(a['exact_sentences'] >= b['exact_sentences'] for a, b in zip(after, before)),
            'coverage_not_lower': all(a['non_rejected_coverage'] >= b['non_rejected_coverage'] for a, b in zip(after, before)),
            'no_inference_failures': all(r['inference_failures'] == 0 for r in before + after),
        }
        eligible = all(checks.values())
        evaluated.append({'settings': config, 'eligible': eligible, 'checks': checks,
                          'pooled_cer': errors / denominator, 'comparisons': comparisons})
        # Equal error counts never justify extra search. Input order is frozen
        # before inference, with the less expensive candidate first.
        if eligible and errors < best_errors:
            best_errors, selected = errors, config
    return {'schema_version': 1, 'selection_partition': 'dev', 'test_used_for_selection': False,
            'baseline_settings': original, 'selected_settings': selected,
            'model_binding': binding(baseline_clean), 'visual_clean': clean, 'visual_stress': stress,
            'baseline_clean': before[0], 'baseline_stress': before[1],
            'selected_pooled_cer': best_errors / denominator, 'candidates': evaluated,
            'automatic_application_activation': False,
            'limitations': ['Oracle candidate scores are diagnostic only and never select a sentence for live input.',
                            'Development gains need a fresh held-out test after settings are frozen.',
                            'Synthetic degradation is not a real webcam or silent-speech test.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--baseline-clean', type=Path, required=True)
    parser.add_argument('--baseline-stress', type=Path, required=True)
    parser.add_argument('--candidate-clean', type=Path, action='append', default=[])
    parser.add_argument('--candidate-stress', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.candidate_clean) != len(args.candidate_stress) or not args.candidate_clean:
        parser.error('provide matched candidate-clean and candidate-stress reports')
    paths = [args.manifest, args.baseline_clean, args.baseline_stress,
             *args.candidate_clean, *args.candidate_stress]
    if args.output.resolve() in {p.resolve() for p in paths}:
        parser.error('output must not overwrite an input')
    try:
        data = {p: p.read_bytes() for p in paths}
        read = lambda p: json.loads(data[p])
        result = select(read(args.baseline_clean), read(args.baseline_stress),
                        [(read(a), read(b)) for a, b in zip(args.candidate_clean, args.candidate_stress)], read(args.manifest))
        result['manifest_sha256'] = hashlib.sha256(data[args.manifest]).hexdigest()
        result['input_sha256'] = {str(p.resolve()): hashlib.sha256(d).hexdigest() for p, d in data.items()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result['selected_settings']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
