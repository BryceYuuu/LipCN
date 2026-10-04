"""Audit stored raw Mandarin candidates in edit space, without model inference.

    python research/scripts/diagnose_chinese_candidates.py report.json \
        --manifest frozen-test.json --output diagnostics.json
    python research/scripts/diagnose_chinese_candidates.py adapter-report.json \
        --stage candidate --split dev --manifest dev.json --include-samples

References are used ONLY for offline error and oracle diagnostics. Oracle scores
are unattainable reference-selected lower bounds, never a decoder, correction,
training target or correctness probability. All failed, rejected and empty
samples remain in denominators. Missing candidate lists use explicitly disclosed
raw-only fallback, with candidate evidence availability reported separately.
Default output is aggregate and contains no reference or predicted text.
"""
from __future__ import annotations

import sys as _path_sys
from pathlib import Path as _SourcePath
_path_sys.path.insert(0, str(_SourcePath(__file__).resolve().parent))
from _lipflow_research_path import REPOSITORY_ROOT

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

from analyze_chinese_errors import alignment, select_rows, _bucket
from compare_chinese_reports import compare
from research.evaluation import character_error, characters

ORACLE_K = (1, 3, 5, 10)
REFERENCE_LENGTH_BINS = ('1-20', '21-40', '41-60', '61+')
DURATION_BINS = ('(0,2]', '(2,4]', '(4,8]', '(8,16]', '(16,+inf)', 'unknown')
EDIT_FIELDS = ('substitutions', 'deletions', 'insertions', 'matches',
               'character_errors', 'reference_characters', 'hypothesis_characters')


def _validated_section(report, manifest, stage, split):
    rows = select_rows(report, stage=stage, split=split)
    nested = 'baseline' in report or 'candidate' in report
    section = report[stage][split] if nested else report
    if (not isinstance(manifest, dict) or type(manifest.get('schema_version')) is not int
            or manifest['schema_version'] != 1 or not isinstance(manifest.get('samples'), list)
            or not manifest['samples']):
        raise ValueError('frozen manifest requires schema_version=1 and nonempty samples')
    for sample in manifest['samples']:
        if not isinstance(sample, dict):
            raise ValueError('manifest samples must be objects')
        digest = sample.get('sha256')
        if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            raise ValueError('every frozen manifest sample requires a lowercase 64-hex sha256')
    # Recompute all recorded raw counts and metrics, and demand exact coverage,
    # references, speaker identities and every declared frozen corpus field.
    checked = compare(section, section, manifest=manifest, resamples=1)
    return rows, checked['baseline']


def _edits(hypothesis, reference):
    evidence = alignment(hypothesis, reference)
    errors, count = character_error(hypothesis, reference)
    if evidence['character_errors'] != errors or evidence['reference_characters'] != count:
        raise ValueError('alignment and canonical CER disagree')
    return {key: evidence[key] for key in EDIT_FIELDS}


def _duration_bucket(value):
    if value is None:
        return 'unknown'
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('duration_seconds must be positive and finite, or null/absent')
    for boundary, name in zip((2, 4, 8, 16), DURATION_BINS):
        if value <= boundary:
            return name
    return '(16,+inf)'


def _candidate_list(row):
    supplied = row.get('hypotheses', [])
    if not isinstance(supplied, list):
        raise ValueError(f'{row["sample_id"]}: hypotheses must be a list when supplied')
    for item in supplied:
        if not isinstance(item, dict) or not isinstance(item.get('text'), str):
            raise ValueError(f'{row["sample_id"]}: each hypothesis requires string text')
        if 'forced_eos' in item and type(item['forced_eos']) is not bool:
            raise ValueError(f'{row["sample_id"]}: forced_eos must be boolean')
        if 'score' in item and (type(item['score']) not in (int, float) or not math.isfinite(item['score'])):
            raise ValueError(f'{row["sample_id"]}: candidate score must be finite numeric evidence')
        reason = item.get('reverse_rerank_skipped_reason')
        if reason is not None and not isinstance(reason, str):
            raise ValueError(f'{row["sample_id"]}: reverse_rerank_skipped_reason must be string or null')
    if supplied and supplied[0]['text'] != row['raw']:
        raise ValueError(f'{row["sample_id"]}: first hypothesis text must agree exactly with raw')
    return supplied


def _diversity(candidates):
    # Deduplicate ONLY for diversity/counting. Oracle rank cutoffs below preserve
    # every stored beam position, so duplicates cannot promote a later answer.
    unique = list(dict.fromkeys(''.join(characters(item['text'])) for item in candidates))
    first_ranks = [next(rank for rank, item in enumerate(candidates, 1)
                        if ''.join(characters(item['text'])) == text) for text in unique]
    per_candidate_distance = [0.0] * len(unique)
    pairs, edits, normalization, normalized_sum = 0, 0, 0, 0.0
    for index, left in enumerate(unique):
        for right in unique[index + 1:]:
            if (len(left) + 1) * (len(right) + 1) > 4_000_000:
                raise ValueError('candidate pair edit space exceeds 4,000,000 cells')
            distance = character_error(left, right)[0] if right else len(left)
            denominator = max(len(left), len(right), 1)
            pairs += 1
            edits += distance
            normalization += denominator
            normalized_sum += distance / denominator
            per_candidate_distance[index] += distance / denominator
            per_candidate_distance[unique.index(right)] += distance / denominator
    means = [total / (len(unique) - 1) for total in per_candidate_distance] if len(unique) > 1 else []
    medoids = [rank for rank, distance in zip(first_ranks, means)
               if abs(distance - min(means)) <= 1e-12] if means else []
    return {'unique_canonical_candidates': len(unique),
            'unique_literal_candidates': len({item['text'] for item in candidates}),
            'pairs': pairs, 'character_edit_distance': edits,
            'normalization_characters': normalization, 'normalized_distance_sum': normalized_sum,
            'edit_space_support': {'available': bool(means), 'references_used': False,
                                   'first_unique_candidate_ranks': first_ranks,
                                   'mean_normalized_distances': means,
                                   'medoid_first_ranks': medoids,
                                   'top1_is_medoid': 1 in medoids if means else None,
                                   'top1_mean_normalized_distance': means[0] if means else None,
                                   'definition': 'mean normalized edit distance to other distinct stored candidates; lower means closer to this list, not more correct'}}


def _sample(row):
    candidates = _candidate_list(row)
    raw_edits = _edits(row['raw'], row['reference'])
    candidate_edits = [_edits(item['text'], row['reference']) for item in candidates]
    oracle = {}
    for k in ORACLE_K:
        best, selected_rank = raw_edits, 1 if candidates else None
        candidate_exact = False
        for rank, edits in enumerate(candidate_edits[:k], 1):
            candidate_exact = candidate_exact or edits['character_errors'] == 0
            if edits['character_errors'] < best['character_errors']:
                best, selected_rank = edits, rank
        oracle[str(k)] = {**best, 'exact': best['character_errors'] == 0,
                          'candidate_list_exact_hit': candidate_exact,
                          'candidate_evidence_available': bool(candidates),
                          'at_least_k_hypotheses': len(candidates) >= k,
                          'observed_hypotheses': min(k, len(candidates)),
                          'selected_rank': selected_rank,
                          'raw_fallback': not candidates,
                          'character_errors_reduction': raw_edits['character_errors'] - best['character_errors']}
    forced = [item['forced_eos'] for item in candidates if 'forced_eos' in item]
    return {'sample_id': row['sample_id'], 'speaker': row['speaker'],
            'reference_length_bin': _bucket(raw_edits['reference_characters']),
            'duration_bin': _duration_bucket(row.get('duration_seconds')),
            'top1': {**raw_edits, 'exact': raw_edits['character_errors'] == 0},
            'inference_failure': bool(row.get('error')),
            'rejected': row.get('action', 'review') == 'retry',
            'empty_raw': raw_edits['hypothesis_characters'] == 0,
            'hypotheses_count': len(candidates), 'candidate_evidence_available': bool(candidates),
            'empty_candidate_texts': sum(not characters(item['text']) for item in candidates),
            'diversity': _diversity(candidates),
            'forced_eos': {'marked_candidates': sum(forced), 'reported_candidates': len(forced),
                           'unreported_candidates': len(candidates) - len(forced),
                           'any_marked': any(forced),
                           'top1_marked': bool(candidates) and candidates[0].get('forced_eos') is True,
                           'reverse_skipped_for_forced_eos': any(item.get('reverse_rerank_skipped_reason') == 'forced_eos_in_beam'
                                                                for item in candidates)},
            'oracle_at_k': oracle}


def _edit_metrics(rows):
    count, totals = len(rows), Counter()
    for row in rows:
        for key in EDIT_FIELDS:
            totals[key] += row[key]
        totals['exact_sentences'] += row['exact']
    references = totals['reference_characters']
    return {'samples': count, **{key: totals[key] for key in EDIT_FIELDS},
            'cer': totals['character_errors'] / references if references else None,
            'exact_sentences': totals['exact_sentences'],
            'exact_sentence_rate': totals['exact_sentences'] / count if count else None,
            'hypothesis_reference_length_ratio': totals['hypothesis_characters'] / references if references else None}


def _metrics(samples):
    count = len(samples)
    top1 = _edit_metrics([sample['top1'] for sample in samples])
    oracle = {}
    for k in ORACLE_K:
        rows = [sample['oracle_at_k'][str(k)] for sample in samples]
        observed = sum(row['candidate_evidence_available'] for row in rows)
        exact_known = sum(row['candidate_list_exact_hit'] for row in rows)
        entry = _edit_metrics(rows)
        entry.update({'available': count > 0 and observed == count,
                      'complete_k_evidence': count > 0 and all(row['at_least_k_hypotheses'] for row in rows),
                      'candidate_evidence_samples': observed, 'raw_fallback_samples': count - observed,
                      'candidate_evidence_available_for_every_sample': count > 0 and observed == count,
                      'samples_with_at_least_k_hypotheses': sum(row['at_least_k_hypotheses'] for row in rows),
                      'samples_with_fewer_than_k_hypotheses': sum(not row['at_least_k_hypotheses'] for row in rows),
                      'candidate_list_exact_hits': exact_known,
                      'candidate_list_exact_hit_rate_full_denominator': exact_known / count if count else None,
                      'candidate_list_exact_hit_rate_available_denominator': exact_known / observed if observed else None,
                      'character_errors_reduction_vs_raw': sum(row['character_errors_reduction'] for row in rows),
                      'raw_inclusive': True, 'score_definition': 'first stored k candidates plus raw; raw-only fallback if candidates unavailable'})
        oracle[str(k)] = entry
    pairs = sum(row['diversity']['pairs'] for row in samples)
    pair_errors = sum(row['diversity']['character_edit_distance'] for row in samples)
    pair_normalization = sum(row['diversity']['normalization_characters'] for row in samples)
    literal_count = sum(row['diversity']['unique_literal_candidates'] for row in samples)
    canonical_count = sum(row['diversity']['unique_canonical_candidates'] for row in samples)
    support = [row['diversity']['edit_space_support'] for row in samples
               if row['diversity']['edit_space_support']['available']]
    return {'available': bool(samples), 'samples': count, 'top1': top1, 'oracle_at_k': oracle,
            'candidate_evidence_samples': sum(row['candidate_evidence_available'] for row in samples),
            'missing_candidate_samples': sum(not row['candidate_evidence_available'] for row in samples),
            'inference_failures': sum(row['inference_failure'] for row in samples),
            'rejected_samples': sum(row['rejected'] for row in samples),
            'empty_raw_samples': sum(row['empty_raw'] for row in samples),
            'empty_candidate_texts': sum(row['empty_candidate_texts'] for row in samples),
            'stored_hypotheses': sum(row['hypotheses_count'] for row in samples),
            'unique_canonical_candidates': canonical_count, 'unique_literal_candidates': literal_count,
            'mean_unique_canonical_candidates_per_sample': canonical_count / count if count else None,
            'mean_unique_literal_candidates_per_sample': literal_count / count if count else None,
            'diversity': {'available': pairs > 0, 'candidate_pairs': pairs,
                          'character_edit_distance': pair_errors,
                          'mean_normalized_edit_distance': sum(row['diversity']['normalized_distance_sum'] for row in samples) / pairs if pairs else None,
                          'pooled_normalized_edit_distance': pair_errors / pair_normalization if pair_normalization else None},
            'edit_space_support': {'available_samples': len(support), 'references_used': False,
                                   'top1_is_medoid_samples': sum(row['top1_is_medoid'] for row in support),
                                   'mean_top1_distance_to_other_candidates': sum(row['top1_mean_normalized_distance'] for row in support) / len(support) if support else None,
                                   'definition': 'candidate-list internal agreement only; not a correctness probability or a reranking prescription'},
            'forced_eos': {'marked_candidates': sum(row['forced_eos']['marked_candidates'] for row in samples),
                           'reported_candidates': sum(row['forced_eos']['reported_candidates'] for row in samples),
                           'unreported_candidates': sum(row['forced_eos']['unreported_candidates'] for row in samples),
                           'samples_with_any_marked': sum(row['forced_eos']['any_marked'] for row in samples),
                           'top1_marked_samples': sum(row['forced_eos']['top1_marked'] for row in samples),
                           'samples_reverse_skipped_for_forced_eos': sum(row['forced_eos']['reverse_skipped_for_forced_eos'] for row in samples)}}


def diagnose(report: dict, manifest: dict, *, stage: str | None = None, split: str | None = None,
             include_samples: bool = False) -> dict:
    """Read-only, label-aware diagnostics; no text leaves the output boundary."""
    if type(include_samples) is not bool:
        raise ValueError('include_samples must be boolean')
    rows, verified_raw = _validated_section(report, manifest, stage, split)
    summaries = [_sample(row) for row in rows]
    summaries.sort(key=lambda row: row['sample_id'])
    groups = {name: defaultdict(list) for name in ('speaker', 'reference_length', 'duration')}
    for row in summaries:
        groups['speaker'][row['speaker']].append(row)
        groups['reference_length'][row['reference_length_bin']].append(row)
        groups['duration'][row['duration_bin']].append(row)
    for name in REFERENCE_LENGTH_BINS:
        groups['reference_length'].setdefault(name, [])
    for name in DURATION_BINS:
        groups['duration'].setdefault(name, [])
    metrics = _metrics(summaries)
    for key in ('samples', 'character_errors', 'reference_characters', 'cer', 'exact_sentence_rate'):
        if metrics['top1'][key] != verified_raw[key]:
            raise ValueError(f'top1 diagnostic {key} differs from independently verified raw report')
    result = {'schema_version': 1, 'diagnostic_only': True,
              'references_used_for_offline_oracle': True, 'references_used_for_decoding': False,
              'references_used_for_training': False, 'new_predictions_generated': False,
              'correctness_probabilities_produced': False, 'failure_causes_established': False,
              'raw_text_included': False, 'per_sample_output_included': include_samples,
              'frozen_manifest_coverage_checked': True, 'video_bytes_rehashed': False,
              'normalization': 'NFKC/casefold Unicode letters and numbers; punctuation and spaces excluded',
              'oracle_k': list(ORACLE_K), 'reference_length_bins': list(REFERENCE_LENGTH_BINS),
              'duration_bins_seconds': list(DURATION_BINS), 'stage': stage, 'split': split,
              'metrics': metrics,
              'per_speaker': {name: _metrics(items) for name, items in sorted(groups['speaker'].items())},
              'per_reference_length': {name: _metrics(groups['reference_length'][name]) for name in REFERENCE_LENGTH_BINS},
              'per_duration': {name: _metrics(groups['duration'][name]) for name in DURATION_BINS},
              'limitations': [
                  'Oracle uses reference text only AFTER predictions are stored; it is an unattainable lower bound, not a decoding or correction method.',
                  'All samples remain in raw and oracle denominators, including failures, rejected outputs and missing candidate evidence.',
                  'Missing candidates use raw-only fallback; candidate evidence availability and exact hits from actual supplied lists are separate.',
                  'Oracle rank k means the first k stored beam positions, including duplicates; diversity deduplicates canonical text without promoting ranks.',
                  'N-best lists with fewer than k positions are incomplete evidence for k; reported counts expose this rather than generating alternatives.',
                  'Pair diversity is normalized character edit distance, not a correctness probability or phonetic/visual dissimilarity.',
                  'Edit-space support/medoid ranks use stored candidate texts only; a candidate list may agree on wrong outputs, and no candidate is changed or applied.',
                  'S/D/I decomposition follows deterministic diagonal/deletion/insertion ties; equally optimal alignments can have different edit types.',
                  'Forced EOS counts reflect only explicit report flags, not inferred truncation or proven failure causes.',
                  'Frozen hashes are compared to report identities; this JSON-only diagnostic does not rehash media or certify label provenance.',
                  'Do not select decoding, corrections or training examples from held-out diagnostics; changes need development data and a fresh independent test.',
                  'No source video, audio, webcam, model, LLM or network is accessed, and no predicted/reference text is included in results.',
              ]}
    if include_samples:
        result['samples'] = summaries
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('report', type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--stage', choices=('baseline', 'candidate'))
    parser.add_argument('--split', choices=('dev', 'test'))
    parser.add_argument('--include-samples', action='store_true', help='Include text-free per-sample edit/rank metadata')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.resolve() in {args.report.resolve(), args.manifest.resolve()}:
            raise ValueError('diagnostic output must not overwrite any input evidence')
        report_bytes, manifest_bytes = args.report.read_bytes(), args.manifest.read_bytes()
        result = diagnose(json.loads(report_bytes), json.loads(manifest_bytes), stage=args.stage,
                          split=args.split, include_samples=args.include_samples)
        result['provenance'] = {'report_sha256': hashlib.sha256(report_bytes).hexdigest(),
                                'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest()}
        serialized = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding='utf-8')
        print(serialized, end='')
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
