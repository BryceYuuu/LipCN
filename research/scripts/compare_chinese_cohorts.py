"""Compare frozen raw Mandarin reports by declared recording cohorts.

    python research/scripts/compare_chinese_cohorts.py cmlr.json cnvsrc.json \
        --manifest frozen-test.json --train-manifest train.json \
        --dev-manifest dev.json --metadata cohort-metadata.json --output paired.json

Only stored JSON evidence is read. Failed, rejected and empty predictions remain
in every relevant denominator. No model, media, camera or LLM is invoked. Cohort
metadata needs explicit provenance; absent metadata stays unknown. Speaker
novelty is relative only to the supplied local adaptation manifests, never to an
unknown base model training corpus. Test results must not select model settings.
"""
from __future__ import annotations

import sys as _path_sys
from pathlib import Path as _SourcePath
_path_sys.path.insert(0, str(_SourcePath(__file__).resolve().parent))
from _lipflow_research_path import REPOSITORY_ROOT

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

from compare_chinese_reports import compare
from research.evaluation import character_error

CAPTURE_TYPES = ('real_webcam', 'pre_cropped', 'public_talk', 'unknown')
CONTENT_STYLES = ('everyday_conversation', 'spontaneous_speech', 'read_speech', 'unknown')
DOMAINS = ('mouth_roi', 'full_face', 'webcam', 'unknown')
ARTICULATIONS = ('silent', 'voiced', 'whispered', 'unknown')
UNKNOWN_METADATA = {'scene': 'unknown', 'capture_type': 'unknown', 'content_style': 'unknown',
                    'metadata_verified': False, 'provenance': None}


def _known_speaker(identity: str) -> bool:
    return identity.strip().casefold() not in ('unknown', '__unknown__', 'unidentified', 'unspecified')


def _visual_config(configuration) -> dict:
    if not isinstance(configuration, dict) or set(configuration) != {'preprocessing', 'stress', 'seed'}:
        raise ValueError('visual evidence requires complete preprocessing/stress/seed configuration')
    if configuration['preprocessing'] not in ('identity', 'stabilize_mild', 'stabilize_moderate'):
        raise ValueError('visual evidence has invalid preprocessing')
    if configuration['stress'] not in ('clean', 'synthetic_mild', 'synthetic_moderate'):
        raise ValueError('visual evidence has invalid stress')
    if type(configuration['seed']) is not int or not 0 <= configuration['seed'] < 2 ** 32:
        raise ValueError('visual evidence has invalid seed')
    return configuration


def _report_pixels(report: dict) -> tuple[dict | None, dict[str, dict | None]]:
    model = report.get('model')
    configuration = None
    if isinstance(model, dict) and 'visual_configuration' in model:
        configuration = _visual_config(model['visual_configuration'])
    evidence_by_id = {}
    for row in report['samples']:
        identity, evidence = row['sample_id'], row.get('visual_input')
        if evidence is None:
            evidence_by_id[identity] = None
            continue
        if not isinstance(evidence, dict):
            raise ValueError(f'{identity}: visual_input must be an object or null')
        observed = _visual_config(evidence.get('configuration'))
        if configuration is None:
            configuration = observed
        if observed != configuration:
            raise ValueError(f'{identity}: visual evidence configuration differs from report configuration')
        for key in ('original_pixels_sha256', 'input_pixels_sha256'):
            _digest(evidence.get(key), identity=f'{identity}: {key}')
        shape = evidence.get('shape')
        if (not isinstance(shape, list) or len(shape) != 3
                or any(type(value) is not int or value < 1 for value in shape)
                or min(shape[1:]) < 2 or evidence.get('dtype') != 'uint8'):
            raise ValueError(f'{identity}: visual evidence requires nonempty T,H,W uint8 shape')
        expected_seed = (observed['seed'] + int.from_bytes(hashlib.sha256(identity.encode('utf-8')).digest()[:8],
                                                         'big')) % (2 ** 32)
        if type(evidence.get('sample_seed')) is not int or evidence['sample_seed'] != expected_seed:
            raise ValueError(f'{identity}: visual sample seed differs from frozen ID/configuration')
        diagnostics = evidence.get('diagnostics')
        if diagnostics is not None:
            if not isinstance(diagnostics, dict):
                raise ValueError(f'{identity}: visual diagnostics must be an object')
            for key, expected in (('input_pixels_sha256', evidence['original_pixels_sha256']),
                                  ('output_pixels_sha256', evidence['input_pixels_sha256']),
                                  ('shape', shape), ('dtype', 'uint8')):
                if key in diagnostics and diagnostics[key] != expected:
                    raise ValueError(f'{identity}: visual diagnostics {key} contradicts evidence')
            if diagnostics.get('text_used') is True or diagnostics.get('audio_used') is True:
                raise ValueError(f'{identity}: visual evidence declares text/audio used by pixel transform')
            if 'synthetic' in diagnostics:
                if type(diagnostics['synthetic']) is not bool or diagnostics['synthetic'] != (observed['stress'] != 'clean'):
                    raise ValueError(f'{identity}: synthetic visual flag contradicts frozen stress')
        if 'real_webcam_evidence' in evidence:
            if type(evidence['real_webcam_evidence']) is not bool:
                raise ValueError(f'{identity}: real_webcam_evidence must be boolean')
            if evidence['real_webcam_evidence'] and observed['stress'] != 'clean':
                raise ValueError(f'{identity}: synthetic pixels cannot claim real webcam evidence')
        evidence_by_id[identity] = evidence
    return configuration, evidence_by_id


def _paired_pixels(baseline: dict, candidate: dict) -> dict:
    before_config, before = _report_pixels(baseline)
    after_config, after = _report_pixels(candidate)
    declared = before_config is not None and after_config is not None
    if declared and any(before_config[key] != after_config[key] for key in ('stress', 'seed')):
        raise ValueError('paired visual inputs require identical stress and seed')
    verified_original, verified_output, different_preprocessing, unknown, samples = [], [], [], [], []
    for identity in sorted(before):
        left, right = before[identity], after[identity]
        if left is None or right is None:
            unknown.append(identity)
            samples.append({'sample_id': identity, 'status': 'unknown_missing_pixel_evidence',
                            'baseline_evidence_present': left is not None, 'candidate_evidence_present': right is not None})
            continue
        for key in ('original_pixels_sha256', 'shape', 'dtype', 'sample_seed'):
            if left[key] != right[key]:
                raise ValueError(f'{identity}: paired original visual pixels {key} mismatch')
        for key in ('stress', 'seed'):
            if left['configuration'][key] != right['configuration'][key]:
                raise ValueError(f'{identity}: paired visual stress/seed mismatch')
        verified_original.append(identity)
        same_preprocessing = left['configuration']['preprocessing'] == right['configuration']['preprocessing']
        if same_preprocessing:
            if left['input_pixels_sha256'] != right['input_pixels_sha256']:
                raise ValueError(f'{identity}: same preprocessing requires identical prepared input pixels')
            verified_output.append(identity)
        else:
            different_preprocessing.append(identity)
        samples.append({'sample_id': identity, 'status': 'original_pixels_verified',
                        'same_preprocessing': same_preprocessing,
                        'prepared_input_pixels_verified_equal': same_preprocessing})
    condition = ('synthetic_stress' if declared and before_config['stress'] != 'clean' else
                 'clean' if declared else 'unknown')
    return {'baseline_visual_configuration': before_config, 'candidate_visual_configuration': after_config,
            'stress_and_seed_verified_equal': True if declared else None,
            'comparison_input_condition': condition,
            'original_pixels_verified_samples': len(verified_original),
            'prepared_same_preprocessing_pixels_verified_samples': len(verified_output),
            'different_preprocessing_samples': len(different_preprocessing),
            'unknown_pixel_evidence_sample_ids': unknown,
            'all_original_pixels_verified': len(verified_original) == len(before),
            'video_or_pixel_bytes_rehashed': False, 'real_webcam_measurement_established': False,
            'samples': samples,
            'scope': 'stored pixel hashes, shape, dtype and deterministic seed evidence; no media is reloaded'}


def _text(row: dict, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{key} must be a nonempty string')
    return value


def _digest(value, *, identity: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(f'{identity}: sha256 must be 64 lowercase hexadecimal characters')
    return value


def _range(row: dict) -> tuple[float, float]:
    start, end = row.get('start', 0), row.get('end')
    if (type(start) not in (int, float) or not math.isfinite(start) or start < 0
            or end is not None and (type(end) not in (int, float) or not math.isfinite(end) or end <= start)):
        raise ValueError('video ranges require finite start >= 0 and end > start')
    return float(start), math.inf if end is None else float(end)


def _manifest_rows(manifest: dict, *, role: str) -> dict[str, dict]:
    if (not isinstance(manifest, dict) or type(manifest.get('schema_version')) is not int
            or manifest['schema_version'] != 1 or not isinstance(manifest.get('samples'), list)
            or not manifest['samples']):
        raise ValueError(f'{role} manifest must have schema_version=1 and nonempty samples')
    rows = {}
    for row in manifest['samples']:
        if not isinstance(row, dict):
            raise ValueError(f'{role} manifest samples must be objects')
        identity = _text(row, 'id')
        if identity in rows:
            raise ValueError(f'{role} manifest IDs must be unique: {identity}')
        _text(row, 'speaker')
        _text(row, 'session')
        _digest(row.get('sha256'), identity=identity)
        _range(row)
        split = row.get('split', manifest.get('split', role))
        allowed = {'test'} if role == 'test' else {'train'} if role == 'train' else {'dev', 'development', 'validation'}
        if not isinstance(split, str) or split not in allowed:
            raise ValueError(f'{identity}: {role} manifest has incorrect split {split!r}')
        if 'video' in row:
            _text(row, 'video')
        if role == 'test':
            character_error('', _text(row, 'reference'))
            if 'label_source' in row:
                _text(row, 'label_source')
            for key in ('label_verified', 'mouth_roi', 'articulation_verified'):
                if key in row and type(row[key]) is not bool:
                    raise ValueError(f'{identity}: {key} must be boolean')
            domain = row.get('domain', 'unknown')
            if domain not in DOMAINS:
                raise ValueError(f'{identity}: invalid domain')
            if 'mouth_roi' in row and domain != 'unknown' and row['mouth_roi'] != (domain == 'mouth_roi'):
                raise ValueError(f'{identity}: mouth_roi must match domain')
            if row.get('articulation', 'unknown') not in ARTICULATIONS:
                raise ValueError(f'{identity}: invalid articulation')
        rows[identity] = row
    return rows


def _metadata(sidecar: dict | None, rows: dict[str, dict]) -> dict[str, dict]:
    result = {identity: dict(UNKNOWN_METADATA) for identity in rows}
    if sidecar is None:
        return result
    if (not isinstance(sidecar, dict) or type(sidecar.get('schema_version')) is not int
            or sidecar['schema_version'] != 1 or not isinstance(sidecar.get('samples'), dict)):
        raise ValueError('metadata must contain schema_version=1 and a samples object keyed by manifest ID')
    if set(sidecar) - {'schema_version', 'samples', 'description'}:
        raise ValueError('unknown metadata document fields')
    if 'description' in sidecar and not isinstance(sidecar['description'], str):
        raise ValueError('metadata description must be a string')
    unexpected = set(sidecar['samples']) - rows.keys()
    if unexpected:
        raise ValueError(f'metadata IDs are not in frozen manifest: {sorted(unexpected)}')
    fields = {'provenance', 'metadata_verified', 'scene', 'capture_type', 'content_style'}
    for identity, record in sidecar['samples'].items():
        if not isinstance(record, dict) or set(record) - fields:
            raise ValueError(f'{identity}: invalid or unknown metadata fields')
        provenance = _text(record, 'provenance')
        if type(record.get('metadata_verified')) is not bool:
            raise ValueError(f'{identity}: metadata_verified must be boolean')
        values = {**UNKNOWN_METADATA, **record, 'provenance': provenance}
        _text(values, 'scene')
        if values['capture_type'] not in CAPTURE_TYPES or values['content_style'] not in CONTENT_STYLES:
            raise ValueError(f'{identity}: invalid capture_type or content_style')
        domain = rows[identity].get('domain', 'unknown')
        if values['capture_type'] == 'real_webcam' and domain != 'webcam':
            raise ValueError(f'{identity}: real_webcam metadata requires webcam manifest domain')
        if values['capture_type'] == 'pre_cropped' and domain != 'mouth_roi':
            raise ValueError(f'{identity}: pre_cropped metadata requires mouth_roi manifest domain')
        result[identity] = values
    return result


def _audit(test: dict[str, dict], train: list[dict], dev: list[dict], manifest: dict) -> tuple[dict, dict[str, str]]:
    groups = {'train': [_manifest_rows(item, role='train') for item in train],
              'dev': [_manifest_rows(item, role='dev') for item in dev]}
    exclusions = [(role, row) for role, corpora in groups.items() for corpus in corpora for row in corpus.values()]
    local_rows = {role: [row for corpus in corpora for row in corpus.values()]
                  for role, corpora in groups.items()}
    train_dev_ids = sorted({row['id'] for row in local_rows['train']} &
                           {row['id'] for row in local_rows['dev']})
    train_dev_hashes = sorted({row['sha256'] for row in local_rows['train']} &
                              {row['sha256'] for row in local_rows['dev']})
    train_dev_speakers = sorted({row['speaker'] for row in local_rows['train']} &
                                {row['speaker'] for row in local_rows['dev']})
    if train_dev_ids or train_dev_hashes:
        raise ValueError(f'train/dev content overlaps: sample_ids={train_dev_ids}, video_sha256={train_dev_hashes}')
    # Embedded exclusions are auditable additional identities, not evidence that
    # all actual train/dev manifests were supplied.
    embedded = manifest.get('development_samples', [])
    if not isinstance(embedded, list):
        raise ValueError('development_samples must be a list')
    for row in embedded:
        if not isinstance(row, dict):
            raise ValueError('development_samples must contain objects')
        for key in ('id', 'speaker', 'session'):
            _text(row, key)
        if 'sha256' in row:
            _digest(row['sha256'], identity=row['id'])
        if 'video' in row:
            _text(row, 'video')
        exclusions.append(('embedded', row))
    ids = {row['id'] for _, row in exclusions}
    hashes = {row['sha256'] for _, row in exclusions if row.get('sha256')}
    speakers = {row['speaker'] for _, row in exclusions}
    sessions = {(row['speaker'], row['session']) for _, row in exclusions}
    id_overlap = sorted(identity for identity in test if identity in ids)
    hash_overlap = sorted(identity for identity, row in test.items() if row['sha256'] in hashes)
    speaker_overlap = sorted({row['speaker'] for row in test.values()} & speakers)
    session_overlap = sorted(identity for identity, row in test.items() if (row['speaker'], row['session']) in sessions)
    if id_overlap or hash_overlap:
        raise ValueError(f'local adaptation content overlaps test: sample_ids={id_overlap}, video_sha256={hash_overlap}')
    values = list(test.values())
    for index, left in enumerate(values):
        for right in values[index + 1:]:
            same_hash = left['sha256'] == right['sha256']
            same_video = left.get('video') and left.get('video') == right.get('video')
            if same_video and not same_hash:
                raise ValueError(f'same test video has inconsistent sha256: {left["id"]}, {right["id"]}')
            if same_hash:
                if left['speaker'] != right['speaker']:
                    raise ValueError(f'same test video has inconsistent speaker: {left["id"]}, {right["id"]}')
                if left['session'] != right['session']:
                    raise ValueError(f'same test video has inconsistent session: {left["id"]}, {right["id"]}')
                a, b = _range(left), _range(right)
                if max(a[0], b[0]) < min(a[1], b[1]):
                    raise ValueError(f'duplicate/overlapping test video ranges: {left["id"]}, {right["id"]}')
    complete_inputs = bool(train) and bool(dev)
    exposure = {identity: ('unknown' if not _known_speaker(row['speaker']) else
                           'seen_relative_to_local_adaptation' if row['speaker'] in speakers else
                           'unseen_relative_to_local_adaptation' if complete_inputs else 'unknown')
                for identity, row in test.items()}
    audit = {'scope': 'only supplied local train/dev manifests and embedded exclusions',
             'train_manifests': len(train), 'dev_manifests': len(dev),
             'train_samples': sum(len(group) for group in groups['train']),
             'dev_samples': sum(len(group) for group in groups['dev']),
             'embedded_exclusions': len(embedded), 'sample_id_overlaps': id_overlap,
             'video_sha256_overlaps': hash_overlap, 'speaker_overlaps': speaker_overlap,
             'train_dev_sample_id_overlaps': train_dev_ids,
             'train_dev_video_sha256_overlaps': train_dev_hashes,
             'train_dev_speaker_overlaps': train_dev_speakers,
             'speaker_session_overlaps': session_overlap,
             'both_local_split_manifests_supplied': complete_inputs,
             'full_adaptation_manifest_inventory_verified': False,
             'base_model_training_exposure': 'unknown',
             'unknown_test_speaker_ids': sorted(identity for identity, row in test.items()
                                               if not _known_speaker(row['speaker'])),
             'unverified_reference_ids': sorted(identity for identity, row in test.items()
                                               if row.get('label_verified') is not True or not row.get('label_source')),
             'video_content_checked_from_frozen_hashes': True,
             'video_bytes_rehashed': False}
    return audit, exposure


def compare_cohorts(baseline: dict, candidate: dict, manifest: dict, *, metadata: dict | None = None,
                    training_manifests=(), development_manifests=(), seed: int = 0,
                    resamples: int = 5000) -> dict:
    """Pair all frozen IDs first, then apply the SAME metadata cohorts to both."""
    rows = _manifest_rows(manifest, role='test')
    declared_metadata = _metadata(metadata, rows)
    audit, exposure = _audit(rows, list(training_manifests), list(development_manifests), manifest)
    overall = compare(baseline, candidate, manifest=manifest, seed=seed, resamples=resamples)
    pixel_audit = _paired_pixels(baseline, candidate)
    before = {row['sample_id']: row for row in baseline['samples']}
    after = {row['sample_id']: row for row in candidate['samples']}

    def cohort(identities: list[str]) -> dict:
        identities = sorted(identities)
        if not identities:
            return {'available': False, 'samples': 0, 'sample_ids': [], 'comparison': None,
                    'reason': 'No samples with the required declared provenance in the frozen paired corpus.'}
        identity_set = set(identities)
        # Aggregate report metrics describe the full corpus; never carry them to
        # a subset or silently filter according to either model's success.
        subset = {'schema_version': 1, 'samples': [rows[identity] for identity in identities]}
        result = compare({'schema_version': 1, 'samples': [before[key] for key in identities]},
                         {'schema_version': 1, 'samples': [after[key] for key in identities]},
                         manifest=subset, seed=seed, resamples=resamples)
        assert len(identity_set) == result['baseline']['samples'] == result['candidate']['samples']
        return {'available': True, 'samples': len(identities), 'sample_ids': identities, 'comparison': result}

    dimensions = {key: defaultdict(list) for key in
                  ('speaker', 'domain', 'articulation', 'scene', 'capture_type', 'content_style',
                   'metadata_verification', 'local_adaptation_exposure')}
    for identity, row in rows.items():
        meta = declared_metadata[identity]
        assignments = {'speaker': row['speaker'], 'domain': row.get('domain', 'unknown'),
                       'articulation': row.get('articulation', 'unknown'), 'scene': meta['scene'],
                       'capture_type': meta['capture_type'], 'content_style': meta['content_style'],
                       'metadata_verification': 'verified' if meta['metadata_verified'] else 'unverified_or_unknown',
                       'local_adaptation_exposure': exposure[identity]}
        for key, value in assignments.items():
            dimensions[key][value].append(identity)
    for key, choices in (('domain', DOMAINS), ('articulation', ARTICULATIONS),
                         ('capture_type', CAPTURE_TYPES), ('content_style', CONTENT_STYLES),
                         ('local_adaptation_exposure', ('unseen_relative_to_local_adaptation',
                                                       'seen_relative_to_local_adaptation', 'unknown'))):
        for choice in choices:
            dimensions[key].setdefault(choice, [])
    targets = {'verified_real_silent_webcam': [], 'full_face_input': [],
               'verified_everyday_conversation': [], 'unseen_relative_to_local_adaptation': [],
               'verified_real_silent_webcam_everyday_unseen_locally': [],
               'explicitly_synthetic_visual_stress': []}
    source_webcam_samples = []
    for identity, row in rows.items():
        meta = declared_metadata[identity]
        webcam = (row.get('domain') == 'webcam' and not row.get('mouth_roi', False)
                  and row.get('articulation') == 'silent' and row.get('articulation_verified') is True
                  and meta['capture_type'] == 'real_webcam' and meta['metadata_verified'])
        if webcam:
            source_webcam_samples.append(identity)
        webcam = webcam and pixel_audit['comparison_input_condition'] == 'clean'
        everyday = meta['content_style'] == 'everyday_conversation' and meta['metadata_verified']
        unseen = exposure[identity] == 'unseen_relative_to_local_adaptation'
        if webcam:
            targets['verified_real_silent_webcam'].append(identity)
        if row.get('domain') in ('full_face', 'webcam') and not row.get('mouth_roi', False):
            targets['full_face_input'].append(identity)
        if everyday:
            targets['verified_everyday_conversation'].append(identity)
        if unseen:
            targets['unseen_relative_to_local_adaptation'].append(identity)
        if webcam and everyday and unseen:
            targets['verified_real_silent_webcam_everyday_unseen_locally'].append(identity)
        if pixel_audit['comparison_input_condition'] == 'synthetic_stress':
            targets['explicitly_synthetic_visual_stress'].append(identity)
    return {'schema_version': 1, 'purpose': 'paired_raw_prediction_cohort_comparison',
            'overall': overall, 'local_adaptation_audit': audit,
            'prepared_pixel_audit': pixel_audit,
            'verified_real_silent_webcam_source_coverage': {'samples': len(source_webcam_samples),
                                                         'sample_ids': sorted(source_webcam_samples),
                                                         'scope': 'original recording provenance; synthetic/unknown model input is not real camera evaluation'},
            'cohorts': {dimension: {name: cohort(keys) for name, keys in sorted(groups.items())}
                        for dimension, groups in dimensions.items()},
            'target_coverage': {name: cohort(keys) for name, keys in targets.items()},
            'sample_metadata': [{'sample_id': identity, **declared_metadata[identity],
                                 'local_adaptation_exposure': exposure[identity]} for identity in sorted(rows)],
            'product_readiness_established': False, 'references_used_for_decoding': False,
            'test_based_configuration_selection': False,
            'limitations': [
                'All paired failures, rejections and empty outputs remain in denominators; metadata cohorts are identical for both models.',
                'Scene, content style and capture provenance are declarations, never inferred from filenames or model predictions.',
                'Unavailable target cohorts have zero samples and no metrics; public talking footage is not real silent webcam evidence.',
                'Unseen means absent from supplied local train/dev speaker identities; base model training exposure remains unknown.',
                'Both train and dev inputs are needed for a local unseen label; supplying them does not certify a complete training inventory.',
                'Frozen SHA-256 identities are compared, but stored video hashes are not independently recomputed by this JSON-only tool.',
                'Prepared pixel equality is verified only where both reports supply consistent hashes/shape/dtype/seed; legacy missing evidence stays unknown.',
                'Synthetic or unknown visual input conditions do not count as verified real silent webcam evaluation, even when the original recording is a webcam.',
                'Verified metadata is a human provenance assertion, not an automatic authenticity or recording-device certification.',
                'Speaker-cluster intervals are descriptive; overlapping cohorts and multiple comparisons must not select settings on test data.',
                'Raw CER comparisons do not prove spontaneous silent-webcam usability or model/data licensing rights.',
            ]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--manifest', type=Path, required=True, help='Frozen test manifest with per-sample sha256')
    parser.add_argument('--metadata', type=Path, help='Explicit provenance sidecar keyed by manifest sample ID')
    parser.add_argument('--train-manifest', action='append', default=[], type=Path)
    parser.add_argument('--dev-manifest', action='append', default=[], type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resamples', type=int, default=5000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        paths = [args.baseline, args.candidate, args.manifest, *args.train_manifest, *args.dev_manifest]
        if args.metadata:
            paths.append(args.metadata)
        if args.output and args.output.resolve() in {path.resolve() for path in paths}:
            raise ValueError('comparison output must not overwrite any input evidence')
        payloads = {path: path.read_bytes() for path in paths}
        objects = {path: json.loads(content) for path, content in payloads.items()}
        result = compare_cohorts(objects[args.baseline], objects[args.candidate], objects[args.manifest],
                                 metadata=objects[args.metadata] if args.metadata else None,
                                 training_manifests=[objects[path] for path in args.train_manifest],
                                 development_manifests=[objects[path] for path in args.dev_manifest],
                                 seed=args.seed, resamples=args.resamples)
        result['provenance'] = {'inputs': [{'path': str(path.resolve()), 'sha256': hashlib.sha256(content).hexdigest()}
                                           for path, content in payloads.items()],
                                'baseline_report_sha256': hashlib.sha256(payloads[args.baseline]).hexdigest(),
                                'candidate_report_sha256': hashlib.sha256(payloads[args.candidate]).hexdigest(),
                                'manifest_sha256': hashlib.sha256(payloads[args.manifest]).hexdigest()}
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
