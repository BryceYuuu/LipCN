"""Compare saved raw predictions with complete S/D/I and declared entity cohorts.

This is diagnostic only: no decoding, text correction, model selection or devices.
It reads manifest metadata and reports, not video pixels or audio.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lipflow_research_path import REPOSITORY_ROOT

import argparse
from collections import defaultdict
import hashlib
import json

from research.evaluation import characters
from analyze_chinese_errors import analyze, _metrics
from compare_chinese_reports import compare


def _rows(report: dict, manifest: dict) -> list[dict]:
    if not isinstance(report, dict) or not isinstance(report.get('samples'), list):
        raise ValueError('Prediction section must contain samples')
    declarations = {row['id']: row for row in manifest['samples']}
    seen, result = set(), []
    for original in report['samples']:
        if not isinstance(original, dict):
            raise ValueError('Prediction samples must be objects')
        row = dict(original)
        identity = row.get('sample_id')
        if identity not in declarations or identity in seen:
            raise ValueError('Prediction IDs must be unique members of the manifest')
        seen.add(identity)
        source = declarations[identity]
        if row.get('reference') != source['reference']:
            raise ValueError('Prediction reference differs from the frozen manifest')
        for key in ('speaker', 'session', 'split', 'domain', 'articulation', 'articulation_verified', 'sha256'):
            target = 'video_sha256' if key == 'sha256' else key
            if key in source:
                if target in row and row[target] != source[key]:
                    raise ValueError(f'Prediction/manifest {key} mismatch')
                row[target] = source[key]
        result.append(row)
    if seen != declarations.keys():
        raise ValueError('Every manifest sample needs a prediction, including failed/empty outputs')
    return result


def _entity_cohorts(manifest: dict) -> dict[str, list[str]]:
    cohorts = {}
    for row in manifest['samples']:
        if row.get('entities_verified') is not True:
            cohorts[row['id']] = ['unannotated']
            continue
        entities = row.get('entities')
        if not isinstance(entities, list):
            raise ValueError('Verified entity annotation requires an explicit entities list')
        kinds = set()
        for entity in entities:
            if not isinstance(entity, dict) or entity.get('kind') not in ('number', 'person_name'):
                raise ValueError('Entity kind must be number or person_name')
            text = ''.join(characters(entity.get('text')))
            if not text or text not in ''.join(characters(row['reference'])):
                raise ValueError('Annotated entity must occur in the reference')
            kinds.add(entity['kind'])
        cohorts[row['id']] = sorted(kinds) or ['no_declared_entities']
    return cohorts


def report(manifest: dict, baseline: dict, candidate: dict) -> dict:
    samples = manifest.get('samples') if isinstance(manifest, dict) else None
    if not isinstance(samples, list) or not samples:
        raise ValueError('Manifest requires nonempty samples')
    if any(not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id'] for row in samples):
        raise ValueError('Manifest sample IDs must be nonempty strings')
    if len({row['id'] for row in samples}) != len(samples):
        raise ValueError('Manifest IDs must be unique')
    before, after = _rows(baseline, manifest), _rows(candidate, manifest)
    paired = compare(dict(baseline, samples=before), dict(candidate, samples=after), manifest=manifest)
    diagnostics = {'baseline': analyze(before, manifest=manifest), 'candidate': analyze(after, manifest=manifest)}
    cohorts = _entity_cohorts(manifest)
    dates = {row['id']: row.get('recording_date', '(unreported)') for row in samples}
    if any(not isinstance(value, str) or not value for value in dates.values()):
        raise ValueError('Recording dates, when present, must be nonempty strings')
    for arm in diagnostics.values():
        entity_groups, date_groups = defaultdict(list), defaultdict(list)
        for row in arm['samples']:
            for group in cohorts[row['sample_id']]:
                entity_groups[group].append(row)
            date_groups[dates[row['sample_id']]].append(row)
        arm['per_entity_sentence_cohort'] = {key: _metrics(rows) for key, rows in sorted(entity_groups.items())}
        arm['per_recording_date'] = {key: _metrics(rows) for key, rows in sorted(date_groups.items())}
    return {
        'schema_version': 1, 'diagnostic_only': True, 'model_acceptance': False,
        'raw_top1_required': True, 'postprocessing_applied_by_tool': False,
        'baseline': diagnostics['baseline'], 'candidate': diagnostics['candidate'],
        'paired_comparison': paired,
        'entity_annotation': {
            'verified_samples': sum(row.get('entities_verified') is True for row in samples),
            'unannotated_samples': sum(row.get('entities_verified') is not True for row in samples),
            'metric': 'Whole-sentence CER/S/D/I grouped by independently annotated number/person-name presence; not entity-only accuracy.',
        },
        'limitations': [
            'Missing entity annotations are unreported, never interpreted as absence of names or numbers.',
            'Entity-containing cohorts may overlap; do not add their sample counts.',
            'Dates are human declarations; different session labels alone do not prove different days.',
            'Raw top-1 output must be supplied by the model; this tool cannot detect upstream rewriting.',
            'A dev comparison, or a previously inspected test, does not establish independent silent accuracy.',
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--baseline-section', help='Optional top-level section of an adapter report')
    parser.add_argument('--candidate-section', help='Optional top-level section of an adapter report')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() in {getattr(args, key).resolve() for key in ('manifest', 'baseline', 'candidate')}:
            raise ValueError('Output must not overwrite an input file')
        contents = {key: getattr(args, key).read_bytes() for key in ('manifest', 'baseline', 'candidate')}
        documents = {key: json.loads(value) for key, value in contents.items()}
        for arm in ('baseline', 'candidate'):
            section = getattr(args, arm + '_section')
            if section:
                documents[arm] = documents[arm][section]
        result = report(**documents)
        result['input_sha256'] = {key: hashlib.sha256(value).hexdigest() for key, value in contents.items()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result['paired_comparison']['delta'], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
