"""Audit local train/dev/test manifests for genuine silent, speaker-independent data.

No model, audio, camera or network is used. Video bytes are hashed, never decoded.
Human declarations remain declarations: this tool cannot prove silent articulation.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lipflow_research_path import REPOSITORY_ROOT

import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json

from research.evaluation import characters, load_manifest


def audit(manifests: dict[str, Path]) -> dict:
    """Check every row and every supplied video's actual SHA256, without filtering."""
    if set(manifests) != {'train', 'dev', 'test'}:
        raise ValueError('Supply exactly train, dev and test manifests')
    issues, summaries, rows_by_split, provenance, review_samples = [], {}, {}, {}, {}
    sessions = defaultdict(set)
    for split, filename in manifests.items():
        path = Path(filename)
        raw = path.read_bytes()
        document = json.loads(raw)
        dataset = load_manifest(path)
        if path.read_bytes() != raw:
            raise ValueError('Manifest changed during verification')
        provenance[split] = hashlib.sha256(raw).hexdigest()
        rows = list(zip(dataset.samples, document['samples']))
        rows_by_split[split] = rows
        counts = Counter()
        speakers, dates_by_speaker = set(), defaultdict(set)
        identities, clips = set(), set()
        if dataset.split != split or any(sample.split != split for sample, _ in rows):
            issues.append(f'{split}: split declaration mismatch')
        if not rows:
            issues.append(f'{split}: empty dataset')
        if document.get('training_overlap_checked') is not True:
            issues.append(f'{split}: human training/tuning overlap review is unconfirmed')
        for sample, row in rows:
            speakers.add(sample.speaker)
            counts['articulation_' + sample.articulation] += 1
            silent = sample.articulation == 'silent' and sample.articulation_verified
            counts['human_confirmed_silent'] += silent
            counts['human_confirmed_labels'] += sample.label_verified
            counts['webcam'] += sample.domain == 'webcam' and not sample.mouth_roi
            if not silent:
                counts['unconfirmed_silent'] += 1
            if not sample.label_verified:
                counts['unconfirmed_labels'] += 1
            if sample.domain != 'webcam' or sample.mouth_roi:
                counts['non_webcam'] += 1
            if not row.get('sha256'):
                counts['missing_declared_sha256'] += 1
            elif row['sha256'] != sample.sha256:
                counts['declared_sha256_mismatch'] += 1
            if sample.id in identities:
                counts['duplicate_ids'] += 1
            identities.add(sample.id)
            identity = (sample.sha256, sample.start, sample.end)
            if identity in clips:
                counts['duplicate_clips'] += 1
            clips.add(identity)
            recorded = row.get('recording_date')
            try:
                if not isinstance(recorded, str) or date.fromisoformat(recorded).isoformat() != recorded:
                    raise ValueError('date must be YYYY-MM-DD')
            except ValueError:
                counts['missing_or_invalid_recording_date'] += 1
            else:
                dates_by_speaker[sample.speaker].add(recorded)
                sessions[(sample.speaker, sample.session)].add(recorded)
        for name in ('unconfirmed_silent', 'unconfirmed_labels', 'non_webcam',
                     'missing_declared_sha256', 'declared_sha256_mismatch', 'duplicate_ids', 'duplicate_clips',
                     'missing_or_invalid_recording_date'):
            if counts[name]:
                issues.append(f'{split}: {name}={counts[name]}')
        summaries[split] = {
            'samples': len(rows), 'speakers': len(speakers), **dict(counts),
            'dates_per_speaker': {speaker: len(dates_by_speaker[speaker]) for speaker in sorted(speakers)},
        }
        # Fixed hash order, two clips per person: reproducible independent review,
        # not a hand-picked set of successful recognition examples.
        review_samples[split] = []
        for speaker in sorted(speakers):
            choices = sorted(((sample, row) for sample, row in rows if sample.speaker == speaker),
                             key=lambda pair: hashlib.sha256(pair[0].id.encode('utf-8')).hexdigest())[:2]
            review_samples[split].extend({'sample_id': sample.id, 'speaker': speaker,
                                          'video': row['video'], 'sha256': sample.sha256,
                                          'recording_date': row.get('recording_date')}
                                         for sample, row in choices)
    for dates in sessions.values():
        if len(dates) > 1:
            issues.append('A speaker/session pair declares multiple recording dates')
    for left, right in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
        a, b = rows_by_split[left], rows_by_split[right]
        for name, identity in (
            ('speakers', lambda sample: sample.speaker),
            ('sample_ids', lambda sample: sample.id),
            ('video_sha256', lambda sample: sample.sha256),
            ('normalized_references', lambda sample: ''.join(characters(sample.reference))),
        ):
            overlap = {identity(sample) for sample, _ in a} & {identity(sample) for sample, _ in b}
            if overlap:
                issues.append(f'{left}/{right}: overlapping {name}={len(overlap)}')
    test = summaries['test']
    if test['samples'] < 50:
        issues.append('test: requires at least 50 samples')
    if test['speakers'] < 5:
        issues.append('test: requires at least 5 speakers')
    if min(test['dates_per_speaker'].values(), default=0) < 2:
        issues.append('test: requires at least 2 distinct recording dates for every speaker')
    return {
        'schema_version': 1, 'scope': 'general_speaker_independent_silent_webcam',
        'data_eligible': not issues, 'model_acceptance': False,
        'partitions': summaries, 'issues': sorted(set(issues)),
        'manifest_sha256': provenance, 'video_bytes_hashed': True,
        'declared_video_hashes_verified': not any(
            row.get('missing_declared_sha256') or row.get('declared_sha256_mismatch') for row in summaries.values()),
        'manual_review_samples': review_samples,
        'manual_review_selection': 'First two sample IDs per person ordered by SHA256(id); video paths are relative to each input manifest.',
        'limitations': [
            'Eligibility validates supplied files and declarations, not recognition accuracy.',
            'A human must verify recording dates, speaker identity, labels and deliberate silent articulation.',
            'Voiced or whispered videos do not become silent data when their audio is removed.',
            'Previously inspected test data cannot be reused to select a new model or threshold.',
            'Freeze models, decoder and acceptance thresholds before running an independent test.',
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ('train', 'dev', 'test'):
        parser.add_argument('--' + split, required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() in {getattr(args, split).resolve() for split in ('train', 'dev', 'test')}:
            raise ValueError('Output must not overwrite an input manifest')
        result = audit({split: getattr(args, split) for split in ('train', 'dev', 'test')})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({'data_eligible': result['data_eligible'], 'issues': result['issues']}, ensure_ascii=False))
    return 0 if result['data_eligible'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
