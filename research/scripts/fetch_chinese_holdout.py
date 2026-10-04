"""Fetch an untouched local Chinese-LiPS speaker cohort, using bounded ZIP ranges.

The local test partition can be drawn from the publisher's train partition.
That fact stays explicit: unseen in our local experiments does not establish
absence from either released model's original pretraining data. These videos
contain normal voiced speech, not silent webcam input. The original test FACE
archive can additionally exercise face tracking on full-size recorded frames.
"""
from __future__ import annotations

import sys as _path_sys
from pathlib import Path as _SourcePath
_path_sys.path.insert(0, str(_SourcePath(__file__).resolve().parent))
from _lipflow_research_path import REPOSITORY_ROOT

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import struct
import zipfile
import zlib

import fetch_chinese_lips as bounded

FULL_FACE_ARCHIVES = {
    # Pinned publisher file metadata; only ordinary test ZIP is supported.
    'test': ('test.zip', 8445567915,
             '0729922e2ed785cf677c23d92613c1b3b869346b57838870e9498ee6c10ed69f'),
}
MAX_FACE_MEMBER_BYTES = 25_000_000


def read_face_member(url: str, size: int, info: zipfile.ZipInfo,
                     budget: bounded.DownloadBudget) -> bytes:
    """Bound exactly one FACE video; verify the local ZIP header, size and CRC."""
    if (info.file_size > MAX_FACE_MEMBER_BYTES or info.compress_size > MAX_FACE_MEMBER_BYTES or
            info.file_size < 1 or info.compress_size < 1):
        raise ValueError('unexpectedly large or empty full-face video member')
    header = bounded.fetch_range(url, size, info.header_offset, 30, budget)
    signature, _, flags, method, _, _, _, _, _, name_len, extra_len = struct.unpack('<4s5H3I2H', header)
    if signature != b'PK\x03\x04' or flags & 1 or method != info.compress_type:
        raise ValueError('invalid or encrypted full-face local ZIP header')
    block = bounded.fetch_range(url, size, info.header_offset + 30,
                                name_len + extra_len + info.compress_size, budget)
    encoding = 'utf-8' if flags & 0x800 else 'cp437'
    if block[:name_len].decode(encoding) != info.filename:
        raise ValueError('full-face local ZIP name differs from central directory')
    compressed = block[name_len + extra_len:]
    if method == zipfile.ZIP_STORED:
        data = compressed
    elif method == zipfile.ZIP_DEFLATED:
        inflater = zlib.decompressobj(-15)
        data = inflater.decompress(compressed, info.file_size + 1)
        if not inflater.eof or inflater.unconsumed_tail or inflater.unused_data:
            raise ValueError('invalid or oversized compressed full-face member')
    else:
        raise ValueError('unsupported full-face archive compression method')
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise ValueError('full-face member size/CRC verification failed')
    return data


def verify_face_dimensions(path: Path) -> tuple[int, int]:
    """Validate decoded geometry, rather than relabeling a 96px mouth crop."""
    import cv2
    video = cv2.VideoCapture(str(path))
    try:
        ok, frame = video.read()
    finally:
        video.release()
    if not ok or frame is None or frame.ndim != 3:
        raise ValueError('full-face source does not decode a video frame')
    height, width = frame.shape[:2]
    if min(width, height) <= 160:
        raise ValueError('full-face source geometry appears to be a mouth crop')
    return width, height


def exclusion_identities(paths: list[Path]) -> tuple[frozenset[str], list[dict]]:
    """Use source identity fields only; labels, predictions and paths are ignored."""
    if not paths:
        raise ValueError('At least one prior manifest is required to reserve unused speakers')
    excluded, evidence = set(), []
    for path in paths:
        encoded = path.read_bytes()
        manifest = json.loads(encoded)
        if (not isinstance(manifest, dict) or manifest.get('schema_version') != 1 or
                not isinstance(manifest.get('samples'), list) or not manifest['samples']):
            raise ValueError('Exclusion manifest must have nonempty schema-1 samples')
        source = manifest.get('source', {})
        if (not isinstance(source, dict) or source.get('repository') != bounded.REPOSITORY or
                source.get('revision') != bounded.REVISION):
            raise ValueError('Every exclusion must use the same pinned Chinese-LiPS source')
        identities = set()
        for sample in manifest['samples']:
            if not isinstance(sample, dict):
                raise ValueError('Invalid exclusion sample')
            speaker, identifier = sample.get('speaker'), sample.get('source_id')
            if (not isinstance(speaker, str) or not re.fullmatch(r'chinese-lips:\d+', speaker) or
                    not isinstance(identifier, str) or not bounded.ID_PATTERN.fullmatch(identifier) or
                    speaker.removeprefix('chinese-lips:') != identifier.split('_')[0]):
                raise ValueError('Exclusion speaker must match its source ID')
            identities.add(identifier.split('_')[0])
        excluded.update(identities)
        evidence.append({'manifest_sha256': hashlib.sha256(encoded).hexdigest(),
                         'sample_count': len(manifest['samples']),
                         'speaker_ids': sorted(identities)})
    return frozenset(excluded), evidence


def prepare_holdout(output: Path, *, source_split: str, count: int, speakers: int,
                    seed: str, maximum: int, exclude_manifests: list[Path],
                    download_workers: int = 4, media: str = 'mouth_roi') -> dict:
    if source_split not in ('train', 'test'):
        raise ValueError('Official source split must be train or test')
    if type(count) is not int or type(speakers) is not int or speakers < 1 or count < speakers:
        raise ValueError('clip count must be >= positive speaker count')
    if type(download_workers) is not int or not 1 <= download_workers <= 16:
        raise ValueError('download workers must be an integer within [1,16]')
    if media not in ('mouth_roi', 'full_face'):
        raise ValueError('media must be mouth_roi or full_face')
    if media == 'full_face' and source_split not in FULL_FACE_ARCHIVES:
        raise ValueError('Bounded full_face download supports only the official test archive')
    excluded, exclusions = exclusion_identities(exclude_manifests)
    budget = bounded.DownloadBudget(maximum)
    meta, meta_size, meta_hash, archive, archive_size, archive_hash = bounded.FILES[source_split]
    if media == 'full_face':
        archive, archive_size, archive_hash = FULL_FACE_ARCHIVES[source_split]
    # Metadata hashes are verified by our existing fetcher. Exclude speakers and
    # validate availability before requesting even the remote ZIP directory.
    rows = bounded._metadata(meta, meta_size, meta_hash, output, budget)
    remaining = [row for row in rows if row.get('ID', '').split('_')[0] not in excluded]
    selected = bounded.select_rows(remaining, count, speakers, seed + ':' + source_split)
    selected_ids = {row['ID'].split('_')[0] for row in selected}
    if selected_ids & excluded:
        raise ValueError('Selected holdout speaker overlaps prior experiments')
    source = {'repository': bounded.REPOSITORY, 'revision': bounded.REVISION,
              'license': bounded.LICENSE, 'official_source_split': source_split,
              'metadata_file': meta, 'metadata_sha256': meta_hash,
              'archive': archive, 'archive_bytes': archive_size,
              'media': media,
              'archive_sha256_from_publisher': archive_hash,
              'whole_archive_hash_verified': False, 'member_crc_verified': True,
              'seed': seed,
              'selection': 'hash-ranked unused speakers and IDs, round-robin; no label-quality or length filtering',
              'exclusion_manifests': exclusions, 'excluded_speaker_ids': sorted(excluded),
              'exclusion_scope': 'supplied_prior_manifests_only',
              'local_speaker_overlap_checked': True,
              'audio_downloaded': None if media == 'full_face' else False,
              'separate_audio_downloaded': False,
              'embedded_audio_may_be_present': media == 'full_face',
              'evaluation_modality_required': 'video_frames_only',
              'slides_downloaded': False,
              'label_provenance': 'Author-provided manual transcript; not independently re-transcribed'}
    remote = bounded.RangeFile(bounded.url_for(archive), archive_size, budget)
    with zipfile.ZipFile(remote) as zipped:
        index = {entry.filename: entry for entry in zipped.infolist()}
    folder = archive.removesuffix('.zip')

    def fetch(row):
        identifier = row['ID']
        if media == 'full_face':
            face = row.get('FACE', '')
            parts = face.split('/') if isinstance(face, str) else []
            if (len(parts) != 4 or parts[0] != row.get('TOPIC') or
                    parts[1] != identifier.rsplit('_', 1)[0] or parts[2] != 'FACE' or
                    parts[3] != identifier + '_FACE.mp4' or '..' in parts):
                raise ValueError('invalid author FACE metadata path')
            # Publisher metadata includes TOPIC; the original test ZIP omits
            # that directory. Require the exact session/FACE/ID suffix.
            canonical = '/'.join(parts[1:])
            matches = [name for name in index if name in (face, canonical) or
                       name.endswith('/' + face) or name.endswith('/' + canonical)]
            if len(matches) != 1:
                raise ValueError(f'author FACE metadata needs one exact archive member: {identifier}')
            member = matches[0]
        else:
            member = f'{folder}/{identifier}.mp4'
        if member not in index:
            raise ValueError(f'author metadata has no matching video: {identifier}')
        info = index[member]
        if media == 'full_face' and (info.file_size > MAX_FACE_MEMBER_BYTES or
                info.compress_size > MAX_FACE_MEMBER_BYTES or info.file_size < 1):
            raise ValueError('unexpectedly large or empty full-face video member')
        path = output / 'videos' / 'test' / f'{identifier}.mp4'
        cached = path.read_bytes() if path.exists() else b''
        if len(cached) == info.file_size and zlib.crc32(cached) == info.CRC:
            data = cached
        else:
            reader = read_face_member if media == 'full_face' else bounded.read_member
            data = reader(bounded.url_for(archive), archive_size, info, budget)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.part')
            temporary.write_bytes(data)
            temporary.replace(path)
        dimensions = verify_face_dimensions(path) if media == 'full_face' else None
        sample = {'id': f'chinese-lips:test:{identifier}', 'source_id': identifier,
                'video': str(path.relative_to(output)), 'reference': row['TEXT'],
                'speaker': f'chinese-lips:{identifier.split("_")[0]}',
                'session': identifier.rsplit('_', 1)[0], 'split': 'test',
                'official_source_split': source_split,
                'domain': media, 'mouth_roi': media == 'mouth_roi',
                'articulation': 'voiced', 'articulation_verified': True,
                'label_source': f'{bounded.SOURCE}/blob/{bounded.REVISION}/{meta}#ID={identifier}',
                'label_verified': True, 'topic': row.get('TOPIC', ''),
                'sha256': hashlib.sha256(data).hexdigest(), 'archive_member': member,
                'archive_member_crc32': f'{info.CRC:08x}'}
        if dimensions:
            sample['source_frame_width'], sample['source_frame_height'] = dimensions
        return sample

    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        samples = list(pool.map(fetch, selected))
    manifest = {'schema_version': 1, 'split': 'test', 'official_source_split': source_split,
                'training_overlap_checked': False,
                'description': ('Speaker cohort excludes every identity supplied in prior manifests. '
                                'Original base-model pretraining overlap remains unknown; normal voiced '
                                'videos do not establish silent webcam accuracy.'),
                'source': source, 'samples': samples}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'test.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    summary = {'schema_version': 1, 'source': bounded.SOURCE, 'revision': bounded.REVISION,
               'license': bounded.LICENSE, 'official_source_split': source_split, 'local_split': 'test',
               'media': media,
               'samples': len(samples), 'speakers': len(selected_ids), 'speaker_ids': sorted(selected_ids),
               'excluded_speaker_ids': sorted(excluded), 'exclusion_manifests': exclusions,
               'manifest': str(output / 'test.json'), 'seed': seed,
               'download_bytes_requested': budget.requested, 'download_budget_bytes': maximum,
               'download_workers': download_workers,
               'download_accounting': 'Requested ranges in this invocation including retries; cached members are reused.',
               'limitations': ['Not the publisher test split when official_source_split=train.',
                               'Excludes only supplied prior manifests; original pretraining overlap unknown.',
                               'Voiced author video; no webcam, deliberate mute, or free conversational speech evidence.',
                               'Original FACE containers may include embedded audio; evaluate decoded video frames only.',
                               'Author labels not independently re-transcribed.']}
    (output / 'provenance.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-noncommercial-license', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-split', choices=('train', 'test'), default='train')
    parser.add_argument('--media', choices=('mouth_roi', 'full_face'), default='mouth_roi')
    parser.add_argument('--exclude-manifest', action='append', type=Path, required=True,
                        help='Repeat for every prior train/dev/test manifest, including identities-only records')
    parser.add_argument('--count', type=int, default=40)
    parser.add_argument('--speakers', type=int, default=10)
    parser.add_argument('--seed', default='lipflow-chinese-local-holdout-v1')
    parser.add_argument('--max-download-mb', type=int, default=100)
    parser.add_argument('--download-workers', type=int, default=4)
    args = parser.parse_args(argv)
    if not args.accept_noncommercial_license:
        parser.error('Read the official CC-BY-NC-SA-4.0 license, then pass --accept-noncommercial-license')
    try:
        result = prepare_holdout(args.output_dir.resolve(), source_split=args.source_split,
                                 count=args.count, speakers=args.speakers, seed=args.seed,
                                 maximum=args.max_download_mb * 1_000_000,
                                 exclude_manifests=args.exclude_manifest,
                                 download_workers=args.download_workers, media=args.media)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
