"""No-network checks for untouched-speaker selection and honest source splits."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile
import types
import sys

import pytest

_spec = importlib.util.spec_from_file_location(
    'fetch_chinese_holdout', Path(__file__).resolve().parents[1] / 'research' / 'scripts' / 'fetch_chinese_holdout.py')
holdout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(holdout)


def prior(tmp_path, name='prior.json', speakers=('001',), split='train'):
    value = {'schema_version': 1, 'split': split,
             'source': {'repository': holdout.bounded.REPOSITORY, 'revision': holdout.bounded.REVISION},
             'samples': [{'speaker': f'chinese-lips:{speaker}', 'source_id': f'{speaker}_25_M_KJ_001'}
                         for speaker in speakers]}
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


def rows():
    return [{'ID': f'{speaker:03}_25_M_KJ_{clip:03}', 'TEXT': '作者原始标签', 'TOPIC': 'KJ'}
            for speaker in range(1, 7) for clip in range(1, 5)]


def options(tmp_path, exclusions, **changes):
    value = dict(source_split='train', count=12, speakers=3, seed='new-cohort',
                 maximum=50_000, exclude_manifests=exclusions, download_workers=2)
    value.update(changes)
    return value


def fake_source(monkeypatch, source_rows=None):
    source_rows = rows() if source_rows is None else source_rows
    encoded = io.BytesIO()
    with zipfile.ZipFile(encoded, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for row in source_rows:
            archive.writestr('processed_train/' + row['ID'] + '.mp4', ('video-' + row['ID']).encode())
    data = encoded.getvalue()
    files = dict(holdout.bounded.FILES)
    original = files['train']
    files['train'] = (*original[:4], len(data), original[5])
    monkeypatch.setattr(holdout.bounded, 'FILES', files)
    monkeypatch.setattr(holdout.bounded, '_metadata', lambda *args: source_rows)
    requested = []
    def read_range(url, size, start, count, budget):
        assert size == len(data)
        requested.append((start, count))
        budget.reserve(count)
        return data[start:start + count]
    monkeypatch.setattr(holdout.bounded, 'fetch_range', read_range)
    return requested


def test_exclusions_union_every_prior_split_without_labels_or_media(tmp_path):
    paths = [prior(tmp_path, 'train.json', ('001', '002'), 'train'),
             prior(tmp_path, 'dev.json', ('003',), 'dev'),
             prior(tmp_path, 'test.json', ('002', '004'), 'test')]
    excluded, evidence = holdout.exclusion_identities(paths)
    assert excluded == {'001', '002', '003', '004'}
    assert [entry['sample_count'] for entry in evidence] == [2, 1, 2]
    assert all(entry['manifest_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
               for entry, path in zip(evidence, paths))
    value = json.loads(paths[0].read_text())
    value['samples'][0].update(reference='不要用内容决定排除', raw='旧预测', video='missing.mp4')
    paths[0].write_text(json.dumps(value))
    assert holdout.exclusion_identities(paths)[0] == excluded


@pytest.mark.parametrize('change', [
    {'schema_version': 2}, {'samples': []}, {'samples': 'bad'},
    {'source': {'repository': 'other', 'revision': holdout.bounded.REVISION}},
    {'source': {'repository': holdout.bounded.REPOSITORY, 'revision': 'wrong'}},
    {'source': None},
    {'samples': [{'speaker': 'chinese-lips:001', 'source_id': '002_25_M_KJ_001'}]},
    {'samples': [{'speaker': 'chinese-lips:001', 'source_id': '../escape'}]},
])
def test_malformed_exclusions_stop_before_metadata_or_network(tmp_path, monkeypatch, change):
    path = prior(tmp_path)
    value = json.loads(path.read_text()); value.update(change)
    path.write_text(json.dumps(value))
    monkeypatch.setattr(holdout.bounded, '_metadata', lambda *args: pytest.fail('must validate prior source first'))
    with pytest.raises(ValueError):
        holdout.prepare_holdout(tmp_path / 'out', **options(tmp_path, [path]))
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('change', [
    {'source_split': 'dev'}, {'download_workers': 0}, {'download_workers': True},
    {'count': 2, 'speakers': 3}, {'count': True}, {'speakers': 0}, {'maximum': 0},
    {'exclude_manifests': []},
])
def test_invalid_options_stop_before_metadata(tmp_path, monkeypatch, change):
    path = prior(tmp_path)
    monkeypatch.setattr(holdout.bounded, '_metadata', lambda *args: pytest.fail('must validate options first'))
    with pytest.raises(ValueError):
        holdout.prepare_holdout(tmp_path / 'out', **options(tmp_path, [path], **change))


def test_not_enough_unused_speakers_refuses_before_archive_network(tmp_path, monkeypatch):
    path = prior(tmp_path, speakers=('001', '002', '003', '004', '005'))
    monkeypatch.setattr(holdout.bounded, '_metadata', lambda *args: rows())
    monkeypatch.setattr(holdout.bounded, 'RangeFile', lambda *args: pytest.fail('must not request ZIP directory'))
    with pytest.raises(ValueError, match='not enough source speakers'):
        holdout.prepare_holdout(tmp_path / 'out', **options(tmp_path, [path]))


def test_fetch_persists_true_publisher_split_and_verified_original_labels(tmp_path, monkeypatch):
    excluded = [prior(tmp_path, speakers=('001', '002'))]
    requests = fake_source(monkeypatch)
    out = tmp_path / 'out'
    summary = holdout.prepare_holdout(out, **options(tmp_path, excluded))
    manifest = json.loads((out / 'test.json').read_text())
    assert summary['samples'] == 12 and summary['speakers'] == 3
    assert summary['official_source_split'] == 'train' and summary['local_split'] == 'test'
    assert manifest['split'] == 'test' and manifest['official_source_split'] == 'train'
    assert manifest['training_overlap_checked'] is False
    assert manifest['source']['local_speaker_overlap_checked'] is True
    assert manifest['source']['audio_downloaded'] is False
    assert manifest['source']['whole_archive_hash_verified'] is False
    assert len(manifest['samples']) == 12
    assert not set(summary['speaker_ids']) & {'001', '002'}
    for sample in manifest['samples']:
        data = (out / sample['video']).read_bytes()
        assert sample['sha256'] == hashlib.sha256(data).hexdigest()
        assert sample['official_source_split'] == 'train'
        assert sample['reference'] == '作者原始标签'
        assert sample['articulation'] == 'voiced' and sample['domain'] == 'mouth_roi'
    assert sum(count for _, count in requests) == summary['download_bytes_requested']
    assert summary['download_bytes_requested'] <= summary['download_budget_bytes']


def test_same_seed_ignores_metadata_order_label_quality_and_length(tmp_path, monkeypatch):
    excluded = [prior(tmp_path)]
    fake_source(monkeypatch)
    holdout.prepare_holdout(tmp_path / 'one', **options(tmp_path, excluded))
    altered = [dict(row, TEXT=('不' * (index + 1))) for index, row in enumerate(reversed(rows()))]
    fake_source(monkeypatch, altered)
    holdout.prepare_holdout(tmp_path / 'two', **options(tmp_path, excluded))
    a, b = [json.loads((tmp_path / name / 'test.json').read_text())['samples'] for name in ('one', 'two')]
    assert [sample['source_id'] for sample in a] == [sample['source_id'] for sample in b]


def test_verified_cache_reuses_members_without_network_reads(tmp_path, monkeypatch):
    excluded = [prior(tmp_path)]
    fake_source(monkeypatch)
    out = tmp_path / 'out'
    first = holdout.prepare_holdout(out, **options(tmp_path, excluded))
    monkeypatch.setattr(holdout.bounded, 'read_member', lambda *args: pytest.fail('valid cached member should be reused'))
    second = holdout.prepare_holdout(out, **options(tmp_path, excluded))
    assert second['download_bytes_requested'] < first['download_bytes_requested']


def test_cli_requires_explicit_license_and_repeated_exclusions(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(holdout, 'prepare_holdout', lambda *args, **kwargs: calls.append(kwargs) or {})
    with pytest.raises(SystemExit) as raised:
        holdout.main(['--output-dir', str(tmp_path), '--exclude-manifest', 'old.json'])
    assert raised.value.code == 2 and not calls
    assert holdout.main(['--accept-noncommercial-license', '--output-dir', str(tmp_path),
                         '--exclude-manifest', 'train.json', '--exclude-manifest', 'test.json']) == 0
    assert calls[0]['exclude_manifests'] == [Path('train.json'), Path('test.json')]


def face_archive():
    row = {'ID': '006_25_M_KJ_001', 'TEXT': '原始完整标签', 'TOPIC': 'KJ',
           'FACE': 'KJ/006_25_M_KJ/FACE/006_25_M_KJ_001_FACE.mp4'}
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr('test/006_25_M_KJ/FACE/006_25_M_KJ_001_FACE.mp4', b'full-face bytes')
        zipped.writestr('test/006_25_M_KJ/WAV/006_25_M_KJ_001.wav', b'never downloaded')
    return row, output.getvalue()


def test_full_face_matches_exact_author_identity_and_preserves_nonwebcam_domain(tmp_path, monkeypatch):
    row, data = face_archive()
    monkeypatch.setattr(holdout, 'FULL_FACE_ARCHIVES', {'test': ('test.zip', len(data), 'publisherhash')})
    monkeypatch.setattr(holdout.bounded, '_metadata', lambda *args: [row])
    reads = []
    def read(url, size, start, count, budget):
        assert url.endswith('/test.zip')
        budget.reserve(count); reads.append((start, count))
        return data[start:start + count]
    monkeypatch.setattr(holdout.bounded, 'fetch_range', read)
    monkeypatch.setattr(holdout, 'verify_face_dimensions', lambda path: (1280, 720))
    out = tmp_path / 'out'
    result = holdout.prepare_holdout(out, **options(tmp_path, [prior(tmp_path)],
                 source_split='test', count=1, speakers=1, media='full_face'))
    sample = json.loads((out / 'test.json').read_text())['samples'][0]
    assert sample['domain'] == 'full_face' and sample['mouth_roi'] is False
    assert sample['articulation'] == 'voiced'
    assert sample['source_frame_width'] == 1280 and sample['source_frame_height'] == 720
    assert sample['archive_member'] == 'test/006_25_M_KJ/FACE/006_25_M_KJ_001_FACE.mp4'
    assert (out / sample['video']).read_bytes() == b'full-face bytes'
    assert result['media'] == 'full_face' and result['official_source_split'] == 'test'


def test_full_face_oversized_member_refuses_before_network(monkeypatch):
    row, data = face_archive()
    with zipfile.ZipFile(io.BytesIO(data)) as zipped:
        info = zipped.infolist()[0]
    info.file_size = holdout.MAX_FACE_MEMBER_BYTES + 1
    monkeypatch.setattr(holdout.bounded, 'fetch_range', lambda *args: pytest.fail('oversized must not request media'))
    with pytest.raises(ValueError, match='large or empty'):
        holdout.read_face_member('https://example.test', len(data), info, holdout.bounded.DownloadBudget(1000))


def test_full_face_crc_corruption_rejected(monkeypatch):
    _, data = face_archive()
    with zipfile.ZipFile(io.BytesIO(data)) as zipped:
        info = zipped.infolist()[0]
    monkeypatch.setattr(holdout.bounded, 'fetch_range', lambda url, size, start, count, budget: data[start:start+count])
    info.CRC ^= 1
    with pytest.raises(ValueError, match='CRC verification'):
        holdout.read_face_member('https://example.test', len(data), info, holdout.bounded.DownloadBudget(1000))


@pytest.mark.parametrize('height,width,ok', [(720, 1280, True), (96, 96, True), (0, 0, False)])
def test_full_face_decoded_geometry_is_checked_and_reader_always_closed(monkeypatch, height, width, ok):
    class Frame:
        ndim = 3
        shape = (height, width, 3)
    released = []
    video = types.SimpleNamespace(read=lambda: (ok, Frame() if ok else None), release=lambda: released.append(True))
    monkeypatch.setitem(sys.modules, 'cv2', types.SimpleNamespace(VideoCapture=lambda path: video))
    if ok and min(width, height) > 160:
        assert holdout.verify_face_dimensions(Path('fake.mp4')) == (width, height)
    else:
        with pytest.raises(ValueError, match='geometry|decode'):
            holdout.verify_face_dimensions(Path('fake.mp4'))
    assert released == [True]


def test_full_face_unsupported_multipart_train_refuses_before_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(holdout.bounded, '_metadata', lambda *args: pytest.fail('unsupported archive must not read metadata'))
    with pytest.raises(ValueError, match='only the official test'):
        holdout.prepare_holdout(tmp_path / 'out', **options(tmp_path, [prior(tmp_path)], media='full_face'))
