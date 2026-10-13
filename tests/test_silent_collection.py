"""Synthetic data only; never opens cameras, microphones or models."""
import builtins
import hashlib
import json
from pathlib import Path
import itertools

import pytest

from research import silent_collection as sc
from research.scripts import collect_silent_chinese as cli


def details(**changes):
    return sc.metadata(**dict(speaker='speaker01', session='day01', split='train',
                              recording_date='2026-10-13', lighting='daylight-front', **changes))


class Recording:
    def __init__(self, directory, payload=b'original image stream'):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / '.unconfirmed-test.mp4'
        self.path.write_bytes(payload)
        self.max_bytes = 1024 * 1024
        self.ts = [i / 25 for i in range(30)]
        self.duration = self.ts[-1]
        self.finished = False

    def finish(self):
        self.finished = True

    def evidence(self):
        return {'capture_timestamps_seconds': self.ts, 'capture_frames': 30,
                'saved_frames': 30, 'saved_fps': 25, 'face_tracking_available': False}


def test_help_imports_no_device_or_model(monkeypatch, capsys):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name in ('cv2', 'research.collection', 'torch', 'AVFoundation', 'sounddevice'):
            pytest.fail('Help imported a device/model dependency')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    with pytest.raises(SystemExit) as info:
        cli.main(['--help'])
    assert info.value.code == 0
    assert 'audit_silent_data' in capsys.readouterr().out


def test_cli_passes_explicit_metadata_and_profile_without_capture(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv('LIPCN_PROFILE_DIR', str(tmp_path))
    monkeypatch.setattr(sc, 'collect', lambda *args, **kwargs: calls.append((args, kwargs)) or tmp_path / 'none')
    assert cli.main(['--speaker', 's1', '--session', 'd1', '--split', 'test',
                     '--date', '2026-10-13', '--lighting', 'window-front']) == 0
    assert calls[0][0][0] == tmp_path / 'silent-data'
    assert calls[0][0][1]['recording_date'] == '2026-10-13'
    assert calls[0][0][1]['split'] == 'test'


@pytest.mark.parametrize('changes', [{'speaker': ''}, {'lighting': ''}, {'recording_date': '2026-10-35'}, {'split': 'unknown'}])
def test_invalid_metadata(changes):
    values = dict(speaker='s1', session='day1', split='train', recording_date='2026-10-13', lighting='front')
    values.update(changes)
    with pytest.raises(ValueError):
        sc.metadata(**values)


def test_prompt_requires_explicit_entity_annotation_and_checks_reference(tmp_path):
    with pytest.raises(ValueError):
        sc.prompt('请联系小林', None)
    with pytest.raises(ValueError):
        sc.prompt('请联系小林', [{'kind': 'person_name', 'text': '小周'}])
    path = tmp_path / 'prompts.json'
    path.write_text(json.dumps([{'reference': '小林明天9点到', 'entities': [
        {'kind': 'person_name', 'text': '小林'}, {'kind': 'number', 'text': '9'}]}]), encoding='utf-8')
    assert len(sc.load_prompts(path)[0]['entities']) == 2


def test_two_confirmations_preserve_full_video_metadata_and_reset_audit(tmp_path):
    rec = Recording(tmp_path / 'train')
    card = sc.prompt('请在九点联系小林', [{'kind': 'number', 'text': '九'}, {'kind': 'person_name', 'text': '小林'}])
    with pytest.raises(ValueError, match='Separate human'):
        sc.commit_clip(rec, tmp_path, details(), card, silent_confirmed=True)
    assert rec.path.exists() and not (tmp_path / 'train/manifest.json').exists()
    original = rec.path.read_bytes()
    row = sc.commit_clip(rec, tmp_path, details(), card, silent_confirmed=True, label_confirmed=True)
    video = tmp_path / 'train' / row['video']
    assert video.read_bytes() == original
    assert row['sha256'] == hashlib.sha256(original).hexdigest()
    assert row['recording_date'] == '2026-10-13' and row['lighting'] == 'daylight-front'
    assert row['articulation_verified'] and row['entities_verified'] and row['label_verified']
    assert row['audio_recorded'] is False and row['mouth_roi'] is False
    assert row['collection']['capture_timestamps_seconds'] == rec.ts
    assert not rec.path.exists()
    assert not json.loads((tmp_path / 'train/manifest.json').read_text(encoding='utf-8'))['training_overlap_checked']


def test_split_speaker_reference_and_date_isolation(tmp_path):
    sc.commit_clip(Recording(tmp_path / 'train'), tmp_path, details(), sc.prompt('明天发送文件', []),
                   silent_confirmed=True, label_confirmed=True)
    current = details()
    with pytest.raises(ValueError, match='another split'):
        sc.check_isolation(tmp_path, dict(current, split='test'))
    with pytest.raises(ValueError, match='reference is already'):
        sc.check_isolation(tmp_path, dict(current, split='test', speaker='speaker02'), '明天，发送文件！')
    with pytest.raises(ValueError, match='one actual recording date'):
        sc.check_isolation(tmp_path, dict(current, recording_date='2026-10-14'))
    sc.check_isolation(tmp_path, dict(current, session='day02', recording_date='2026-10-14'))


def test_failed_manifest_write_preserves_previous_and_removes_new_video(tmp_path, monkeypatch):
    from research import collection
    sc.commit_clip(Recording(tmp_path / 'train'), tmp_path, details(), sc.prompt('请发文件', []),
                   silent_confirmed=True, label_confirmed=True)
    path = tmp_path / 'train/manifest.json'
    previous = path.read_bytes()
    def fail(*args):
        raise OSError('write failed')
    monkeypatch.setattr(collection, '_atomic_json', fail)
    with pytest.raises(OSError, match='write failed'):
        sc.commit_clip(Recording(tmp_path / 'train', b'another original stream'), tmp_path, details(),
                       sc.prompt('请勿发文件', []), silent_confirmed=True, label_confirmed=True)
    assert path.read_bytes() == previous
    assert len(list((tmp_path / 'train/videos').glob('*.mp4'))) == 1
    assert not (tmp_path / '.silent-collection.lock').exists()


def test_too_short_or_existing_lock_cannot_publish(tmp_path):
    rec = Recording(tmp_path / 'train')
    rec.duration = .1
    with pytest.raises(ValueError, match='too short'):
        sc.commit_clip(rec, tmp_path, details(), sc.prompt('请发文件', []), silent_confirmed=True, label_confirmed=True)
    rec.duration = 1.
    (tmp_path / '.silent-collection.lock').touch()
    with pytest.raises(RuntimeError, match='Another collector'):
        sc.commit_clip(rec, tmp_path, details(), sc.prompt('请发文件', []), silent_confirmed=True, label_confirmed=True)
    assert not (tmp_path / 'train/manifest.json').exists()


def test_duplicate_or_incompatible_manifest_cannot_publish(tmp_path):
    sc.commit_clip(Recording(tmp_path / 'train'), tmp_path, details(), sc.prompt('请发文件', []),
                   silent_confirmed=True, label_confirmed=True)
    with pytest.raises(ValueError, match='Duplicate video'):
        sc.commit_clip(Recording(tmp_path / 'train'), tmp_path, details(), sc.prompt('请勿发文件', []),
                       silent_confirmed=True, label_confirmed=True)
    (tmp_path / 'dev').mkdir()
    (tmp_path / 'dev/manifest.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='incompatible'):
        sc.check_isolation(tmp_path, details())


@pytest.mark.parametrize('keys,save', [
    ([27], False),
    ([32, 255, 27], False),
    ([32] + [255] * 12 + [32, 13, ord('s'), 13], True),
    ([32] + [255] * 12 + [32, ord('s'), ord('r'), 27], False),
])
def test_camera_loop_requires_explicit_start_and_two_confirmations_and_cleans_up(tmp_path, monkeypatch, keys, save):
    import numpy as np
    from research import collection
    cv2 = collection.cv2
    class Camera:
        released = False
        def isOpened(self): return True
        def set(self, *args): pass
        def read(self): return True, np.full((16, 16, 3), 7, np.uint8)
        def release(self): self.released = True
    class Writer:
        frames = []
        def __init__(self, path, *args): self.path = Path(path)
        def isOpened(self): return True
        def write(self, frame):
            self.frames.append(frame.copy())
            with self.path.open('ab') as stream: stream.write(frame.tobytes())
        def release(self): pass
    camera, key_iterator, clock = Camera(), iter(keys), itertools.count(100, .1)
    monkeypatch.setattr(sc, '_resolve_camera', lambda value: 0)
    monkeypatch.setattr(collection, '_camera_permission', lambda: None)
    monkeypatch.setattr(cv2, 'VideoCapture', lambda *args: camera)
    monkeypatch.setattr(cv2, 'VideoWriter', Writer)
    monkeypatch.setattr(cv2, 'imshow', lambda *args: None)
    monkeypatch.setattr(cv2, 'waitKey', lambda *args: next(key_iterator))
    monkeypatch.setattr(cv2, 'getWindowProperty', lambda *args: 1)
    monkeypatch.setattr(cv2, 'destroyAllWindows', lambda: None)
    monkeypatch.setattr(cv2, 'putText', lambda frame, *args: frame.fill(99))
    monkeypatch.setattr(sc.time, 'monotonic', lambda: next(clock))
    result = sc.collect(tmp_path, details(), cards=[sc.prompt('请发文件', [])])
    assert result.exists() is save
    assert camera.released
    assert not list((tmp_path / 'train').glob('.unconfirmed-*'))
    # Preview overlays must never enter the video.
    assert all(np.all(frame == 7) for frame in Writer.frames)
