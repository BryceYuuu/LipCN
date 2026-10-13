"""Local, explicitly operated silent capture; imports no devices until collect()."""
from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

from research.evaluation import characters

SPLITS = ('train', 'dev', 'test')
WINDOW = 'LipCN silent collection'


def metadata(speaker, session, split, recording_date, lighting):
    for key, value in (('speaker', speaker), ('session', session), ('lighting', lighting)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{key} must be a nonempty human declaration')
    if split not in SPLITS:
        raise ValueError('Choose an explicit train, dev or test split')
    if not isinstance(recording_date, str) or date.fromisoformat(recording_date).isoformat() != recording_date:
        raise ValueError('recording_date must be YYYY-MM-DD')
    return dict(speaker=speaker.strip(), session=session.strip(), split=split,
                recording_date=recording_date, lighting=lighting.strip())


def prompt(reference, entities):
    if not characters(reference):
        raise ValueError('Reference must contain letters or numbers')
    if not isinstance(entities, list):
        raise ValueError('Explicit entities list required (empty means no names/numbers)')
    checked = []
    for entity in entities:
        if not isinstance(entity, dict) or entity.get('kind') not in ('number', 'person_name'):
            raise ValueError('Entity kind must be number or person_name')
        text = entity.get('text')
        normalized = ''.join(characters(text))
        if not normalized or normalized not in ''.join(characters(reference)):
            raise ValueError('Every entity must occur verbatim after CER normalization in the reference')
        checked.append({'kind': entity['kind'], 'text': text})
    return {'reference': reference, 'entities': checked}


def load_prompts(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, list) or not data:
        raise ValueError('Prompt file must contain a nonempty JSON array')
    if any(not isinstance(row, dict) for row in data):
        raise ValueError('Every prompt must be an object')
    return [prompt(row.get('reference'), row.get('entities')) for row in data]


def check_isolation(root, details, reference=None):
    """Fail closed against all sibling manifests, including changes since preview."""
    for split in SPLITS:
        manifest = Path(root) / split / 'manifest.json'
        if not manifest.exists():
            continue
        data = json.loads(manifest.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or data.get('schema_version') != 1 or data.get('split') != split or not isinstance(data.get('samples'), list):
            raise ValueError('Existing collection manifest is incompatible')
        for row in data['samples']:
            if not isinstance(row, dict) or row.get('split') != split:
                raise ValueError('Existing collection row has an inconsistent split')
            if row.get('speaker') == details['speaker']:
                if split != details['split']:
                    raise ValueError('This speaker already belongs to another split; do not rename them to bypass isolation')
                if row.get('session') == details['session'] and row.get('recording_date') != details['recording_date']:
                    raise ValueError('A session must identify one actual recording date')
            if reference is not None and split != details['split'] and characters(row.get('reference')) == characters(reference):
                raise ValueError('This reference is already in another split; use independently authored prompts')


def commit_clip(recording, root, details, card, *, silent_confirmed=False, label_confirmed=False):
    """Atomically publish original full-frame video plus explicit human metadata."""
    if silent_confirmed is not True or label_confirmed is not True:
        raise ValueError('Separate human confirmation of silence and exact reference/entities is required')
    details = metadata(**details)
    card = prompt(**card)
    root = Path(root).resolve()
    directory = root / details['split']
    if recording.directory.resolve() != directory:
        raise ValueError('Recording directory differs from the declared split')
    from research.collection import _atomic_json, _dataset, directory_bytes
    recording.finish()
    if recording.duration < .6 or len(recording.ts) < 12:
        raise ValueError('Clip is too short; record at least 0.6 seconds and 12 camera frames')
    lock = root / '.silent-collection.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError('Another collector is updating this dataset; inspect a stale lock before removing it') from None
    os.close(fd)
    final, published = None, False
    try:
        check_isolation(root, details, card['reference'])
        manifest = directory / 'manifest.json'
        data = _dataset(manifest, details['split'])
        digest = hashlib.sha256()
        with recording.path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if any(row.get('sha256') == digest.hexdigest() for row in data['samples']):
            raise ValueError('Duplicate video content is already present')
        videos = directory / 'videos'
        videos.mkdir(exist_ok=True, mode=0o700)
        identity = uuid.uuid4().hex
        final = videos / f'{identity}.mp4'
        evidence = recording.evidence()
        row = {'id': identity, 'video': final.relative_to(directory).as_posix(), **details, **card,
               'label_verified': True, 'entities_verified': True,
               'label_source': 'Operator prepared reference before recording and separately confirmed exact articulation/entities',
               'domain': 'webcam', 'mouth_roi': False, 'articulation': 'silent',
               'articulation_verified': True, 'audio_recorded': False,
               'confirmed_at': datetime.now().astimezone().isoformat(),
               'sha256': digest.hexdigest(), 'collection': evidence}
        data['samples'].append(row)
        data['training_overlap_checked'] = False
        encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
        if directory_bytes(directory) + len(encoded) + 4096 > recording.max_bytes:
            raise ValueError('Collection output quota exceeded')
        os.link(recording.path, final)
        published = True
        _atomic_json(manifest, data)
        published = False  # manifest now owns the final file, even if temp cleanup fails
        recording.path.unlink()
        return row
    except BaseException:
        if published:
            final.unlink(missing_ok=True)
        raise
    finally:
        lock.unlink(missing_ok=True)


def _interactive_card():
    reference = input('准确原句（空行结束）：').strip()
    if not reference:
        return None
    entities = []
    for kind, label in (('person_name', '人名'), ('number', '数字/数词（含中文数字）')):
        text = input(f'{label}，按原句填写，多个用 | 分隔，无则回车：')
        entities.extend({'kind': kind, 'text': item.strip()} for item in text.split('|') if item.strip())
    return prompt(reference, entities)


def _resolve_camera(camera):
    """Select a physical device without importing the runtime face/model stack."""
    if isinstance(camera, int) and not isinstance(camera, bool) and camera >= 0:
        return camera
    if isinstance(camera, str) and camera.isdecimal():
        return int(camera)
    if sys.platform != 'darwin':
        if camera == 'auto':
            return 0
        raise ValueError('On this platform select the camera by nonnegative device number')
    from AVFoundation import AVCaptureDevice, AVMediaTypeVideo, AVMediaTypeMuxed
    devices = list(AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)) + list(AVCaptureDevice.devicesWithMediaType_(AVMediaTypeMuxed))
    devices.sort(key=lambda device: device.uniqueID())  # OpenCV AVFoundation index order
    if not devices:
        raise RuntimeError('No physical cameras are available')
    if camera != 'auto':
        for index, device in enumerate(devices):
            if camera == str(device.uniqueID()) or str(camera).casefold() in str(device.localizedName()).casefold():
                return index
        raise ValueError('Requested physical camera was not found')
    for index, device in enumerate(devices):
        if 'BuiltIn' in str(device.deviceType()):
            return index
    return next((index for index, device in enumerate(devices) if 'iphone' not in str(device.localizedName()).casefold()), 0)


def collect(root, details, cards=None, camera='auto', max_seconds=15., max_megabytes=512.):
    """SPACE starts each clip; S then ENTER confirms; P replays; R/ESC discard."""
    details = metadata(**details)
    if not math.isfinite(max_seconds) or not .6 <= max_seconds <= 60 or not math.isfinite(max_megabytes) or max_megabytes <= 0:
        raise ValueError('max_seconds must be .6..60 and max_megabytes must be positive')
    if isinstance(camera, str) and (Path(camera).is_file() or '://' in camera):
        raise ValueError('Use a physical camera, not a video or network URL')
    root = Path(root).expanduser().resolve()
    directory = root / details['split']
    check_isolation(root, details)
    if cards is not None:
        cards = [prompt(**card) for card in cards]
        for card in cards:
            check_isolation(root, details, card['reference'])
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    print(f'仅摄像头，不打开麦克风。完整画面保存在：{directory / "videos"}', flush=True)
    print('请让完整脸部入镜。原句与画面必须一致，不能发声或耳语。SPACE 每句开始/停止；S 确认无声；Enter 确认原句和标注；P 回放；R 重录；Esc 丢弃退出。', flush=True)
    from research.collection import Recording, _camera_permission, directory_bytes
    import cv2
    if directory_bytes(directory) >= max_megabytes * 1024 * 1024:
        raise ValueError('Collection output quota exceeded')
    _camera_permission()
    source = _resolve_camera(camera)
    backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, backend)
    recording = None
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open camera; check permissions and close other camera apps')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        iterator = iter(cards) if cards is not None else None
        while True:
            card = next(iterator, None) if iterator is not None else _interactive_card()
            if card is None:
                break
            check_isolation(root, details, card['reference'])
            print('准备原句：' + card['reference'], flush=True)
            print('人名/数字：' + json.dumps(card['entities'], ensure_ascii=False), flush=True)
            review, silent, frame = False, False, None
            while True:
                if not review:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError('Camera stopped returning images')
                    now = time.monotonic()
                    if recording is not None:
                        if recording.ts and now - recording.ts[0] >= max_seconds:
                            recording.finish()
                            review = True
                        else:
                            recording.push(frame, now)
                preview = frame.copy()
                state = ('S confirm SILENT; ENTER confirm TEXT+ENTITIES; P replay; R retake' if review else 'SPACE start / stop; R discard; ESC quit')
                if silent:
                    state = 'SILENT confirmed; ENTER saves TEXT+ENTITIES; P replay; R retake'
                cv2.putText(preview, state, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
                if recording:
                    cv2.putText(preview, f'{recording.duration:.1f}s / {max_seconds}s', (12, 54), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                cv2.imshow(WINDOW, preview)
                key = cv2.waitKey(20) & 0xff
                if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    return directory / 'manifest.json'
                if key in (ord('r'), ord('R')):
                    if recording:
                        recording.discard()
                    recording, review, silent = None, False, False
                elif key == 32 and not review:
                    if recording is None:
                        recording = Recording(directory, int(max_megabytes * 1024 * 1024), tracking_available=False)
                    elif recording.ts:
                        recording.finish()
                        review = True
                elif review and key in (ord('p'), ord('P')):
                    replay = cv2.VideoCapture(str(recording.path))
                    try:
                        while replay.isOpened():
                            ok, replay_frame = replay.read()
                            if not ok:
                                break
                            cv2.imshow(WINDOW, replay_frame)
                            if cv2.waitKey(40) & 0xff == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                                return directory / 'manifest.json'
                    finally:
                        replay.release()
                elif review and key in (ord('s'), ord('S')):
                    silent = True
                    print('已声明没有发声或耳语；请再确认原句逐字一致，且人名/数字标注完整，Enter 保存。', flush=True)
                elif review and key in (10, 13):
                    if not silent:
                        print('先按 S 明确确认没有发声或耳语。', flush=True)
                        continue
                    if recording.duration < .6 or len(recording.ts) < 12:
                        print('片段过短，请 R 重录（至少 0.6 秒、12 帧）。', flush=True)
                        continue
                    row = commit_clip(recording, root, details, card, silent_confirmed=True, label_confirmed=True)
                    recording = None
                    print(f'已保存：{directory / row["video"]}', flush=True)
                    break
    finally:
        if recording is not None:
            recording.discard()
        cap.release()
        cv2.destroyAllWindows()
    return directory / 'manifest.json'
