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


def commit_clip(recording, root, details, card, *, silent_confirmed=False, label_confirmed=False,
                articulation='silent'):
    """Atomically publish original full-frame video plus explicit human metadata."""
    if silent_confirmed is not True or label_confirmed is not True:
        raise ValueError('Separate human confirmation of silence and exact reference/entities is required')
    if articulation not in ('silent', 'voiced'):
        raise ValueError('articulation must be silent or voiced')
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
               'domain': 'webcam', 'mouth_roi': False, 'articulation': articulation,
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


FONT_PATHS = ('/System/Library/Fonts/Hiragino Sans GB.ttc', '/System/Library/Fonts/STHeiti Medium.ttc',
              '/System/Library/Fonts/STHeiti Light.ttc', 'C:/Windows/Fonts/msyh.ttc',
              '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
SCREEN_WIDTH = 880


def _font(size):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return None


def _wrap(text, font, width):
    lines, line = [], ''
    for character in text:
        if line and font.getlength(line + character) > width:
            lines.append(line)
            line = ''
        line += character
    return lines + ([line] if line else [])


def screen(frame, *, progress, sentence, hint, detail='', recording=False):
    """One window: progress and the sentence above a mirrored preview, keys below.

    Only this display copy is drawn on; recorded frames are never modified.
    """
    import cv2
    import numpy as np

    height = max(1, round(SCREEN_WIDTH * frame.shape[0] / max(1, frame.shape[1])))
    preview = cv2.flip(cv2.resize(frame, (SCREEN_WIDTH, height)), 1)
    if recording:
        cv2.rectangle(preview, (0, 0), (SCREEN_WIDTH - 1, height - 1), (40, 40, 230), 8)
    small, large = _font(22), _font(46)
    lines = _wrap(sentence, large, SCREEN_WIDTH - 56)[:2] if large is not None else []
    header, footer = 72 + 62 * max(1, len(lines)), 96
    canvas = np.full((header + height + footer, SCREEN_WIDTH, 3), 32, np.uint8)
    canvas[header:header + height] = preview
    if small is None or large is None:  # No CJK font: keep keys usable, sentence stays in the terminal.
        cv2.putText(canvas, 'SPACE start/stop  ENTER save  R retake  P replay  ESC quit',
                    (16, header + height + 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (230, 230, 230), 2)
        return canvas
    from PIL import Image, ImageDraw
    image = Image.fromarray(canvas[..., ::-1].copy())
    draw = ImageDraw.Draw(image)
    draw.text((28, 18), progress, font=small, fill=(170, 170, 170))
    for row, line in enumerate(lines):
        draw.text((28, 58 + row * 62), line, font=large, fill=(255, 255, 255))
    draw.text((28, header + height + 16), hint, font=small,
              fill=(235, 80, 80) if recording else (255, 255, 255))
    if detail:
        draw.text((28, header + height + 54), detail, font=small, fill=(250, 200, 90))
    return np.asarray(image)[..., ::-1].copy()


def pending_cards(root, split, cards):
    """Drop prompts whose text is already saved in this split, so sessions can resume."""
    manifest = Path(root) / split / 'manifest.json'
    if not manifest.exists():
        return list(cards), 0
    data = json.loads(manifest.read_text(encoding='utf-8'))
    saved = {tuple(characters(row.get('reference', ''))) for row in data.get('samples', [])
             if isinstance(row, dict)}
    pending = [card for card in cards if tuple(characters(card['reference'])) not in saved]
    return pending, len(cards) - len(pending)


def collect(root, details, cards=None, camera='auto', max_seconds=15., max_megabytes=512.,
            articulation='silent'):
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
        cards, skipped = pending_cards(root, details['split'], cards)
        if skipped:
            print(f'已跳过本分组中已保存的 {skipped} 句，还剩 {len(cards)} 句。', flush=True)
    else:
        skipped = 0
    total = skipped + len(cards) if cards is not None else None
    role = {'train': '训练句', 'dev': '测试句', 'test': '验收句'}[details['split']]
    manner = '默念（不出声）' if articulation == 'silent' else '正常说话'
    confirm = ('按 Enter 即确认：念的就是上面这句，并且没有出声' if articulation == 'silent'
               else '按 Enter 即确认：说的就是上面这句')
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    print(f'仅摄像头，不打开麦克风。完整画面保存在：{directory / "videos"}', flush=True)
    print(f'请让完整脸部入镜，{manner}。窗口里：空格 开始/结束，Enter 保存，R 重录，P 回放，Esc 退出。', flush=True)
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
        number = skipped
        while True:
            card = next(iterator, None) if iterator is not None else _interactive_card()
            if card is None:
                break
            number += 1
            check_isolation(root, details, card['reference'])
            print(f'第 {number} 句：' + card['reference'], flush=True)
            progress = f'{role} 第 {number} / {total} 句　·　{manner}' if total else f'{role} 第 {number} 句　·　{manner}'
            review, notice, frame = False, '', None
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
                if review:
                    hint, detail = 'Enter 保存　·　R 重录　·　P 回放　·　Esc 退出', notice or confirm
                elif recording is not None:
                    hint, detail = f'● 录制中 {recording.duration:.1f} 秒　·　念完按 空格 结束', ''
                else:
                    hint, detail = '按 空格 开始，念完再按 空格 结束　·　Esc 退出', notice
                cv2.imshow(WINDOW, screen(frame, progress=progress, sentence=card['reference'], hint=hint,
                                          detail=detail, recording=recording is not None and not review))
                key = cv2.waitKey(20) & 0xff
                if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    return directory / 'manifest.json'
                if key in (ord('r'), ord('R')):
                    if recording:
                        recording.discard()
                    recording, review, notice = None, False, ''
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
                elif review and key in (10, 13):
                    if recording.duration < .6 or len(recording.ts) < 12:
                        notice = '太短了（至少 0.6 秒），请按 R 重录'
                        continue
                    # One explicit Enter confirms both the articulation and the exact
                    # prepared sentence, as stated on screen before it is pressed.
                    row = commit_clip(recording, root, details, card, silent_confirmed=True,
                                      label_confirmed=True, articulation=articulation)
                    recording = None
                    print(f'已保存第 {number} 句', flush=True)
                    break
    finally:
        if recording is not None:
            recording.discard()
        cap.release()
        cv2.destroyAllWindows()
    return directory / 'manifest.json'
