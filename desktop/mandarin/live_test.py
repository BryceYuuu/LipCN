#!/usr/bin/env python3
"""Local Mandarin dictation and English lip reading with reviewed wording.

Microphone/camera clips and recognized text stay in memory. No focused-field context,
network inference, automatic paste, transcript history, or personal training.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import time

from config import ADAPTER, CHECKPOINT, ENGLISH_MODELS, FORMATTER_MODEL, PROFILE, ROOT, SOURCE, STATE

# The retained lipflow Python package reads this legacy variable on import.
# Keep this product's profile separate even when launched directly with Python.
os.environ['LIPFLOW_HOME'] = str(PROFILE)
os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('OPENCV_AVFOUNDATION_SKIP_AUTH', '1')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'research' / 'scripts'))

import cv2
import numpy as np
import objc
from AppKit import (
    NSApplication, NSApplicationActivationPolicyRegular, NSBackingStoreBuffered,
    NSButton, NSColor, NSFont, NSImageScaleProportionallyUpOrDown, NSImageView,
    NSMakeRect, NSMenu, NSMenuItem, NSPopUpButton, NSScrollView, NSTextField, NSTextView,
    NSWindow, NSWindowStyleMaskClosable, NSWindowStyleMaskMiniaturizable,
    NSWindowStyleMaskTitled,
)
from Foundation import NSObject, NSTimer
from PyObjCTools import AppHelper

from lipflow.camera import Camera, mouth_view
from lipflow.dictation import TAIL_SECONDS, clip_problem, rois_for
from lipflow.hud import _nsimage
from hybrid_audio import AudioCapture, LocalASR
from hybrid_candidates import CandidateFormatter
from hybrid_hotkey import CommandToggle
from hybrid_output import OutputTarget
from english_lips import load_english_reader
from languages import DEFAULT_LANGUAGE, LANGUAGE_CHOICES, recognition_language

try:
    STATE.mkdir(parents=True, exist_ok=True)
except OSError:
    pass  # Operational telemetry must never prevent the visible app from starting.
STATUS = STATE / 'status.json'
MAX_SECONDS = 20.0


def label(root, text, frame, size=15, color=None):
    view = NSTextField.wrappingLabelWithString_(text)
    view.setFrame_(NSMakeRect(*frame))
    view.setFont_(NSFont.systemFontOfSize_(size))
    if color is not None:
        view.setTextColor_(color)
    root.addSubview_(view)
    return view


class LiveTest(NSObject):
    def init(self):
        self = objc.super(LiveTest, self).init()
        if self is None:
            return None
        self.reader = None
        self.reader_language = 'zh'
        self.language = DEFAULT_LANGUAGE
        self.language_switching = False
        self.language_worker = None
        self.camera = None
        self.closed = False
        self.suspended = False
        self.loading = True
        self.speech_ready = False
        self.lip_ready = False
        self.wording_ready = False
        self.startup_complete = False
        self.startup_stage = '加载语音模型'
        self.startup_started = time.monotonic()
        self.startup_stage_started = self.startup_started
        self.model_executor = None
        self.model_worker = None
        self.inference_worker = None
        self.cleanup_future = None
        self._executor_shutdown = False
        self._termination_waiting = False
        self.processing = False
        self.recording = False
        self.pending_stop = None
        self.capture_guidance = ''
        self.token = 0
        self.frames = 0
        self.face_present = False
        self.last_frame_at = 0.0
        self.last_ui_frame = 0.0
        self.ui_frame_pending = False
        self.last_status_write = 0.0
        self.completed = 0
        self.model_seconds = None
        self.adapter_sha256 = None
        self.last_latency = None
        self.last_stage_seconds = {}
        self.warmup_seconds = {}
        self.error = None
        self.model_failed = False
        self.audio = AudioCapture()
        self.asr = LocalASR(compute_type='float32', cpu_threads=4)
        self.formatter = CandidateFormatter(model_path=FORMATTER_MODEL)
        self.output_target = OutputTarget()
        self.microphone_allowed = False
        self.microphone_note = '麦克风权限待确认'
        self.hotkey = None
        self.hotkey_note = '快捷键正在初始化'
        self.last_route = None
        self.result_variants = []
        self.pending_begin = False
        self.camera_permission_pending = False
        self._build()
        self._status('starting')
        return self

    @objc.python_method
    def _build(self):
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 1000, 800),
            NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable,
            NSBackingStoreBuffered, False)
        self.window.setTitle_('LipCN')
        self.window.setReleasedWhenClosed_(False)
        self.window.setDelegate_(self)
        root = self.window.contentView()
        label(root, 'LipCN', (26, 752, 650, 30), 25)
        label(root, '语言', (670, 752, 50, 25), 15)
        self.language_popup = NSPopUpButton.alloc().initWithFrame_pullsDown_(
            NSMakeRect(725, 747, 249, 32), False)
        self.language_popup.addItemsWithTitles_([title for _, title in LANGUAGE_CHOICES])
        self.language_popup.selectItemAtIndex_(0)
        self.language_popup.setTarget_(self)
        self.language_popup.setAction_('changeLanguage:')
        self.language_popup.setEnabled_(False)
        root.addSubview_(self.language_popup)
        self.video = NSImageView.alloc().initWithFrame_(NSMakeRect(26, 345, 640, 360))
        self.video.setImageScaling_(NSImageScaleProportionallyUpOrDown)
        root.addSubview_(self.video)
        self.mouth = NSImageView.alloc().initWithFrame_(NSMakeRect(686, 505, 288, 180))
        self.mouth.setImageScaling_(NSImageScaleProportionallyUpOrDown)
        root.addSubview_(self.mouth)
        self.access_label = label(root, '', (686, 371, 288, 48), 11, NSColor.secondaryLabelColor())
        for x, title, action in ((686, '启用快捷键', 'requestHotkey:'), (836, '麦克风权限', 'requestMicrophone:')):
            b = NSButton.buttonWithTitle_target_action_(title, self, action)
            b.setFrame_(NSMakeRect(x, 336, 136, 29))
            root.addSubview_(b)
        self.state_label = label(root, '加载语音模型 · 0.0 秒', (26, 311, 945, 28), 17)
        self.start_button = NSButton.buttonWithTitle_target_action_('开始说一句', self, 'begin:')
        self.start_button.setFrame_(NSMakeRect(26, 254, 240, 46))
        self.start_button.setFont_(NSFont.boldSystemFontOfSize_(19))
        self.start_button.setEnabled_(False)
        root.addSubview_(self.start_button)
        self.stop_button = NSButton.buttonWithTitle_target_action_('结束并识别', self, 'finish:')
        self.stop_button.setFrame_(NSMakeRect(280, 254, 240, 46))
        self.stop_button.setFont_(NSFont.boldSystemFontOfSize_(19))
        self.stop_button.setEnabled_(False)
        root.addSubview_(self.stop_button)
        self.cancel_button = NSButton.buttonWithTitle_target_action_('取消本句', self, 'cancel:')
        self.cancel_button.setFrame_(NSMakeRect(536, 254, 170, 46))
        self.cancel_button.setKeyEquivalent_('\x1b')
        self.cancel_button.setEnabled_(False)
        root.addSubview_(self.cancel_button)
        quit_button = NSButton.buttonWithTitle_target_action_('关闭', self, 'quit:')
        quit_button.setFrame_(NSMakeRect(800, 254, 174, 46))
        root.addSubview_(quit_button)
        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(26, 90, 948, 152))
        scroll.setHasVerticalScroller_(True)
        self.result = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, 928, 154))
        self.result.setEditable_(False)
        self.result.setSelectable_(True)
        self.result.setFont_(NSFont.systemFontOfSize_(19))
        self.result.setString_('')
        self.result.setVerticallyResizable_(True)
        self.result.setHorizontallyResizable_(False)
        self.result.textContainer().setWidthTracksTextView_(True)
        scroll.setDocumentView_(self.result)
        root.addSubview_(scroll)
        self.choice_buttons = []
        for index in range(3):
            for offset, action, title in ((0, 'copyChoice:', '复制'), (139, 'insertChoice:', '输入')):
                b = NSButton.buttonWithTitle_target_action_(f'{title}方案{"一二三"[index]}', self, action)
                b.setFrame_(NSMakeRect(26 + index * 319 + offset, 49, 132, 32))
                b.setTag_(index)
                b.setEnabled_(False)
                root.addSubview_(b)
                self.choice_buttons.append(b)
        self.window.center()

    @objc.python_method
    def launch(self):
        self.window.makeKeyAndOrderFront_(None)
        app = NSApplication.sharedApplication()
        app.activate() if hasattr(app, 'activate') else app.activateIgnoringOtherApps_(True)
        self.timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            .2, self, 'tick:', None, True)
        self._request_microphone()
        self.hotkey = CommandToggle(self._toggle_command, self._hotkey_error, on_cancel=self._escape)
        if self.hotkey.start():
            self.hotkey_note = '右 Command：按一下开始，再按结束'
        self._refresh_access()
        self.startup_started = time.monotonic()
        self.startup_stage_started = self.startup_started
        self.model_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='lipcn-model')
        self.model_worker = self.model_executor.submit(self._load_model)

    @objc.python_method
    def _request_microphone(self):
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
        status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)
        if status == 0:
            AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                AVMediaTypeAudio, lambda granted: AppHelper.callAfter(self._microphone_permission, bool(granted)))
        else:
            self._microphone_permission(status == 3)

    @objc.python_method
    def _microphone_permission(self, granted):
        self.microphone_allowed = granted
        self.microphone_note = '语音优先已启用' if granted else '麦克风未授权：仅唇语；可在系统设置中允许'
        self._refresh_access()
        if granted and getattr(self, 'speech_ready', False):
            self.error = None

    @objc.python_method
    def _refresh_access(self):
        notes = []
        if getattr(self, 'language', DEFAULT_LANGUAGE) != 'en' and not self.microphone_allowed:
            notes.append('麦克风未启用')
        if self.hotkey is None or not getattr(self.hotkey, 'is_listening', False):
            notes.append('右 Command 未启用')
        self.access_label.setStringValue_(' · '.join(notes))

    @objc.python_method
    def _refresh_language_control(self):
        popup = getattr(self, 'language_popup', None)
        if popup is not None:
            popup.setEnabled_(not (self.closed or self.loading or self.pending_begin or
                                  self.recording or self.processing or self.language_switching))

    def changeLanguage_(self, sender):
        index = int(self.language_popup.indexOfSelectedItem())
        previous = self.language
        if (self.closed or self.loading or self.pending_begin or self.recording or
                self.processing or self.language_switching or not 0 <= index < len(LANGUAGE_CHOICES)):
            self.language_popup.selectItemAtIndex_(
                next(i for i, (code, _) in enumerate(LANGUAGE_CHOICES) if code == previous))
            return
        language = LANGUAGE_CHOICES[index][0]
        if language == previous:
            return
        self.language = language
        self.token += 1
        self.error = None
        self.result_variants = []
        self.result.setString_('')
        self.last_route = None
        for button in self.choice_buttons:
            button.setEnabled_(False)
        self._refresh_access()
        model_language = recognition_language(language)
        if model_language != self.reader_language or not self.lip_ready:
            self.language_switching = True
            self.loading = True
            self.lip_ready = False
            self.startup_stage = '加载英文口型模型' if model_language == 'en' else '加载中文口型模型'
            self.startup_started = time.monotonic()
            self.startup_stage_started = self.startup_started
            self.start_button.setEnabled_(False)
            try:
                self.language_worker = self.model_executor.submit(
                    self._load_language_model, model_language, self.token)
            except (AttributeError, RuntimeError):
                self._language_model_finished(self.token, model_language, None, 0.0,
                                              '识别组件未就绪，请重新打开。')
        self._refresh_language_control()
        self._status('language_changed')

    @objc.python_method
    def _load_language_model(self, model_language, token):
        """Replace the active visual model on the same worker used for inference."""
        started = time.monotonic()
        digest, error = None, None
        try:
            if self.closed:
                return
            import gc
            import torch
            if self.reader is not None and getattr(self.reader.enc_device, 'type', '') == 'mps':
                torch.mps.synchronize()
            self.reader = None
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if self.closed:
                return
            if model_language == 'en':
                reader = load_english_reader(ENGLISH_MODELS)
            else:
                from evaluate_cnvsrc import _reader, _load_adapter
                reader = _reader(CHECKPOINT, SOURCE, 40, .1, 'mps')
                metadata = _load_adapter(reader, ADAPTER)
                digest = metadata['file_sha256']
                reader.encode(np.zeros((25, 96, 96), dtype=np.uint8))
            self.reader = reader
            self.reader_language = model_language
            if reader.enc_device.type == 'mps':
                torch.mps.empty_cache()
        except Exception as exc:
            # Paths and model errors may contain private data; show a fixed message.
            error = ('英文口型模型未能加载，请检查英文模型是否已安装。' if model_language == 'en'
                     else '中文口型模型未能加载，请重新打开后重试。')
            self.lip_load_error = type(exc).__name__
        finally:
            AppHelper.callAfter(self._language_model_finished, token, model_language,
                                digest, time.monotonic() - started, error)

    @objc.python_method
    def _language_model_finished(self, token, model_language, digest, seconds, error):
        if self.closed:
            return
        # Cancelling or hiding does not cancel a model load. Readiness still needs
        # to settle, but a late load must not overwrite a newer result or error.
        self.language_switching = False
        self.loading = False
        self.lip_ready = self.reader is not None and self.reader_language == model_language and error is None
        self.adapter_sha256 = digest
        self.warmup_seconds['lips'] = seconds
        self.startup_stage = '就绪'
        self.model_failed = (not self.lip_ready if model_language == 'en'
                             else not (self.speech_ready or self.lip_ready))
        if token == self.token:
            self.error = error
            if error:
                self.result.setString_(error)
        self.start_button.setEnabled_(self._can_begin())
        self._refresh_language_control()
        self._status('language_ready' if self.lip_ready else 'language_unavailable')

    @objc.python_method
    def _hotkey_error(self, message):
        self.hotkey_note = '右 Command 未启用：系统设置 → 隐私与安全性 → 输入监控；仍可点击按钮'
        self._refresh_access()

    def requestHotkey_(self, sender):
        import Quartz
        from Foundation import NSURL
        from AppKit import NSWorkspace
        if not Quartz.CGPreflightListenEventAccess():
            Quartz.CGRequestListenEventAccess()
        if self.hotkey.start():
            self.hotkey_note = '右 Command：按一下开始，再按结束'
            self._refresh_access()
        else:
            NSWorkspace.sharedWorkspace().openURL_(NSURL.URLWithString_(
                'x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent'))

    def requestMicrophone_(self, sender):
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
        if AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio) in (1, 2):
            from AppKit import NSWorkspace
            from Foundation import NSURL
            NSWorkspace.sharedWorkspace().openURL_(NSURL.URLWithString_(
                'x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone'))
        self._request_microphone()

    @objc.python_method
    def _toggle_command(self):
        if self.closed or self.processing:
            return
        if self.recording:
            self.finish_(None)
            return
        if self.pending_begin:
            self.cancel_(None)
            return
        try:
            self.output_target.capture()
        except Exception:
            self.output_target.target = None
        self.window.deminiaturize_(None)
        self.suspended = False
        self.window.makeKeyAndOrderFront_(None)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        self.begin_(None)

    @objc.python_method
    def _escape(self):
        if self.pending_begin or self.recording or self.processing:
            self.cancel_(None)

    @objc.python_method
    def _open_camera(self):
        if self.closed or self.suspended or self.camera is not None:
            return
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
        status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
        if status == 0:
            self.state_label.setStringValue_('请在系统弹窗中允许摄像头权限。')
            self.camera_permission_pending = True
            token = self.token
            AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                AVMediaTypeVideo, lambda granted: AppHelper.callAfter(self._camera_permission, bool(granted), token))
            return
        if status != 3:
            self.error = '请在系统设置 → 隐私与安全性 → 摄像头中允许此应用，再重新打开。'
            self.result.setString_(self.error)
            self._status('camera_permission_denied')
            return
        self.face_present = False
        self.last_frame_at = 0
        self.camera = Camera('4K USB Camera', on_frame=self._camera_frame)
        self.camera.track_always = True
        self.camera.ensure_open()

    @objc.python_method
    def _camera_permission(self, granted, token):
        if token != self.token:
            return
        self.camera_permission_pending = False
        if (not self.closed and not self.suspended and granted and token == self.token
                and (self.pending_begin or self.recording)):
            self._open_camera()
            if self.recording and self.camera is not None:
                self.camera.start_recording()

    @objc.python_method
    def _close_camera(self):
        camera, self.camera = self.camera, None
        self.face_present = False
        if camera is not None:
            camera.stop_recording()
            camera.close()
            tracker = getattr(camera, '_tracker', None)
            if tracker is not None and not camera.is_open:
                tracker.close()

    @objc.python_method
    def _camera_frame(self, frame, observation, is_recording):
        if self.closed or self.suspended:
            return
        now = time.monotonic()
        self.frames += 1
        self.last_frame_at = now
        self.face_present = observation is not None
        if observation is None:
            self.capture_guidance = '请正对镜头，让嘴部完整入镜'
        if self.ui_frame_pending or now - self.last_ui_frame < .10:
            return
        self.ui_frame_pending = True
        self.last_ui_frame = now
        try:
            if observation is not None:
                lips = observation.outer_lips
                mouth_pixels = float(np.ptp(lips[:, 0]))
                x0, y0 = np.floor(lips.min(0)).astype(int)
                x1, y1 = np.ceil(lips.max(0)).astype(int)
                h, w = frame.shape[:2]
                patch = frame[max(0, y0):min(h, y1 + 1), max(0, x0):min(w, x1 + 1)]
                brightness = float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).mean()) if patch.size else None
                advice = []
                if mouth_pixels < 35:
                    advice.append('请靠近镜头')
                if brightness is not None and brightness < 35:
                    advice.append('嘴部偏暗，请补光')
                elif brightness is not None and brightness > 225:
                    advice.append('嘴部过亮，请调整光线')
                self.capture_guidance = ' · '.join(advice) or f'嘴部宽度 {mouth_pixels:.0f} px'
            preview = cv2.flip(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA), 1)
            mouth = mouth_view(frame, observation, 288, 180)
            AppHelper.callAfter(self._draw_frame, preview, mouth)
        except Exception:
            self.ui_frame_pending = False
            raise

    @objc.python_method
    def _draw_frame(self, preview, mouth):
        try:
            if not self.closed and not self.suspended:
                self.video.setImage_(_nsimage(preview))
                self.mouth.setImage_(_nsimage(mouth))
        finally:
            self.ui_frame_pending = False

    @objc.python_method
    def _load_model(self):
        """Initialize and later use every model on the same persistent worker."""
        started = time.monotonic()
        wording_ok = False
        wording_seconds = 0.0
        try:
            if self.closed:
                return
            AppHelper.callAfter(self._startup_phase, '加载语音模型', time.monotonic())
            speech_started = time.monotonic()
            try:
                speech_ok = bool(self.asr.warmup())
            except Exception:
                speech_ok = False
            AppHelper.callAfter(self._speech_loaded, speech_ok, time.monotonic() - speech_started)
            if self.closed:
                return

            AppHelper.callAfter(self._startup_phase, '加载口型模型', time.monotonic())
            lip_started = time.monotonic()
            try:
                import torch
                torch.set_num_threads(4)
                from evaluate_cnvsrc import _reader, _load_adapter
                reader = _reader(CHECKPOINT, SOURCE, 40, .1, 'mps')
                metadata = _load_adapter(reader, ADAPTER)
                reader.encode(np.zeros((25, 96, 96), dtype=np.uint8))
                if reader.enc_device.type == 'mps':
                    torch.mps.empty_cache()
                # Publish the worker-owned model before a queued inference can run;
                # the main-thread callback only updates UI readiness and metadata.
                self.reader = reader
                AppHelper.callAfter(self._model_loaded, None, metadata['file_sha256'],
                                    time.monotonic() - lip_started)
            except Exception as exc:
                AppHelper.callAfter(self._model_error, type(exc).__name__)
            if self.closed:
                return

            AppHelper.callAfter(self._startup_phase, '加载表达模型', time.monotonic())
            wording_started = time.monotonic()
            try:
                wording_ok = bool(self.formatter.warmup())
            except Exception:
                wording_ok = False
            wording_seconds = time.monotonic() - wording_started
        finally:
            if not self.closed:
                AppHelper.callAfter(self._startup_finished, wording_ok, wording_seconds,
                                    time.monotonic() - started)

    @objc.python_method
    def _startup_phase(self, stage, started):
        if self.closed:
            return
        self.startup_stage = stage
        self.startup_stage_started = started
        self._status('loading')

    @objc.python_method
    def _speech_loaded(self, available, seconds):
        if self.closed:
            return
        self.speech_ready = bool(available)
        self.warmup_seconds['speech'] = seconds
        self.start_button.setEnabled_(self._can_begin())
        self._status('speech_ready' if available else 'speech_unavailable')

    @objc.python_method
    def _model_loaded(self, reader, digest, seconds):
        if self.closed:
            return
        if reader is not None:  # Compatibility with callers that supply a reader.
            self.reader = reader
        self.lip_ready = True
        self.adapter_sha256 = digest
        self.warmup_seconds['lips'] = seconds
        self.error = None
        self.start_button.setEnabled_(self._can_begin())
        self._status('model_loaded')

    @objc.python_method
    def _model_error(self, error):
        if self.closed:
            return
        self.lip_ready = False
        self.lip_load_error = error
        self._status('lip_unavailable')

    @objc.python_method
    def _startup_finished(self, wording_ok, wording_seconds, total_seconds):
        if self.closed:
            return
        self.wording_ready = bool(wording_ok)
        self.warmup_seconds['wording'] = wording_seconds
        self.model_seconds = total_seconds
        self.startup_complete = True
        self.loading = False
        self.startup_stage = '就绪'
        self.model_failed = not (getattr(self, 'speech_ready', False) or
                                 getattr(self, 'lip_ready', False))
        if self.model_failed:
            self.error = '识别模型加载失败，请关闭后重新打开。'
            self.state_label.setStringValue_(self.error)
            self.result.setString_(self.error)
        else:
            self.error = None
        self.start_button.setEnabled_(self._can_begin())
        self._status('error' if self.model_failed else 'ready')

    @objc.python_method
    def _can_begin(self):
        if getattr(self, 'language_switching', False):
            return False
        if getattr(self, 'language', DEFAULT_LANGUAGE) == 'en':
            available = (getattr(self, 'lip_ready', False) and
                         getattr(self, 'reader_language', 'zh') == 'en')
        elif hasattr(self, 'speech_ready') or hasattr(self, 'lip_ready'):
            available = ((getattr(self, 'speech_ready', False) and
                          getattr(self, 'microphone_allowed', False)) or
                         getattr(self, 'lip_ready', False))
        else:
            # Compatibility for older test/controller doubles without staged state.
            available = not self.loading and not getattr(self, 'model_failed', False)
        return (not self.closed and not self.suspended and available and
                not self.processing and not self.recording and not self.pending_begin)

    def tick_(self, timer):
        if self.closed:
            return
        self._refresh_language_control()
        self.start_button.setEnabled_(self._can_begin())
        self.stop_button.setEnabled_(self.recording and self.pending_stop is None)
        self.cancel_button.setEnabled_(self.recording or self.processing or self.pending_begin)
        if self.suspended:
            state = 'paused'
        elif self.pending_begin:
            self.state_label.setStringValue_('准备采集…')
            state = 'preparing'
        elif self.recording:
            seconds = time.monotonic() - self.record_started
            if self.pending_stop is not None:
                self.state_label.setStringValue_('结束中…')
                state = 'finishing'
            else:
                self.state_label.setStringValue_(f'采集中 · {seconds:.1f} 秒')
                state = 'recording'
            if seconds >= MAX_SECONDS - TAIL_SECONDS and self.pending_stop is None:
                self.finish_(None)
        elif self.processing:
            label = '识别中' if getattr(self, 'startup_complete', not self.loading) else '等待模型就绪'
            self.state_label.setStringValue_(f'{label} · {time.monotonic() - self.inference_started:.1f} 秒')
            state = 'processing'
        elif self.loading:
            elapsed = time.monotonic() - getattr(self, 'startup_started', time.monotonic())
            phase = getattr(self, 'startup_stage', '加载模型')
            ready = '可以开始 · ' if self._can_begin() else ''
            self.state_label.setStringValue_(f'{ready}{phase} · {elapsed:.1f} 秒')
            state = 'loading'
        elif getattr(self, 'model_failed', False):
            self.state_label.setStringValue_(self.error or '识别模型加载失败，请重新打开。')
            state = 'error'
        elif self._can_begin():
            self.state_label.setStringValue_('就绪')
            state = 'ready'
        else:
            self.state_label.setStringValue_('请允许麦克风后开始。')
            state = 'waiting_for_microphone'
        if time.monotonic() - self.last_status_write >= 1.0:
            self._status(state)

    def begin_(self, sender):
        if not self._can_begin():
            return
        self.token += 1
        self.pending_stop = None
        self.error = None
        self.pending_begin = True
        self._refresh_language_control()
        self.camera_permission_pending = False
        if sender is not None:
            try:
                self.output_target.capture()
            except Exception:
                self.output_target.target = None
        self._open_camera()
        self.start_button.setEnabled_(False)
        self.cancel_button.setEnabled_(True)
        self.result.setString_('')
        self._begin_when_ready(self.token, time.monotonic() + 4.0)

    @objc.python_method
    def _begin_when_ready(self, token, deadline):
        if token != self.token or not self.pending_begin or self.closed or self.suspended:
            return
        waiting = ((self.camera is not None and not self.camera.ready.is_set() and not self.camera.error)
                   or self.camera_permission_pending)
        if waiting and time.monotonic() < deadline:
            AppHelper.callLater(.05, self._begin_when_ready, token, deadline)
            return
        self.pending_begin = False
        self.record_started = time.monotonic()
        if self.camera is not None:
            self.camera.start_recording()
        if self.microphone_allowed and getattr(self, 'language', DEFAULT_LANGUAGE) != 'en':
            if not self.audio.start():
                self.microphone_note = '麦克风不可用，本句转为唇语'
                self._refresh_access()
        if not self.audio.capturing and (self.camera is None or self.camera.error):
            self._close_camera()
            self.recording = False
            self.cancel_button.setEnabled_(False)
            self.result.setString_(
                'English 模式需要可用的摄像头，请允许摄像头权限后重试。'
                if getattr(self, 'language', DEFAULT_LANGUAGE) == 'en' else
                '摄像头和麦克风均不可用。请在系统设置 → 隐私与安全性中允许设备，再开始。')
            self._status('capture_unavailable')
            return
        self.recording = True
        self.result_variants = []
        for b in self.choice_buttons:
            b.setEnabled_(False)
        self.start_button.setEnabled_(False)
        self.stop_button.setEnabled_(True)
        self.cancel_button.setEnabled_(True)
        self.result.setString_('')
        # The total recording cap includes the tail. Tokens make stale callbacks harmless.
        AppHelper.callLater(max(0.0, MAX_SECONDS - TAIL_SECONDS), self._auto_finish, self.token)
        self._status('recording')

    @objc.python_method
    def _auto_finish(self, token):
        if token == self.token and self.recording and self.pending_stop is None and not self.closed and not self.suspended:
            self.finish_(None)

    def finish_(self, sender):
        if not self.recording or self.pending_stop is not None:
            return
        self.pending_stop = self.token
        self.stop_button.setEnabled_(False)
        self.state_label.setStringValue_('结束中…')
        # Manual and automatic stops use the same tail, shortened at the 20-second cap.
        remaining = max(0.0, MAX_SECONDS - (time.monotonic() - self.record_started))
        AppHelper.callLater(min(TAIL_SECONDS, remaining), self._finish_after_tail, self.token)
        self._status('finishing')

    @objc.python_method
    def _finish_after_tail(self, token):
        if (token != self.token or self.pending_stop != token or not self.recording or
                self.closed or self.suspended):
            return
        self.pending_stop = None
        rec = self.camera.stop_recording() if self.camera is not None else None
        audio = self.audio.stop()
        self.recording = False
        self.stop_button.setEnabled_(False)
        self._close_camera()
        # A busy UI may deliver its timer late. Keep inference input within the cap.
        if rec is not None:
            count = sum(timestamp <= rec.started + MAX_SECONDS for timestamp in rec.ts)
            for name in ('ts', 'grays', 'anchors', 'mouth_open', 'mouth_pixels'):
                setattr(rec, name, getattr(rec, name)[:count])
        self.processing = True
        self.inference_started = time.monotonic()
        self.result.setString_('')
        try:
            self.inference_worker = self.model_executor.submit(self._infer, self.token, rec, audio)
        except (AttributeError, RuntimeError):
            self._infer_error(self.token, '识别组件未就绪，请重新打开。')
            return
        self._status('processing')

    @objc.python_method
    def _infer(self, token, rec, audio):
        started = time.monotonic()
        stage_seconds = {'queue': max(0.0, started - getattr(self, 'inference_started', started))}
        try:
            if token != self.token or self.closed:
                AppHelper.callAfter(self._discarded, token)
                return
            stage_started = time.monotonic()
            language = getattr(self, 'language', DEFAULT_LANGUAGE)
            # English is the upstream visual recognizer, never Chinese ASR or a translation.
            speech = self.asr.transcribe(audio) if language != 'en' else None
            stage_seconds['speech'] = time.monotonic() - stage_started
            audio_duration = len(audio) / 16000.0
            del audio
            if token != self.token or self.closed:
                AppHelper.callAfter(self._discarded, token)
                return
            duration = rec.duration if rec is not None else audio_duration
            count = len(rec.ts) if rec is not None else 0
            widths = [value for value in rec.mouth_pixels if value > 0] if rec is not None else []
            metadata = {'fps': (count - 1) / duration if duration > 0 else 0.0,
                        'mouth_pixels': float(np.median(widths)) if widths else 0.0,
                        'face_ratio': rec.face_ratio if rec is not None else 0.0,
                        'route': 'speech' if speech is not None and speech.accepted else 'lips',
                        'audio_reason': speech.reason if speech is not None else 'english_visual_only',
                        'language': language}
            if speech is not None and speech.accepted:
                texts = [speech.text]
                del rec
            else:
                problem = clip_problem(rec) if rec is not None else ('No camera', '')
                if problem:
                    translations = {'Too short': '本句太短，请至少说或默念一秒。',
                                    "Can't see your face": '没有可靠语音，且多数画面未检测到脸，请正对镜头重试。',
                                    'No lip movement': '没有可靠语音，也未检测到足够嘴部动作，请重试。',
                                    'No camera': '没有可靠语音或可用口型，请检查麦克风、摄像头权限。'}
                    if language == 'en':
                        translations.update({
                            "Can't see your face": '多数画面未检测到脸，请正对镜头重试。',
                            'No lip movement': '未检测到足够嘴部动作，请重试。',
                            'No camera': '没有可用口型，请检查摄像头权限。'})
                    raise ValueError(translations.get(problem[0], '画面质量不足，请重试。'))
                if self.reader is None:
                    raise ValueError('没有可靠语音，且唇语模型尚不可用。')
                if getattr(self, 'reader_language', 'zh') != recognition_language(language):
                    raise ValueError('当前语言的口型模型尚未就绪，请稍后重试。')
                stage_started = time.monotonic()
                rois = rois_for(rec)
                stage_seconds['mouth_crop'] = time.monotonic() - stage_started
                del rec
                if token != self.token or self.closed:
                    AppHelper.callAfter(self._discarded, token)
                    return
                if rois is None or len(rois) < 12:
                    raise ValueError('本句缺少可用的嘴部画面，请重新录制。')
                stage_started = time.monotonic()
                encoded = self.reader.encode(rois)
                stage_seconds['lip_encoder'] = time.monotonic() - stage_started
                stage_started = time.monotonic()
                if language == 'en':
                    hypotheses = self.reader.hypotheses(encoded, nbest=3)
                else:
                    from evaluate_cnvsrc import configured_hypotheses
                    hypotheses, _ = configured_hypotheses(
                        self.reader, encoded, beam_size=40, ctc_weight=.1,
                        nbest=3, reverse_weight=.3, reverse_scoring='batched',
                        candidate_pool_size=10, pre_beam_ratio=1.5)
                stage_seconds['lip_decoder'] = time.monotonic() - stage_started
                texts = [hypothesis.text for hypothesis in hypotheses]
                del rois
            if not any(text.strip() for text in texts):
                raise ValueError('没有识别出可用文字，请重新说或默念这一句。')
            if token != self.token or self.closed:
                AppHelper.callAfter(self._discarded, token)
                return
            stage_started = time.monotonic()
            formatted = self.formatter.format(texts, source=metadata['route'], language=language)
            stage_seconds['wording'] = time.monotonic() - stage_started
            metadata['stage_seconds'] = stage_seconds
            AppHelper.callAfter(self._inferred, token, formatted,
                               time.monotonic() - getattr(self, 'inference_started', started),
                               duration, count, metadata)
        except Exception as exc:
            # Errors remain in the UI, never serialize user-provided text from exceptions.
            message = str(exc) if isinstance(exc, ValueError) else '本地识别组件暂不可用，请重试。'
            AppHelper.callAfter(self._infer_error, token, message)

    @objc.python_method
    def _discarded(self, token):
        self.processing = False

    @objc.python_method
    def _inferred(self, token, formatted, seconds, duration, frames, metadata):
        self.processing = False
        if self.closed or token != self.token:
            return
        self.completed += 1
        self.last_latency = seconds
        self.last_stage_seconds = metadata.get('stage_seconds', {})
        self.cancel_button.setEnabled_(False)
        self.last_route = metadata['route']
        self.result_variants = [v.text for v in formatted.variants]
        lines = [f'方案{"一二三"[index]}：{variant.text}'
                 for index, variant in enumerate(formatted.variants[:3])]
        for b in self.choice_buttons:
            b.setEnabled_(b.tag() < len(self.result_variants) and bool(self.result_variants[b.tag()]))
        self.result.setString_('\n\n'.join(lines))
        self._status('result_shown')

    def copyChoice_(self, sender):
        index = sender.tag()
        if index < len(self.result_variants):
            self.output_target.copy(self.result_variants[index])
            self.state_label.setStringValue_(f'方案 {index + 1} 已复制。')

    def insertChoice_(self, sender):
        index = sender.tag()
        if index < len(self.result_variants):
            try:
                self.output_target.insert(self.result_variants[index])
                self.state_label.setStringValue_(f'方案 {index + 1} 已输入原应用。')
            except Exception as exc:
                self.result.setString_(self.result.string() + '\n' + str(exc))

    @objc.python_method
    def _infer_error(self, token, error):
        self.processing = False
        if self.closed or token != self.token:
            return
        self.error = error
        self.result.setString_('本句识别失败：' + error)
        self.cancel_button.setEnabled_(False)
        self._status('inference_error')

    def cancel_(self, sender):
        self.token += 1
        self.pending_stop = None
        self.pending_begin = False
        self.camera_permission_pending = False
        self.audio.cancel()
        self.result_variants = []
        for b in self.choice_buttons:
            b.setEnabled_(False)
        if self.camera is not None:
            self.camera.stop_recording()
        self.recording = False
        self._close_camera()
        self.stop_button.setEnabled_(False)
        self.cancel_button.setEnabled_(False)
        self.result.setString_('本句已取消。' + ('正在等待当前计算结束…' if self.processing else '可以重新开始。'))
        self._status('cancelled')

    @objc.python_method
    def _suspend(self):
        if self.closed or self.suspended:
            return
        self.cancel_(None)
        self.suspended = True
        self._close_camera()
        self.video.setImage_(None)
        self.mouth.setImage_(None)
        self._status('paused')

    @objc.python_method
    def _resume(self):
        if not self.closed and self.suspended:
            self.suspended = False

    def windowDidMiniaturize_(self, note):
        self._suspend()

    def windowDidDeminiaturize_(self, note):
        self._resume()

    def applicationDidHide_(self, note):
        self._suspend()

    def applicationDidUnhide_(self, note):
        if not self.window.isMiniaturized():
            self._resume()

    def applicationShouldHandleReopen_hasVisibleWindows_(self, app, visible):
        self.window.deminiaturize_(None)
        self.window.makeKeyAndOrderFront_(None)
        self._resume()
        app.activateIgnoringOtherApps_(True)
        return True

    def windowWillClose_(self, note):
        self.quit_(None)

    def applicationWillTerminate_(self, note):
        self._shutdown()

    def applicationShouldTerminateAfterLastWindowClosed_(self, app):
        return False  # Stop the event loop after worker cleanup; let Python unwind.

    def applicationShouldTerminate_(self, app):
        self._shutdown()
        if not getattr(self, '_termination_waiting', False):
            self._termination_waiting = True
            AppHelper.callLater(.1, self._finish_shutdown)
        # Cocoa termination calls native exit directly. Cancel that path and let
        # main() return normally after the Python/native models have been released.
        return 0  # NSTerminateCancel

    @objc.python_method
    def _workers_alive(self):
        for worker in (getattr(self, 'model_worker', None),
                       getattr(self, 'language_worker', None),
                       getattr(self, 'inference_worker', None),
                       getattr(self, 'cleanup_future', None)):
            if worker is None:
                continue
            done = getattr(worker, 'done', None)
            if callable(done):
                if not done():
                    return True
            elif getattr(worker, 'is_alive', lambda: False)():
                return True  # Older controller/test doubles may still use Thread.
        return False

    @objc.python_method
    def _cleanup_models(self):
        """Release model resources on their sole owning worker, after queued work."""
        import gc
        errors = {}
        mx = sys.modules.get('mlx.core')
        torch = sys.modules.get('torch')
        if mx is not None:
            try:
                generation = sys.modules.get('mlx_lm.generate')
                stream = getattr(generation, 'generation_stream', None)
                if stream is not None:
                    mx.synchronize(stream)
                mx.synchronize()
            except Exception as exc:
                errors['mlx_sync'] = type(exc).__name__
        if torch is not None:
            try:
                torch.mps.synchronize()
            except Exception as exc:
                errors['mps_sync'] = type(exc).__name__
        self.reader = None
        try:
            close = getattr(self.asr, 'close', None)
            if callable(close):
                close()
            else:
                self.asr._model = None
                self.asr._speech_detector = None
                self.asr._converter = None
        except Exception as exc:
            errors['speech_close'] = type(exc).__name__
        if getattr(self, 'formatter', None) is not None:
            self.formatter._model = None
        gc.collect()
        if mx is not None:
            try:
                mx.clear_cache()
                mx.clear_streams()
            except Exception as exc:
                errors['mlx_close'] = type(exc).__name__
        if torch is not None:
            try:
                torch.mps.empty_cache()
            except Exception as exc:
                errors['mps_close'] = type(exc).__name__
        self.cleanup_errors = errors

    @objc.python_method
    def _finish_shutdown(self):
        if self._workers_alive():
            AppHelper.callLater(.1, self._finish_shutdown)
            return
        executor = getattr(self, 'model_executor', None)
        if executor is not None and not getattr(self, '_executor_shutdown', False):
            executor.shutdown(wait=True, cancel_futures=True)
            self._executor_shutdown = True
        self._status('closed')
        self._termination_waiting = False
        if getattr(self, 'window', None) is not None:
            self.window.orderOut_(None)
        # AppHelper.stopEventLoop delegates to NSApp.terminate_ for GUI loops,
        # which would re-enter our cancelled Cocoa termination request.
        app = NSApplication.sharedApplication()
        app.stop_(None)
        from AppKit import NSEvent, NSEventTypeApplicationDefined
        wake = NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
            NSEventTypeApplicationDefined, (0.0, 0.0), 0, time.monotonic(),
            0, None, 0, 0, 0)
        app.postEvent_atStart_(wake, False)

    def quit_(self, sender):
        self.applicationShouldTerminate_(NSApplication.sharedApplication())

    @objc.python_method
    def _shutdown(self):
        if self.closed:
            return
        self.closed = True
        self.token += 1
        self.pending_stop = None
        self.pending_begin = False
        self.camera_permission_pending = False
        self.recording = False
        self.audio.cancel()
        if self.hotkey is not None:
            self.hotkey.stop()
        if hasattr(self, 'timer'):
            self.timer.invalidate()
        self._close_camera()
        for future in (getattr(self, 'model_worker', None), getattr(self, 'language_worker', None),
                       getattr(self, 'inference_worker', None)):
            cancel = getattr(future, 'cancel', None)
            if callable(cancel):
                cancel()
        executor = getattr(self, 'model_executor', None)
        if executor is not None and getattr(self, 'cleanup_future', None) is None:
            self.cleanup_future = executor.submit(self._cleanup_models)
        self._status('closing')

    @objc.python_method
    def _status(self, state):
        self.last_status_write = time.monotonic()
        data = {'schema_version': 1, 'pid': os.getpid(), 'state': state,
                'updated_at_unix': time.time(),
                'language': getattr(self, 'language', DEFAULT_LANGUAGE),
                'model': ('Auto-AVSR LRS3 English' if getattr(self, 'reader_language', 'zh') == 'en'
                          else 'CNVSRC Round8 research candidate'),
                'reader_language': getattr(self, 'reader_language', 'zh'),
                'language_switching': getattr(self, 'language_switching', False),
                'window_visible': bool(self.window.isVisible()), 'window_minimized': bool(self.window.isMiniaturized()), 'model_loaded': self.reader is not None, 'model_load_seconds': self.model_seconds,
                'adapter_sha256': self.adapter_sha256, 'camera_open': bool(self.camera and self.camera.is_open),
                'camera_frames_seen': self.frames, 'face_present': self.face_present,
                'ready_to_record': self._can_begin(), 'recording': self.recording,
                'processing': self.processing, 'completed_utterances': self.completed,
                'last_inference_seconds': self.last_latency, 'error': bool(self.error),
                'last_stage_seconds': getattr(self, 'last_stage_seconds', {}),
                'warmup_seconds': getattr(self, 'warmup_seconds', {}),
                'startup_complete': getattr(self, 'startup_complete', not self.loading),
                'startup_stage': getattr(self, 'startup_stage', ''),
                'startup_elapsed_seconds': time.monotonic() - getattr(self, 'startup_started', time.monotonic()),
                'speech_ready': getattr(self, 'speech_ready', False),
                'lip_ready': getattr(self, 'lip_ready', self.reader is not None),
                'wording_ready': getattr(self, 'wording_ready', False),
                'cleanup_errors': getattr(self, 'cleanup_errors', {}),
                'speech_model_loaded': bool(getattr(self.asr, 'loaded', False)),
                'wording_model_loaded': getattr(self.formatter, '_model', None) is not None,
                'speech_cpu_threads': getattr(self.asr, 'cpu_threads', None),
                'speech_compute_type': getattr(self.asr, 'compute_type', None),
                'beam_size': 4 if getattr(self, 'reader_language', 'zh') == 'en' else 40,
                'ctc_weight': .1,
                'reverse_weight': None if getattr(self, 'reader_language', 'zh') == 'en' else .3,
                'build': '2026-10-10-languages', 'tail_seconds': TAIL_SECONDS, 'max_utterance_seconds': MAX_SECONDS,
                'microphone_authorized': self.microphone_allowed, 'last_route': self.last_route,
                'hotkey_listening': bool(self.hotkey and self.hotkey.is_listening),
                'microphone_capturing': self.audio.capturing,
                'audio_persisted': False, 'frames_persisted': False, 'transcripts_persisted': False,
                'cleanup_local_only': True, 'clipboard_requires_choice': True, 'context_used': False}
        temporary = STATUS.with_suffix('.json.tmp')
        try:
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            temporary.replace(STATUS)
            self.status_write_error = None
        except OSError as exc:
            self.status_write_error = f'{type(exc).__name__}: {exc}'
            # Status is diagnostic metadata, not application state. Never abort the UI.



def main():
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    controller = LiveTest.alloc().init()
    app.setDelegate_(controller)
    menu_bar = NSMenu.alloc().init()
    app_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_('LipCN', None, '')
    app_menu = NSMenu.alloc().initWithTitle_('LipCN')
    quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_('退出 LipCN', 'quit:', 'q')
    quit_item.setTarget_(controller)
    app_menu.addItem_(quit_item)
    app_item.setSubmenu_(app_menu)
    menu_bar.addItem_(app_item)
    app.setMainMenu_(menu_bar)
    controller.launch()
    AppHelper.runEventLoop()


if __name__ == '__main__':
    main()
