"""Lipflow: hold a key, mouth the words, let go — the text appears at your cursor."""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import objc
import Quartz
from AppKit import (
    NSApplication, NSApplicationActivationPolicyAccessory, NSMenu, NSMenuItem, NSStatusBar,
    NSVariableStatusItemLength,
)
from Foundation import NSObject
from PyObjCTools import AppHelper

from .camera import Camera, Recording, mouth_view
from .cleanup import CHOICES, Cleaner, unavailable
from .face import mouth_rois
from .hotkey import KEYS, PushToTalk
from .hud import HUD, symbol
from AppKit import NSFontWeightRegular
from .paste import copy_text, paste_text
from .vsr import LipReader

from .paths import HOME
from .paths import WHO

from .dictation import (  # noqa: F401  (onboarding/settings import the settings helpers from here)
    HISTORY, JOIN_WINDOW, MAX_SECONDS, MIN_SECONDS, PREVIEW_EVERY, SETTINGS, TAIL_SECONDS, clip_problem,
    keep_clip, load_settings, log_history, rois_for, save_settings, train_on_face,
)


@dataclass
class Options:
    key: str = "right_option"
    beam: int = 4  # measured on 38 sentences: beam 10 25.2% WER 1.66 s/clip, beam 4 26.1% 0.99 s
    backend: str = "auto"
    camera: "int | str" = "auto"  # 'auto' = the Mac's built-in camera
    paste: bool = True
    live_preview: bool = True
    onboard: bool = False


def ui(fn, *args, **kw):
    AppHelper.callAfter(fn, *args, **kw)


class Lipflow(NSObject):
    def initWithOptions_(self, opts: Options):
        self = objc.super(Lipflow, self).init()
        if self is None:
            return None
        self.opts = opts
        self.reader: LipReader | None = None
        self.settings = load_settings()
        # --cleanup wins for this run; otherwise what you last picked in the menu
        self.cleanup_choice = opts.backend if opts.backend != "auto" else self.settings.get("cleanup", "auto")
        self.cleaner = Cleaner(self.cleanup_choice)
        self.jobs: "queue.Queue" = queue.Queue()
        self.session = 0          # bumps on every start/cancel so stale previews are dropped
        self.preview_busy = False
        self.last_output = ""
        self.last_paste_at = 0.0
        self.context: list[str] = []
        self.hands_free = False
        self.pending_stop = None
        from .mic import Mic
        self.mic = Mic()
        self.av_reader = None        # whisper mode's audio-visual model, loaded when switched on
        self._ui_busy = False        # a video frame is waiting to be drawn on the main thread
        self.onboarding = None       # the setup window while it's collecting practice clips
        self.onboarding_text = ""
        self.setup = None
        self.loading = True
        cam = opts.camera if opts.camera != "auto" else self.settings.get("camera", "auto")
        if opts.key == "right_option" and self.settings.get("key"):
            opts.key = self.settings["key"]
        self.camera = Camera(cam, on_frame=self.onFrame)
        return self

    # -- setup ---------------------------------------------------------------------
    @objc.python_method
    def start(self):
        self.hud = HUD.alloc().init()
        self._build_menu()
        self.ptt = PushToTalk(self.opts.key, self.on_start, self.on_stop, self.on_cancel)
        try:
            self.ptt.install()
        except PermissionError as e:
            print(f"\n[lipflow] {e}\n")
            Quartz.CGRequestListenEventAccess()
            self.hud.show("error", "Needs Input Monitoring", f"Allow {WHO}, then restart lipflow")
        if self.opts.paste and not Quartz.CGPreflightPostEventAccess():
            Quartz.CGRequestPostEventAccess()
            print(f"[lipflow] Allow {WHO} under Privacy & Security → Accessibility so Lipflow can paste.")
        self._request_camera()
        self.hud.show("reading", "Lipflow", "Loading the lip-reading model…")
        threading.Thread(target=self._worker, name="lipflow-model", daemon=True).start()
        self.jobs.put(("load",))

    @objc.python_method
    def _request_camera(self):
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
        status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
        if status == 0:  # not determined: show the system prompt now, from the main thread
            AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                AVMediaTypeVideo, lambda granted: print(f"[lipflow] camera access {'granted' if granted else 'denied'}"))
        elif status in (1, 2):
            print(f"[lipflow] camera access is denied: System Settings → Privacy & Security → Camera → enable {WHO}")
            self.hud.show("error", "No camera access", f"Enable {WHO} in Settings → Privacy → Camera", 6.0)

    @objc.python_method
    def _build_menu(self):
        self.status = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        self._set_status_icon(False)
        menu = NSMenu.alloc().init()
        self.state_item = self._item(menu, "Loading model…", None, icon="hourglass")
        key_name = self.opts.key.replace("_", " ").title()
        self._item(menu, f"Hold {key_name} to dictate, double-tap for hands-free", None, icon="keyboard")
        self._cleanup_menu(menu)
        n = len(self.cleaner.personal.phrases)
        self._item(menu, f"Personalised from {n:,} of your phrases" if n else
                   "Not personalised yet: run lipflow import-wispr", None, icon="person.text.rectangle")
        menu.addItem_(NSMenuItem.separatorItem())
        self._camera_menu(menu)
        self.last_item = self._item(menu, "Copy last dictation", "copyLast:", icon="doc.on.clipboard")
        self._item(menu, "Open history", "openHistory:", icon="clock.arrow.circlepath")
        self._item(menu, "Edit custom words…", "openWords:", icon="character.book.closed")
        self._item(menu, "Practice & train more…", "trainMore:", icon="person.crop.square")
        self._item(menu, "Settings…", "openSettings:", ",", icon="gearshape")
        menu.addItem_(NSMenuItem.separatorItem())
        self._item(menu, "Quit Lipflow", "quit:", "q", icon="power")
        self.status.setMenu_(menu)

    @objc.python_method
    def _camera_menu(self, menu):
        from .camera import list_cameras, resolve_camera
        parent = self._item(menu, "Camera", None, icon="camera")
        parent.setEnabled_(True)
        sub = NSMenu.alloc().init()
        self.cam_items = []
        try:
            cams = list_cameras()
        except Exception:
            cams = []
        current = resolve_camera(self.camera.index) if not isinstance(self.camera.index, str) or \
            not os.path.exists(self.camera.index) else None
        for c in cams:
            it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(c["name"], "pickCamera:", "")
            it.setTarget_(self)
            it.setRepresentedObject_(c["id"])
            it.setState_(1 if c["index"] == current else 0)
            sub.addItem_(it)
            self.cam_items.append(it)
        parent.setSubmenu_(sub)

    @objc.python_method
    def _cleanup_menu(self, menu):
        self.cleanup_item = self._item(menu, f"Cleanup: {self.cleaner.describe()}", None, icon="text.badge.checkmark")
        self.cleanup_item.setEnabled_(True)
        sub = NSMenu.alloc().init()
        sub.setAutoenablesItems_(False)  # so items that aren't set up stay greyed out
        self.cleanup_items = []
        for name, label in CHOICES.items():
            why = unavailable(name)
            it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(label + (f" ({why})" if why else ""),
                                                                          "pickCleanup:", "")
            it.setTarget_(self)
            it.setRepresentedObject_(name)
            it.setEnabled_(why is None)
            it.setState_(1 if name == self.cleanup_choice else 0)
            sub.addItem_(it)
            self.cleanup_items.append(it)
        self.cleanup_item.setSubmenu_(sub)

    def pickCleanup_(self, sender):
        name = sender.representedObject()
        self.cleanup_choice = self.settings["cleanup"] = name
        save_settings(self.settings)
        for it in self.cleanup_items:
            it.setState_(1 if it.representedObject() == name else 0)
        self.jobs.put(("cleanup", name))  # on the model thread, between dictations

    def pickCamera_(self, sender):
        cam_id = sender.representedObject()
        self.settings["camera"] = cam_id
        save_settings(self.settings)
        self.camera.set_source(cam_id)
        for it in self.cam_items:
            it.setState_(1 if it.representedObject() == cam_id else 0)
        print(f"[lipflow] camera: {sender.title()}")

    @objc.python_method
    def _set_status_icon(self, listening: bool):
        img = symbol("mouth.fill" if listening else "mouth", 15)
        img.setTemplate_(True)  # follows light/dark menu bar
        self.status.button().setImage_(img)

    @objc.python_method
    def _item(self, menu, title, action, key="", icon=None):
        it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, key)
        if icon:
            it.setImage_(symbol(icon, 13, NSFontWeightRegular))
        if action:
            it.setTarget_(self)
        else:
            it.setEnabled_(False)
        menu.addItem_(it)
        return it

    # -- menu actions --------------------------------------------------------------
    def copyLast_(self, sender):
        if self.last_output:
            copy_text(self.last_output)

    def openHistory_(self, sender):
        os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
        open(HISTORY, "a").close()
        os.system(f'open -t "{HISTORY}"')

    def openSetup_(self, sender):
        self.show_setup()

    def trainMore_(self, sender):
        self.show_setup(start_at="practice")

    def openSettings_(self, sender):
        from .settings_window import Settings
        if getattr(self, "settings_win", None) is None:
            self.settings_win = Settings.alloc().initWithApp_(self)
        self.settings_win.show()

    @objc.python_method
    def show_setup(self, start_at: str = "welcome"):
        from .onboarding import Onboarding
        if self.setup is None:
            self.setup = Onboarding.alloc().initWithApp_(self)
        self.camera.track_always = True
        self.setup.show()
        if start_at == "practice":
            self.setup.goPractice_(None)

    def openWords_(self, sender):
        from . import vocab
        vocab.load()
        os.system(f'open -t "{vocab.PATH}"')

    def quit_(self, sender):
        self.camera.close()
        NSApplication.sharedApplication().terminate_(None)

    # -- hotkey callbacks (main thread) -------------------------------------------
    @objc.python_method
    def on_start(self, hands_free: bool):
        if self.loading:
            self.hud.show("error", "Still loading", "The model is almost ready…", hide_after=1.5)
            return
        if self.pending_stop is not None:  # pressed again during the tail: finish the last one now
            self._finish_stop(self.pending_stop)
        if hands_free and self.camera.recording is not None:
            self.hands_free = True  # the second tap of a double-tap: keep the recording going
            self.hud.show("listening", "Hands-free · tap to finish", self.hud.body.stringValue())
            return
        self.session += 1
        self.hands_free = hands_free
        from .context import Context, capture
        # the app you're typing into is frontmost right now
        self.ctx = capture() if self.settings.get("use_context", True) else Context()
        rec = self.camera.start_recording()
        if self.whisper_on:
            self.mic.start()
        self._set_status_icon(True)
        title = "Hands-free · tap to finish" if hands_free else "Listening"
        self.hud.show("listening", title, "" if self.camera.ready.is_set() else "Starting camera…")
        threading.Thread(target=self._preview_loop, args=(self.session, rec), daemon=True).start()

    @objc.python_method
    def on_stop(self):
        self.hands_free = False
        if self.camera.recording is None:
            return
        self.session += 1
        self.pending_stop = self.session
        self._set_status_icon(False)
        self.hud.show("reading", "Reading your lips", self.hud.body.stringValue())
        AppHelper.callLater(TAIL_SECONDS, self._finish_stop, self.session)

    @objc.python_method
    def _finish_stop(self, token):
        if self.pending_stop != token:
            return
        self.pending_stop = None
        rec = self.camera.stop_recording()
        audio = self.mic.stop() if self.whisper_on else []
        if rec is not None:
            rec.audio = audio
            self.jobs.put(("final", rec))

    @objc.python_method
    def on_cancel(self, silent: bool = False):
        self.pending_stop = None
        self.mic.stop()
        self._set_status_icon(False)
        self.session += 1
        self.hands_free = False
        self.camera.stop_recording()
        if silent:
            self.hud.hide()
        else:
            self.hud.show("error", "Cancelled", "", hide_after=0.8)

    # -- camera thread -------------------------------------------------------------
    @objc.python_method
    def onFrame(self, frame, obs, recording):
        """Camera thread. Video previews are *coalesced*: at most one frame update waits on the main
        thread at a time. Queuing every frame let the backlog delay the push-to-talk key handling
        (measured: 8 s practice clips came out 0.5 s long)."""
        rec = self.camera.recording
        if recording and rec is not None and rec.duration > MAX_SECONDS:
            ui(self.on_stop)
            return
        if self._ui_busy or (not recording and self.onboarding is None):
            return
        setup = mouth_view(frame, obs, 208, 130) if self.onboarding is not None else None
        pill = mouth_view(frame, obs) if recording else None
        level = obs.mouth_open * 2.6 if obs else 0.0
        self._ui_busy = True
        ui(self._show_frame, setup, pill, level)

    @objc.python_method
    def _show_frame(self, setup, pill, level):
        try:
            if setup is not None and self.onboarding is not None:
                self.onboarding.set_frame(setup)
            if pill is not None:
                self.hud.set_frame(pill, level)
        finally:
            self._ui_busy = False

    # -- model thread --------------------------------------------------------------
    @objc.python_method
    def _preview_loop(self, session: int, rec: Recording):
        if not self.opts.live_preview:
            return
        waited = 0.0
        while self.session == session:
            time.sleep(PREVIEW_EVERY)
            waited += PREVIEW_EVERY
            if not rec.ts and (self.camera.error or waited > 4):
                msg = self.camera.error or "The camera isn't sending frames"
                print(f"[lipflow] camera problem: {msg}")
                ui(self.hud.show, "error", "Camera problem", msg[:90], 5.0)
                return
            if self.session != session or self.preview_busy or len(rec.ts) < 15:
                continue
            self.preview_busy = True
            self.jobs.put(("preview", session, rec))

    @objc.python_method
    def _worker(self):
        while True:
            job = self.jobs.get()
            try:
                if job[0] == "load":
                    self._load()
                elif job[0] == "preview":
                    self._preview(*job[1:])
                elif job[0] == "final":
                    self._final(job[1])
                elif job[0] == "train":
                    self._train(job[1])
                elif job[0] == "whisper":
                    self._load_whisper()
                elif job[0] == "cleanup":
                    self.cleaner = Cleaner(job[1])
                    self._warm_cleanup()
                    print(f"[lipflow] cleanup: {self.cleaner.describe()}")
                    ui(self.cleanup_item.setTitle_, f"Cleanup: {self.cleaner.describe()}")
                elif job[0] == "reload":  # e.g. face model reset from Settings
                    self.reader = LipReader(beam_size=self.opts.beam)
                    self.reader.warmup()
            except Exception as e:
                import traceback
                traceback.print_exc()
                ui(self.hud.show, "error", "Something went wrong", str(e)[:80], 3.0)
            finally:
                if job[0] == "preview":
                    self.preview_busy = False

    @objc.python_method
    def _warm_cleanup(self):
        if self.cleaner.backend != "local":
            return
        try:
            self.cleaner.warmup()
        except Exception as e:
            print(f"[lipflow] local cleanup model unavailable ({e}); using basic cleanup")
            self.cleaner.backend, self.cleaner.model = "basic", None

    @objc.python_method
    def _load(self):
        t = time.time()
        self.reader = LipReader(beam_size=self.opts.beam)
        self.reader.warmup()
        if self.cleaner.backend == "local":
            ui(self.hud.set_text, "Loading the text-cleanup model…")
        self._warm_cleanup()
        ui(self.cleanup_item.setTitle_, f"Cleanup: {self.cleaner.describe()}")
        self.loading = False
        print(f"[lipflow] model ready in {time.time() - t:.1f}s "
              f"(encoder on {self.reader.enc_device}, cleanup: {self.cleaner.describe()})")
        name = self.opts.key.replace("_", " ").title()
        ui(self.state_item.setTitle_, "Ready")
        ui(self.state_item.setImage_, symbol("checkmark.circle", 13, NSFontWeightRegular))
        if self.settings.get("whisper"):
            self.jobs.put(("whisper",))
        if self.opts.onboard or not self.settings.get("onboarded"):
            ui(self.show_setup)
        else:
            ui(self.hud.show, "done", "Lipflow is ready", f"Hold {name} and mouth your words", 2.5)

    @property
    def whisper_on(self) -> bool:
        return bool(self.settings.get("whisper")) and self.av_reader is not None and self.onboarding is None

    @objc.python_method
    def _load_whisper(self):
        """Model thread: download (first time, 1.8 GB) and load the audio-visual model."""
        from . import av
        if not av.available():
            ui(self.hud.show, "reading", "Whisper mode", "Downloading the audio-visual model (1.8 GB)…")
            try:
                av.download(lambda pct: ui(self.hud.set_text, f"Downloading the audio-visual model… {pct:.0f}%"))
            except Exception as e:
                ui(self.hud.show, "error", "Whisper mode", f"Download failed: {e}"[:80], 5.0)
                return
        ui(self.hud.show, "reading", "Whisper mode", "Loading…")
        self.av_reader = av.AVReader(beam_size=self.opts.beam)
        self.av_reader.warmup_av()
        print("[lipflow] whisper mode ready (lips + audio)")
        ui(self.hud.show, "done", "Whisper mode on", "Whisper or speak softly while you mouth the words", 3.0)

    @objc.python_method
    def _av_candidates(self, rec, rois):
        """Lips + audio when whisper mode has audio for this clip, else None."""
        from .mic import segment
        if not self.whisper_on or not getattr(rec, "audio", None):
            return None
        ts, _, _ = rec.snapshot()
        wave = segment(rec.audio, ts[0], rois.shape[0])
        if wave is None:
            return None
        return self.av_reader.beam_search(self.av_reader.encode_av(rois, wave), nbest=5)

    @objc.python_method
    def _rois(self, rec: Recording):
        return rois_for(rec)

    @objc.python_method
    def _preview(self, session: int, rec: Recording):
        if self.session != session:
            return
        rois = self._rois(rec)
        if rois is None:
            ui(self.hud.set_text, "Can't see your face…")
            return
        text = self.reader.greedy(self.reader.encode(rois))
        if self.session == session and text:
            ui(self.hud.set_text, text.lower())
    @objc.python_method
    def _final(self, rec: Recording):
        t0 = time.time()
        problem = clip_problem(rec)
        if problem and self.onboarding is not None:
            print(f"[lipflow] practice clip rejected ({rec.duration:.1f}s, {len(rec.ts)} frames, face in "
                  f"{rec.face_ratio:.0%}): {problem[0]}")
            ui(self.onboarding.clip_done, False, f"{problem[0]}. {problem[1]}.")
            ui(self.hud.hide)
            return
        if problem:
            print(f"[lipflow] skipped {rec.duration:.1f}s clip ({len(rec.ts)} frames, face in "
                  f"{rec.face_ratio:.0%}): {problem[0]}")
            ui(self.hud.show, "error", problem[0], problem[1], 2.2)
            return
        rois = self._rois(rec)
        enc = self.reader.encode(rois)
        t_enc = time.time() - t0
        if self.onboarding is not None:  # practice clip: keep it with its known text, don't paste
            raw = self.reader.greedy(enc)
            print(f"[lipflow] practice clip saved ({rec.duration:.1f}s): {raw!r}")
            ui(self.onboarding.clip_done, True, "", rois, self.onboarding_text, raw)
            ui(self.hud.hide)
            return
        # Whisper mode reads empty when there's no audible whisper (silent mouthing): fall back to lips
        candidates = self._av_candidates(rec, rois)
        if not candidates or not candidates[0]:
            if candidates is not None:
                print("[lipflow] lips + audio read nothing, using lips only")
            candidates = self.reader.beam_search(enc, nbest=5)
        t_beam = time.time() - t0 - t_enc
        if not candidates or not candidates[0]:
            print(f"[lipflow] {rec.duration:.1f}s clip: nothing read")
            ui(self.hud.show, "error", "Couldn't read that", "Try again, a little slower", 2.2)
            return
        ui(self.hud.set_text, candidates[0].lower())
        ctx = getattr(self, "ctx", None)
        text = self.cleaner(candidates, context=" ".join(self.context[-3:]), names=ctx.names if ctx else None)
        t_all = time.time() - t0
        if ctx and ctx.names:
            print(f"[lipflow] context: {ctx.app}, {len(ctx.names)} names")
        print(f"[lipflow] {rec.duration:.1f}s clip → raw: {candidates[0]!r}\n"
              f"          → typed: {text!r}  (encode {t_enc:.2f}s, beam {t_beam:.2f}s, total {t_all:.2f}s)")
        if not text:
            ui(self.hud.show, "error", "Couldn't read that", "Try again, a little slower", 2.2)
            return
        out = text
        if self.last_paste_at and time.time() - self.last_paste_at < JOIN_WINDOW:
            out = " " + text
        self.last_output = text
        self.last_paste_at = time.time()
        self.context.append(text)
        self._log(rec, candidates, text, t_all)
        self._save_clip(rois, candidates, text)
        if self.opts.paste:
            ui(paste_text, out)
            if ctx is not None and ctx.element is not None and self.settings.get("learn_corrections", True):
                from .corrections import Watcher
                ui(lambda: Watcher(ctx.element, ctx.value, out, rois, candidates).start())
        else:
            ui(copy_text, text)
        ui(self.hud.show, "done", "Pasted" if self.opts.paste else "Copied", text, 2.4)

    @objc.python_method
    def _save_clip(self, rois, candidates, text):
        keep_clip(rois, candidates, text, self.settings)

    @objc.python_method
    def _train(self, ob):
        """Onboarding: personal LM (if phrases) then face adaptation with a held-out check."""
        self.loading = True
        r = train_on_face(self.opts.beam, ob.report)
        if r["after"] is None:
            self.loading = False
            ob.finished(0, None, False, r["note"])
            return
        self.reader = LipReader(beam_size=self.opts.beam)
        self.reader.warmup()
        self.loading = False
        self.settings["training"] = {"before": r["before"], "after": r["after"], "kept": r["kept"],
                                     "clips": r["clips"], "at": time.time()}
        save_settings(self.settings)
        ob.finished(r["before"], r["after"], r["kept"], r["note"])

    @objc.python_method
    def _log(self, rec, candidates, text, secs):
        log_history(rec, candidates, text, secs, self.cleaner.describe())


class AppDelegate(NSObject):
    """Opening Lipflow.app while it's already running brings up Settings (or setup, first time),
    since the menu-bar icon can be hidden behind the notch when the menu bar is full."""

    def applicationShouldHandleReopen_hasVisibleWindows_(self, app, visible):
        if self.lf is not None and not self.lf.loading:
            if self.lf.settings.get("onboarded"):
                self.lf.openSettings_(None)
            else:
                self.lf.show_setup()
        return False


def run(opts: Options):
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    lf = Lipflow.alloc().initWithOptions_(opts)
    delegate = AppDelegate.alloc().init()
    delegate.lf = lf
    app.setDelegate_(delegate)
    lf._delegate = delegate
    lf.start()
    print(f"[lipflow] hold {opts.key.replace('_', ' ')} and mouth your words · double-tap for hands-free · "
          f"Esc cancels · Ctrl-C quits")
    AppHelper.runEventLoop(installInterrupt=True)
