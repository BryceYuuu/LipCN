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
from .cleanup import Cleaner
from .face import mouth_rois
from .hotkey import KEYS, PushToTalk
from .hud import HUD, symbol
from AppKit import NSFontWeightRegular
from .paste import copy_text, paste_text
from .vsr import LipReader

from .paths import HOME

HISTORY = os.path.join(HOME, "history.jsonl")
SETTINGS = os.path.join(HOME, "settings.json")


def load_settings() -> dict:
    try:
        return json.load(open(SETTINGS))
    except (OSError, ValueError):
        return {}


def save_settings(d: dict):
    os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
    json.dump(d, open(SETTINGS, "w"), indent=2)
MIN_SECONDS = 0.6
MAX_SECONDS = 60.0
PREVIEW_EVERY = 0.45
TAIL_SECONDS = 0.4  # keep filming after release: the last word needs the frames after it
JOIN_WINDOW = 45.0  # dictations this close together get a separating space


@dataclass
class Options:
    key: str = "right_option"
    beam: int = 10
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
        self.cleaner = Cleaner(opts.backend)
        self.jobs: "queue.Queue" = queue.Queue()
        self.session = 0          # bumps on every start/cancel so stale previews are dropped
        self.preview_busy = False
        self.last_output = ""
        self.last_paste_at = 0.0
        self.context: list[str] = []
        self.hands_free = False
        self.pending_stop = None
        self._ui_busy = False        # a video frame is waiting to be drawn on the main thread
        self.onboarding = None       # the setup window while it's collecting practice clips
        self.onboarding_text = ""
        self.setup = None
        self.loading = True
        self.settings = load_settings()
        cam = opts.camera if opts.camera != "auto" else self.settings.get("camera", "auto")
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
            self.hud.show("error", "Needs Input Monitoring", "Allow your terminal, then restart lipflow")
        if self.opts.paste and not Quartz.CGPreflightPostEventAccess():
            Quartz.CGRequestPostEventAccess()
            print("[lipflow] Allow your terminal under Privacy & Security → Accessibility so Lipflow can paste.")
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
            print("[lipflow] camera access is denied: System Settings → Privacy & Security → Camera → enable your terminal")
            self.hud.show("error", "No camera access", "Enable your terminal in Settings → Privacy → Camera", 6.0)

    @objc.python_method
    def _build_menu(self):
        self.status = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        self._set_status_icon(False)
        menu = NSMenu.alloc().init()
        self.state_item = self._item(menu, "Loading model…", None, icon="hourglass")
        key_name = self.opts.key.replace("_", " ").title()
        self._item(menu, f"Hold {key_name} to dictate, double-tap for hands-free", None, icon="keyboard")
        self._item(menu, f"Cleanup: {self.cleaner.describe()}", None, icon="text.badge.checkmark")
        n = len(self.cleaner.personal.phrases)
        self._item(menu, f"Personalised from {n:,} of your phrases" if n else
                   "Not personalised yet: run lipflow import-wispr", None, icon="person.text.rectangle")
        menu.addItem_(NSMenuItem.separatorItem())
        self._camera_menu(menu)
        self.last_item = self._item(menu, "Copy last dictation", "copyLast:", icon="doc.on.clipboard")
        self._item(menu, "Open history", "openHistory:", icon="clock.arrow.circlepath")
        self._item(menu, "Edit custom words…", "openWords:", icon="character.book.closed")
        self._item(menu, "Set up / train on my face…", "openSetup:", icon="person.crop.square")
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

    @objc.python_method
    def show_setup(self):
        from .onboarding import Onboarding
        if self.setup is None:
            self.setup = Onboarding.alloc().initWithApp_(self)
        self.camera.track_always = True
        self.setup.show()

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
        rec = self.camera.start_recording()
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
        if rec is not None:
            self.jobs.put(("final", rec))

    @objc.python_method
    def on_cancel(self, silent: bool = False):
        self.pending_stop = None
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
            except Exception as e:
                import traceback
                traceback.print_exc()
                ui(self.hud.show, "error", "Something went wrong", str(e)[:80], 3.0)
            finally:
                if job[0] == "preview":
                    self.preview_busy = False

    @objc.python_method
    def _load(self):
        t = time.time()
        self.reader = LipReader(beam_size=self.opts.beam)
        self.reader.warmup()
        if self.cleaner.backend == "local":
            ui(self.hud.set_text, "Loading the text-cleanup model…")
            try:
                self.cleaner.warmup()
            except Exception as e:
                print(f"[lipflow] local cleanup model unavailable ({e}); using basic cleanup")
                self.cleaner.backend, self.cleaner.model = "basic", None
        self.loading = False
        print(f"[lipflow] model ready in {time.time() - t:.1f}s "
              f"(encoder on {self.reader.enc_device}, cleanup: {self.cleaner.describe()})")
        name = self.opts.key.replace("_", " ").title()
        ui(self.state_item.setTitle_, "Ready")
        ui(self.state_item.setImage_, symbol("checkmark.circle", 13, NSFontWeightRegular))
        if self.opts.onboard or not self.settings.get("onboarded"):
            ui(self.show_setup)
        else:
            ui(self.hud.show, "done", "Lipflow is ready", f"Hold {name} and mouth your words", 2.5)

    @objc.python_method
    def _rois(self, rec: Recording):
        ts, grays, anchors = rec.snapshot()
        idx = LipReader.resample(ts, len(ts))
        return mouth_rois([grays[i] for i in idx], [anchors[i] for i in idx])

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
        problem = None
        if rec.duration < MIN_SECONDS or len(rec.ts) < 12:
            problem = ("Too short", "Hold the key while you mouth the words")
        elif rec.face_ratio < 0.4:
            problem = ("Can't see your face", "Face the camera with your mouth in view")
        elif np.std([m for m in rec.mouth_open if m > 0] or [0]) < 0.012:
            problem = ("No lip movement", "Mouth the words clearly — no sound needed")
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
        candidates = self.reader.beam_search(enc, nbest=5)
        t_beam = time.time() - t0 - t_enc
        if not candidates or not candidates[0]:
            ui(self.hud.show, "error", "Couldn't read that", "Try again, a little slower", 2.2)
            return
        ui(self.hud.set_text, candidates[0].lower())
        text = self.cleaner(candidates, context=" ".join(self.context[-3:]))
        t_all = time.time() - t0
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
        else:
            ui(copy_text, text)
        ui(self.hud.show, "done", "Pasted" if self.opts.paste else "Copied", text, 2.4)

    @objc.python_method
    def _save_clip(self, rois, candidates, text):
        """Keep the mouth crops of each dictation (local only) so accuracy changes can be measured on
        your real clips. Turn off with "save_clips": false in settings.json."""
        if not self.settings.get("save_clips", True):
            return
        d = os.path.join(os.path.dirname(HISTORY), "clips", "dictations")
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, f"{int(time.time() * 1000)}.npz"), rois=rois,
                            raw=np.array(candidates), text=text)

    @objc.python_method
    def _train(self, ob):
        """Onboarding: personal LM (if phrases) then face adaptation with a held-out check."""
        import random as _r
        from .bench import wer
        from .onboarding import N_HELD_OUT, saved_clips
        from .personal import PHRASES
        from .train_vsr import finetune, save
        self.loading = True
        note = ""
        if os.path.exists(PHRASES):
            ob.report(3, "Learning how you talk from your phrases…")
            from .train_lm import train as train_lm
            try:
                r = train_lm(epochs=3)
                note = (f"Your phrasing: {r['before']['yours']:.0f} → {r['after']['yours']:.0f} perplexity. "
                        if r["saved"] else "")
            except Exception as e:
                print(f"[lipflow] train-lm failed: {e}")
        clips = saved_clips()
        if len(clips) < N_HELD_OUT + 6:
            ob.finished(0, None, False, "Not enough practice clips to train on. Run setup again from the menu.")
            self.loading = False
            return
        _r.Random(1).shuffle(clips)
        test, train = clips[:N_HELD_OUT], clips[N_HELD_OUT:]
        ob.report(15, f"Measuring the standard model on {len(test)} of your sentences…")
        base = LipReader(beam_size=self.opts.beam, personal=False)

        def score(reader):
            e = n = 0
            for c in test:
                a, b = wer(reader.beam_search(reader.encode(c["rois"])), c["text"])
                e, n = e + a, n + b
            return e / max(n, 1)

        before = score(base)
        ob.report(25, f"Training on {len(train)} of your clips…")
        finetune(train, reader=base, log=lambda *a: None,
                 on_epoch=lambda k, n: ob.report(25 + 65 * k / n, f"Training on your face: pass {k} of {n}"))
        ob.report(92, "Checking it on the sentences it didn't see…")
        after = score(base)
        kept = after < before
        if kept:
            save(base)
        del base
        self.reader = LipReader(beam_size=self.opts.beam)
        self.reader.warmup()
        self.loading = False
        print(f"[lipflow] onboarding: held-out WER {before:.1%} → {after:.1%} ({'kept' if kept else 'discarded'})")
        ob.finished(before, after, kept, note)

    @objc.python_method
    def _log(self, rec, candidates, text, secs):
        os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
        with open(HISTORY, "a") as f:
            f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(rec.duration, 2),
                                "raw": candidates, "text": text, "latency": round(secs, 2),
                                "cleanup": self.cleaner.describe()}) + "\n")


def run(opts: Options):
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    lf = Lipflow.alloc().initWithOptions_(opts)
    lf.start()
    print(f"[lipflow] hold {opts.key.replace('_', ' ')} and mouth your words · double-tap for hands-free · "
          f"Esc cancels · Ctrl-C quits")
    AppHelper.runEventLoop(installInterrupt=True)
