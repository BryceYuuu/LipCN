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
from .hud import HUD
from .paste import backspace, copy_text, paste_text, type_text
from .stream import Broadcaster, serve
from .vsr import LipReader

HISTORY = os.path.expanduser("~/Library/Application Support/Lipflow/history.jsonl")
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
    camera: int = 0
    paste: bool = True
    live_preview: bool = True
    live_type: bool = False    # type stable words into the focused app while you talk
    port: int = 8765           # SSE stream of partial/final text; 0 disables


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
        self.loading = True
        self.camera = Camera(opts.camera, on_frame=self.onFrame)
        self.bus = Broadcaster()
        self.live_typed = ""      # what live typing has put in the focused app this dictation
        self.prev_partial: list[str] = []
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
        if self.opts.port:
            try:
                serve(self.bus, self.opts.port)
                print(f"[lipflow] live stream: http://127.0.0.1:{self.opts.port}/  (SSE at /events)")
            except OSError as e:
                print(f"[lipflow] couldn't start the live stream on port {self.opts.port}: {e}")
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
        self.status.button().setTitle_("👄")
        menu = NSMenu.alloc().init()
        self.state_item = self._item(menu, "Loading model…", None)
        key_name = self.opts.key.replace("_", " ").title()
        self._item(menu, f"Hold {key_name} to dictate · double-tap for hands-free", None)
        self._item(menu, f"Cleanup: {self.cleaner.describe()}", None)
        self.live_item = self._item(menu, "Type while I talk (live)", "toggleLive:")
        self.live_item.setState_(1 if self.opts.live_type else 0)
        if self.opts.port:
            self._item(menu, "Open live stream page", "openStream:")
        menu.addItem_(NSMenuItem.separatorItem())
        self.last_item = self._item(menu, "Copy last dictation", "copyLast:")
        self._item(menu, "Open history", "openHistory:")
        self._item(menu, "Edit custom words (names, terms)…", "openWords:")
        menu.addItem_(NSMenuItem.separatorItem())
        self._item(menu, "Quit Lipflow", "quit:", "q")
        self.status.setMenu_(menu)

    @objc.python_method
    def _item(self, menu, title, action, key=""):
        it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, key)
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

    def toggleLive_(self, sender):
        self.opts.live_type = not self.opts.live_type
        sender.setState_(1 if self.opts.live_type else 0)

    def openStream_(self, sender):
        os.system(f"open http://127.0.0.1:{self.opts.port}/")

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
        self.live_typed, self.prev_partial = "", []
        self.bus.publish("start", id=self.session)
        rec = self.camera.start_recording()
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
        self.bus.publish("cancel", id=self.session)
        if self.live_typed:  # take back what live typing already put in the app
            backspace(len(self.live_typed))
            self.live_typed = ""
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
        if not recording:
            return
        rec = self.camera.recording
        if rec is not None and rec.duration > MAX_SECONDS:
            ui(self.on_stop)
            return
        thumb = mouth_view(frame, obs)
        level = obs.mouth_open * 2.6 if obs else 0.0
        ui(self.hud.set_frame, thumb, level)

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
            self.bus.publish("partial", id=session, text=text.lower())
            if self.opts.live_type and self.opts.paste:
                ui(self._live_type, session, text.lower().split())

    @objc.python_method
    def _live_type(self, session: int, words: list[str]):
        """Type words that two consecutive live reads agree on, never the newest word."""
        if self.session != session:
            return
        prev, self.prev_partial = self.prev_partial, words
        stable = []
        for a, b in zip(prev, words[:-1]):
            if a != b:
                break
            stable.append(a)
        want = " ".join(stable)
        if stable and want.startswith(self.live_typed) and len(want) > len(self.live_typed):
            new = want[len(self.live_typed):]
            if self.last_paste_at and not self.live_typed and time.time() - self.last_paste_at < JOIN_WINDOW:
                new = " " + new
                want = " " + want
            type_text(new)
            self.live_typed += new

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
        if problem:
            ui(self._drop_live)
            print(f"[lipflow] skipped {rec.duration:.1f}s clip ({len(rec.ts)} frames, face in "
                  f"{rec.face_ratio:.0%}): {problem[0]}")
            ui(self.hud.show, "error", problem[0], problem[1], 2.2)
            return
        rois = self._rois(rec)
        enc = self.reader.encode(rois)
        t_enc = time.time() - t0
        candidates = self.reader.beam_search(enc, nbest=3)
        t_beam = time.time() - t0 - t_enc
        if not candidates or not candidates[0]:
            ui(self._drop_live)
            ui(self.hud.show, "error", "Couldn't read that", "Try again, a little slower", 2.2)
            return
        ui(self.hud.set_text, candidates[0].lower())
        text = self.cleaner(candidates, context=" ".join(self.context[-3:]))
        t_all = time.time() - t0
        print(f"[lipflow] {rec.duration:.1f}s clip → raw: {candidates[0]!r}\n"
              f"          → typed: {text!r}  (encode {t_enc:.2f}s, beam {t_beam:.2f}s, total {t_all:.2f}s)")
        if not text:
            ui(self._drop_live)
            ui(self.hud.show, "error", "Couldn't read that", "Try again, a little slower", 2.2)
            return
        out = text
        if self.last_paste_at and time.time() - self.last_paste_at < JOIN_WINDOW:
            out = " " + text
        self.last_output = text
        self.last_paste_at = time.time()
        self.context.append(text)
        self._log(rec, candidates, text, t_all)
        self.bus.publish("final", id=self.session, text=text, raw=candidates, latency=round(t_all, 2))
        if self.opts.paste:
            ui(self._insert_final, out)
        else:
            ui(copy_text, text)
        ui(self.hud.show, "done", "Typed" if self.opts.paste else "Copied", text, 2.4)

    @objc.python_method
    def _drop_live(self):
        if self.live_typed:
            backspace(len(self.live_typed))
            self.live_typed = ""

    @objc.python_method
    def _insert_final(self, out: str):
        typed, self.live_typed = self.live_typed, ""
        if typed:
            # swap the rough live words for the cleaned sentence, keeping any shared start
            common = 0
            while common < min(len(typed), len(out)) and typed[common] == out[common]:
                common += 1
            backspace(len(typed) - common)
            type_text(out[common:])
        else:
            paste_text(out)

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
