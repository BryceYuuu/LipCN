"""The floating pill at the bottom of the screen, à la Wispr Flow.

States: listening (mouth preview + lip-motion bars + live words), reading (spinner),
done (the text that was typed, fades out), error/hint (fades out).
All methods must be called on the main thread (use AppHelper.callAfter).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import objc
import Quartz
from AppKit import (
    NSBackingStoreBuffered, NSBezierPath, NSBitmapImageRep, NSColor, NSFont, NSImage, NSImageView,
    NSMakeRect, NSPanel, NSScreen, NSTextField, NSView, NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary, NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel, NSLineBreakByTruncatingHead, NSStatusWindowLevel,
    NSAnimationContext, NSTimer, NSImageScaleProportionallyUpOrDown, NSTextAlignmentCenter, NSLineBreakByTruncatingTail,
)
from Foundation import NSObject

W, H = 440, 64
VW, VH = 240, 150  # live mouth video above the pill
THUMB = 48
ACCENT = (0.98, 0.36, 0.45)  # lip pink


def _rgb(r, g, b, a=1.0):
    return NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, a)


def _nsimage(bgr: np.ndarray) -> NSImage:
    ok, buf = cv2.imencode(".png", bgr)
    from Foundation import NSData
    data = NSData.dataWithBytes_length_(buf.tobytes(), len(buf))
    rep = NSBitmapImageRep.imageRepWithData_(data)
    img = NSImage.alloc().initWithSize_(rep.size())
    img.addRepresentation_(rep)
    return img


class PillView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(PillView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.levels = [0.0] * 14
        self.mode = "idle"
        self.phase = 0.0
        return self

    def isFlipped(self):
        return False

    def drawRect_(self, rect):
        b = self.bounds()
        path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(b, H / 2, H / 2)
        _rgb(0.07, 0.07, 0.09, 0.92).setFill()
        path.fill()
        _rgb(1, 1, 1, 0.10).setStroke()
        path.setLineWidth_(1.0)
        path.stroke()
        if self.mode not in ("listening", "reading"):
            return
        # Lip-motion bars on the right edge
        x0 = b.size.width - 24 - len(self.levels) * 5
        mid = b.size.height / 2
        for i, lv in enumerate(self.levels):
            if self.mode == "reading":
                lv = 0.25 + 0.25 * math.sin(self.phase * 2.2 - i * 0.55)
            h = 4 + lv * 30
            color = _rgb(*ACCENT, 0.95) if self.mode == "listening" else _rgb(1, 1, 1, 0.55)
            color.setFill()
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(x0 + i * 5, mid - h / 2, 3, h), 1.5, 1.5).fill()


class HUD(NSObject):
    def init(self):
        self = objc.super(HUD, self).init()
        if self is None:
            return None
        screen = NSScreen.mainScreen().visibleFrame()
        x = screen.origin.x + (screen.size.width - W) / 2
        y = screen.origin.y + 28
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, W, H), NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered, False)
        p = self.panel
        p.setLevel_(NSStatusWindowLevel)
        p.setOpaque_(False)
        p.setBackgroundColor_(NSColor.clearColor())
        p.setHasShadow_(True)
        p.setIgnoresMouseEvents_(True)
        p.setHidesOnDeactivate_(False)
        p.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorFullScreenAuxiliary)

        # Live mouth video, floating just above the pill
        self.video_panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x + (W - VW) / 2, y + H + 10, VW, VH),
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel, NSBackingStoreBuffered, False)
        vp = self.video_panel
        vp.setLevel_(NSStatusWindowLevel)
        vp.setOpaque_(False)
        vp.setBackgroundColor_(NSColor.clearColor())
        vp.setHasShadow_(True)
        vp.setIgnoresMouseEvents_(True)
        vp.setHidesOnDeactivate_(False)
        vp.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorFullScreenAuxiliary)
        self.video = NSImageView.alloc().initWithFrame_(NSMakeRect(0, 0, VW, VH))
        self.video.setImageScaling_(NSImageScaleProportionallyUpOrDown)
        self.video.setWantsLayer_(True)
        self.video.layer().setCornerRadius_(18)
        self.video.layer().setMasksToBounds_(True)
        self.video.layer().setBorderWidth_(1.0)
        self.video.layer().setBorderColor_(Quartz.CGColorCreateSRGB(1, 1, 1, 0.15))
        self.video.layer().setBackgroundColor_(Quartz.CGColorCreateSRGB(0.07, 0.07, 0.09, 0.92))
        vp.setContentView_(self.video)

        self.view = PillView.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))
        p.setContentView_(self.view)

        self.thumb = NSImageView.alloc().initWithFrame_(NSMakeRect(8, (H - THUMB) / 2, THUMB, THUMB))
        self.thumb.setImageScaling_(NSImageScaleProportionallyUpOrDown)
        self.thumb.setWantsLayer_(True)
        self.thumb.layer().setCornerRadius_(THUMB / 2)
        self.thumb.layer().setMasksToBounds_(True)
        self.thumb.layer().setBackgroundColor_(Quartz.CGColorCreateSRGB(1, 1, 1, 0.08))
        self.view.addSubview_(self.thumb)

        self.glyph = NSTextField.alloc().initWithFrame_(NSMakeRect(8, (H - THUMB) / 2 + 11, THUMB, 26))
        for f in (self.glyph,):
            f.setBezeled_(False); f.setDrawsBackground_(False); f.setEditable_(False); f.setSelectable_(False)
            f.setAlignment_(NSTextAlignmentCenter)
            f.setFont_(NSFont.systemFontOfSize_(20))
            f.setTextColor_(_rgb(1, 1, 1, 0.9))
        self.view.addSubview_(self.glyph)

        text_x = 8 + THUMB + 12
        text_w = W - text_x - 24 - 14 * 5 - 10
        self.title = self._label(NSMakeRect(text_x, H / 2 + 1, text_w, 18), 11, 0.55, bold=True)
        self.body = self._label(NSMakeRect(text_x, H / 2 - 21, text_w, 20), 14, 0.95)
        self.body.cell().setLineBreakMode_(NSLineBreakByTruncatingHead)
        self._hide_timer = None
        self._anim_timer = None
        return self

    @objc.python_method
    def _label(self, frame, size, alpha, bold=False):
        t = NSTextField.alloc().initWithFrame_(frame)
        t.setBezeled_(False)
        t.setDrawsBackground_(False)
        t.setEditable_(False)
        t.setSelectable_(False)
        t.setTextColor_(_rgb(1, 1, 1, alpha))
        t.setFont_(NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size))
        self.view.addSubview_(t)
        return t

    # -- public (main thread) ------------------------------------------------------
    @objc.python_method
    def show(self, mode: str, title: str, body: str = "", hide_after: float | None = None):
        self._cancel_hide()
        self.view.mode = mode
        self.title.setStringValue_(title.upper())
        self.body.setStringValue_(body)
        self.body.cell().setLineBreakMode_(NSLineBreakByTruncatingHead if mode in ("listening", "reading")
                                           else NSLineBreakByTruncatingTail)
        bars = mode in ("listening", "reading")
        f = self.body.frame()
        full = W - f.origin.x - 22
        self.body.setFrameSize_((full - (14 * 5 + 12 if bars else 0), f.size.height))
        self.title.setFrameSize_((full, self.title.frame().size.height))
        self.glyph.setStringValue_({"done": "✓", "error": "!", "reading": "👄", "listening": "●"}.get(mode, ""))
        self.glyph.setTextColor_(_rgb(0.45, 0.9, 0.6) if mode == "done" else _rgb(*ACCENT) if mode in ("error", "listening")
                                 else _rgb(1, 1, 1, 0.9))
        self.thumb.setImage_(None)
        if mode == "listening":
            self.video_panel.setAlphaValue_(1.0)
            self.video_panel.orderFrontRegardless()
        else:
            self.video_panel.orderOut_(None)
            self.video.setImage_(None)
        if mode != "listening":
            self.view.levels = [0.0] * len(self.view.levels)
        if mode == "reading":
            self._start_anim()
        else:
            self._stop_anim()
        self.view.setNeedsDisplay_(True)
        self.panel.setAlphaValue_(1.0)
        self.panel.orderFrontRegardless()
        if hide_after:
            self._hide_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                hide_after, self, "fadeOut:", None, False)

    @objc.python_method
    def set_text(self, body: str):
        self.body.setStringValue_(body)

    @objc.python_method
    def set_frame(self, video_bgr, level: float):
        if video_bgr is not None and self.view.mode == "listening":
            self.video.setImage_(_nsimage(video_bgr))
        lv = self.view.levels
        lv.pop(0)
        lv.append(float(max(0.0, min(1.0, level))))
        self.view.setNeedsDisplay_(True)

    @objc.python_method
    def hide(self):
        self._cancel_hide()
        self._stop_anim()
        self.panel.orderOut_(None)
        self.video_panel.orderOut_(None)
        self.video.setImage_(None)

    # -- internals -----------------------------------------------------------------
    def fadeOut_(self, timer):
        self._hide_timer = None
        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(0.35)
        self.panel.animator().setAlphaValue_(0.0)
        self.video_panel.animator().setAlphaValue_(0.0)
        NSAnimationContext.endGrouping()
        self._hide_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.4, self, "orderOut:", None, False)

    def orderOut_(self, timer):
        self._hide_timer = None
        self.hide()

    def tick_(self, timer):
        self.view.phase += 0.25
        self.view.setNeedsDisplay_(True)

    @objc.python_method
    def _start_anim(self):
        if self._anim_timer is None:
            self._anim_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                1 / 30, self, "tick:", None, True)

    @objc.python_method
    def _stop_anim(self):
        if self._anim_timer is not None:
            self._anim_timer.invalidate()
            self._anim_timer = None

    @objc.python_method
    def _cancel_hide(self):
        if self._hide_timer is not None:
            self._hide_timer.invalidate()
            self._hide_timer = None
