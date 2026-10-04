"""Shared decision and focus checks for both front ends."""
from .cleanup import basic_cleanup
from .confidence import Assessment, Quality, assess


def capture_target():
    """Remember the foreground app/window only; never read focused text or a title."""
    import sys
    from .context import Context
    ctx = Context()
    try:
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.WinDLL('user32')
            user32.GetForegroundWindow.restype = wintypes.HWND
            ctx.target = int(user32.GetForegroundWindow() or 0)
        else:
            from AppKit import NSWorkspace
            app = NSWorkspace.sharedWorkspace().frontmostApplication()
            if app is not None:
                ctx.target = int(app.processIdentifier())
    except Exception:
        pass  # No target: a reviewed result will be copied instead of pasted.
    return ctx


def context_for_recording(settings, policy):
    from .context import Context, capture
    if settings.get('use_context', True):
        return capture()
    # English direct paste needs no context access. Review can retain a target
    # identity without asking Accessibility for a focused field or reading text.
    return Context() if policy == 'off' else capture_target()


def quality_for(rec, rois):
    import numpy as np
    pixels = getattr(rec, 'mouth_pixels', [])
    return Quality(rec.face_ratio, float(np.mean(rois)), float(np.std(rois)),
                   float(np.median(pixels)) if pixels else 0.0)


def choose_result(hypotheses, greedy, quality, cleaner, ctx, context, policy='off', min_margin=0.5):
    language = getattr(cleaner, 'language', 'en')
    if getattr(cleaner, 'mode', 'faithful') == 'polish':
        policy = 'review'
    if language == 'en' and policy == 'off':
        # Keep the original English flow: common clip checks happen upstream,
        # then cleanup is pasted directly, with no new quality or margin gate.
        decision = (Assessment('auto', 'Direct paste (English default)', None, False)
                    if hypotheses and hypotheses[0].text.strip() else
                    Assessment('retry', "Couldn't read that; try again a little slower", None, False))
    else:
        decision = assess(hypotheses, greedy, quality, policy, min_margin, language=language)
    if decision.action == 'retry':
        return decision, None, []
    candidates = [h.text for h in hypotheses]
    result = cleaner.process(candidates, context=context, names=ctx.names if ctx else None)
    if not result.text:
        return Assessment('retry', "Couldn't read that; try again a little slower", None, False), None, []
    if language == 'en' and policy == 'off':
        return decision, result, []
    choices = []
    for label, text in [('Cleanup / 纠错建议', result.proposed),
                        ('Raw / 原始识别', basic_cleanup(result.raw)),
                        *[('Alternative / 其他候选', basic_cleanup(c)) for c in candidates[1:]]]:
        if text and text not in [t for _, t in choices]:
            choices.append((label, text))
        if len(choices) == 3:
            break
    return decision, result, choices


def target_is_current(ctx):
    if ctx is None or not ctx.target:
        return False
    import sys
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        u = ctypes.WinDLL('user32')
        u.GetForegroundWindow.restype = wintypes.HWND
        return u.GetForegroundWindow() == ctx.target
    from AppKit import NSWorkspace
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    if app is None or app.processIdentifier() != ctx.target:
        return False
    if ctx.element is not None:
        from .context import _ax
        from ApplicationServices import AXUIElementCreateApplication, CFEqual
        focused = _ax(AXUIElementCreateApplication(ctx.target), 'AXFocusedUIElement')
        return focused is not None and bool(CFEqual(focused, ctx.element))
    return True


def restore_target(ctx):
    """Called only after choosing a focused review window; never for a stale automatic result."""
    if ctx is None or not ctx.target:
        return False
    import sys
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        u = ctypes.WinDLL('user32')
        u.SetForegroundWindow.argtypes = [wintypes.HWND]
        return bool(u.SetForegroundWindow(ctx.target))
    from AppKit import NSRunningApplication, NSApplicationActivateIgnoringOtherApps
    app = NSRunningApplication.runningApplicationWithProcessIdentifier_(ctx.target)
    return bool(app and app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps))
