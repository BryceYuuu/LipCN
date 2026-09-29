"""What you're typing into: app, window title, and the text around the cursor.

Read once, locally, through the Accessibility API when you press the push-to-talk key (the app
you're dictating into is frontmost at that moment). Only names and terms are pulled out, to help
with the words lip reading gets wrong most often — who you're writing to, the thread you're in.
Nothing is saved to disk or sent anywhere.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_STOP = set("""a an the and or but if of to in on at for with from by as is are was were be been am i
you he she it we they me my your our their this that these those do does did have has had not no so
just can will would could should there here what when where who how why all about up out then than
too very really also like get got go going new re fwd inbox search compose sent drafts home today
monday tuesday wednesday thursday friday saturday sunday january february march april may june july
august september october november december untitled message messages chat thread channel reply""".split())


@dataclass
class Context:
    app: str = ""
    title: str = ""
    near_text: str = ""
    names: list[str] = field(default_factory=list)

    def describe(self) -> str:
        bits = [self.app] + ([f'"{self.title[:60]}"'] if self.title else [])
        return " · ".join(b for b in bits if b)


def _ax(el, attr):
    from ApplicationServices import AXUIElementCopyAttributeValue
    err, val = AXUIElementCopyAttributeValue(el, attr, None)
    return val if err == 0 else None


def extract_names(*texts: str, limit: int = 30) -> list[str]:
    """Capitalised words that aren't sentence starts or common words: names, products, places."""
    seen, out = set(), []
    for t in texts:
        for sent in re.split(r"[.!?\n|•·—\-–:]+", t or ""):
            toks = re.findall(r"[A-Za-z][A-Za-z'\-]+", sent)
            for i, w in enumerate(toks):
                if not w[0].isupper() or w.isupper() and len(w) > 4:
                    continue
                if w.lower() in _STOP or len(w) < 3:
                    continue
                if i == 0 and w.lower() in _STOP:
                    continue
                if w.lower() not in seen:
                    seen.add(w.lower())
                    out.append(w)
    return out[:limit]


def capture(max_chars: int = 600) -> Context:
    """Snapshot of the frontmost app. Never raises: context is a bonus, not a requirement."""
    ctx = Context()
    try:
        from AppKit import NSWorkspace
        from ApplicationServices import AXUIElementCreateApplication
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return ctx
        ctx.app = str(app.localizedName() or "")
        ax_app = AXUIElementCreateApplication(app.processIdentifier())
        win = _ax(ax_app, "AXFocusedWindow")
        if win is not None:
            ctx.title = str(_ax(win, "AXTitle") or "")
        focused = _ax(ax_app, "AXFocusedUIElement")
        if focused is not None:
            val = _ax(focused, "AXValue")
            if isinstance(val, str):
                ctx.near_text = val[-max_chars:]
            if not ctx.near_text:
                ph = _ax(focused, "AXPlaceholderValue")  # e.g. Slack's "Message Miguel"
                if isinstance(ph, str):
                    ctx.near_text = ph
        ctx.names = extract_names(ctx.title, ctx.near_text)
    except Exception as e:  # permissions, sandboxed apps, odd elements
        print(f"[lipflow] context unavailable: {e.__class__.__name__}")
    return ctx
