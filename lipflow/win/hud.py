"""The floating pill above the taskbar: what Lipflow is doing, live words, and your mouth while you talk.

A borderless always-on-top tk window that never takes focus (WS_EX_NOACTIVATE), so Ctrl+V still
lands in the app you're dictating into. Main (tk) thread only.
"""
from __future__ import annotations

import sys
import tkinter as tk

import numpy as np

W, H = 420, 86
VIDEO_W, VIDEO_H = 112, 70
KEY = "#ff00fe"  # transparent colour: everything outside the rounded rectangle
BG, FG, DIM = "#1d1b20", "#ffffff", "#b9b4bf"
ACCENT, GREEN, AMBER, RED = "#ff5473", "#4dd98c", "#ffb840", "#ff6b5e"
FONT = "Segoe UI"
STATES = {"listening": ACCENT, "reading": AMBER, "done": GREEN, "error": RED}


def rounded_rect(c: tk.Canvas, x0, y0, x1, y1, r, **kw):
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
           x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return c.create_polygon(pts, smooth=True, **kw)


def photo(bgr: np.ndarray):
    """BGR array → tk PhotoImage (keep a reference or tk drops it)."""
    from PIL import Image, ImageTk
    return ImageTk.PhotoImage(Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1])))


def no_activate(win: tk.Toplevel) -> "int | None":
    """Make a tk toplevel a tool window that never steals focus. Returns its HWND."""
    if sys.platform != "win32":
        return None
    import ctypes
    user32 = ctypes.WinDLL("user32")
    user32.GetParent.restype = ctypes.c_void_p
    user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    user32.GetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int]
    user32.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
    win.update_idletasks()
    hwnd = user32.GetParent(win.winfo_id()) or win.winfo_id()
    GWL_EXSTYLE, WS_EX_TOPMOST, WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE = -20, 0x8, 0x80, 0x08000000
    ex = user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, ex | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
    return hwnd


class HUD:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.configure(bg=KEY)
        if sys.platform == "win32":
            self.win.attributes("-transparentcolor", KEY)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        self.win.geometry(f"{W}x{H}+{(sw - W) // 2}+{sh - H - 96}")  # above the taskbar
        c = self.c = tk.Canvas(self.win, width=W, height=H, bg=KEY, highlightthickness=0)
        c.pack()
        rounded_rect(c, 2, 2, W - 2, H - 2, 26, fill=BG, outline="#3a3640")
        self.dot = c.create_oval(22, 22, 32, 32, fill=AMBER, outline="")
        self.video_item = c.create_image(18 + VIDEO_W // 2, H // 2, state="hidden")
        self.title = c.create_text(42, 27, anchor="w", fill=DIM, font=(FONT, 9, "bold"), text="")
        self.body = c.create_text(22, 54, anchor="w", fill=FG, font=(FONT, 12), text="", width=W - 44)
        self._img = None
        self._hide_job = None
        self.mode = ""
        self.body_text = ""
        self.hwnd = no_activate(self.win)
        self._visible = True
        self.hide()

    # -- public (main thread) --------------------------------------------------------------
    def show(self, mode: str, title: str, body: str = "", hide_after: "float | None" = None):
        self._cancel_hide()
        self.mode = mode
        self.c.itemconfigure(self.dot, fill=STATES.get(mode, AMBER))
        self.c.itemconfigure(self.title, text=title.upper())
        if mode != "listening":
            self.c.itemconfigure(self.video_item, state="hidden")
            self._img = None
            self._layout(video=False)
        self.set_text(body)
        self._show_window()
        if hide_after:
            self._hide_job = self.root.after(int(hide_after * 1000), self.hide)

    def set_text(self, body: str):
        self.body_text = body
        live = self.mode in ("listening", "reading")
        limit = 46 if self.c.itemcget(self.video_item, "state") == "hidden" else 34
        if len(body) > limit:  # live words: keep the newest; messages: keep the start
            body = "…" + body[-limit:] if live else body[:limit] + "…"
        self.c.itemconfigure(self.body, text=body)

    def set_frame(self, video_bgr, level: float = 0.0):
        if video_bgr is None or self.mode != "listening":
            return
        self._img = photo(video_bgr)
        self.c.itemconfigure(self.video_item, image=self._img, state="normal")
        self._layout(video=True)

    def hide(self):
        self._cancel_hide()
        self.mode = ""
        self._img = None
        self.c.itemconfigure(self.video_item, state="hidden")
        if not self._visible:
            return
        self._visible = False
        if self.hwnd is not None:
            import ctypes
            ctypes.WinDLL("user32").ShowWindow(ctypes.c_void_p(self.hwnd), 0)  # SW_HIDE
        else:
            self.win.withdraw()

    # -- internals ---------------------------------------------------------------------------
    def _layout(self, video: bool):
        x = 18 + VIDEO_W + 14 if video else 22
        self.c.coords(self.dot, x, 22, x + 10, 32)
        self.c.coords(self.title, x + 20, 27)
        self.c.coords(self.body, x, 54)
        self.c.itemconfigure(self.body, width=W - x - 22)

    def _show_window(self):
        if self._visible:
            return
        self._visible = True
        if self.hwnd is not None:
            import ctypes
            user32 = ctypes.WinDLL("user32")
            user32.ShowWindow(ctypes.c_void_p(self.hwnd), 4)  # SW_SHOWNOACTIVATE
            # HWND_TOPMOST, no move/size/activate
            user32.SetWindowPos(ctypes.c_void_p(self.hwnd), ctypes.c_void_p(-1), 0, 0, 0, 0, 0x1 | 0x2 | 0x10)
        else:
            self.win.deiconify()

    def _cancel_hide(self):
        if self._hide_job is not None:
            self.root.after_cancel(self._hide_job)
            self._hide_job = None


def tray_image(listening: bool = False, size: int = 64):
    """Pink rounded square with a white mouth; solid lips while listening."""
    from PIL import Image, ImageDraw
    s = size * 4  # draw big, scale down for smooth edges
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([s * 0.04, s * 0.04, s * 0.96, s * 0.96], radius=s * 0.22,
                        fill=(255, 84, 115, 255) if not listening else (214, 41, 97, 255))
    box = [s * 0.2, s * 0.34, s * 0.8, s * 0.68]
    if listening:
        d.ellipse(box, fill="white")
        d.line([s * 0.24, s * 0.51, s * 0.76, s * 0.51], fill=(214, 41, 97, 255), width=int(s * 0.05))
    else:
        d.ellipse(box, outline="white", width=int(s * 0.07))
        d.line([s * 0.24, s * 0.51, s * 0.76, s * 0.51], fill="white", width=int(s * 0.05))
    return img.resize((size, size), Image.LANCZOS)
