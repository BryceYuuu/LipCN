"""Where each person's data lives. Shared model weights stay in the repo's models/; everything
learned from *you* (clips, phrases, personal models, settings) goes here.

Override with LIPFLOW_HOME (tests use a temp dir so they never touch your real data)."""
import os
import sys

WINDOWS = sys.platform == "win32"
LINUX = sys.platform == "linux"
DESKTOP_TRAY = WINDOWS or LINUX

if WINDOWS:  # %APPDATA%\Lipflow
    _DEFAULT_HOME = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "Lipflow")
elif LINUX:
    _DEFAULT_HOME = os.path.join(
        os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
        "Lipflow",
    )
else:
    _DEFAULT_HOME = os.path.expanduser("~/Library/Application Support/Lipflow")
HOME = os.environ.get("LIPFLOW_HOME") or _DEFAULT_HOME
PERSONAL_MODELS = os.path.join(HOME, "models")
PERSONAL_VSR = os.path.join(PERSONAL_MODELS, "vsr_face.pth")
PERSONAL_LM = os.path.join(PERSONAL_MODELS, "lm_phrasing.pth")

# The app bundle's launcher sets LIPFLOW_APP=1: permissions then belong to "Lipflow", not the terminal.
# On Windows nothing is granted per app, so the name only shows up in messages.
WHO = "Lipflow" if os.environ.get("LIPFLOW_APP") or DESKTOP_TRAY else "your terminal"


def personal_vsr(language="en"):
    return PERSONAL_VSR if language == "en" else os.path.join(PERSONAL_MODELS, language, "vsr_face.pth")


def language_home(language="en"):
    """English retains its legacy data; Mandarin has an explicit namespace."""
    if language not in {"en", "zh"}:
        raise ValueError("language must be en or zh")
    return HOME if language == "en" else os.path.join(HOME, "languages", "zh")


def phrases_path(language="en"):
    return os.path.join(language_home(language), "phrases.txt")


def default_language():
    try:
        with open(os.path.join(HOME, "active-language.txt"), encoding="utf-8") as f:
            language = f.read().strip()
        return language if language in {"en", "zh"} else "en"
    except OSError:
        return "en"


def select_language(language):
    language_home(language)  # validate before writing
    os.makedirs(HOME, exist_ok=True)
    with open(os.path.join(HOME, "active-language.txt"), "w", encoding="utf-8") as f:
        f.write(language)
