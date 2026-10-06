"""Import standalone desktop modules without installing or loading native backends."""
from pathlib import Path
import sys

DESKTOP_ROOT = str(Path(__file__).resolve().parents[1])
if DESKTOP_ROOT not in sys.path:
    sys.path.insert(0, DESKTOP_ROOT)
