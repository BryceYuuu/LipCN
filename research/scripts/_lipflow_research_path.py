"""Find local source dependencies without depending on the caller's directory."""
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# A source checkout takes precedence over a different installed Lipflow version.
if (REPOSITORY_ROOT / 'lipflow' / '__init__.py').is_file():
    sys.path.insert(0, str(REPOSITORY_ROOT))
