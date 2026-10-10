"""Load the original English lip reader from explicitly prepared local assets."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lipflow.vsr import LipReader


def load_english_reader(model_dir: str | Path) -> LipReader:
    """Build and warm the reader on the caller's model worker; never download files."""
    root = Path(model_dir).expanduser()
    required = ('vsr/model.json', 'vsr/model.pth', 'lm/model.json', 'lm/model.pth')
    missing = [name for name in required
               if not (root / name).is_file() or (root / name).stat().st_size == 0]
    if missing:
        raise FileNotFoundError(
            'English lip-reading assets are missing or empty: ' + ', '.join(missing)
            + '. Prepare them locally and set LIPCN_ENGLISH_MODELS to their parent directory.'
        )

    import numpy as np
    from lipflow.vsr import LipReader

    reader = LipReader(language='en', model_dir=str(root), beam_size=4, personal=False)
    reader.encode(np.zeros((25, 96, 96), dtype=np.uint8))
    return reader
