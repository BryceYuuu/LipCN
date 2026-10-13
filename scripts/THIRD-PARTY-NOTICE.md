# Third-party runtime notices

The LipCN standalone application includes the CPython runtime and third-party
Python/native dependencies installed from the repository's `uv.lock`. LipCN
retains their distributed copyright notices and license texts. Those components
remain under their respective licenses; LipCN's project license does not
relicense them.

Within `LipCN.app/Contents/Resources`:

- `source/LICENSE`, `source/NOTICE` and `source/LICENSES/` contain the project and
  bundled source-code notices, including Amy Zhou's original Lipflow work,
  ESPnet, Auto-AVSR and Chaplin attribution.
- `runtime/lib/python3.12/LICENSE.txt` contains the CPython license and notices.
- `runtime/lib/python3.12/site-packages/*.dist-info/` retains wheel metadata,
  `LICENSE*`, `COPYING*`, `NOTICE*` and `licenses/` directories when supplied by
  each dependency. Further third-party notices distributed inside individual
  package directories are also retained.
- `BUILD-MANIFEST.json` records the exact installed package versions and lock
  hash. The full lock file is included at `source/uv.lock`.

Major components include NumPy, SciPy, PyTorch, MLX/MLX-LM, MediaPipe, OpenCV,
PyObjC, faster-whisper, CTranslate2, ONNX Runtime, OpenCC and sounddevice, together
with their transitive dependencies. Refer to the retained license texts for
their specific terms and subcomponents.

Model weights, training data, private adapters and personal configuration are
not included. Model setup downloads separately selected public assets from the
pinned origins listed in `source/lipcn/assets_manifest.json`. Their terms are
separate from the application and runtime licenses. English LRS3 and Mandarin
visual research restrictions remain in force; accepting the setup notice is
not a grant of additional rights.
