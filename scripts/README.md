# LipCN macOS distribution

The new `build_lipcn.py` produces an Apple Silicon `.app` with its own CPython 3.12
and locked Python/native dependencies. It does not require the source checkout,
uv, Homebrew or a separately installed Python on the destination Mac. macOS 14+
is required. The historical `make_app.py` builds a development launcher and is
not the standalone distribution builder.

## Build

Use an Apple Silicon Mac with Xcode Command Line Tools and uv. Keep the build
directory outside cloud-synced folders, with at least 6 GB free for the runtime,
temporary environment and compressed archive. From the repository root:

```sh
uv python install 3.12.14
UV_PROJECT_ENVIRONMENT="$HOME/.cache/lipcn/build-env" \
  uv sync --frozen --python 3.12.14 --extra desktop --no-editable --no-dev
"$HOME/.cache/lipcn/build-env/bin/python" scripts/build_lipcn.py \
  --python "$HOME/.cache/lipcn/build-env/bin/python" \
  --output "$HOME/.cache/lipcn/dist-2.3.0"
```

Use a fresh output directory. The builder stages code from a newly built wheel,
copies dependency license files, checks native library links for external
development paths, ad-hoc signs the bundle, relocates it, runs `--version` and
the read-only `doctor`, and imports its native dependencies from inside the
bundle. It then verifies the signature and streams a ZIP with executable permissions and a SHA256 sidecar.
Compression preserves at least 512 MiB of free disk space; if that threshold is
reached, only the new partial archive is removed and the verified app remains.
`BUILD-MANIFEST.json` records package versions, lock hash and validation scope.
The procedure is reproducible from the lock and source; ZIP bytes need not be
identical because timestamps and ad-hoc signatures can differ.

For a source-only refresh with the same dependency lock, use `--reuse-runtime`
and the same output path, omitting `--python`. The saved local build metadata
allows the temporary build environment to be removed after staging. This
refresh rebuilds the source wheel and repeats validation/signing without another
runtime copy. The local `.build-environment.json` is outside the app and archive;
it contains build paths and is not a release artifact.

These checks do not open the GUI, load models or exercise cameras/microphones.
Release validation must additionally cover the actual desktop on supported
hardware. This build is **not Apple notarized** and has no Developer ID signature;
macOS may require the user to allow opening it in Privacy & Security. Do not
describe it as notarized or silently disable Gatekeeper.

## Install and prepare models

Extract the ZIP and move `LipCN.app` to Applications. No models, private adapters,
recordings or personal profile are included. The bundled command is available at:

```sh
/Applications/LipCN.app/Contents/MacOS/LipCN doctor --mode speech
/Applications/LipCN.app/Contents/MacOS/LipCN setup --mode speech --download
/Applications/LipCN.app/Contents/MacOS/LipCN setup --mode english --download --accept-research-license
```

Running `setup` without `--download` only inspects local state. Downloads use the
fixed revisions, sizes and SHA256 hashes in `lipcn/assets_manifest.json`. An
interrupted `.lipcn-part` download resumes when the server supports ranges; only
verified complete bytes replace the final asset. Re-running setup recovers a
failed transfer. See the source links reported by `doctor --json` before accepting
research-model terms. The English LRS3 and Mandarin visual models have research
restrictions; downloading does not grant a commercial model license.

`speech` prepares Mandarin audio recognition, the local wording model and the
face model used by the desktop camera preview. `english` prepares the English visual
model, face model and wording model. `mandarin`/`all` also need a compatible
**locally supplied Mandarin adapter**: the private development adapter is never
downloaded or shipped. A fresh install therefore cannot claim Mandarin visual
readiness merely because public downloads succeeded. `doctor` reports the manual
asset as missing until it is supplied; metadata checks do not validate adapter
compatibility or recognition quality.

Model files live under `~/.cache/lipcn` by default, with the existing `LIPCN_*`
overrides supported. Personal state belongs to `~/Library/Application Support/LipCN`
(or `LIPCN_PROFILE_DIR`) outside the app. Removing/replacing the app does not
remove these files. CLI source installs use `lipcn`; the `lipflow` compatibility
command continues to use the original CLI. `lipcn collect --help` describes the
explicit, local silent-data research tool and does not start capture.

## Third-party licenses

See [THIRD-PARTY-NOTICE.md](THIRD-PARTY-NOTICE.md). Keep the full runtime license
files and `source/LICENSES` in redistributions; the package-version inventory
does not replace those licenses.
