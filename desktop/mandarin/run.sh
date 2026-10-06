#!/bin/sh
# Run from this checkout with dependencies installed in its virtual environment.
set -eu

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
repo_dir="$(CDPATH= cd -- "$script_dir/../.." && pwd)"
python_bin="${LIPCN_PYTHON:-$repo_dir/.venv/bin/python}"

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    echo "The Mandarin desktop app requires macOS on Apple Silicon." >&2
    exit 2
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python not found: $python_bin" >&2
    echo "Install this checkout's dependencies in .venv, or set LIPCN_PYTHON." >&2
    exit 2
fi

# Loading is local-only; set model paths before launch as described in README.md.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
# The retained upstream Python package still reads LIPFLOW_HOME internally.
# Do not inherit the original app's profile into this separate product.
export LIPFLOW_HOME="${LIPCN_PROFILE_DIR:-$HOME/Library/Application Support/LipCN}"
exec "$python_bin" "$script_dir/live_test.py" "$@"
