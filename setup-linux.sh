#!/usr/bin/env bash
# Linux setup: uv deps + model weights (~1.2 GB). No app bundle.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null; then
  echo "Installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync
if [[ "${1:-}" == "--samples" ]]; then
  ./scripts/download-models.sh --samples
else
  ./scripts/download-models.sh
fi

echo
echo "Done. Run: uv run lipflow doctor && uv run lipflow"
