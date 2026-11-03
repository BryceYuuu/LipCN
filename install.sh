#!/usr/bin/env bash
# Install LipCN: dependencies in .venv, then pinned, SHA-256-verified models.
#   ./install.sh                         # speech models (default)
#   ./install.sh --mode english --accept-research-license
#   ./install.sh --mode all --accept-research-license --with-wording
# Options after the script name go to `lipcn setup`. The Mandarin lip adapter is
# never downloaded; place your own copy as described in desktop/mandarin/README.md.
# (./setup.sh is the retained legacy Lipflow installer, not this app.)
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "LipCN desktop requires macOS 14+ on Apple Silicon." >&2
  exit 2
fi

if ! command -v uv >/dev/null; then
  echo "Installing uv (Python package manager)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --extra desktop

args=("$@")
if [[ " ${args[*]-} " != *" --mode "* && " ${args[*]-} " != *" --mode="* ]]; then
  args=(--mode speech "${args[@]+"${args[@]}"}")
fi
status=0
.venv/bin/lipcn setup --download "${args[@]}" || status=$?

echo
if [[ $status -eq 0 ]]; then
  echo "Done. Start LipCN with: .venv/bin/lipcn"
else
  echo "Some assets are missing (see above). Re-run ./install.sh to resume, or check with: .venv/bin/lipcn doctor"
fi
exit $status
