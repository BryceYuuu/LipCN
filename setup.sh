#!/usr/bin/env bash
# Install dependencies and download the models (~1.2 GB). Add --samples for the test clips.
set -euo pipefail
cd "$(dirname "$0")"

command -v uv >/dev/null || { echo "Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
uv sync

get() {  # url dest
  [ -s "$2" ] && { echo "✓ $2"; return; }
  mkdir -p "$(dirname "$2")"
  echo "↓ $2"
  curl -fL --progress-bar -o "$2.part" "$1" && mv "$2.part" "$2"
}

HF=https://huggingface.co
# Auto-AVSR visual-only model trained on LRS3 (WER 19.1%) + subword RNN language model
get $HF/Amanvir/LRS3_V_WER19.1/resolve/main/model.json models/vsr/model.json
get $HF/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth  models/vsr/model.pth
get $HF/Amanvir/lm_en_subword/resolve/main/model.json  models/lm/model.json
get $HF/Amanvir/lm_en_subword/resolve/main/model.pth   models/lm/model.pth
# MediaPipe face landmarker
get https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
    models/face_landmarker.task

if [[ "${1:-}" == "--samples" ]]; then
  # Public-domain White House weekly addresses (Wikimedia Commons), used by tests/test_pipeline.py
  C=https://upload.wikimedia.org/wikipedia/commons/transcoded
  get "$C/c/ce/2016-03-12_President_Obama%27s_Weekly_Address.webm/2016-03-12_President_Obama%27s_Weekly_Address.webm.360p.mpeg4.mov" samples/2016-03-12.mov
  get "$C/2/29/2017-01-07_President_Obama%27s_Weekly_Address.webm/2017-01-07_President_Obama%27s_Weekly_Address.webm.360p.mpeg4.mov" samples/2017-01-07.mov
fi

echo
uv run lipflow doctor || true
echo
echo "Start it with:  uv run lipflow"
