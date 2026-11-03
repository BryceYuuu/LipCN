#!/usr/bin/env bash
# Train a personal Mandarin lip adapter from your own recordings, starting from
# the installed Round8 adapter, and install it only if it beats Round8 on your
# own dev clips. The desktop app then prefers it (desktop/mandarin/config.py).
#
#   research/scripts/train_personal_adapter.sh            # train + install if better
#   TRAIN=/path/train/manifest.json DEV=/path/dev/manifest.json research/scripts/train_personal_adapter.sh
#   RESUME=1 RUN=<previous run folder> research/scripts/train_personal_adapter.sh   # continue after an interruption
#
# Guide: docs/validation/personal-adaptation.md. Research-only weights; see NOTICE.
set -euo pipefail
cd "$(dirname "$0")/../.."

PROFILE="${LIPCN_PROFILE_DIR:-$HOME/Library/Application Support/LipCN}"
MODELS="${LIPCN_MANDARIN_MODELS:-$HOME/.cache/lipcn/models/mandarin}"
SOURCE="${LIPCN_CNVSRC_SOURCE:-$HOME/.cache/lipcn/CNVSRC2025}"
TRAIN="${TRAIN:-$PROFILE/personal-data/train/manifest.json}"
DEV="${DEV:-$PROFILE/silent-data/dev/manifest.json}"
BASE_ADAPTER="${BASE_ADAPTER:-$MODELS/round8-encoder-adapter.pth}"
INSTALL="${INSTALL:-$MODELS/personal-encoder-adapter.pth}"
RUN="${RUN:-$PROFILE/personal-adapter/run-$(date +%Y%m%d-%H%M%S)}"
EPOCHS="${EPOCHS:-6}"
MAX_SECONDS="${MAX_SECONDS:-1800}"  # trainer cap (cumulative across resumes); best finished epoch is kept

for file in "$TRAIN" "$DEV" "$MODELS/model_avg_cncvs_2_3_cnvsrc.pth"; do
  [[ -f "$file" ]] || { echo "Missing: $file" >&2; exit 2; }
done
# Round8 is a local research adapter that is not distributed; without it,
# train from the public CNVSRC base model instead.
initial=()
if [[ -f "$BASE_ADAPTER" ]]; then
  initial=(--initial-adapter "$BASE_ADAPTER")
  echo "Starting from adapter: $BASE_ADAPTER"
else
  echo "No starting adapter found; training from the CNVSRC base model."
fi
count() { .venv/bin/python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["samples"]))' "$1"; }
echo "Train clips: $(count "$TRAIN")   Dev clips: $(count "$DEV")"

mkdir -p "$(dirname "$RUN")"
resume=()
[[ "${RESUME:-0}" == 1 ]] && resume=(--resume)
.venv/bin/python research/scripts/train_chinese_joint_adapter.py --accept-research-license \
  --checkpoint "$MODELS/model_avg_cncvs_2_3_cnvsrc.pth" --source-dir "$SOURCE" \
  --train-manifest "$TRAIN" --dev-manifest "$DEV" --scope personal \
  ${initial[@]+"${initial[@]}"} --ctc-weight 0.1 --device auto \
  --epochs "$EPOCHS" --max-seconds "$MAX_SECONDS" --output "$RUN" ${resume[@]+"${resume[@]}"}

if [[ -f "$RUN/encoder_adapter.pth" ]]; then
  cp "$RUN/encoder_adapter.pth" "$INSTALL.tmp" && mv "$INSTALL.tmp" "$INSTALL"
  echo
  echo "Installed personal adapter: $INSTALL"
  echo "It beat the starting adapter on your dev clips; restart LipCN to use it."
  echo "To go back to Round8, delete that file or set LIPCN_LIP_ADAPTER=$BASE_ADAPTER."
else
  echo
  echo "No epoch beat the starting adapter on your dev clips; nothing was installed."
fi
echo "Report: $RUN/report.json"
# A shareable summary for a GitHub "training report" issue: numbers only.
.venv/bin/python - "$RUN/report.json" "$TRAIN" "$DEV" <<'PY'
import json, sys
report, train, dev = (json.load(open(path)) for path in sys.argv[1:4])
articulation = sorted({row.get('articulation', '?') for row in train['samples']})
print('\n--- 可复制到 GitHub Issue 的训练成绩 ---')
print(f"训练句：{len(train['samples'])}　测试句：{len(dev['samples'])}　方式：{'/'.join(articulation)}")
print(f"训练前字错误率：{report['baseline']['cer']:.1%}　训练后：{report['candidate']['cer']:.1%}"
      f"　最佳轮次：{report['selected_epoch']}（训练程序评测，不含语言模型）")
PY
