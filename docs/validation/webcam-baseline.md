# 个人摄像头基准 / Personal webcam baseline

用自己的摄像头和真实默念建立一套固定测试集，之后每次改模型、解码参数或草稿策略，都在同一批视频上比较字错误率（CER）和延迟。没有这套数据，就不能说“准确率提高了”。

Record a fixed set of your own silently mouthed sentences once, then compare CER and latency on exactly those clips whenever the model, decoder or draft strategy changes.

## 1. 录制 / Record

[`research/examples/webcam-baseline-prompts.json`](../../research/examples/webcam-baseline-prompts.json) 含 40 句日常输入：30 句通用句子，后 10 句含台湾常用词（捷运、机车、软体、影片、便利商店等）。参考文本统一写成简体，因为模型原始输出是简体，CER 按字比较，简繁不能混用；台湾用语在“繁體中文”模式下由 OpenCC `s2twp` 转换显示。

The 40 prompts cover everyday dictation; the last 10 use common Taiwan terms. References are written in simplified characters because raw model output is simplified and CER compares characters.

```bash
.venv/bin/lipcn collect --speaker me01 --session dev-$(date +%Y%m%d) --split dev \
  --lighting indoor-front --prompts research/examples/webcam-baseline-prompts.json
```

- 窗口上方用大字显示要念的句子和进度。正对摄像头、光线照向脸部：按 `空格` 开始，念完再按 `空格` 结束，按 `Enter` 保存；不满意按 `R` 重录，`P` 回放，`Esc` 退出（下次运行会自动跳过已保存的句子）。
- 录制方式要和平时使用 LipCN 一致：平时默念就默念；平时会出声，就在命令后加 `--voiced`（仍然只录画面，不录声音）。
- 建议分两天、换一次光线录制；同一个 `--speaker` 在同一个数据目录里只能出现在一个分组。之后做[个人适配](personal-adaptation.md)时，训练句录在另一个目录、使用不同的 `--session`。
- 视频保存在 `~/Library/Application Support/LipCN/silent-data/dev/`，只在本机，不要提交到 Git。

## 2. 评测 / Evaluate

与桌面应用相同的识别设置（CNVSRC 基座 + 本地 adapter，beam 40、CTC 0.1、反向 0.3、候选池 10）：

```bash
M=~/.cache/lipcn/models/mandarin
DATA="$HOME/Library/Application Support/LipCN/silent-data"
.venv/bin/python research/scripts/evaluate_cnvsrc.py --accept-research-license \
  --checkpoint $M/model_avg_cncvs_2_3_cnvsrc.pth --source-dir ~/.cache/lipcn/CNVSRC2025 \
  --adapter $M/round8-encoder-adapter.pth --manifest "$DATA/dev/manifest.json" --device mps \
  --beam-size 40 --ctc-weight 0.1 --nbest 10 --candidate-pool-size 10 \
  --reverse-weight 0.3 --reverse-scoring batched --output baseline.json

.venv/bin/python -m research.scripts.report_silent_errors \
  --manifest "$DATA/dev/manifest.json" --baseline baseline.json --output errors.json
```

`errors.json` 给出整体 CER、替换/漏字/多字、人名与数字句子、按日期分组的结果。改动后用相同命令生成 `candidate.json`，再加 `--candidate candidate.json` 做成对比较。

草稿与完整搜索的准确率和耗时对比：

```bash
.venv/bin/python research/scripts/benchmark_draft_decoding.py \
  --manifest "$DATA/dev/manifest.json" --checkpoint $M/model_avg_cncvs_2_3_cnvsrc.pth \
  --source-dir ~/.cache/lipcn/CNVSRC2025 --adapter $M/round8-encoder-adapter.pth
```

## 3. 当前参考数字 / Reference numbers (2026-11-03)

尚无个人摄像头数据。公开 Chinese-LiPS 开发子集（33 段、11 人、1432 字，正常发声的嘴部裁剪视频，非摄像头、非默念）上，`benchmark_draft_decoding.py` 的结果：

| 解码 / Decode | CER | 平均 / mean | p95 |
| --- | --- | --- | --- |
| CTC 草稿 / greedy draft | 60.5% | 0.003 s | 0.005 s |
| beam 4（nbest 1） | 57.7% | 0.84 s | 1.23 s |
| 完整搜索 / full search | 57.1% | 3.35 s | 5.14 s |

编码另需平均 0.27 s（三者共用）。这些数字不代表默念或个人摄像头上的表现。

Encoding adds 0.27 s on average, shared by every route. These public-clip numbers say nothing about silent speech or your own webcam.
