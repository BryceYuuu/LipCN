# 个人口型适配 / Personal lip adaptation

不同的人嘴型差别很大：公开开发集上同一个模型对不同说话者的字错误率从 25% 到 95% 不等。用你自己的摄像头和默念方式训练一个个人 adapter，是目前最可能明显提高准确率的办法。本机有 Round8 adapter 时从它出发，没有时（Round8 不随仓库发布）从公开的 CNVSRC 基座出发；只训练编码器最后两层，只有在你自己的测试句上比起点更好时才会安装。没有 adapter 的用户装上个人 adapter 后，就能使用中文纯口型模式。

People's lips differ a lot: on the public development set the same model ranges from 25% to 95% CER across speakers. A personal adapter, trained from Round8 on your own webcam and silent articulation, is the most promising way to raise accuracy. It is installed only if it beats Round8 on your own dev clips.

## 需要什么 / What you need

| 步骤 | 内容 | 时间 |
| --- | --- | --- |
| 1. 测试句 | [摄像头基准](webcam-baseline.md)的 40 句（`--split dev`） | 约 15 分钟 |
| 2. 训练句 | [`personal-train-prompts.json`](../../research/examples/personal-train-prompts.json) 179 句（`--split train`） | 约 45–60 分钟，可分几次 |
| 3. 训练 | 一条命令，自动比较并安装 | 最多 30 分钟（训练程序的累计上限） |

两组必须使用**同一个 `--speaker`、不同的 `--session`**，最好不在同一天录；否则训练脚本会拒绝开始，避免同一次录制的光线和状态同时出现在训练和测试里。

Both sets must use the same `--speaker` and different `--session` values, ideally on different days.

## 1. 录测试句（已录可跳过） / Record dev prompts

```bash
.venv/bin/lipcn collect --speaker me01 --session dev-$(date +%Y%m%d) --split dev \
  --lighting indoor-front --prompts research/examples/webcam-baseline-prompts.json
```

## 2. 录训练句 / Record training prompts

训练句录在单独的目录 `personal-data`，并放宽容量上限：

```bash
.venv/bin/lipcn collect --speaker me01 --session train-$(date +%Y%m%d) --split train \
  --output "$HOME/Library/Application Support/LipCN/personal-data" --max-megabytes 2048 \
  --lighting indoor-front --prompts research/examples/personal-train-prompts.json
```

- 像平时使用一样默念：同样的距离、角度和嘴型幅度。录得“标准”但和日常不同，反而学不到你的习惯。
- 每句按 `空格` 开始，念完再按 `空格` 结束，按 `Enter` 保存；`R` 重录，`P` 回放，`Esc` 退出。
- 平时使用时会出声的话，两组录制命令都加 `--voiced`（只录画面，不录声音）；测试句和训练句要用同一种方式。
- 可以中途 `Esc` 退出，之后用相同命令继续：已经保存过的句子会自动跳过，并显示还剩多少句。换一天继续时，换一个 `--session` 名（例如 `train-20261105`）。
- 句子里的人名、数字已在提示文件中标好；如果你念的和提示不同，请按实际念的内容修改后再确认。

## 3. 训练并安装 / Train and install

```bash
research/scripts/train_personal_adapter.sh
```

脚本会：

1. 检查两组数据：同一个人、不同场次、没有重复句子或视频；训练最多 30 分钟（累计），数据多时跑不完 6 轮也会保留已完成轮次中最好的一轮；
2. 先用 Round8 在你的测试句上测出起点；
3. 每训练一轮就在测试句上评估，保留最好的一轮；
4. 只有比 Round8 更好时，才安装到 `~/.cache/lipcn/models/mandarin/personal-encoder-adapter.pth`。

重启 LipCN 后自动使用个人 adapter。训练中断后可以继续：

```bash
RESUME=1 RUN="$HOME/Library/Application Support/LipCN/personal-adapter/run-<时间>" research/scripts/train_personal_adapter.sh
```

想回到 Round8：删除 `personal-encoder-adapter.pth`，或启动前设置 `LIPCN_LIP_ADAPTER=~/.cache/lipcn/models/mandarin/round8-encoder-adapter.pth`。

## 怎么看结果 / Reading the result

`report.json` 中 `baseline_cer` 是 Round8 在你的测试句上的字错误率，`selected_cer` 是个人 adapter 的结果。因为最好的一轮也是按这 40 句挑出来的，这个数字会略微偏乐观。想要不带偏差的数字，可以换一天再录一组新的句子作为 `--split test`（放在另一个目录），用[摄像头基准](webcam-baseline.md)里的评测命令比较两个 adapter。

`baseline_cer` is Round8 on your dev clips and `selected_cer` is the personal adapter. Because the best epoch is chosen on those clips, the gain is slightly optimistic; record a separate test set on another day for an unbiased comparison.

## 已验证 / Verified (2026-11-03)

完整流程（采集器格式的清单 → 完整脸部视频的脸部跟踪 → 从 Round8 继续训练 → 按测试句选择 → 安装 → 桌面应用的 adapter 加载器）用 8 段公开完整脸部视频空跑通过，每轮约 27 秒。尚未用真实的个人默念数据训练，因此还没有个人准确率数字。

The full pipeline was dry-run on 8 public full-face clips (about 27 s per epoch). No personal silent data has been trained on yet, so there is no personal accuracy number.
