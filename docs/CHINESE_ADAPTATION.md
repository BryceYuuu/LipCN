# 公开数据适配与本地无声采集

目标是纯无声、自由中文句子。**目前仍未达到日常输入的可用标准。** 本流程分别记录公开数据研究与真实无声摄像头验收，训练报告不会自动替换 GUI 模型或解除中文候选确认。

## 公开数据

使用作者发布的 [BAAI/Chinese-LiPS](https://huggingface.co/datasets/BAAI/Chinese-LiPS)，固定版本 `db96948538811029011eee44602438a26710ecd9`，许可为 CC-BY-NC-SA-4.0。原始参考文本来自作者人工整理；下载工具不会重新核对每个字，也不会生成标签。数据是普通说话的嘴部裁剪，不能证明刻意不发声的口型识别能力。[作者论文](https://arxiv.org/abs/2504.15066)说明了数据来源和划分。

```bash
uv run python scripts/fetch_chinese_lips.py --accept-noncommercial-license \
  --output-dir samples/chinese_lips
```

默认按固定哈希规则选取 train 180 条/12 人、dev 30 条/6 人、test 50 条/10 人。选择不看识别结果。工具只读取嘴部视频与文本元数据，不读取音频或幻灯片。通过 HTTP Range 读取指定压缩成员；服务器忽略范围请求时直接失败，不回退下载整个多 GB 压缩包。元数据 SHA256、成员 CRC 和本地视频 SHA256 会核对；整个压缩包的发布者 SHA256 仅记录，未下载整包，因此不声称已验证整包。

`samples/` 已由 Git 忽略，视频和参考标签不随应用分发。三分区是本轮适配的独立分区；尚未完整核实基座预训练的身份/视频重叠，`training_overlap_checked` 保持 false。

## 固定训练实验

单独下载并阅读 [CNVSRC2025 源码许可](https://github.com/liu12366262626/CNVSRC2025/blob/main/VSR/LICENSE)和[基座权重](https://huggingface.co/ReflectionL/CNVSRC2025Baseline)。源码版本、配置、词表和权重哈希由研究脚本核对。实验使用现有 ESPnet 模块实现兼容结构，不导入作者源码、不将研究权重装入产品。

CNVSRC 使用固定字符词表。先列出无法表示的训练目标；明确选择排除这些**完整训练片段**，保留原始文本和排除记录。这个操作只适用于 train，dev/test 中不支持的字符仍须保留在完整计分中：

```bash
uv run python scripts/prepare_chinese_training.py \
  --manifest samples/chinese_lips/train.json \
  --vocabulary /local/CNVSRC2025/VSR/datamodule/char_units.txt \
  --exclude-unsupported-training-samples \
  --output samples/chinese_lips/train-supported.json

uv run python scripts/train_chinese_adapter.py --accept-research-license \
  --checkpoint /local/model_avg_cncvs_2_3_cnvsrc.pth \
  --source-dir /local/CNVSRC2025 \
  --train-manifest samples/chinese_lips/train-supported.json \
  --dev-manifest samples/chinese_lips/dev.json \
  --test-manifest samples/chinese_lips/test.json \
  --scope general --output samples/chinese_lips/adapter-run \
  --epochs 2 --last-layers 1 --learning-rate 0.00001 \
  --beam-size 40 --ctc-weight 0.5 --seed 0
```

本轮默认划分保留 160 条可表示训练片段。训练只改最后一个视觉编码块和输出归一化，冻结前端、CTC 头、解码器与 BatchNorm 统计。损失是完整字符目标的 CTC，MPS 编码器与 CPU CTC 之间保留梯度。过短目标帧数或非有限损失会停止实验，不会悄悄跳过。推理模式产生的位置编码缓存会在训练前转换为可反传张量。

实验固定两个 epoch，不按验证/测试分数选择 epoch。基线和最终候选分别在完整 dev/test 各计分一次。只有两套数据的原始 CER 都降低、整句正确率和候选覆盖率不下降且无推理失败，才保存 `encoder_adapter.pth`；否则只保存拒绝报告。相对改善仍不等于产品可用。报告保留基座、数据、排除记录与参数，部分权重保留基座研究用途限制。

观察过测试结果后，若更改参数、增强、模型或解码方式，需要新的独立测试集。不要重复使用同一测试集挑选最佳模型。

保存的部分权重可以在独立研究脚本中显式加载；基座 SHA256、语言、允许参数、形状、类型和有限值均会先核对：

```bash
uv run python scripts/evaluate_cnvsrc.py --accept-research-license \
  --checkpoint /local/model_avg_cncvs_2_3_cnvsrc.pth \
  --source-dir /local/CNVSRC2025 \
  --adapter samples/chinese_lips/adapter-run/encoder_adapter.pth \
  --manifest /local/new-independent-test/manifest.json \
  --output samples/chinese_lips/new-test-results.json
```

## 稍后采集真正无声的摄像头片段

采集不需要加载识别权重。只在你主动执行以下命令时打开摄像头；没有麦克风或云调用：

```bash
uv run lipflow collect-chinese --output samples/chinese_private/train \
  --speaker bryce --session day01 --split train
```

录制前，在终端输入你准备无声说的准确句子。按 SPACE 开始/停止；R 丢弃并重录；Esc 丢弃当前片段并退出。停止后，只有当你确实无声发音、没有耳语/发声且说的文字与提示一致时才按 Enter 保存。说错或漏字请重录。也可用 `--sentences phrases.txt` 提供独立编写的 UTF-8 每行一句文本。不要采用模型识别结果作为参考答案。

预览提示绘制在副本上，保存的是未加文字的原始摄像头画面。文件按实际采集时间戳流式重采样为 25 fps；记录人脸、光线、口型运动和时间戳用于后续审计。单段默认最多 15 秒，输出目录默认配额 512 MiB，可用参数调整。质量检查不可用时会明确记录未知质量。未确认片段在正常退出或重录时删除。

`speaker` 必须是同一人的一致标识，`session` 表示一次真实录制场次，不能每条视频虚构一个新场次。train/dev/test 用独立目录，个性化模型使用同一人在不同日期的片段；通用模型使用不同说话人。test 句子不得来自训练或调参。默认清单不声称完成重叠审计。

采集工具的合成测试已通过；真实 macOS 摄像头权限、录制操作和 Windows 界面仍需人工验证。你稍后录制的实际片段才可用于验证这个目标。批量验收见[中文纯无声验收流程](CHINESE_EVALUATION.md)，本轮结果见[验证记录](VALIDATION.md)。
