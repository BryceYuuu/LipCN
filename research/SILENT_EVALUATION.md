# 真正无声中文：本地采集与验收

本轮交付采集、数据审计和原始错误分析能力，**没有得到真正无声准确率提升的证据，也没有更换默认模型**。目前尚无经确认的真实无声样本；发声视频删掉音轨、低声或耳语都不能冒充无声表达。

## 1. 本地采集

先关闭其他使用摄像头的程序，在终端显式运行：

```bash
lipcn collect --speaker speaker01 --session day01 --split train --lighting daylight-front
```

源码入口等价：`python -m research.scripts.collect_silent_chinese`，后面使用相同参数。`lipcn collect --help` 不访问摄像头、不加载识别模型。采集器没有麦克风或音频输入、上传和识别调用。本轮开发测试使用合成帧，没有开启摄像头或麦克风。

默认保存到 `~/Library/Application Support/LipCN/silent-data/`；`LIPCN_PROFILE_DIR` 可改变 profile，`--output /本地目录` 可指定整个数据集根目录。原始完整画面写到 `<根目录>/<train|dev|test>/videos/<id>.mp4`，元数据写到同级 `manifest.json`。图像保留完整脸部和背景，不裁成嘴部，不写入提示文字；采用 MP4 压缩和 25 fps 最近邻采样，并记录采集时间戳，**不是原生帧率的无损相机文件**。数据始终留在本地，不应提交 Git；文件可用普通本地视频播放器回放。

每句流程：

1. 输入真实打算无声表达的原句，再标注其中的人名、数字或中文数词。多人名/数字用 `|` 分隔，没有就回车。不要按模型预测倒推标签。
2. 让完整脸部入镜，光线照向正面；按 `SPACE` 开始该句，再按 `SPACE` 停止，最长默认 15 秒。
3. `P` 回放、`R` 丢弃重录。确认确实**没有发声或耳语**才按 `S`；确认原句逐字准确且人名/数字标注完整，再按 `Enter` 保存。
4. `Esc` 丢弃未确认片段并退出。正常退出清理未确认临时文件；异常断电可能留下 `.unconfirmed-*.mp4`，人工检查后可以删除，它们不在清单内。

`--date YYYY-MM-DD` 默认当天；必须填写真实日期。`--session` 表示同一天同次录制，换日期必须换 session；`--speaker` 使用固定匿名编号，不要因换日期或分组改名；`--lighting` 保存真实光线条件。清单同时保存人工确认、SHA256、完整帧信息、时间戳和日期。因为没有录音，工具无法自动证明“没有发声”，人工确认是证据边界。

可预先准备 JSON 提示词：

```bash
lipcn collect --speaker speaker01 --session day01 --split train --lighting daylight-front \
  --prompts research/examples/silent-prompts-train.json
```

提供 [train](examples/silent-prompts-train.json)、[dev](examples/silent-prompts-dev.json)、[test](examples/silent-prompts-test.json) 三份格式示例。人名都是虚构占位名。**这些少量公开提示仅演示格式，不是合格验收集**；独立验收需另行准备未用于调参的新句子和人物。`entities: []` 表示人工确认没有人名/数字，不能把未检查的标签标成已确认。不要混用阿拉伯数字与中文数词来美化分数，CER 会保留这种差异。

## 2. 冻结数据与独立审计

根目录固定分为 `train`、`dev`、`test`，同一人只能归属一个分组，归一化后相同的句子不能跨组；采集器保存前会重新校验已有清单。训练只读 train，模型/前处理/解码选择只看 dev；先冻结模型、参数、阈值和文件哈希，再由独立验收人员打开新 test。看过 test 后再修改方案，这批 test 就不再是新方案的独立验收集。单人多天样本可以做个人适配研究，但不能宣称跨人物可用。

清单的 `training_overlap_checked` 初始为 `false`，每次新增片段都会重置。采集完毕后，由操作者检查全部训练、调参历史与三份清单，确认没有重叠且 test 未被用于选择方案，**实际完成后**才将对应清单该字段设为 `true`，并保存审核说明。不要为了通过检查跳过人工审计。

```bash
python -m research.scripts.audit_silent_data \
  --train /本地数据/train/manifest.json \
  --dev /本地数据/dev/manifest.json \
  --test /本地数据/test/manifest.json \
  --output /本地报告/data-audit.json
```

工具逐文件重算 SHA256，核对声明和分组，拒绝未确认无声/标签、非 webcam 或仅嘴部数据、缺失日期、重复片段、跨组同人/相同视频/相同文本。跨人物测试的最低采集门槛沿用项目协议并补充跨日期：test 至少 50 条、5 人、每人至少两个真实日期。这只是最低数据门槛，**不是统计把握度或识别质量保证**。当人数或日期不足时仍保存完整问题报告，并返回退出码 2。

报告含 `manual_review_samples`：每人按 sample ID 的 SHA256 排序固定抽取两条，连同视频相对路径和内容哈希，便于另一个人独立回放检查完整脸部、标签、日期与采集记录。抽样不是所有标签的逐条核验，更不能仅凭无音轨证明未发声。`data_eligible=true` 也永远不会设置 `model_acceptance=true`。

## 3. 原始模型错误报告

向评测器输入视频，保存**未经改写的模型 top-1**。每个清单 ID 必须保留一条预测，包括失败或空输出；不能过滤困难样本。使用既有 `research/scripts/evaluate_chinese.py --help` 了解本地模型评测参数。下面工具只读取已保存预测和清单，不加载模型、不打开设备，也不改变输出：

```bash
python -m research.scripts.report_silent_errors \
  --manifest /本地数据/dev/manifest.json \
  --baseline /本地报告/baseline.json \
  --candidate /本地报告/candidate.json \
  --output /本地报告/errors.json
```

预测格式为 `{"samples": [{"sample_id": "…", "reference": "…", "raw": "…", "error": null}]}`。错误报告检查 ID 全覆盖、标签和已有身份/哈希一致性，重算字符错误率 CER、替换 S、漏字 D、多字 I、空输出、失败数，按人物、原句长度、真实日期、人名/数字句群分别统计，并给按人物重采样的成对差异区间。未标注人名/数字的样本归入 `unannotated`，不会被当成“没有”；人名/数字分组是**包含该实体的整句 CER/S/D/I，不是实体专属准确率**，两种分组可以重叠。仅设置数据字段不能证明原始预测未经过外部改写，原始推理日志仍需保留核验。

模型替换至少需要冻结后、未参与选择的真正无声 test 证据，结合事先定下的绝对 CER/整句正确率、失败、覆盖和延迟要求。现有 `research/evaluation.py` 的阈值是项目验收门槛，不是当前模型已达到的性能。不能以三种润色、候选重排或删改原句取代模型原始准确率。

## 4. 2026-10-13 现有证据盘点

| 本地项目数据 | 条数 / 人数 | 能说明什么 |
| --- | --- | --- |
| Round4 Chinese-LiPS train / supported | 1200 / 40；1105 / 40 | 公开发声嘴部视频，可用于训练 |
| Round4 dev / 历史 test | 100 / 11；100 / 10 | 发声数据；历史 test 已被查看，不可当新验收 |
| Round5 fresh / full-face holdout | 40 / 10；20 / 10 | 发声嘴部 / 全脸视频，均是历史证据 |
| Round6 fresh holdout | 20 / 10 | 发声嘴部视频，历史证据 |
| Round8 train / dev | 240 / 40；33 / 11 | 发声开发实验，没有真实录制日期或人工实体标注 |
| 人工确认真正无声、跨人物跨日期样本 | 0 | 当前无法验证真正无声准确率提升 |

Round8 使用已有 adapter 初始化，最后两个 encoder blocks，joint loss、学习率 `1e-6`、seed 7、240 步/一轮。历史报告记录参数实际变化，当前本机 adapter 的 SHA256 为 `341040600460c5a0b30064c017f574f6d4757a79d04e759adf31a2166db246d5`。本轮只重算已有预测，没有训练或进行新推理。

[可复核汇总](docs/round8-raw-error-audit.json) 记录输入清单/报告哈希和按人、句长分组。33 条 dev 共 1432 个参考字符：

| 原始错误 | 基线 | Round8 candidate |
| --- | ---: | ---: |
| 替换 / 漏字 / 多字 | 706 / 66 / 56 | 704 / 66 / 55 |
| 总错字 / CER | 828 / 57.821% | 825 / 57.612% |
| 整句正确 | 0 / 33 | 0 / 33 |
| 41–60 字长句总错字 | 621 | 623 |

按 11 人做 5000 次 bootstrap，候选减基线 CER 的 95% 区间为 **−1.089 至 +0.392 个百分点**，包含零；而长句有回退。该候选本身是在 dev 上选择的，区间只是历史诊断，不能证明可推广的增益。全部是发声数据，因此不能称为无声准确率改善。下一步真正模型实验应等人工采集、审核和冻结的新数据具备之后再进行。
