# 中文模型同视频对照 / Mandarin models on the same videos

范围说明 / Scope: 本页记录历史冻结实验；涉及“应用默认入口”“未装入 GUI”的描述均指当时的默认旧版运行方式。新增 `desktop/mandarin/` 是显式选择的 LipCN 研究桌面入口，不改变这些历史指标或模型、数据许可。 This page records frozen historical experiments. Statements about the default application or absence of GUI integration refer to the legacy runtime at the time. The explicitly selected `desktop/mandarin/` LipCN research desktop entry does not change these historical measurements or model/data licenses.

## 中文

本轮把当前 CMLR 运行配方、原始 CNVSRC2025 模型、此前固定的 CNVSRC 编码器适配器放到同一批视频上比较。新增工作是我们编写的视觉稳定化实验、开发集选择、有界数据采样、共享视觉输入、分组评测及反向解码批处理。它们位于独立研究目录，应用没有切换模型，也没有根据测试结果自动启用新处理。

**状态：已完成本地冻结对照。适配配方在两组数据及合成退化下都优于 CMLR 与原始 CNVSRC；整句正确率仍为 0，不能据此宣布自由无声中文可用。汇总数据见 [round5 JSON](benchmarks/mandarin-round5-summary.json)。**

### 数据覆盖与说话者身份

全部视频来自 [Chinese-LiPS 作者发布的数据](https://huggingface.co/datasets/BAAI/Chinese-LiPS)，固定 revision `db96948538811029011eee44602438a26710ecd9`。采样按 seed 对说话者及样本身份进行哈希排序，再轮流选择，不按标签难度、句长或模型识别结果筛选。作者原始标签保持不变；错误、拒绝、空输出和失败都留在计分分母中。

| 数据组 | 视频 / 说话者 | 原始分区与画面 | 本地身份隔离 | 能支持的结论 |
| --- | --- | --- | --- | --- |
| fresh-holdout | 40 / 10 | 官方 train；96×96 嘴部，25fps | 排除此前训练、开发及已查看测试中的全部 82 名说话者 | 尚未接触过的本地说话者对照；不是官方测试集 |
| full-face-holdout | 20 / 10 | 官方 test；1280×720 原始 FACE 视频，30fps | 排除全部 62 名本地训练/开发说话者；其中 9 人此前参与过测试 | 未参与本地适配的说话者及全画面跟踪测试；不是全新的测试说话者组 |
| 开发子集 | 33 / 11 | 现有官方 valid 中每人固定 3 段 | 只用于本轮预先限定的图像处理选择 | 开发选择，不能混作新测试成绩 |

历史接触记录包含 1652 个样本身份，仅使用身份及来源文件哈希恢复。fresh-holdout 的 10 名说话者与这些记录均不重叠。full-face-holdout 的 10 人未参与本地训练或开发，但 9 人的其他视频已经测试过，这一差别保留在来源说明中。两组都不能保证与原始 CMLR/CNVSRC 模型的预训练内容完全独立，基础预训练重叠仍为未知。

已检查 60 个本地文件的 SHA-256、ZIP 成员 CRC、首帧解码、分辨率及帧率。40 个处理后嘴部文件只有视频流；20 个原始 FACE 容器均带有内嵌音轨，下载这些容器也下载了该音轨。识别和视觉处理只读取视频帧，不解码音频，不调用 ASR、麦克风或 LLM；没有额外下载 WAV 或幻灯片。不能把“没有使用音频”写成“原始 FACE 文件没有音频”。

本轮所有视频都是正常发声时的口型。**已验证的真实 webcam 样本为 0、刻意无声发音样本为 0、日常自由对话样本为 0。**无法证实的采集设备和表达方式保留 `unknown`。全画面公开视频不等于用户摄像头，合成模糊或曝光变化也不等于真实无声口语测试。

### 三个固定模型配方

| Arm | 权重 | 解码配方 |
| --- | --- | --- |
| cmlr | 作者 CMLR 视觉及语言模型，关闭个人权重 | 当前中文入口：beam 10、CTC 0.1、LM 0.3、length bonus 0.3 |
| cnvsrc_base | 原始 CNVSRC2025 基座 | beam 40、CTC 0.5、reverse 0、n-best 10、无外部 LM |
| cnvsrc_candidate | 同一基座，加此前 joint 第 2 轮编码器适配器 | beam 40、CTC 0.1、reverse 0.3、n-best 10、无外部 LM；图像处理只按本轮开发集选择 |

候选的权重及解码参数在本轮测试前固定，未按新测试重训或调参。它与原始基座之间的对照衡量整个固定配方，不能把差异全部归因于本轮新增图像处理、适配器或某一个解码参数。CMLR 也保留既有语言模型配方，因此是实际运行配方对照，不能称作参数量、语言模型或训练数据完全匹配的架构实验。

CNVSRC 基座 SHA-256 为 `577cd9558eea111683a406bc25d69c7161cdb79534c2273fc0d0f044c356231c`；此前固定适配器 SHA-256 为 `e85ccabd47dc62fbcfbbfc0f4a040e322f4e2f86ba15714b5e341a2e950c2457`。CMLR 的四个本地文件、源码、开发选择和每份测试清单都由运行生成的 `frozen-plan.json` 保存哈希，最终记录已随汇总 JSON 保存。

### 开发集上的视觉处理选择

预先限定 `identity`、`stabilize_mild`、`stabilize_moderate` 三组处理，分别测试原始画面和固定 `synthetic_mild` 合成退化；seed 0。退化模拟分辨率损失、模糊与曝光波动，保留原始正常发声的口型。稳定化仅对整段视频中的亮度和对比度做有界调整，不移动像素坐标、不插帧、不依据文字调整图像。

33 段开发视频共 1432 个计分字符；下表来自本地开发报告及 `development-selection.json`，不是新测试结果。每组的两种条件全部保留，不能只展示较好的原始画面结果。

| 处理 | 原始画面错误数 / CER | 合成退化错误数 / CER | 合计错误 / 字符 | 选择 |
| --- | --- | --- | --- | --- |
| identity | 820 / 57.26% | 847 / 59.15% | 1667 / 2864 | 保留 |
| stabilize_mild | 803 / 56.08% | 864 / 60.34% | 1667 / 2864 | 合计没有严格改善，未启用 |
| stabilize_moderate | 822 / 57.40% | 877 / 61.24% | 1699 / 2864 | 原始画面及合计均退步，未启用 |

六次开发运行均为 0/33 整句正确、无推理失败、候选覆盖 100%。选择规则要求合计原始错误数严格减少，原始画面 CER 不变差，两种条件的整句正确数和覆盖率都不降低，且没有推理失败。最终选择 **identity**；代码中保留另外两种研究工具，但未将它们宣布为有效优化或启用到应用。

另一个独立改动把已有候选的反向打分合成一个批次。真实权重上的固定开发位置 0、8、16、24 共 4 段验证保持排名、第一候选文本和动作一致，最大反向分数差不超过 `3.82e-6`；固定容差 `0.001` 时接近排序边界的结果退回串行计算。开发测得反向打分子步骤中位加速约 2.14 倍，**这不是整个识别流程的加速比例，也不是 CER 改善**。批处理只有在提供匹配的冻结开发验证记录时才允许用于对照。

### 相同输入、计分和延迟

`benchmark_chinese_models.py` 先读取每段视频一次，将嘴部灰度图、跟踪、对齐和采样结果存入有内存上限的只读缓存，然后给三个模型复用。模型内部归一化仍遵循各自配方。文字在图像入口被清空，仅保留在最后的评测层；没有参考文本提示或 LLM 纠错。

每份预测记录包括原始像素哈希、实际输入像素哈希、形状、类型及按样本身份确定的 seed。分组对照验证源视频身份、参考文本和这些像素证据。相同处理必须拥有相同输入像素；处理不同的实验只声称源像素相同。本轮冻结选择为 identity，使三个配方在同一条件下接收相同的准备后图像。合成条件必须具有相同退化和 seed，不能和原始条件混算为同一个测试。

本地运行固定 PyTorch CPU 线程数为 4；支持时编码器使用 MPS，解码在 CPU。每段共享准备耗时都计入三个模型各自的处理时间，模型验证、加载和预热另记 startup。由于准备结果被缓存、模型依次运行，这些数值是固定协议下的处理测量，不是三次独立冷启动或 GUI 完整输入延迟。报告同时保留 CER、整句正确率、候选覆盖率、失败数和延迟；候选分数不解释为正确概率。

### 冻结测试结果

CER 是字错误率，越低越好。共 60 段不同视频、20 名说话者，完成 9 组报告、300 次识别。所有失败和拒绝都参与计分。

| 数据组 / 条件 | Arm | 错误 / 字符 | CER | 整句正确 | 候选覆盖 | 推理失败 | warm p95 | RTF p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 陌生人物 / 原始 | cmlr | 1605 / 1686 | 95.20% | 0/40 | 100.0% | 0 | 1.419s | 0.139 |
| 陌生人物 / 原始 | cnvsrc_base | 1057 / 1686 | 62.69% | 0/40 | 97.5% | 0 | 5.887s | 0.450 |
| 陌生人物 / 原始 | cnvsrc_candidate | 949 / 1686 | 56.29% | 0/40 | 100.0% | 0 | 5.849s | 0.424 |
| 陌生人物 / 合成退化 | cmlr | 1606 / 1686 | 95.26% | 0/40 | 97.5% | 0 | 1.357s | 0.154 |
| 陌生人物 / 合成退化 | cnvsrc_base | 1064 / 1686 | 63.11% | 0/40 | 100.0% | 0 | 5.897s | 0.437 |
| 陌生人物 / 合成退化 | cnvsrc_candidate | 982 / 1686 | 58.24% | 0/40 | 100.0% | 0 | 6.220s | 0.451 |
| 完整人脸 / 原始 | cmlr | 821 / 853 | 96.25% | 0/20 | 95.0% | 0 | 4.339s | 0.335 |
| 完整人脸 / 原始 | cnvsrc_base | 559 / 853 | 65.53% | 0/20 | 100.0% | 0 | 8.771s | 0.649 |
| 完整人脸 / 原始 | cnvsrc_candidate | 474 / 853 | 55.57% | 0/20 | 100.0% | 0 | 8.496s | 0.601 |

下表是适配配方相对原版 CNVSRC 的配对变化。负值表示少错；95% 区间按整个说话者重采样 5000 次，seed 0。每组仅 10 名说话者，共享录制条件，区间不能外推为实际无声口语表现。

| 条件 | 少错字数 | CER 变化（百分点） | 95% 区间（百分点） | 改善 / 退步 / 持平说话者 |
| --- | --- | --- | --- | --- |
| 陌生人物 / 原始 | 108 | -6.41 | [-10.07, -3.05] | 9 / 1 / 0 |
| 陌生人物 / 合成退化 | 82 | -4.86 | [-7.81, -1.73] | 9 / 1 / 0 |
| 完整人脸 / 原始 | 85 | -9.96 | [-14.85, -5.34] | 10 / 0 / 0 |

三模型同条件的像素证据全部一致：陌生人物原始 40/40、合成退化 40/40、完整人脸 20/20。训练/开发与两组测试的已提供样本身份、源文件及说话者均无重叠。基础预训练来源仍未知。完整人脸的相机跟踪准备全部成功；此处只证实录制视频处理，没有打开实体摄像头。

所有报告推理失败数均为 0。原版 CNVSRC 在陌生人物原始组有 1 次“不清楚或重复”拒绝；CMLR 在合成退化组、完整人脸组各有 1 次同类拒绝。拒绝的原始文本保留参与 CER，不等同模型异常。适配版两组原始和合成条件均提供候选，但全部仍需确认，候选覆盖不是正确率。

适配版 100 次候选评分中 97 次使用批量计算，3 次因最终分数接近而完整退回串行，没有批量异常退回。所有配方均没有整句正确的样本；适配版 warm p95 为 5.85–8.50 秒（含共享画面准备），本轮没有证明整个流程提速。它优于 CMLR 的结果是固定配方的对照证据，新增批处理不承担准确率改善的归因。

实际测试使用的 58 个冻结源码文件已在本地保留。运行完成后只加强了评测入口的 schema、开发报告身份、反向证明依赖及二进制资产检查，不改模型、候选生成或评分；完整报告保留运行当时的源码哈希。

### 可复现操作与来源

把下面 `/local/...` 路径替换为自己的已安装环境、当前源码、合法取得的权重和本地证据目录。命令通过已冻结的研究脚本运行；研究许可开关表示操作者已经阅读并接受原始来源条款。每次模型对照需使用一个尚未包含结果的新输出目录。

```bash
/local/lipflow/.venv/bin/python /local/lipflow/research/scripts/fetch_chinese_holdout.py \
  --accept-noncommercial-license --output-dir /local/round5/fresh-holdout \
  --source-split train --media mouth_roi --count 40 --speakers 10 \
  --seed lipflow-chinese-local-holdout-v1 --max-download-mb 100 \
  --exclude-manifest /local/round5/prior-local-exposure.json

/local/lipflow/.venv/bin/python /local/lipflow/research/scripts/fetch_chinese_holdout.py \
  --accept-noncommercial-license --output-dir /local/round5/full-face-holdout \
  --source-split test --media full_face --count 20 --speakers 10 \
  --seed lipflow-round5-full-face-v1 --max-download-mb 100 \
  --exclude-manifest /local/round5/prior-development-exposure.json

/local/lipflow/.venv/bin/python /local/lipflow/research/scripts/benchmark_chinese_models.py \
  --accept-research-license --manifest /local/round5/fresh-holdout/test.json \
  --cmlr-dir /local/round5/cmlr \
  --checkpoint /local/model_avg_cncvs_2_3_cnvsrc.pth \
  --source-dir /local/CNVSRC2025 \
  --adapter /local/encoder_adapter.pth \
  --selection /local/evidence/development-selection.json \
  --ctc-weight 0.1 --reverse-weight 0.3 \
  --reverse-scoring batched --reverse-validation /local/evidence/reverse-batch-validation-bound.json \
  --cpu-threads 4 --device auto --seed 0 --synthetic-stress \
  --output-dir /local/results/fresh-comparison-new

/local/lipflow/.venv/bin/python /local/lipflow/research/scripts/compare_chinese_cohorts.py \
  /local/results/fresh-comparison-new/cmlr-clean.json \
  /local/results/fresh-comparison-new/cnvsrc_candidate-clean.json \
  --manifest /local/round5/fresh-holdout/test.json \
  --train-manifest /local/round5/all-local-train-identities.json \
  --dev-manifest /local/round5/all-local-dev-identities.json \
  --metadata /local/round5/fresh-holdout/cohort-metadata.json \
  --seed 0 --resamples 5000 --output /local/results/fresh-paired.json
```

全画面对照替换测试清单与输出目录即可；比较原始 CNVSRC 时，将候选报告换为 `cnvsrc_base-clean.json`。只有缺少匹配的批处理验证时使用 `--reverse-scoring sequential`；改变冻结输入需开启新运行，不能续写已有对照结果。数据采样还接受多个 `--exclude-manifest`，应覆盖自己的全部既有实验，不能只沿用这次 82 人的记录。

新增研究模块由我们独立编写，不导入作者仓库的 Python 实现。模型兼容层使用 Lipflow 已有的 ESPnet 实现，读取已校验的原始配置和词表，并加载作者发布的预训练权重；此前适配器也是这些权重的派生物。这是有署名的兼容和对照研究，**不是自研全部模型或从零训练**。保留 [NOTICE](../../NOTICE) 中的来源及原有许可；作者权重、配置、词表、适配器和数据各自的条款仍适用。研究代码的独立实现不能取消权重的研究及非商业限制。

来源：[CMLR 模型项目](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages)、[CNVSRC2025 项目](https://github.com/liu12366262626/CNVSRC2025)、[CNVSRC2025 作者权重](https://huggingface.co/ReflectionL/CNVSRC2025Baseline)、[Chinese-LiPS 作者数据](https://huggingface.co/datasets/BAAI/Chinese-LiPS)。仓库只保存源码、协议和汇总指标；视频、权重、逐段参考文本及原始预测留在本地，不随代码发布。本轮没有更改应用默认模型，也没有提交或推送这些研究修改。

## English

This local research comparison evaluates the current CMLR recipe, the original CNVSRC2025 base, and the previously frozen CNVSRC joint encoder adapter on identical videos. New work comprises independently written visual experiments, development-only selection, bounded sampling, shared visual preparation, cohort reporting and batched reverse scoring. It does not switch the application model or activate a candidate from test results.

**Frozen local runs are complete. The candidate improves raw CER in both cohorts and synthetic stress, but exact-sentence accuracy remains zero. This does not establish usable free silent Mandarin. Aggregate evidence: [round5 JSON](benchmarks/mandarin-round5-summary.json).**

### Data and scope

The publisher is [BAAI/Chinese-LiPS](https://huggingface.co/datasets/BAAI/Chinese-LiPS), pinned at revision `db96948538811029011eee44602438a26710ecd9`. Speakers and clip IDs are hash-ranked with a fixed seed and selected round-robin, without selecting by text difficulty, length or model output. Original labels remain unchanged. Failed, rejected and empty outputs remain in scoring.

- **Fresh local holdout:** 40 clips from 10 speakers, excluding all 82 speakers previously used in local training/development or inspected tests. The files come from the publisher's **train** partition, with 96×96 mouth crops at 25fps. This is a locally untouched cohort, not the publisher's official test set.
- **Full-face cohort:** 20 original FACE clips from 10 official-test speakers, all 1280×720 at 30fps. All 62 historical local training/development speakers are excluded. Nine of these 10 speakers previously appeared in local tests; only one is entirely new locally. This cohort evaluates recorded full-frame tracking and speakers unseen by local adaptation, rather than another wholly fresh test-speaker population.
- **Development subset:** a fixed 33 clips, three per each of 11 existing development speakers, used only for the predeclared preprocessing comparison.

The identity recovery contains 1652 source IDs and source-file hashes, without predictions or reference text. File SHA-256, ZIP-member CRC, decoded dimensions and frame rates were checked for all 60 test files. The 40 mouth crops have video-only streams. All 20 original FACE containers contain embedded audio: downloading them downloads those streams too. Evaluation consumes video frames only and does not decode audio, use ASR/LLM cleanup, or access microphones; no separate WAV or slide assets were downloaded.

These are **voiced recordings**. Verified real-webcam clips, deliberate silent-articulation clips and everyday free-conversation clips are each **zero**. Unverified capture and speaking-style metadata remain `unknown`. A full-face recording or synthetic blur/exposure stress does not establish real webcam or silently mouthed conversational accuracy. Original CMLR/CNVSRC pretraining overlap is unknown for both cohorts.

### Frozen recipes and development choice

The CMLR arm retains the application's published visual model and LM: beam 10, CTC 0.1, LM 0.3 and length bonus 0.3, without personal weights. The original CNVSRC arm uses its published base, beam 40, CTC 0.5, reverse 0, n-best 10 and no external LM. The candidate uses the same base and previously fixed joint epoch-2 encoder adapter, beam 40, CTC 0.1, reverse 0.3 and n-best 10, without an external LM. These are complete recipe comparisons; differences cannot isolate architecture, the adapter, decoding, or this iteration's preprocessing. The models were not trained from scratch this iteration.

The fixed development experiment evaluates `identity`, `stabilize_mild` and `stabilize_moderate` under clean and `synthetic_mild` conditions, seed 0. The pixel-only stabilization bounds changes in clip brightness/contrast, preserves coordinates and frame order, and never uses labels. Synthetic resolution, blur and exposure degradation preserves the original voiced articulation and is explicitly diagnostic.

The Chinese development table above is also the authoritative English table: each condition has 33 clips and 1432 scored characters. Identity has 820 clean and 847 stressed errors; mild stabilization has 803 and 864; moderate stabilization has 822 and 877. Mild stabilization improves clean CER but ties the pooled error count and worsens synthetic stress. Moderate stabilization regresses. All six runs have zero exact sentences, zero inference failures and 100% candidate coverage. The predeclared gate requires strictly fewer pooled errors without lower clean accuracy, exact matches or coverage. **Identity is therefore retained**, and neither stabilization preset is promoted or enabled in the application.

Batched reverse scoring is a separate execution improvement. Fixed real-checkpoint development positions 0, 8, 16 and 24 preserve candidate order, top text and action, with maximum reverse-score difference below `3.82e-6`. Near ties use sequential scoring with a fixed `0.001` tolerance. The measured isolated reverse-scoring median speedup is approximately 2.14×, not end-to-end latency or CER improvement. Benchmark activation requires a matching frozen development-validation record.

### Shared pixels, timing and reproduction

The benchmark reads each clip once into a bounded, read-only visual cache, sharing the same tracking, alignment, grayscale crops and resampling across arms. References are removed at the image boundary and used only for scoring. Per-sample evidence includes original/prepared pixel hashes, shape, dtype and deterministic seed. Identical preprocessing must produce identical prepared pixels. Different preprocessing is reported as a comparison of the same source pixels rather than identical final inputs; the frozen identity choice currently provides identical prepared images for all three recipes. Clean and synthetic conditions are reported separately.

PyTorch uses four CPU threads, MPS encoding where supported and CPU decoding. Every model's per-sample processing time is charged the full measured shared preparation cost; model verification/loading/warmup is reported separately as startup. Cached preparation and serial arm execution do not simulate independent cold-start or GUI input latency. CER, exact sentences, candidate coverage, failures and warm latency are all retained; decoder scores are not calibrated correctness probabilities. Speaker-cluster paired intervals should accompany completed results, with their small-cohort scope stated.

The commands above are reproducible templates: replace `/local/...` with the installed environment, current checkout, legally obtained source artifacts and local evidence paths. Each benchmark needs a new output directory. The fetcher accepts repeated `--exclude-manifest` options so another operator can exclude all their own prior experiments. The comparison script consumes frozen prediction JSONs, training/development identity manifests and explicit metadata without running a model. Change the cohort manifest/output directory for full-face data and the report arm for base-model comparisons. Without matching batched validation, use sequential reverse scoring in a new frozen run.

Our new research modules do not import Python implementation code from the authors' repositories. Compatibility uses Lipflow's existing ESPnet implementation, verified source configuration/vocabulary and attributed released weights. The adapter is derived from those weights. Independent experimental code does not remove model/data licensing restrictions or make the entire model an original from-scratch implementation. Preserve [NOTICE](../../NOTICE) and each original license; the weights and derived adapters remain restricted to their permitted research/non-commercial uses. Only code, protocols and aggregate measurements belong in the repository; media, weights, per-clip references and raw predictions stay local. This iteration does not switch the application model, commit changes or push them.

### Completed paired results

The measured table above applies to both language versions: 60 distinct clips from 20 speakers, nine model reports and 300 recognition executions. On fresh speakers, raw CER is 95.20% for CMLR, 62.69% for original CNVSRC and 56.29% for the candidate. Under synthetic stress it is 95.26%, 63.11% and 58.24%; on full-face recorded footage it is 96.25%, 65.53% and 55.57%. All reports have zero inference failures and zero exact sentences. Rejected raw predictions remain in scoring.

Candidate-minus-base CER changes are −6.41 percentage points (95% speaker-cluster interval [−10.07, −3.05]) on fresh clean footage, −4.86 ([−7.81, −1.73]) under synthetic stress, and −9.96 ([−14.85, −5.34]) on full-face footage. Improvements cover 9/10, 9/10 and 10/10 speakers respectively. Five thousand whole-speaker resamples use seed 0; ten speakers and shared recording conditions limit interpretation. Every same-condition pair verifies identical prepared-pixel evidence: 40/40 clean, 40/40 synthetic and 20/20 full-face. Provided local train/dev identities, source files and speakers do not overlap these tests; base pretraining overlap remains unknown.

The candidate uses batch reverse scoring in 97/100 utterances and whole-utterance sequential fallback for three close score gaps, with no batch error. Warm p95 is 5.85–8.50 seconds including shared preparation. This is neither proof of end-to-end speedup nor a silent-webcam result. Original CNVSRC rejects one fresh-clean prediction as unclear/repetitive; CMLR rejects one synthetic and one full-face prediction for the same reason. All remain in the CER denominator.

The 58 source files actually used for inference are archived locally. Subsequent changes harden only schema, dev identities, reverse-proof dependencies and binary-asset verification; they do not change model inference, candidate generation or scoring. Original report hashes retain the actual runtime source identity. Verified real silent webcam and everyday conversation remain zero, so they have no claimed accuracy metric.

### 本地验证 / Local validation

最终测试 **678 通过、13 跳过**；跳过项涉及 Windows 原生交互、缺少特定测试权重/视频及真实粘贴验证。独立研究对照已实际加载三套权重。加严后的基准入口复跑一段已测完整人脸视频，三套模型的原始文本、动作、计分和视频身份与原记录一致；这是入口回归，不是新增准确率样本。入口 `--help` 在源码目录外以 `Python -S` 运行通过，编译与 diff 格式检查通过。

Final suite: **678 passed, 13 skipped** for Windows-native interaction, specific test artifacts and real-paste checks. All three research-model weight sets were exercised independently. The hardened benchmark repeated one already tested full-face clip with identical raw text, action, scoring and video identity for every arm; this checks the entry point and is not new accuracy evidence. Standalone help outside the checkout, compilation and diff checks passed.
