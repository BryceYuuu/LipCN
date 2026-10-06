# Mandarin research / 中文唇语研究

## English

This directory contains optional data preparation, adaptation, decoder comparison,
evaluation, benchmark reports, and manual recording tools. The default legacy
runtime in `lipflow/` does not import these commands, and the research directory
is excluded from its application wheel. The explicitly selected
[LipCN research desktop entry](../desktop/mandarin/) reuses the CNVSRC reader
and scoring helpers from this directory with separately obtained local weights.
The experiments do not automatically replace the default legacy CMLR runtime
or redistribute model weights.

Use a checkout and its installed Python environment. Examples below run from the
repository root. An absolute script path also works from another directory; relative
input/output arguments resolve against the directory from which you invoke it.

```bash
python research/scripts/evaluate_chinese.py --help
python research/scripts/train_chinese_joint_adapter.py --help
python research/scripts/compare_chinese_decoders.py --help
```

See the [public-data workflow](docs/chinese-public-adaptation.md),
[silent evaluation protocol](docs/CHINESE_EVALUATION.md), and
[same-video model comparison](docs/PAIRED_MODEL_COMPARISON.md), alongside the
[historical measured benchmark](docs/MANDARIN_BENCHMARK.md). New comparison tools:

```bash
python research/scripts/fetch_chinese_holdout.py --help
python research/scripts/benchmark_chinese_models.py --help
python research/scripts/select_chinese_visual.py --help
python research/scripts/compare_chinese_cohorts.py --help
python research/scripts/diagnose_chinese_candidates.py --help
python research/scripts/select_chinese_search.py --help
```

The [candidate-search guide](docs/SEARCH_DIAGNOSTICS.md) separates raw edit errors,
offline oracle candidate bounds, and development-only selection of a larger
reverse-scoring pool or CTC shortlist. Application defaults remain unchanged.
The [measured search results](docs/SEARCH_EXPERIMENT_RESULTS.md) record both gains
and regressions; this round retains the original settings.

The existing numerical benchmark
and frozen experiment identities are historical evidence, preserved unchanged.
Relocating executable sources changes the continuation identity: an old saved
training state must be resumed with its original frozen source checkout. This
checkout rejects that state rather than treating a changed implementation as the
same experiment. A new experiment must select on train/dev and freeze its settings
before evaluating an independent test set.

Recording is an optional, separate human-operated command:

```bash
python research/scripts/collect_chinese.py --output samples/private/train \
  --speaker speaker01 --session day01 --split train
```

The operator enters the reference before recording and confirms the exact words
and silent articulation before saving. SPACE starts/stops, R discards/retakes, and
Esc discards/exits. The collector records images only; it uses neither audio nor a
recognition model to generate labels. `--help` does not request camera permission
or open a device. See the [collection guide](docs/CHINESE_ADAPTATION.md).

Keep datasets, private recordings, model weights, adaptation tensors, and raw
predictions outside Git. Their source licenses remain applicable; see [NOTICE](../NOTICE).

## 中文

本目录集中存放可选的数据准备、适配训练、解码比较、评测、基准报告和人工录制
工具。`lipflow/` 中的默认旧版运行入口不会导入这些命令，其应用 wheel 也不包含研究目录。
显式选择的[LipCN 中文研究桌面入口](../desktop/mandarin/)会复用这里的 CNVSRC 读取和评分组件，
并使用操作者另行取得的本地研究权重。研究实验不会自动替换默认旧版 CMLR 模型，
也不会随源码分发权重。

使用源码检出目录及已安装依赖的 Python 环境。上面的命令示例从仓库根目录运行；
也可以在任意目录使用脚本的绝对路径。输入、输出的相对路径以执行命令时的目录
为准。研究流程见[公开数据操作说明](docs/chinese-public-adaptation.md)、
[纯无声验收协议](docs/CHINESE_EVALUATION.md)、[同视频模型对照](docs/PAIRED_MODEL_COMPARISON.md)
和[历史实测报告](docs/MANDARIN_BENCHMARK.md)。新增命令覆盖排除既有说话者的数据
采样、共享画面的三模型对照、开发集图像处理选择及按场景/说话者分组的评测，
可使用上面的 `--help` 检查入口。

[候选搜索与错误诊断](docs/SEARCH_DIAGNOSTICS.md)进一步统计替换、漏字、多字、
人物及句长差异，并用开发集比较反向评分池和 CTC 预筛范围。应用默认行为不变。
[本轮实测](docs/SEARCH_EXPERIMENT_RESULTS.md)记录了改善和回退，最终保留原配置。

已有基准数值和冻结实验身份保留为原始历史证据。移动可执行源码会改变续训身份：
历史保存状态只能在原来的冻结源码目录中恢复。本目录会拒绝将其当作同一实验
续训。新实验仍需先用 train/dev 选择参数，在独立 test 评测前冻结完整配置。

录制是独立的人工操作，可按上面的 `collect_chinese.py` 示例显式启动。录制前
输入参考句，保存前确认嘴型逐字一致且未发声或耳语。SPACE 开始/停止，R 丢弃
重录，Esc 丢弃退出。工具仅录制画面，不采集音频，不使用识别模型生成标签；
`--help` 不会请求相机权限或打开设备。详细操作见[采集说明](docs/CHINESE_ADAPTATION.md)。

数据集、私人录像、模型权重、适配参数和原始预测应保存在 Git 之外，继续遵守
原来源许可，见[NOTICE](../NOTICE)。
