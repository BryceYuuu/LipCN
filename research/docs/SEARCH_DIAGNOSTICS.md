# 候选生成与错误诊断 / Candidate generation and error diagnostics

## 中文

本轮先分析既有开发视频，再做两个独立的搜索实验。模型、词表、编码器适配参数、
CTC 权重与反向评分权重保持一致；没有训练新权重，也没有用参考句改写识别结果。
这些命令只属于 CNVSRC 研究入口，英文默认输入及中文 CMLR 应用入口不受影响。

### 先区分错误现象与原因

`diagnose_chinese_candidates.py` 从保存的原始报告计算替换、删除、插入、整句正确率、
输出/参考长度比，以及按人物、参考句长、记录时长分组的统计。使用与主评测相同的
NFKC/casefold 字符规则，去掉标点和空格；S/D/I 使用固定的最小编辑距离回溯规则。
同一最小距离可能有多种对齐，因此 S/D/I 不是唯一的语言学解释。

前 1/3/5/10 个候选的 oracle CER 只在预测完成后用参考答案计算。它表示这份候选
列表能够达到的错误下界，不能直接用于输入、训练目标或正确概率。候选重复不改变
原始排名；缺失候选时保留原始输出参与分母，并明确报告证据缺失。候选之间的编辑
距离和中心候选排名仅描述列表内部的一致性，一致的候选也可能全都错。
本轮保存的是重排后返回的前 10 个候选；oracle@10 不代表内部 40 个候选的 oracle。

输入报告必须覆盖冻结清单的所有样本，逐条核对身份、参考文本、人物及源视频哈希。
失败、拒绝、空文本均不移出分母。诊断不会打开视频，时长分组使用报告中记录的
秒数，不独立验证拍摄时长；输出不包含原句或候选全文。

### 两个独立搜索参数

| 参数 | 原有行为 | 可选实验 | 改变的环节 |
| --- | --- | --- | --- |
| `--candidate-pool-size` | 等于 `--nbest`，本轮基线为 10 | 40 | 收集更多已生成候选，全部反向评分后再返回前 10 个 |
| `--pre-beam-ratio` | 1.5；beam 40 时预筛 60 个词表项 | 3.0；预筛 120 项 | 让更多字进入 CTC 前缀评分，然后仍保留 beam 40 |

扩大反向评分池不会扩大前向 beam；扩大预筛选也不会扩大最终 beam。默认参数不变。
候选池要求 `nbest <= pool <= beam`；预筛比例限于有限数值 `[1,4]`。CTC 缓冲区
估算超过现有 1 GiB 研究预算时，在创建 scorer 前报错，不悄悄缩小参数。这个估算
不是整个模型进程的内存上限。

完整 beam 的强制 EOS 检查继续保留，即使问题候选落在返回列表之外也不会漏掉。
批量反向评分存在异常或整个候选池任意相邻最终分数差小于 `1e-3` 时，整池回退
串行评分，再截取输出数量。报告记录申请和实际收集的池大小。

### 运行与选择

下面路径均为示例，替换为自己合法取得的本地数据和权重。相同的开发清单分别跑
`pool=10/40`、`pre-beam-ratio=1.5/3.0` 四种组合；每种同时跑 `clean` 和
`synthetic_mild`，其余参数固定。合成退化不能代表真实摄像头或刻意无声说话。

```bash
python research/scripts/evaluate_cnvsrc.py --accept-research-license \
  --checkpoint /local/model_avg_cncvs_2_3_cnvsrc.pth \
  --source-dir /local/CNVSRC2025 --adapter /local/encoder_adapter.pth \
  --manifest /local/dev.json --output /local/pool40-clean.json \
  --device mps --beam-size 40 --ctc-weight 0.1 --reverse-weight 0.3 \
  --reverse-scoring batched --nbest 10 --candidate-pool-size 40 \
  --pre-beam-ratio 1.5 --visual-preprocessing identity --visual-stress clean

python research/scripts/diagnose_chinese_candidates.py /local/pool40-clean.json \
  --manifest /local/dev.json --output /local/diagnostics.json

python research/scripts/select_chinese_search.py --manifest /local/dev.json \
  --baseline-clean /local/baseline-clean.json \
  --baseline-stress /local/baseline-stress.json \
  --candidate-clean /local/pool40-clean.json \
  --candidate-stress /local/pool40-stress.json \
  --output /local/search-selection.json
```

可以重复传入成对的 candidate 参数。选择器只接受开发集，绑定权重、适配器、其他
解码参数以及相同的处理后像素；只允许上述两个搜索参数变化。必须减少两种条件
的合计原始错误，且任一条件的错误、整句正确数、候选覆盖和失败数不得退步。
并列时保留预先列在前面的配置。冻结选择结果之后再运行新测试，不能按测试成绩
反复调参。评测命令退出码 `2` 表示未达到既定产品验收条件，并不表示推理崩溃。

## English

This round adds stored-candidate diagnostics and two independently configurable
search controls to the optional CNVSRC research path. It does not train new
weights, rewrite predictions using references, change English defaults, or
replace the CMLR application model.

The diagnostic command reports substitutions/deletions/insertions, exact match,
length ratios, and speaker/reference-length/recorded-duration groups. It uses
the evaluation pipeline's canonical character normalization. Minimum-edit
alignments have deterministic ties; edit categories describe errors without
establishing their causes. Recorded durations are not independently measured.

Oracle@1/3/5/10 uses references only after inference and is a lower error bound
for the supplied candidate list. It is not a live decoder or a correctness
probability. Stored ranks retain duplicates. Missing lists explicitly fall back
to raw text without dropping failures or changing the denominator. Edit-space
agreement and medoid ranks describe internal candidate similarity, not truth.
This round stores the final returned ten candidates; oracle@10 is not an oracle
over the entire internal forty-candidate pool.
Complete manifest coverage and all declared identities/hashes are checked;
neither reference text nor predicted strings appear in the output.

`--candidate-pool-size 40` collects up to 40 unique completed hypotheses, scores
the entire pool with the reverse decoder, then returns `--nbest 10`.
`--pre-beam-ratio 3.0` widens the decoder shortlist from 60 to 120 vocabulary
items before CTC prefix scoring at beam 40. These are different operations;
neither raises the final beam width. Defaults remain pool=nbest and ratio=1.5.
Parameters are bounded and the existing 1 GiB estimated CTC workspace budget is
enforced before scorer construction, without silently changing the protocol.
This estimate is not a bound on total process memory.

Whole-beam forced-EOS detection is preserved beyond the returned pool. Batched
reverse errors or adjacent final-score gaps below `1e-3` anywhere in the pool
trigger whole-pool sequential fallback before truncation. Reports include both
requested and observed pool sizes.

The commands above illustrate reproduction using local licensed artifacts.
Evaluate the four fixed pool/shortlist combinations on the same development
clips under clean and synthetic-mild conditions. The selector permits only these
two settings to vary and binds the model, adapter, other decoder settings and
identical prepared pixels. It requires fewer pooled raw errors without worsening
either condition's raw errors, exact sentences, coverage or failures. Ties keep
the first prespecified configuration. Freeze the choice before a new holdout;
test labels must never drive another selection. Exit code 2 from evaluation
means product criteria were not met, not that inference crashed.

Original [model/data attributions and licenses](../../NOTICE) still apply.
Datasets, weights, adapters and raw per-clip predictions stay outside Git.
