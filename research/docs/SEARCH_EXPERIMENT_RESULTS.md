# 中文候选搜索实测 / Mandarin candidate-search results

## 中文

**结论：本轮完成错误诊断、两个独立搜索参数和 330 次真实权重推理，但没有找到同时改善原始画面与合成退化的配置。因此保留原默认参数，不把单组小幅改善当作稳定提升。**

新增代码和复现命令见[候选搜索说明](SEARCH_DIAGNOSTICS.md)，数值和指纹见
[round6 汇总 JSON](benchmarks/mandarin-round6-summary.json)。本轮只在本地修改与测试。

### 错误在哪里

固定的开发子集有 33 段视频、11 人，每人 3 段，共 1432 个参考字符。
基线错误 820 个：替换 706、漏字 55、多字 59。替换占错误约 **86.1%**。
输出共 1436 字，输出/参考长度比 1.0028；全部 330 个基线候选都正常结束，
没有被标记为强制 EOS。当前证据不支持优先添加长度奖励或强制延长句子。

| 参考句长 | 视频数 | 错误 / 字符 | CER |
| --- | --- | --- | --- |
| 1–20 字 | 2 | 9 / 24 | 37.50% |
| 21–40 字 | 8 | 197 / 259 | 76.06% |
| 41–60 字 | 23 | 614 / 1149 | 53.44% |
| 61 字以上 | 0 | 无样本 | 无法判断 |

不能据此断言“句子越长越差”：中等长度组反而更差，且人物、内容、画面条件没有
交叉控制。按记录时长分组，2–4 秒为 37.50%（2 段），4–8 秒为 76.47%（4 段），
8–16 秒为 55.86%（27 段）；也没有单调退化的证据。

| 匿名人物 | CER | 匿名人物 | CER |
| --- | --- | --- | --- |
| 060 | 25.17% | 070 | 80.95% |
| 113 | 93.96% | 119 | 41.51% |
| 127 | 35.25% | 139 | 40.91% |
| 155 | 50.65% | 168 | 94.79% |
| 175 | 70.32% | 189 | 72.50% |
| 190 | 44.37% | — | — |

人物差异很大，但每人只有 3 段且句子不同，不能将差异完全归因于身份本身。
S/D/I 为固定编辑对齐得到的描述性统计，不是已证实的模型错误原因。

### 这轮改了什么、测出什么

1. 新增候选诊断，覆盖 S/D/I、句长、记录时长、人物、长度比、候选差异及
   oracle@1/3/5/10。失败、拒绝、空结果与缺失候选均保留在分母中。
2. 将反向评分池大小与最终展示数量分开。允许先评分 20/40 个已生成候选，再返回 10 个。
3. 允许独立扩大 CTC 前的字词预筛选范围，保留完整 beam 的 EOS 检查、整池评分回退
   和内存预算检查。
4. 新增开发集选择器，核对完整清单、模型和适配器指纹、其他解码参数与相同的处理后
   像素。图像处理选择器也开始绑定新增搜索参数，避免比较中偷偷改变另一项设置。

固定模型仍是此前的 CNVSRC 基座与 joint 第 2 轮编码器适配器；beam 40、CTC 0.1、
reverse 0.3、返回 10 个候选、不接外部语言模型、不调用 AI 纠错。没有训练新权重。
初始网格在推理前声明；40 候选在退化组回退后，又单独声明并完成一次 20 候选补测。
两阶段均只使用开发集；验收规则始终不变。

| 配置：反向池 / 预筛比例 | 原始画面错误 / CER | 合成退化错误 / CER | 是否采用 |
| --- | --- | --- | --- |
| **10 / 1.5（原配置）** | **820 / 57.26%** | **847 / 59.15%** | **保留** |
| 20 / 1.5 | 814 / 56.84% | 850 / 59.36% | 不采用：退化组多错 3 字 |
| 40 / 1.5 | 814 / 56.84% | 848 / 59.22% | 不采用：退化组多错 1 字 |
| 10 / 3.0 | 820 / 57.26% | 847 / 59.15% | 不采用：没有收益 |
| 40 / 3.0 | 814 / 56.84% | 848 / 59.22% | 不采用：退化组回退 |

每格分母均为 1432 字。共 10 组 × 33 段 = **330 次推理**，失败均为 0，
原始输出整句正确率均为 0；候选均需确认。所有同条件比较均核对了相同处理后像素。
旧配置复跑的 66 个原始结果及每段前 10 候选顺序，与上一轮完全一致。

原始画面基线 oracle@10 为 **761 / 1432 = 53.14%**。20 候选池重排后返回的
前 10 个候选，其 oracle 为 **760 / 1432 = 53.07%**，仅少 1 个错误；40 候选池
返回列表的 oracle 为 **765 / 1432 = 53.42%**。所有配置的 oracle@10 整句正确率
仍为 0。这些下界只覆盖最终存储的 10 个候选，不覆盖内部整个 20/40 候选池。

实测 warm p95 在 5.47–8.22 秒之间，包含逐段视频处理、编码与解码，模型加载和预热
单独记录。各配置只有一次顺序运行，受机器状态影响；本轮不宣称提速。

### 测试、盲测与范围

完整测试：**796 通过，13 跳过**，另有 3 条依赖告警；其中相关专项测试 188 项通过。
跳过项包含不适用的平台和需要显式配置的集成测试，不代表这些能力已被实机验证。

已准备一组 20 段 / 10 位新人物的视频，每人 2 段，排除此前全部 93 位本地接触人物，
完成 SHA/CRC 与结构检查。由于没有新配置通过开发验收，**本轮没有对这组视频运行
模型，也没有用它选择参数**，保留给后续候选。它来自官方 train 中的正常发声嘴部
视频，不能保证基础预训练从未见过，更不是实际无声或摄像头测试。

本轮优先级结论是：继续单纯扩大这套候选搜索，已观察到的收益有限。下一轮应把
重点转向视觉模型适配、人物差异与训练数据覆盖的验证；这是一项待验证方向，
不是已经确定了替换错误的根因。英文默认流程、中文 CMLR 产品入口与原识别配置均保留。

## English

**This round adds diagnostics and configurable search controls, but does not establish a robust accuracy gain. The original pool=10 / pre-beam ratio=1.5 remains selected.**

The fixed development subset contains 33 clips from 11 speakers, three clips each,
with 1,432 reference characters. Baseline errors comprise **706 substitutions,
55 deletions and 59 insertions**: 820 errors, or 57.26% CER. Substitutions account
for 86.1% of errors. Output/reference length is 1.0028, and no baseline candidate
has a forced EOS. Blanket length rewards are not supported by this evidence.

Reference-length bins have CERs of 37.50% (1–20 characters, two clips), 76.06%
(21–40, eight clips), and 53.44% (41–60, 23 clips), with no longer samples.
Recorded-duration bins similarly do not show monotonic degradation. Speaker CER
ranges from 25.17% to 94.79%, but different utterances and recording conditions
confound attribution. Edit categories reflect one deterministic alignment, not
established causal diagnoses.

The implementation separates the reverse-scoring pool from the returned n-best
list and permits a wider decoder shortlist before CTC. Whole-beam EOS detection,
whole-pool numerical fallback and workspace bounds remain in place. The new
development selector binds complete manifests, model/adapter fingerprints,
other decoder settings and identical prepared pixels. The visual selector now
also binds these search settings. Diagnostics preserve failures and missing
candidates and never turn oracle scores into live corrections or probabilities.

The table above contains all five configurations under clean and synthetic-mild
conditions, with 1,432 reference characters in every cell. Pool 20 and pool 40
both reduce clean errors from 820 to 814, but increase stressed errors from 847
to 850 and 848 respectively. Widening the shortlist alone changes neither raw
error count. No candidate passes the unchanged rule requiring fewer pooled errors
without worsening either condition. Pool 20 was one declared development-only
follow-up after the initial four-configuration grid; no holdout informed it.

All **330 real-weight inferences** complete without failure, with zero exact
sentences. The 66 baseline raw predictions and returned candidate orders match
the prior round. Baseline clean oracle@10 is 53.14%; the pool-20 returned list
reaches 53.07% (one fewer error), while pool 40 returns 53.42%. These are bounds
over the final stored ten candidates, not the entire internal pool. Oracle exact
match remains zero in every condition. Warm processing p95 ranges from 5.47 to
8.22 seconds; these sequential single runs do not establish a speedup.

The complete suite passes **796 tests with 13 skips and three dependency
warnings**; 188 relevant focused tests pass. A fresh 20-clip/10-speaker holdout
excluding all 93 previously exposed local speakers was structurally checked but
**never inferred or used for selection**, because no new recipe passed development.
It remains reserved for a future candidate. Public voiced crops do not establish
silent speech, real webcam, everyday-conversation or unknown pretraining isolation.

No weights were trained and no application defaults changed. Further work should
evaluate visual adaptation and speaker/data coverage; the present experiment
does not establish the root cause of substitution errors. See the
[guide](SEARCH_DIAGNOSTICS.md), [aggregate evidence](benchmarks/mandarin-round6-summary.json)
and [source attributions](../../NOTICE).
