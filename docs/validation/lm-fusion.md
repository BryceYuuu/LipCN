# 中文字级语言模型融合 / Mandarin character-LM fusion (2026-11-03)

## 做了什么 / What changed

中文口型完整搜索加入 CMLR 字级 Transformer 语言模型（16 层，3363 字），在 beam search 每一步与识别模型共同打分（浅融合）。CMLR 与 CNVSRC 都按单字切分，[`research/lm_fusion.py`](../../research/lm_fusion.py) 把识别模型的 4470 个字映射到语言模型字表，73.2% 的识别字表字符在语言模型中，其余使用 `<unk>` 概率。草稿（CTC）不经过语言模型，仍然最先显示。

The Mandarin full search now adds the CMLR character Transformer LM as a shallow-fusion scorer. Both vocabularies are single characters; 73.2% of recognizer characters exist in the LM and the rest use its `<unk>` probability. The CTC draft is unchanged and still appears first.

设置 / Settings: LM 权重 0.1、长度奖励 0.5，其余不变（beam 40、CTC 0.1、反向 0.3、候选池 10）。

## 选择与验证 / Selection and confirmation

参数只在开发子集上选择，然后在未参与选择的 round5 fresh-holdout 上验证一次。两组都是 Chinese-LiPS 正常发声的嘴部裁剪视频，不是摄像头默念。

Settings were chosen on the development subset only, then confirmed once on the untouched round5 held-out set. Both are voiced Chinese-LiPS mouth crops, not webcam silent speech.

开发子集 / Development (33 clips, 11 speakers, 1432 characters), selected from 10 rescoring and 4 fusion settings:

| 设置 / Setting | 错字 / Errors | CER | 平均 / mean |
| --- | --- | --- | --- |
| 原搜索 / baseline | 818 | 57.12% | 3.98 s |
| 候选重打分 LM 0.05 / rescoring | 798 | 55.73% | 3.98 s |
| **融合 LM 0.1、长度 0.5 / fusion** | **777** | **54.26%** | 6.40 s |

验证集 / Held-out (round5 fresh-holdout, 40 clips, 10 new speakers, 1686 characters):

| 设置 / Setting | 错字 / Errors | CER | 平均 / mean | p95 |
| --- | --- | --- | --- | --- |
| 原搜索 / baseline | 965 | 57.24% | 3.24 s | 5.11 s |
| 候选重打分 / rescoring 0.05 | 957 | 56.76% | 3.29 s | 5.18 s |
| **融合 / fusion 0.1 / 0.5** | **940** | **55.75%** | 4.94 s | 8.32 s |

验证集上 10 人中 7 人变好、2 人变差、1 人不变；40 段中 19 段变好、8 段变差、13 段相同。提升约 1.5 个百分点，小于开发集，说明开发集结果含有选择偏差；整句正确率仍为 0。round6 盲测集未使用，留作以后验收。

On held-out data 7 of 10 speakers improved and 2 got worse; 19 clips improved, 8 got worse. The 1.5-point gain is smaller than on development data, as expected after selection. Exact-sentence accuracy remains zero. The round6 blind set was not opened.

耗时为完整搜索（不含编码），在 Apple M4 上测得；语言模型使完整结果平均慢约 1.7 秒，草稿出现时间不受影响。

## 使用与关闭 / Use and opt out

把 CMLR `lm/model.pth` 与 `lm/model.json` 放到 `~/.cache/lipcn/models/mandarin/cmlr-lm/`，或用 `LIPCN_CHINESE_LM` 指向该目录。加载时校验 SHA-256；缺少文件或校验失败时自动使用原搜索。`LIPCN_LM_FUSION=0` 关闭融合，换取更快的完整结果。CMLR 权重仅限研究/个人学习，不得用于商业产品开发，见 [NOTICE](../../NOTICE)。

Place the CMLR LM files in `~/.cache/lipcn/models/mandarin/cmlr-lm/` or point `LIPCN_CHINESE_LM` at them. Files are SHA-256 verified; missing or mismatched files fall back to the original search. `LIPCN_LM_FUSION=0` disables fusion. CMLR weights are for research/private study only; see NOTICE.

复现 / Reproduce:

```bash
.venv/bin/python research/scripts/benchmark_lm_fusion.py --manifest <manifest.json> \
  --checkpoint <cnvsrc.pth> --source-dir <CNVSRC2025> --adapter <adapter.pth> \
  --lm <cmlr/lm> --rescore 0.05:0 --fusion 0.1:0.5
```
