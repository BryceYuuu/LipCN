# Validation of experimental Mandarin support

Local validation on 2026-10-02, Apple M4, macOS, Python 3.12.14. This record describes tests actually run; it is not a claim about user-camera accuracy.

## Third iteration: public-data adaptation improves CER but remains unusable

The user deferred camera recording and requested continuing with public data. No physical webcam or microphone was opened. Upstream PR #4 remains closed.

### Public corpus and fixed adaptation

A deterministic subset of [official Chinese-LiPS](https://huggingface.co/datasets/BAAI/Chinese-LiPS) was fetched from revision `db96948538811029011eee44602438a26710ecd9`: 180 train videos/12 speakers, 30 dev/6 and 50 test/10. These are ordinary voiced speech mouth crops at 96×96/25 fps. Only video and author-provided labels were used; no audio, slides, OCR or LLM. The [dataset paper](https://arxiv.org/abs/2504.15066) describes manual transcription; this experiment did not independently correct possible label errors. CC-BY-NC-SA-4.0 applies.

The partitions have no shared speaker IDs, video hashes or normalized reference text. Base pretraining overlap remains unknown. Twenty whole training records contain targets outside the fixed CNVSRC vocabulary and were explicitly excluded before training. All 160 retained references are unchanged; all 30 dev and 50 test records remain in scoring, including OOV labels. Source hashes and every exclusion are recorded. The preparation script reproduces the same ordered 160 IDs/references. See [the reproducible workflow](CHINESE_ADAPTATION.md).

Model choice used the dev partition: [ViSpeR](https://github.com/YasserdahouML/visper) with the author's Chinese prefix, beam 40/CTC 0.1, made 942/1293 errors (72.85% CER, 0/30 exact); CNVSRC beam 40/CTC 0.5 made 901/1293 (69.68%). Deterministic traditional-to-simplified conversion did not change the ViSpeR dev score. ViSpeR stayed an isolated author-source comparison, with no source/weights integrated into Lipflow. Source revision `772fe4688fad5cea104308f51ccbfe3141b440a1`, [weight revision](https://huggingface.co/tiiuae/visper/tree/1a7d37da9d67980951b9ab181e05ca62c40766d1) `1a7d37da9d67980951b9ab181e05ca62c40766d1`, checkpoint SHA256 `d6b45e0a9988ae3e747496dd0f24c7b3f5e019358e743df8e66579a38e755608`. Both published license versions restrict commercial use; the repository and model card differ on NC license version, so no deployment is inferred.

The selected CNVSRC experiment fixed seed 0, two epochs, learning rate 1e-5, final encoder block plus output norm, CTC-only loss, frozen frontend/head/decoders/BN statistics and no external LM. An initial execution failed on the first training step because MPS warmup cached inference-mode positional tensors. The implementation was repaired with a regression that reproduces the failure and verifies backward gradients. The same protocol was restarted without observing failed-run test scores or changing training/decoding parameters. The completed execution scored baseline and candidate once each per dev/test partition; the earlier failed execution had also run baseline inference. No epoch or checkpoint was selected from test scores.

| Partition | Before errors / characters | Before CER | After errors / characters | After CER | Exact sentences |
| --- | --- | --- | --- | --- | --- |
| Dev: 30 clips / 6 speakers | 901 / 1293 | 69.68% | 859 / 1293 | 66.43% | 0 / 30 before and after |
| Test: 50 clips / 10 speakers | 1277 / 1829 | 69.82% | 1235 / 1829 | 67.52% | 0 / 50 before and after |

Both relative-improvement gates passed, with unchanged candidate coverage (dev 29/30, test 50/50) and zero inference failures. Mean training CTC loss was 3.6466 then 3.4350. All 39 selected parameter tensors changed. The saved 57,010,134-byte research adapter was successfully loaded into the strictly verified CPU base, checking complete parameter keys, shape/dtype, finite values, language and base/config/vocabulary/source provenance. No additional inference was performed for this loading check. Adapter SHA256: `ee20df62ee5032a719366c06e08d0210aa3267c451bf319336665c2fbc82fde3`.

**Product readiness is false.** Adapted test CER 67.52% greatly exceeds the declared 10% goal, exact rate is 0% versus the 70% goal, and warm test p95 processing is 5.05 s versus the 2 s goal (p95 RTF 0.398). Every speaker has only one source session, source video is cropped and voiced, and base pretraining independence has not been established. The modest improvement on this subset does not establish generalization to deliberately silent webcam speech. There is no automatic activation or paste; the GUI continues using its existing review-only Chinese path. Future tuning requires new independent test data.

### Crop diagnostics and collection

Twelve predeclared global crop/grayscale transforms were run on all eight previously inspected AISHELL6 demos (96 inferences, zero crashes). Baseline was 84.18% CER, mirror 79.11%, and exact author float grayscale 83.54%; every transform had 0/8 exact. These are reused development diagnostics, not fresh acceptance data. No transform was promoted into product defaults. Missing full-face landmarks/context cannot be reconstructed from existing crops.

`lipflow collect-chinese` now supports operator-confirmed image-only silent webcam collection, timestamps, untouched raw frames, retake/discard, quotas, unique IDs, SHA256 and atomic manifest updates. It does not open a camera until explicitly run. Actual text and silent articulation require human confirmation. A new readiness gate rejects voiced/whispered/unknown or unverified articulation even when predictions are perfect. Closing an audio track does not establish silent articulation.

Third-iteration checks: **199 passed, 10 skipped** in the complete local suite, including English real-video regression and Chinese strict-load/synthetic inference; **79 passed, 1 optional OpenCV skip** in the Python 3.11 minimal-dependency CI subset. Python 3.11/3.12 compilation and `git diff --check` passed. Synthetic collection tests verify that preview text never enters saved images and discard/rollback preserve earlier data. Real camera permission/recording and Windows UI remain unverified. No live audio/camera test is claimed.

## Second iteration: pure visual free-form Mandarin is not ready

The user selected fully silent, freely spoken Mandarin sentences. Upstream PR #4 was closed before this iteration. No new merge request is submitted by this change.

Engineering corrections:

- Steady 25 fps now preserves every frame. The previous floating-point/searchsorted path could map 150 frames to only 141 distinct frames.
- CMLR decoding now uses the original author's 0.3 length bonus; English decoding parameters stay the same.
- Encoder-memory attention caches are cleared for each search, using the active beam decoder. Regression tests reproduce reused storage across sentences and cover the separate AV decoder.
- Mandarin cannot automatically paste even with large score margins and CTC agreement. Multi-character phrase loops are rejected. The startup UI identifies unvalidated Chinese silent input as a test mode.
- Batch evaluation retains failed/rejected samples, labels/provenance, file/model hashes and explicit readiness failures. Full-face videos use the live no-face/no-motion/short-clip rules. See [the declared protocol](CHINESE_EVALUATION.md).

### Expanded diagnostic results

All eight author-published AISHELL6 demos were tested: S0128, S0075, S0218 and S0140, both normal and whispered speech. They are 96×96 mouth crops. Only video frames were read; no audio, LLM cleanup, personal weights or reference-derived prompt was used. References are used only for scoring. These previously inspected diagnostic assets are **not independent webcam acceptance data**, and ordinary/whispered mouth movements do not establish deliberately silent-speech performance.

| Model | Raw character errors | CER | Exactly correct sentences | Offered candidates | Warm p50 / p95 processing |
| --- | --- | --- | --- | --- | --- |
| CMLR, beam 20, LM 0.3, CTC 0.1, length bonus 0.3 | 156 / 158 | 98.73% | 0 / 8 | 8 / 8 | 0.76 / 1.30 s |
| CNVSRC2025 research baseline, beam 40, CTC 0.5, no external LM | 133 / 158 | 84.18% | 0 / 8 | 7 / 8 | 1.16 / 1.51 s |

Both reports return `not_ready` and exit 2: accuracy, exact sentence rate, data size, speaker/session coverage, independence and webcam domain fail the declared criteria. Zero automatic acceptance is enforced. Candidate coverage includes wrong text and is not a correctness metric. The newer model is a separately runnable benchmark, not a replacement integrated into the GUI. Its vocabulary covers every reference character; CMLR cannot represent four reference characters (`拂`, `漾`, `嬉`, `籁`), but that alone does not account for the very high error rate.

CNVSRC weights were strictly loaded (850 state entries including the reverse training decoder), with verified checkpoint/config/vocabulary hashes. Two examples were also run using the official implementation: top-five hypotheses matched, with only small floating-point score differences. Clearing cross-utterance caches did not change the eight-model comparison predictions. The official forward-default CTC 0.1 diagnostic was worse (139 / 158 errors); no tuning on these samples is claimed as independent validation.

Sources and pinned model identities:

- [Author demo and dataset provenance](https://zutm.github.io/AISHELL6-Whisper/).
- CMLR author source commit `5e1405db`, and the verified archives in `lipflow/chinese.py`.
- [CNVSRC2025 source](https://github.com/liu12366262626/CNVSRC2025/tree/main/VSR), commit `e5c4454016ba4eef9e586e77dd58e8981bb5c3e1`.
- [Separate CNVSRC research weights](https://huggingface.co/ReflectionL/CNVSRC2025Baseline), revision `b16f238d0df860da7e3b9834f959780b1d388f44`; checkpoint SHA256 `577cd9558eea111683a406bc25d69c7161cdb79534c2273fc0d0f044c356231c`.

To compare the separate research model, obtain the pinned author sources and weights locally, review their license, then run:

```bash
uv run python scripts/evaluate_cnvsrc.py --accept-research-license \
  --checkpoint /local/model_avg_cncvs_2_3_cnvsrc.pth \
  --source-dir /local/CNVSRC2025 --manifest eval/manifest.json \
  --output eval/cnvsrc-results.json
```

The script verifies the exact three inputs, does not import the author's code, never downloads weights implicitly and does not read audio. Model weights and demo media are not redistributed in this repository.

## Automated checks

- Second iteration full suite: 122 passed, 10 skipped, including the English real-video regression and CMLR strict-load/synthetic-inference tests. This tests implementation, not Mandarin recognition accuracy.
- Python 3.11 minimal-dependency confidence/guard/evaluation suite: 60 passed, 1 optional OpenCV test skipped.
- Native macOS review panel instantiated, displayed Chinese choices, exposed 1/2/Esc key equivalents, chose the raw candidate and cancelled another review successfully. No text was injected into another app during this check.
- Source compilation and `git diff --check` passed.
- Windows GUI, real microphone/camera recording, keyboard-driven selection, actual cross-app paste and personal Chinese face-training accuracy have not been manually validated. The existing Windows workflow exercises shared and platform tests; remote CI is not asserted by these local checks.

## First iteration public Mandarin demo experiment

Source: the authors' [AISHELL6-Whisper demo](https://zutm.github.io/AISHELL6-Whisper/), S0128 and S0075. These published videos contain **96×96 mouth crops**, not full faces; full-face detection is inappropriate. No video is redistributed in this PR.

S0128 reference: `微风轻拂树叶沙沙作响带来了一丝丝宁静` (18 characters).

| Path | Raw result after optional simplified-script conversion | Character errors | CER |
| --- | --- | --- | --- |
| CMLR VSR, direct mouth crop, normal speech, beam 10 | 是否能幸福我国天下这个过程的一个可能性 | 18 / 18 | 100% |
| Whisper large-v3-turbo, CPU int8, normal speech | 微风轻浮 树叶沙沙作响带来了一丝丝宁静 | 1 / 18 | 5.6% |
| Whisper large-v3-turbo, CPU int8, whisper speech | 微风轻浮鼠也沙沙作响带来了一丝丝宁静 | 3 / 18 | 16.7% |

The small Whisper model initially returned nothing with voiced-speech VAD on this quiet clip. Bounded audio gain and disabling that VAD allowed recognition, but the small model still made substantial errors. The stronger model is the default Chinese quiet-speech backend. Silence/invalid audio is rejected before loading a model, and decoder no-speech filtering remains enabled. Results always require confirmation.

Using the evaluation script after weights were cached, total time including loading was ~8.9 s for normal speech and ~9.0 s for whispered speech on this machine. This is not a realtime or GPU performance claim. First-run model download is additional.

The CMLR model also made 18 errors over 18 characters on the S0075 normal-speech demo (`小船在湖上荡漾宛如一片白云在水面徘徊`). These two out-of-domain tests do **not** reproduce CMLR's published in-domain benchmark. They show why the silent path is experimental and review-first. Different crop pipelines/domain conditions may contribute; no conclusion about trained benchmark performance is inferred.

Reproduce with locally obtained author demo files:

```bash
uv run python scripts/evaluate_chinese.py S0128_normal_av.mp4 --mouth-roi \
  --reference '微风轻拂树叶沙沙作响带来了一丝丝宁静'
uv run --extra chinese-whisper python scripts/evaluate_chinese.py S0128_whisper_av.mp4 \
  --mode whisper --reference '微风轻拂树叶沙沙作响带来了一丝丝宁静'
```

## What remains before production use

- Held-out, independently labelled user webcam sessions across lighting, dates and speakers; CER, sentence acceptance, manual corrections and latency measurements.
- Empirical calibration of routing features; do not interpret decoder margins as probabilities or enable unattended auto input based on this demo.
- Validation of face adaptation on independent sessions, not just the inherited six-clip holdout gate.
- Chinese-English code switching in a genuine multilingual visual or audio-visual model. CMLR's Han vocabulary cannot provide it.
- End-to-end Windows review/paste validation and control-level focus checking.
