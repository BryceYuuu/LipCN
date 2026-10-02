# Validation of experimental Mandarin support

Local validation on 2026-10-02, Apple M4, macOS, Python 3.12.14. This record describes tests actually run; it is not a claim about user-camera accuracy.

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
