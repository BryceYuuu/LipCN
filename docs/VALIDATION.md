# Validation of experimental Mandarin support

Local validation on 2026-10-02, Apple M4, macOS, Python 3.12.14. This record describes tests actually run; it is not a claim about user-camera accuracy.

## Automated checks

- Existing tests plus new confidence, guard, Chinese text/training, data separation and delivery tests: 72 passed, 10 skipped, including the English real-video regression. The optional CMLR strict-load/synthetic-inference test was included.
- Python 3.11 minimal-dependency confidence/guard suite 21 passed.
- Native macOS review panel instantiated, displayed Chinese choices, exposed 1/2/Esc key equivalents, chose the raw candidate and cancelled another review successfully. No text was injected into another app during this check.
- Source compilation and `git diff --check` passed.
- Windows GUI, real microphone/camera recording, keyboard-driven selection, actual cross-app paste and personal Chinese face-training accuracy have not been manually validated. The existing Windows workflow exercises shared and platform tests, but results must be inspected on the submitted commit.

## Public Mandarin demo experiment

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
