# Distinct lip-reading hypotheses

The desktop previously obtained three decoder hypotheses, then passed only the first through three wording styles. Conservative cleanup often left the same mistaken sentence in every box. Lip results now use `CandidateFormatter.recognition_choices` and never call the wording generator, including when `LIPCN_WORDING_MODE=model` is selected. Accepted speech retains the existing wording path.

Candidates stay in model rank order. The display uses up to three distinct texts after conversion to the selected Chinese script, with conservative terminal punctuation only. Sentence punctuation, redundant whitespace and case do not create extra alternatives. Numeric separators, signs, English word boundaries and internal apostrophes remain meaningful. Missing hypotheses leave empty, non-editable slots with copy/insertion disabled. No candidate is rewritten, fabricated or repeated to fill the panel.

The first validated candidate is delivered before the complete result. Both deliveries use the same prepared text, and user edits are preserved. Cancelled or superseded callbacks cannot restore old candidates. The existing Simplified Chinese/Lip only startup and continuous-preview behavior remain in place.

Mandarin now returns all 10 hypotheses from the already-existing 10-candidate rerank pool, before display deduplication; beam 40, CTC 0.1, reverse weight 0.3 and batched reverse scoring are unchanged. English returns up to four hypotheses from its existing beam of four. Neither change runs another encoder, beam search or reverse-scoring pass. There is a small additional cost for result objects and text handling.

Decoder control-flow comparisons with fixed synthetic inputs verified equal model-call counts and identical first-three results for normal batched reranking, near-score sequential fallback, batch-error fallback, forced EOS, and fewer than three candidates. A small randomly initialized English decoder likewise kept identical first-three hypotheses and search iterations. No pretrained weights or user recordings were involved in those comparisons.

This is a candidate-presentation fix, not evidence of better top-one recognition accuracy. Decoder scores are not calibrated correctness probabilities, and neither a large score gap nor duplicate hypotheses can establish 100% certainty. Independent silent-video accuracy validation is still outstanding.

Validation: 1,505 tests passed and 24 skipped. A native isolated harness with a fixed synthetic video and injected decoder hypotheses displayed three distinct choices (tea, water and coffee), dropping the punctuation-only duplicate. It exercised the real `_infer`, candidate formatter and result widgets, without real camera/microphone input or a pretrained recognition model. This validates presentation behavior, not recognition accuracy.
