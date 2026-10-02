import math
from lipflow.confidence import Hypothesis as H, Quality as Q, assess


def test_review_is_default_even_for_large_gap():
    assert assess([H('hello', -1, 1), H('yellow', -4, 1)], 'hello', Q()).action == 'review'


def test_opt_in_auto_requires_margin_and_ctc_agreement():
    hs = [H('你好', -2, 2), H('你们', -6, 2)]
    assert assess(hs, '你好', Q(), 'auto').action == 'auto'
    assert assess(hs, '你们', Q(), 'auto').action == 'review'
    assert assess([H('你好', -2, 2), H('你们', -2.1, 2)], '你好', Q(), 'auto').action == 'review'


def test_missing_nonfinite_or_single_hypothesis_never_auto():
    for hs in ([H('hello', -1, 1)], [H('hello', math.nan, 1), H('yellow', -5, 1)],
               [H('hello', -1, 0), H('yellow', -5, 1)]):
        assert assess(hs, 'hello', Q(), 'auto').action == 'review'
    assert assess([], '', Q()).action == 'retry'


def test_quality_advice_takes_priority_over_decoder_scores():
    hs = [H('hello', -1, 1), H('yellow', -9, 1)]
    for q, expected in [(Q(face_ratio=.5), 'facing'), (Q(mouth_pixels=20), 'closer'),
                        (Q(brightness=20), 'lighting'), (Q(contrast=1), 'lighting')]:
        result = assess(hs, 'hello', q, 'auto')
        assert result.action == 'retry' and expected in result.reason


def test_nonfinite_quality_and_unknown_output_require_retry():
    hs = [H('hello', -1, 1), H('yellow', -9, 1)]
    assert assess(hs, 'hello', Q(brightness=math.nan), 'auto').action == 'retry'
    assert assess([H('<unk>', -1, 1)], '<unk>', Q()).action == 'retry'
