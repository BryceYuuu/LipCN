from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research' / 'scripts'))
from evaluate_cnvsrc import configured_hypotheses
from lipflow.confidence import Hypothesis


def test_zero_reverse_preserves_original_hypotheses():
    expected = [Hypothesis('原始', -2.0, 2)]
    calls = []
    reader = SimpleNamespace(hypotheses=lambda encoded, nbest: calls.append((encoded, nbest)) or expected)
    actual, evidence = configured_hypotheses(reader, 'encoded', beam_size=40, ctc_weight=.5,
                                            nbest=10, reverse_weight=0)
    assert actual is expected
    assert evidence == [asdict(expected[0])]
    assert calls == [('encoded', 10)]


def test_frozen_reverse_uses_candidates_and_keeps_original_scores(monkeypatch):
    import compare_chinese_decoders as helper
    first = helper.Candidate('甲', -2, (1,), -2, -2, beam_score=-2)
    second = helper.Candidate('乙', -3, (2,), -3, -3, beam_score=-3)
    reader = SimpleNamespace(token_list=['blank', '甲', '乙', 'eos'],
                             model=SimpleNamespace(r_decoder='reverse-decoder'))
    calls = []
    monkeypatch.setattr(helper, 'decode_candidates',
                        lambda *args: calls.append(args) or [first, second])
    def reverse(decoder, encoded, ids, eos):
        calls.append((decoder, encoded, ids, eos))
        return -100 if ids == (1,) else 0
    monkeypatch.setattr(helper, 'reverse_log_probability', reverse)
    actual, evidence = configured_hypotheses(reader, 'visual-encoded', beam_size=40,
                                            ctc_weight=.3, nbest=10, reverse_weight=.3)
    assert actual[0].text == '乙'
    assert evidence[0]['beam_score'] == -3
    assert evidence[0]['reverse_score'] == 0
    assert actual[0].score == pytest.approx(-3 + .7 * .3 * 3)
    assert actual[0].token_count == 1
    assert calls[0] == (reader, 'visual-encoded', .3, 40, 10)
    assert calls[1:] == [('reverse-decoder', 'visual-encoded', (1,), 3),
                        ('reverse-decoder', 'visual-encoded', (2,), 3)]


@pytest.mark.parametrize('condition', ['ordinary', 'close_tie', 'batch_error', 'forced_eos'])
def test_batch_layout_preserves_sequential_fallback_and_forced_eos(monkeypatch, condition):
    import compare_chinese_decoders as helper
    import research.reverse_batch as batch
    original = [helper.Candidate('甲', -2, (1,), -2, -2, forced_eos=condition == 'forced_eos'),
                helper.Candidate('乙', -3, (2,), -3, -3)]
    reader = SimpleNamespace(token_list=['blank', '甲', '乙', 'eos'],
                             model=SimpleNamespace(r_decoder='decoder'))
    monkeypatch.setattr(helper, 'decode_candidates', lambda *args: original)
    calls = []
    def parallel(*args):
        calls.append('batch')
        if condition == 'batch_error':
            raise RuntimeError('batch failed')
        # Second candidate exactly ties first after weighted reranking.
        return [-2, -3 + 1 / .27] if condition == 'close_tie' else [-100, 0]
    def serial(decoder, encoded, ids, eos):
        calls.append(ids)
        return -100 if ids == (1,) else 0
    monkeypatch.setattr(batch, 'reverse_log_probabilities', parallel)
    monkeypatch.setattr(helper, 'reverse_log_probability', serial)
    hypotheses, evidence = configured_hypotheses(reader, 'encoded', beam_size=40, ctc_weight=.1,
                                                reverse_weight=.3, reverse_scoring='batched')
    if condition == 'forced_eos':
        assert calls == [] and hypotheses[0].text == '甲'
        assert evidence[0]['reverse_scoring'] == 'skipped'
        assert evidence[0]['reverse_rerank_skipped_reason'] == 'forced_eos_in_beam'
    else:
        assert hypotheses[0].text == '乙'
        assert evidence[0]['beam_score'] == -3
        assert calls == ['batch'] if condition == 'ordinary' else calls == ['batch', (1,), (2,)]
        assert evidence[0]['reverse_scoring'] == ('batched' if condition == 'ordinary' else 'sequential')
        assert evidence[0]['reverse_scoring_fallback'] == {
            'ordinary': None, 'close_tie': 'close_final_score_gap', 'batch_error': 'batch_error:RuntimeError'}[condition]


@pytest.mark.parametrize('flags', [['--reverse-weight', 'nan'], ['--reverse-weight', '-.1'],
                                  ['--nbest', '0'], ['--beam-size', '2', '--nbest', '3'],
                                  ['--nbest', '10', '--candidate-pool-size', '9'],
                                  ['--candidate-pool-size', '41'], ['--pre-beam-ratio', '.99'],
                                  ['--pre-beam-ratio', '4.01'], ['--pre-beam-ratio', 'nan']])
def test_invalid_frozen_parameters_fail_before_model_load(monkeypatch, flags):
    import evaluate_cnvsrc
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--accept-research-license', '--checkpoint', 'missing',
                                    '--source-dir', 'missing', '--manifest', 'missing', *flags])
    monkeypatch.setattr(evaluate_cnvsrc, '_reader', lambda *args: pytest.fail('model must not load'))
    with pytest.raises(SystemExit) as exc:
        evaluate_cnvsrc.main()
    assert exc.value.code == 2


@pytest.mark.parametrize('reverse_scoring', ['sequential', 'batched'])
def test_expanded_pool_reranked_before_output_truncation_and_discloses_counts(monkeypatch, reverse_scoring):
    import compare_chinese_decoders as helper
    import research.reverse_batch as batch
    rows = [helper.Candidate(text, -i, (i,), -i, -i)
            for i, text in enumerate(('甲', '乙', '丙'), 1)]
    reader = SimpleNamespace(token_list=['blank', '甲', '乙', '丙', 'eos'],
                             model=SimpleNamespace(r_decoder='decoder'))
    calls = []
    def decode(*args, **kwargs):
        calls.append(('decode', args, kwargs))
        return rows
    def serial(decoder, encoded, ids, eos):
        calls.append(('serial', ids))
        return 0 if ids == (3,) else -100
    def parallel(decoder, encoded, sequences, eos):
        calls.append(('batch', sequences))
        return [-100, -100, 0]
    monkeypatch.setattr(helper, 'decode_candidates', decode)
    monkeypatch.setattr(helper, 'reverse_log_probability', serial)
    monkeypatch.setattr(batch, 'reverse_log_probabilities', parallel)
    hypotheses, evidence = configured_hypotheses(reader, 'encoded', beam_size=4, ctc_weight=.1,
        nbest=1, reverse_weight=.3, reverse_scoring=reverse_scoring,
        candidate_pool_size=4, pre_beam_ratio=3.)
    assert calls[0] == ('decode', (reader, 'encoded', .1, 4, 1),
                        {'candidate_pool_size': 4, 'pre_beam_ratio': 3.})
    assert len(hypotheses) == len(evidence) == 1
    assert hypotheses[0].text == '丙' and evidence[0]['beam_score'] == -3
    assert evidence[0]['candidate_pool_size_requested'] == 4
    assert evidence[0]['candidate_pool_size_collected'] == 3
    assert evidence[0]['pre_beam_ratio'] == 3.
    assert calls[1:] == ([('batch', [(1,), (2,), (3,)])] if reverse_scoring == 'batched'
                         else [('serial', (1,)), ('serial', (2,)), ('serial', (3,))])


def test_forced_eos_outside_final_output_skips_entire_expanded_pool(monkeypatch):
    import compare_chinese_decoders as helper
    rows = [helper.Candidate('甲', -1, (1,), -1, -1),
            helper.Candidate('乙', -2, (2,), -2, -2, forced_eos=True)]
    reader = SimpleNamespace(token_list=['blank', '甲', '乙', 'eos'],
                             model=SimpleNamespace(r_decoder='decoder'))
    monkeypatch.setattr(helper, 'decode_candidates', lambda *args, **kwargs: rows)
    monkeypatch.setattr(helper, 'reverse_log_probability', lambda *args: pytest.fail('must skip whole pool'))
    hypotheses, evidence = configured_hypotheses(reader, 'encoded', beam_size=2, ctc_weight=.1,
        nbest=1, candidate_pool_size=2, reverse_weight=.3)
    assert len(hypotheses) == len(evidence) == 1 and hypotheses[0].score == -1
    assert evidence[0]['reverse_rerank_skipped_reason'] == 'forced_eos_in_beam'
    assert evidence[0]['candidate_pool_size_collected'] == 2


def test_close_tie_outside_output_still_falls_back_on_entire_pool(monkeypatch):
    import compare_chinese_decoders as helper
    import research.reverse_batch as batch
    rows = [helper.Candidate(text, -i, (i,), -i, -i)
            for i, text in enumerate(('甲', '乙', '丙'), 1)]
    reader = SimpleNamespace(token_list=['blank', '甲', '乙', '丙', 'eos'],
                             model=SimpleNamespace(r_decoder='decoder'))
    monkeypatch.setattr(helper, 'decode_candidates', lambda *args, **kwargs: rows)
    # Top candidate is isolated; two discarded candidates have a close tie.
    monkeypatch.setattr(batch, 'reverse_log_probabilities',
                        lambda *args: [-1, -2, -3 + .9995 / .27])
    calls = []
    monkeypatch.setattr(helper, 'reverse_log_probability',
                        lambda decoder, encoded, ids, eos: calls.append(ids) or -100)
    hypotheses, evidence = configured_hypotheses(reader, 'encoded', beam_size=3, ctc_weight=.1,
        nbest=1, candidate_pool_size=3, reverse_weight=.3, reverse_scoring='batched')
    assert calls == [(1,), (2,), (3,)] and len(hypotheses) == 1
    assert evidence[0]['reverse_scoring_fallback'] == 'close_final_score_gap'


def test_prebeam_only_experiment_uses_fresh_decoder_without_reverse(monkeypatch):
    import compare_chinese_decoders as helper
    rows = [helper.Candidate('甲', -1, (1,), -1, -1), helper.Candidate('乙', -2, (2,), -2, -2)]
    reader = SimpleNamespace(token_list=['blank', '甲', '乙', 'eos'],
                             model=SimpleNamespace(),
                             hypotheses=lambda *args, **kwargs: pytest.fail('must use requested shortlist'))
    calls = []
    monkeypatch.setattr(helper, 'decode_candidates',
                        lambda *args, **kwargs: calls.append(kwargs) or rows)
    monkeypatch.setattr(helper, 'reverse_log_probability', lambda *args: pytest.fail('reverse disabled'))
    hypotheses, evidence = configured_hypotheses(reader, 'encoded', beam_size=3, ctc_weight=.1,
        nbest=1, candidate_pool_size=2, pre_beam_ratio=3., reverse_weight=0)
    assert [row.text for row in hypotheses] == ['甲']
    assert calls == [{'candidate_pool_size': 2, 'pre_beam_ratio': 3.}]
    assert evidence[0]['reverse_score'] is None and evidence[0]['reverse_scoring'] == 'skipped'


def test_cli_frozen_single_condition_passes_resolved_search_controls(monkeypatch):
    import evaluate_cnvsrc
    calls = []
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--accept-research-license', '--checkpoint', 'missing',
        '--source-dir', 'missing', '--manifest', 'missing', '--nbest', '10',
        '--candidate-pool-size', '40', '--pre-beam-ratio', '3', '--reverse-weight', '.3'])
    monkeypatch.setattr(evaluate_cnvsrc, 'run',
                        lambda args: calls.append(args) or {'readiness': {'ready': False}})
    assert evaluate_cnvsrc.main() == 2
    assert len(calls) == 1 and calls[0].nbest == 10
    assert calls[0].candidate_pool_size == 40 and calls[0].pre_beam_ratio == 3.
