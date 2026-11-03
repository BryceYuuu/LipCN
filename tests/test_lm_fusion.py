"""Character-LM fusion contracts with a tiny random TransformerLM; no real weights."""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research' / 'scripts'))

from espnet.nets.pytorch_backend.lm.transformer import TransformerLM
from research.lm_fusion import MappedLM

LM_VOCAB = {'<blank>': 0, '<unk>': 1, '我': 2, '们': 3, '好': 4, '<eos>': 5}
TOKENS = ['<blank>', '我', '你', '们', '好', '<eos>']  # recognizer order; '你' is missing from the LM


@pytest.fixture
def mapped():
    torch.manual_seed(0)
    args = argparse.Namespace(layer=2, unit=16, att_unit=8, embed_unit=8, head=2, dropout_rate=0.0,
                              att_dropout_rate=0.0, emb_dropout_rate=0.0, tie_weights=False,
                              pos_enc='sinusoidal')
    return MappedLM(TransformerLM(len(LM_VOCAB), args).eval(), LM_VOCAB, TOKENS)


def test_mapping_uses_lm_eos_and_unk_for_missing_characters(mapped):
    assert mapped.mapping.tolist() == [1, 2, 1, 3, 4, 5]
    assert mapped.coverage == pytest.approx(3 / 4)


def test_batch_score_returns_recognizer_ordered_log_probabilities(mapped):
    ys = torch.tensor([[5, 1], [5, 4]])  # SOS 我 / SOS 好 in recognizer IDs
    logp, states = mapped.batch_score(ys, [None, None], torch.zeros(2, 1, 1))
    raw, _ = mapped.lm.batch_score(torch.tensor([[5, 2], [5, 4]]), [None, None], torch.zeros(2, 1, 1))
    assert logp.shape == (2, len(TOKENS)) and len(states) == 2
    assert torch.allclose(logp[:, 1], raw[:, 2]) and torch.allclose(logp[:, 2], raw[:, 1])
    assert torch.allclose(logp[:, 5], raw[:, 5])


def test_sequence_probability_matches_incremental_scores(mapped):
    ids = [1, 3, 4]  # 我们好
    total, state, prefix = 0.0, None, [5]
    for token in ids + [5]:
        logp, state = mapped.score(torch.tensor(prefix), state, None)
        total += float(logp[token])
        prefix.append(token)
    batched = mapped.sequence_log_probability([ids, [4]], eos=5)
    assert batched[0] == pytest.approx(total, abs=1e-4)
    assert len(batched) == 2 and batched[1] < 0


def test_configured_search_passes_fusion_to_candidate_decoder(monkeypatch):
    import compare_chinese_decoders
    import evaluate_cnvsrc

    calls = []
    monkeypatch.setattr(compare_chinese_decoders, 'decode_candidates',
                        lambda *args, **kwargs: calls.append(kwargs) or [])
    lm = object()
    evaluate_cnvsrc.configured_hypotheses(
        SimpleNamespace(token_list=TOKENS), None, beam_size=40, ctc_weight=.1, nbest=10,
        reverse_weight=0.0, candidate_pool_size=10, lm=lm, lm_weight=.1, length_bonus=.5)
    assert calls[-1]['lm'] is lm and calls[-1]['lm_weight'] == .1 and calls[-1]['length_bonus'] == .5


@pytest.mark.parametrize('name,value', [('lm_weight', -0.1), ('length_bonus', float('nan')),
                                        ('lm_weight', True)])
def test_decoder_rejects_invalid_fusion_weights(name, value):
    from compare_chinese_decoders import decode_candidates

    reader = SimpleNamespace(token_list=TOKENS)
    with pytest.raises(ValueError, match=name):
        decode_candidates(reader, torch.zeros(3, 2), .1, 4, 2, **{name: value})
