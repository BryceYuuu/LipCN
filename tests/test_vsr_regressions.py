"""Sampling and cross-utterance decoder regressions, without model assets."""
import copy

import numpy as np
import pytest
import torch

from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from lipflow.vsr import LipReader


@pytest.mark.parametrize("timestamps, expected", [
    ([], []),
    ([3.0], [0]),
    ([0.0, 0.03, 0.08], [0, 1, 2]),
    ([0.0, 0.08], [0, 0, 1]),
    ([0.0, 0.0, 0.08], [0, 1, 2]),
    ([0.0, 0.04, 0.02, 0.08], [0, 1, 2, 3]),
    ([0.08, 0.04, 0.0], [0, 1, 2]),
])
def test_nearest_frame_and_clock_steps(timestamps, expected):
    assert LipReader.resample(timestamps, len(timestamps)) == expected


def test_irregular_capture_selects_nearest_frames():
    timestamps = np.array([0.0, 0.013, 0.037, 0.077, 0.1, 0.159, 0.201])
    grid = np.arange(0.0, timestamps[-1] + 1e-9, 0.04)
    expected = np.abs(timestamps[:, None] - grid).argmin(axis=0).tolist()
    assert LipReader.resample(timestamps.tolist(), len(timestamps)) == expected


def test_large_wall_clock_at_model_frame_rate():
    timestamps = [1_790_000_000.0 + i / 25 for i in range(50)]
    assert LipReader.resample(timestamps, len(timestamps)) == list(range(50))


def make_beam(decoder):
    return BatchBeamSearch(
        scorers={"decoder": decoder}, weights={"decoder": 1.0},
        beam_size=3, vocab_size=6, sos=5, eos=5,
    ).eval()


@torch.inference_mode()
@pytest.mark.parametrize("replaced_scorer", [False, True])
def test_reused_encoder_storage_matches_fresh_decoder(replaced_scorer):
    torch.manual_seed(7)
    decoder = Decoder(6, attention_dim=8, attention_heads=2,
                      linear_units=16, num_blocks=2, dropout_rate=0,
                      positional_dropout_rate=0).eval()
    fresh = copy.deepcopy(decoder)
    beam = make_beam(decoder)
    if replaced_scorer:
        beam = make_beam(copy.deepcopy(decoder))
        beam.full_scorers["decoder"] = decoder
        beam.scorers["decoder"] = decoder
    memory = torch.randn(6, 8)
    beam(memory, maxlenratio=-4)
    assert any(getattr(m, "_mem_kv", None) is not None for m in decoder.modules())
    pointer = memory.data_ptr()
    memory.copy_(torch.randn_like(memory) * 3)
    assert memory.data_ptr() == pointer
    actual = beam(memory, maxlenratio=-4)
    expected = make_beam(fresh)(memory, maxlenratio=-4)
    assert len(actual) == len(expected) > 0
    for a, b in zip(actual, expected):
        assert torch.equal(a.yseq, b.yseq)
        torch.testing.assert_close(a.score, b.score)


@torch.inference_mode()
def test_memory_projection_still_cached_within_utterance():
    decoder = Decoder(6, attention_dim=8, attention_heads=2,
                      linear_units=16, num_blocks=1, dropout_rate=0,
                      positional_dropout_rate=0).eval()
    attention = decoder.decoders[0].src_attn
    calls = []
    handle = attention.linear_k.register_forward_hook(lambda *args: calls.append(1))
    try:
        memory = torch.randn(1, 6, 8).expand(3, -1, -1)
        decoder.batch_score(torch.tensor([[5], [5], [5]]), [None] * 3, memory)
        decoder.batch_score(torch.tensor([[5, 1], [5, 2], [5, 3]]), [None] * 3, memory)
        assert len(calls) == 1
    finally:
        handle.remove()
