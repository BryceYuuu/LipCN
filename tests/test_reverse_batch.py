"""Batch/serial reverse-decoder equivalence without research weights or labels."""
import inspect

import pytest
import torch

from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from research.reverse_batch import reverse_log_probabilities
from research.scripts.compare_chinese_decoders import reverse_log_probability


def small_decoder(dtype=torch.float32):
    torch.manual_seed(54)
    return Decoder(odim=13, attention_dim=16, attention_heads=2, linear_units=32, num_blocks=2,
                   dropout_rate=0, positional_dropout_rate=0, self_attention_dropout_rate=0,
                   src_attention_dropout_rate=0).to(dtype=dtype).eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("bodies", [[(), (1,), (2, 3, 4), (5, 6, 7, 8, 9)],
                                   [(1, 2), (1, 2), ()], [()]])
def test_real_decoder_matches_serial_for_variable_length_and_empty_candidates(dtype, bodies):
    decoder = small_decoder(dtype)
    encoded = torch.randn(9, 16, dtype=dtype)
    original = encoded.clone()
    expected = [reverse_log_probability(decoder, encoded, body, 12) for body in bodies]
    actual = reverse_log_probabilities(decoder, encoded, bodies, 12)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(torch.tensor(actual, dtype=dtype), torch.tensor(expected, dtype=dtype),
                               rtol=0, atol=tolerance)
    torch.testing.assert_close(encoded, original, rtol=0, atol=0)
    assert not any(hasattr(module, "_mem_kv") for module in decoder.modules())


def test_right_padding_masks_and_shared_memory_are_used_without_scoring_padding():
    class InspectingDecoder(torch.nn.Module):
        def forward(self, inputs, mask, memory, memory_mask):
            assert not torch.is_grad_enabled() and torch.is_inference_mode_enabled()
            assert inputs.tolist() == [[5, 2, 1, 0], [5, 0, 0, 0], [5, 4, 3, 2]]
            assert memory.shape == (3, 7, 8) and memory.stride(0) == 0
            assert memory[0].data_ptr() == memory[1].data_ptr()
            assert memory_mask is None
            # The shortest candidate may attend only to its SOS key; padded
            # queries are irrelevant, and no valid query can see future tokens.
            assert mask[1].tolist() == [[True, False, False, False]] * 4
            assert mask[0].tolist() == [[True, False, False, False],
                                        [True, True, False, False],
                                        [True, True, True, False],
                                        [True, True, True, False]]
            return torch.zeros((3, 4, 6), dtype=memory.dtype, device=memory.device), mask

    scores = reverse_log_probabilities(InspectingDecoder().eval(), torch.zeros(7, 8), [(1, 2), (), (2, 3, 4)], 5)
    logp = -torch.log(torch.tensor(6.0)).item()
    assert scores == pytest.approx([3 * logp, logp, 4 * logp], abs=1e-6)


def test_same_memory_storage_can_be_reused_for_another_utterance():
    decoder = small_decoder()
    memory = torch.randn(7, 16)
    bodies = [(1, 2), (3,), ()]
    before = reverse_log_probabilities(decoder, memory, bodies, 12)
    memory.mul_(3).add_(5)
    after = reverse_log_probabilities(decoder, memory, bodies, 12)
    expected = [reverse_log_probability(decoder, memory, body, 12) for body in bodies]
    assert before != after
    assert after == pytest.approx(expected, abs=2e-5)


def test_empty_batch_is_empty_and_stale_projection_cache_is_cleared():
    decoder = small_decoder()
    decoder.decoders[0].src_attn._mem_kv = object()
    assert reverse_log_probabilities(decoder, torch.zeros(3, 16), [], 12) == []
    assert not hasattr(decoder.decoders[0].src_attn, "_mem_kv")


@pytest.mark.parametrize("bodies", [[(0,)], [(12,)], [(-1,)], [(13,)], [(1.0,)], [(True,)],
                                   ["hello"], [None], "hello", None])
def test_nonordinary_or_malformed_candidate_ids_fail_before_decode(bodies):
    decoder = small_decoder()
    decoder.decoders[0].src_attn._mem_kv = object()
    with pytest.raises(ValueError, match="token|candidate|Candidate"):
        reverse_log_probabilities(decoder, torch.zeros(3, 16), bodies, 12)
    assert not hasattr(decoder.decoders[0].src_attn, "_mem_kv")


@pytest.mark.parametrize("boundary", [0, 1, True, 12.0, 11, 13])
def test_sos_eos_must_match_original_vocabulary(boundary):
    with pytest.raises(ValueError, match="SOS/EOS"):
        reverse_log_probabilities(small_decoder(), torch.zeros(3, 16), [(1, 2)], boundary)


@pytest.mark.parametrize("memory", [torch.zeros(0, 16), torch.zeros(2, 0), torch.zeros(1, 2, 16),
                                    torch.zeros(2, 16, dtype=torch.long), torch.full((3, 16), float("nan")),
                                    torch.full((3, 16), float("inf")), None])
def test_malformed_memory_is_rejected(memory):
    with pytest.raises(ValueError, match="encoded"):
        reverse_log_probabilities(small_decoder(), memory, [(1,)], 12)


def test_training_mode_and_dtype_mismatch_are_rejected_without_changing_model_mode():
    decoder = small_decoder().train()
    with pytest.raises(ValueError, match="evaluation"):
        reverse_log_probabilities(decoder, torch.zeros(2, 16), [(1,)], 12)
    assert decoder.training
    decoder.eval()
    with pytest.raises(ValueError, match="dtype"):
        reverse_log_probabilities(decoder, torch.zeros(2, 16, dtype=torch.float64), [(1,)], 12)


@pytest.mark.parametrize("failure", ["raise", "nan", "shape"])
def test_decoder_failure_always_clears_projection_cache(failure):
    class FailingDecoder(torch.nn.Module):
        def forward(self, inputs, mask, memory, memory_mask):
            self._mem_kv = object()
            if failure == "raise":
                raise RuntimeError("fake model failure")
            if failure == "shape":
                return torch.zeros((len(inputs), inputs.shape[1], 7)), mask
            return torch.full((len(inputs), inputs.shape[1], 6), float("nan")), mask

    decoder = FailingDecoder().eval()
    with pytest.raises((RuntimeError, ValueError)):
        reverse_log_probabilities(decoder, torch.zeros(7, 8), [(1, 2), ()], 5)
    assert not hasattr(decoder, "_mem_kv")


def test_api_exposes_no_labels_and_preserves_candidate_order():
    assert list(inspect.signature(reverse_log_probabilities).parameters) == ["decoder", "encoded", "token_sequences", "sos_eos"]
    decoder = small_decoder()
    encoded = torch.randn(5, 16)
    bodies = [(1, 2, 3), (), (4, 5)]
    scores = reverse_log_probabilities(decoder, encoded, bodies, 12)
    shuffled = reverse_log_probabilities(decoder, encoded, bodies[::-1], 12)
    assert shuffled == pytest.approx(scores[::-1], abs=2e-5)
