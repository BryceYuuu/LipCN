"""Reference-free, batched reverse-candidate scoring for local research.

This helper changes execution layout only. It does not generate candidates,
select a model, choose weights, read labels/audio or enable an application mode.
Floating-point batching can slightly change sums. An integration that promises
the original ranking should score close final-reranking ties sequentially.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch


def _clear_memory(decoder):
    for module in decoder.modules():
        if hasattr(module, "_mem_kv"):
            del module._mem_kv


@torch.inference_mode()
def reverse_log_probabilities(decoder, encoded, token_sequences, sos_eos) -> list[float]:
    """Score each reversed candidate body followed by EOS in its original order.

    ``encoded`` is the single clip's T,D floating-point encoder output, already
    on the decoder's device/dtype. Each tuple contains ordinary vocabulary IDs
    only: no blank/SOS/EOS. The original vocabulary uses its final ID for both
    SOS and EOS. An empty tuple scores EOS given SOS, matching the serial helper.

    All candidates share a broadcast view of encoder memory. Right padding is
    hidden from self-attention keys and excluded from the log-probability sum;
    it never becomes a scored token. Existing encoder-memory projection caches
    are cleared before and after the batch, including failed input/model calls.
    """
    if not isinstance(decoder, torch.nn.Module):
        raise ValueError("decoder must be a torch module")
    _clear_memory(decoder)
    try:
        if type(sos_eos) is not int or sos_eos <= 1:
            raise ValueError("SOS/EOS must be the final vocabulary ID, greater than 1")
        if (not isinstance(encoded, torch.Tensor) or encoded.layout != torch.strided or encoded.ndim != 2
                or min(encoded.shape) < 1 or not encoded.is_floating_point() or not torch.isfinite(encoded).all()):
            raise ValueError("encoded must be finite nonempty T,D floating-point memory")
        if any(module.training for module in decoder.modules()):
            raise ValueError("reverse scoring requires an evaluation-mode decoder")
        parameter = next(decoder.parameters(), None)
        if parameter is not None and (parameter.device != encoded.device or parameter.dtype != encoded.dtype):
            raise ValueError("encoded and decoder must share device and floating-point dtype")
        output = getattr(decoder, "output_layer", None)
        if hasattr(output, "out_features") and output.out_features != sos_eos + 1:
            raise ValueError("SOS/EOS must match the decoder's final vocabulary ID")
        if (not isinstance(token_sequences, Sequence) or isinstance(token_sequences, (str, bytes))):
            raise ValueError("token_sequences must be a sequence of ordinary-token sequences")
        bodies = []
        for body in token_sequences:
            if not isinstance(body, Sequence) or isinstance(body, (str, bytes)):
                raise ValueError("Each candidate body must be an ordinary-token sequence")
            if any(type(token) is not int or not 0 < token < sos_eos for token in body):
                raise ValueError("Candidate bodies must contain integer ordinary vocabulary tokens")
            bodies.append(tuple(reversed(body)))
        if not bodies:
            return []

        batch = len(bodies)
        lengths = torch.tensor([len(body) + 1 for body in bodies], device=encoded.device)
        width = max(len(body) for body in bodies) + 1
        inputs = torch.zeros((batch, width), dtype=torch.long, device=encoded.device)
        targets = torch.zeros_like(inputs)
        inputs[:, 0] = sos_eos
        for row, body in enumerate(bodies):
            count = len(body)
            if count:
                tokens = torch.tensor(body, dtype=torch.long, device=encoded.device)
                inputs[row, 1:count + 1] = tokens
                targets[row, :count] = tokens
            targets[row, count] = sos_eos
        valid = torch.arange(width, device=encoded.device).unsqueeze(0) < lengths.unsqueeze(1)
        causal = torch.ones((width, width), dtype=torch.bool, device=encoded.device).tril()
        attention_mask = causal.unsqueeze(0) & valid.unsqueeze(1)
        memory = encoded.unsqueeze(0).expand(batch, -1, -1)
        logits, _ = decoder(inputs, attention_mask, memory, None)
        if (not isinstance(logits, torch.Tensor) or logits.shape != (batch, width, sos_eos + 1)
                or logits.device != encoded.device or logits.dtype != encoded.dtype
                or not torch.isfinite(logits).all()):
            raise ValueError("Unexpected/non-finite reverse-decoder output")
        token_logp = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        scores = token_logp.masked_fill(~valid, 0).sum(-1)
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite reverse candidate score")
        return scores.tolist()
    finally:
        _clear_memory(decoder)
