"""Character language-model fusion for the CNVSRC Mandarin recognizer.

The CMLR character LM (3363 tokens) and the CNVSRC recognizer (4470 tokens) are
both single-character vocabularies. ``MappedLM`` translates recognizer token IDs
into LM IDs and gathers LM log-probabilities back into the recognizer's order,
so the LM can join ESPnet beam search as an ordinary batch scorer or rescore a
finished candidate list. Recognizer characters missing from the LM use the
LM's ``<unk>`` probability. No reference text is ever seen here.
"""
from __future__ import annotations

from pathlib import Path


# The CMLR character LM from the author's model zoo (research/private study only;
# see NOTICE). These are the files evaluated in docs/validation/lm-fusion.md.
CMLR_LM_SHA256 = {
    'model.json': 'bcfb190a447936524fbc95543ae4d8151fbffb812aaf241c2fdaabb81ddefae7',
    'model.pth': '85470cc3e98c1875a007113780ea2966ada73394f6d86e0658adbf02f0dfca4c',
}
# Chosen on the 33-clip development subset, confirmed on the round5 held-out set.
LM_WEIGHT = 0.1
LENGTH_BONUS = 0.5


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_cmlr_lm(root: Path, device='cpu', *, verify=True):
    """Load ``root/model.pth`` + ``root/model.json`` (ESPnet TransformerLM)."""
    import json

    from espnet.asr.asr_utils import get_model_conf, torch_load
    from espnet.nets.lm_interface import dynamic_import_lm

    root = Path(root)
    if verify:
        for name, expected in CMLR_LM_SHA256.items():
            if _sha256(root / name) != expected:
                raise ValueError(f'Unexpected CMLR LM file: {name}')
    args = get_model_conf(str(root / 'model.pth'), str(root / 'model.json'))
    lm_class = dynamic_import_lm(getattr(args, 'model_module', 'default'), args.backend)
    lm = lm_class(args.n_vocab, args)
    torch_load(str(root / 'model.pth'), lm)
    lm.to(device).eval()
    vocabulary = json.loads((root / 'model.json').read_text(encoding='utf-8'))['char_list_dict']
    return lm, vocabulary


try:  # Settings above stay importable without torch/ESPnet (headless desktop tests).
    from espnet.nets.scorer_interface import BatchScorerInterface as _ScorerBase
except ImportError:
    _ScorerBase = object


class MappedLM(_ScorerBase):
    """ESPnet (batch) scorer exposing an LM in another character vocabulary."""

    def __init__(self, lm, lm_vocabulary: dict[str, int], tokens: list[str]):
        import torch

        self.lm = lm
        eos = lm_vocabulary['<eos>']
        unk = lm_vocabulary['<unk>']
        mapping = []
        for index, token in enumerate(tokens):
            if index == len(tokens) - 1:  # recognizer SOS/EOS
                mapping.append(eos)
            elif index == 0:  # CTC blank is never emitted by attention search
                mapping.append(unk)
            else:
                mapping.append(lm_vocabulary.get(token, unk))
        self.mapping = torch.tensor(mapping, dtype=torch.long)
        self.coverage = sum(token in lm_vocabulary for token in tokens[1:-1]) / max(1, len(tokens) - 2)

    # ScorerInterface
    def init_state(self, x):
        return None

    def select_state(self, state, i, new_id=None):
        return None if state is None else state[i]

    def score(self, y, state, x):
        logp, state = self.lm.score(self.mapping.to(y.device)[y], state, x)
        return logp[self.mapping.to(logp.device)], state

    # BatchScorerInterface
    def batch_init_state(self, x):
        return None

    def batch_score(self, ys, states, xs):
        logp, states = self.lm.batch_score(self.mapping.to(ys.device)[ys], states, xs)
        return logp[:, self.mapping.to(logp.device)], states

    def final_score(self, state):
        return 0.0

    def sequence_log_probability(self, token_ids: list[list[int]], eos: int) -> list[float]:
        """Teacher-forced log P(tokens + EOS) for each candidate, in one batch."""
        import torch

        if not token_ids:
            return []
        width = max(len(ids) for ids in token_ids) + 1
        inputs = torch.full((len(token_ids), width), eos, dtype=torch.long)
        targets = torch.full((len(token_ids), width), -1, dtype=torch.long)
        for row, ids in enumerate(token_ids):
            inputs[row, 1:len(ids) + 1] = torch.tensor(ids, dtype=torch.long)
            targets[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            targets[row, len(ids)] = eos
        mapped_inputs, mapped_targets = self.mapping[inputs], self.mapping[targets.clamp(min=0)]
        with torch.inference_mode():
            mask = self.lm._target_mask(mapped_inputs)
            hidden, _ = self.lm.encoder(self.lm.embed(mapped_inputs), mask)
            logp = self.lm.decoder(hidden).log_softmax(-1)
            picked = logp.gather(-1, mapped_targets.unsqueeze(-1)).squeeze(-1)
            picked = picked.masked_fill(targets < 0, 0.0)
        return [float(value) for value in picked.sum(-1)]
