"""Tiny CPU checkpoint parity/integrity tests; no real model or device."""
import hashlib

import pytest
import torch

from research.scripts import evaluate_cnvsrc as subject


class Decoder(torch.nn.Linear):
    def __init__(self, **kwargs):
        super().__init__(3, 3)


class Model(torch.nn.Module):
    def __init__(self, odim, args):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 3)
        self.decoder = Decoder()

    def scorers(self):
        return {'decoder': self.decoder}


class Beam(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs


def assets(tmp_path, monkeypatch):
    from espnet.nets import batch_beam_search
    from espnet.nets.pytorch_backend import e2e_asr_transformer
    from espnet.nets.pytorch_backend.transformer import decoder
    monkeypatch.setattr(batch_beam_search, 'BatchBeamSearch', Beam)
    monkeypatch.setattr(e2e_asr_transformer, 'E2E', Model)
    monkeypatch.setattr(decoder, 'Decoder', Decoder)
    model = Model(4470, None)
    model.r_decoder = Decoder()
    checkpoint = tmp_path / 'model.pth'
    torch.save(model.state_dict(), checkpoint)
    for relative, content, constant in [
        ('conf/train_cncvs_2_3_cnvsrc.yaml', 'tiny test configuration', 'CONFIG_SHA256'),
        ('datamodule/char_units.txt', '\n'.join(chr(0x4e00 + i) + ' 1' for i in range(4468)), 'VOCABULARY_SHA256'),
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        monkeypatch.setattr(subject, constant, hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(subject, 'CHECKPOINT_SHA256', hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    return checkpoint, model


def test_mapped_loading_preserves_every_tensor_forward_and_owned_storage(tmp_path, monkeypatch):
    checkpoint, baseline = assets(tmp_path, monkeypatch)
    old_bytes = checkpoint.read_bytes()
    original, calls = torch.load, []
    def observe(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, 'load', observe)
    reader = subject._reader(checkpoint, tmp_path, 40, .1, 'cpu')
    assert calls == [{'map_location': 'cpu', 'weights_only': True, 'mmap': True}]
    assert reader.tensor_count == len(baseline.state_dict())
    assert all(torch.equal(value, reader.model.state_dict()[name]) for name, value in baseline.state_dict().items())
    value = torch.tensor([[.1, .2, .3]])
    assert torch.equal(baseline.encoder(value), reader.model.encoder(value))
    assert reader.beam.config['beam_size'] == 40
    assert reader.beam.config['weights'] == {'decoder': .9, 'ctc': .1, 'length_bonus': 0.}
    with torch.no_grad():
        reader.model.encoder.weight.add_(1)
    assert checkpoint.read_bytes() == old_bytes
    assert torch.equal(original(checkpoint, weights_only=True)['encoder.weight'], baseline.encoder.weight)


def test_hash_mismatch_still_rejects_before_deserialization(tmp_path, monkeypatch):
    checkpoint, _ = assets(tmp_path, monkeypatch)
    checkpoint.write_bytes(checkpoint.read_bytes() + b'tampered')
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: pytest.fail('Unverified checkpoint must not load'))
    with pytest.raises(ValueError, match='Unexpected source/checkpoint SHA256'):
        subject._reader(checkpoint, tmp_path, 40, .1, 'cpu')


def test_strict_model_structure_remains_enforced(tmp_path, monkeypatch):
    checkpoint, model = assets(tmp_path, monkeypatch)
    state = model.state_dict()
    state.pop('decoder.weight')
    torch.save(state, checkpoint)
    monkeypatch.setattr(subject, 'CHECKPOINT_SHA256', hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    with pytest.raises(RuntimeError, match='Missing key'):
        subject._reader(checkpoint, tmp_path, 40, .1, 'cpu')
