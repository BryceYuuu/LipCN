"""Whisper mode: read lips *and* a faint whisper with the Auto-AVSR audio-visual model.

Same vocabulary and language model as the lip reader; the encoder fuses a visual stream (the mouth
crops) with an audio stream (16 kHz waveform, 640 samples per 25 fps video frame). Audio is
layer-normalised per clip, so a whisper's low volume doesn't matter, only its shape.
Weights: nguyenvulebinh/auto_avsr_av_trlrwlrs2lrs3vox2avsp_base (downloaded by setup.sh).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .vsr import MODELS, LipReader

AV_DIR = os.path.join(MODELS, "av")
SAMPLES_PER_FRAME = 640  # 16 kHz / 25 fps


URL = "https://huggingface.co/nguyenvulebinh/auto_avsr_av_trlrwlrs2lrs3vox2avsp_base/resolve/main/"


def download(progress=None):
    """Fetch config + weights (1.8 GB) into models/av, resumable via a .part file."""
    import urllib.request
    os.makedirs(AV_DIR, exist_ok=True)
    urllib.request.urlretrieve(URL + "config.json", os.path.join(AV_DIR, "config.json"))
    dest = os.path.join(AV_DIR, "model.safetensors")
    part = dest + ".part"
    with urllib.request.urlopen(URL + "model.safetensors") as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if progress and total and done % (50 << 20) < (1 << 20):
                progress(100 * done / total)
    os.replace(part, dest)


def available() -> bool:
    return os.path.exists(os.path.join(AV_DIR, "model.safetensors"))


class AVReader(LipReader):
    """A LipReader whose encoder also listens. Reuses LipReader's beam search, LMs and helpers;
    the AV model's own decoder and CTC head replace the lip-only ones."""

    def __init__(self, beam_size: int = 4, **kw):
        super().__init__(beam_size=beam_size, **kw)
        from safetensors.torch import load_file
        from espnet.nets.pytorch_backend.e2e_asr_transformer_av import E2E as AVE2E
        from espnet.nets.scorers.ctc import CTCPrefixScorer
        cfg = json.load(open(os.path.join(AV_DIR, "config.json"), encoding="utf-8"))
        args = argparse.Namespace(**cfg, report_cer=False, report_wer=False, char_list=None,
                                  sym_space=" ", sym_blank="<blank>")
        av = AVE2E(cfg["odim"], args)
        sd = {k[len("avsr."):]: v for k, v in load_file(os.path.join(AV_DIR, "model.safetensors")).items()}
        av.load_state_dict(sd)
        av.eval()
        self.av = av
        for m in (av.encoder, av.aux_encoder, av.fusion):
            m.to(self.enc_device)
        av.decoder.to(self.device)
        av.ctc.to(self.device)
        ctc = CTCPrefixScorer(av.ctc, av.eos)
        for d in (self.beam.full_scorers, self.beam.scorers):
            d["decoder"] = av.decoder
        for d in (self.beam.part_scorers, self.beam.scorers):
            d["ctc"] = ctc
        self.model.ctc = av.ctc
        # the lip-only encoder isn't used here (the app keeps a separate lip reader for live words)
        del self.model.encoder
        import gc
        gc.collect()

    @staticmethod
    def audio_tensor(wave: np.ndarray, n_frames: int) -> torch.Tensor:
        """16 kHz mono float → (T_samples, 1), trimmed/padded to the video length, layer-normed."""
        want = n_frames * SAMPLES_PER_FRAME
        w = np.asarray(wave, dtype=np.float32)[:want]
        if len(w) < want:
            w = np.pad(w, (0, want - len(w)))
        x = torch.from_numpy(w)
        x = torch.nn.functional.layer_norm(x, x.shape, eps=1e-5)  # eps: pure silence would be 0/0
        return x.unsqueeze(-1)

    @torch.inference_mode()
    def encode_av(self, rois: np.ndarray, wave: np.ndarray) -> torch.Tensor:
        v = self.to_tensor(rois).unsqueeze(0).to(self.enc_device)
        a = self.audio_tensor(wave, rois.shape[0]).unsqueeze(0).to(self.enc_device)
        feat, _ = self.av.encoder(v, None)
        aux, _ = self.av.aux_encoder(a, None)
        n = min(feat.shape[1], aux.shape[1])
        fused = self.av.fusion(torch.cat((feat[:, :n], aux[:, :n]), dim=-1))
        return fused.squeeze(0).to(self.device)

    def warmup_av(self):
        self.read_av(np.zeros((25, 96, 96), np.uint8), np.random.default_rng(0).normal(0, 0.01, 16000))

    def read_av(self, rois, wave) -> tuple[str, float]:
        t0 = time.monotonic()
        return self.beam_search(self.encode_av(rois, wave)), time.monotonic() - t0


def load_audio(path: str, start: float, end: float) -> np.ndarray:
    """16 kHz mono float32 audio from a video file (for tests and evaluation)."""
    import av as pyav
    c = pyav.open(path)
    rs = pyav.AudioResampler(format="flt", layout="mono", rate=16000)
    chunks = []
    for frame in c.decode(audio=0):
        t = float(frame.pts * frame.time_base)
        if t > end + 0.5:
            break
        for f in rs.resample(frame):
            chunks.append(f.to_ndarray().reshape(-1))
    c.close()
    wave = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    s = int(start * 16000)
    return wave[s:s + int((end - start) * 16000)]
