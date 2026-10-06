"""Bounded, in-memory Mandarin audio-first input for the local research UI.

Uses python-sounddevice, faster-whisper and its bundled Silero VAD. No audio or
transcript is saved, logged, or uploaded here; model loading is local-only.
Acceptance is a routing heuristic, not a calibrated transcript probability.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from config import ASR_MODEL as DEFAULT_MODEL

SAMPLE_RATE = 16000
MAX_SECONDS = 20.0


def _use_cached_packages() -> None:
    """Compatibility hook; dependencies now come from the active environment."""


@dataclass(frozen=True)
class ASRResult:
    accepted: bool
    text: str = field(default='', repr=False)
    reason: str = 'unavailable'
    # Deliberately no transcript/token strings in diagnostic metadata.
    segments: tuple[dict, ...] = ()
    duration_seconds: float = 0.0
    speech_seconds: float = 0.0
    elapsed_seconds: float = 0.0


class AudioCapture:
    """A single explicit recording, at most 20 seconds. No stream opens in init.

    start() -> bool; stop() -> mono float32 16 kHz array; cancel() discards it.
    A failed/overflowed recording yields an empty array so callers can use lips.
    The caller owns the returned array and should release it after inference.
    """

    def __init__(self, *, max_seconds: float = MAX_SECONDS, device=None,
                 sounddevice_module=None):
        self.max_seconds = min(MAX_SECONDS, max(0.1, float(max_seconds)))
        self.device = device
        self._sd = sounddevice_module
        self._stream = None
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._samples = 0
        self._sample_rate = SAMPLE_RATE
        self.capturing = False
        self.last_error: str | None = None

    def start(self) -> bool:
        if self._stream is not None:
            return False
        self.last_error = None
        self._chunks.clear()
        self._samples = 0
        try:
            if self._sd is None:
                import sounddevice
                self._sd = sounddevice
            sd = self._sd
            try:
                sd.check_input_settings(device=self.device, channels=1,
                                        dtype='float32', samplerate=SAMPLE_RATE)
                self._sample_rate = SAMPLE_RATE
            except Exception:
                native = sd.query_devices(self.device, 'input')
                self._sample_rate = int(round(native['default_samplerate']))
                if self._sample_rate < 8000 or self._sample_rate > 192000:
                    raise ValueError('unsupported input sampling rate')
            self.capturing = True
            self._stream = sd.InputStream(device=self.device, channels=1,
                dtype='float32', samplerate=self._sample_rate, callback=self._callback)
            self._stream.start()
            return True
        except Exception as exc:
            self.last_error = 'microphone_unavailable:' + type(exc).__name__
            self._close_stream()
            self._chunks.clear()
            self._samples = 0
            return False

    def _callback(self, indata, frames, timing, status):
        with self._lock:
            if not self.capturing:
                return
            if status:
                self.last_error = 'microphone_overflow'
            remaining = int(self.max_seconds * self._sample_rate) - self._samples
            amount = min(len(indata), max(0, remaining))
            if amount:
                self._chunks.append(np.asarray(indata[:amount, 0], dtype=np.float32).copy())
                self._samples += amount
            reached_limit = self._samples >= int(self.max_seconds * self._sample_rate)
            if reached_limit:
                self.capturing = False
        if reached_limit:
            raise self._sd.CallbackStop

    def _close_stream(self) -> None:
        with self._lock:
            self.capturing = False
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def stop(self) -> np.ndarray:
        self._close_stream()
        with self._lock:
            chunks, self._chunks = self._chunks, []
            self._samples = 0
        if not chunks or self.last_error:
            return np.empty(0, dtype=np.float32)
        audio = np.concatenate(chunks)
        if self._sample_rate != SAMPLE_RATE:
            try:
                from scipy.signal import resample_poly
                divisor = math.gcd(self._sample_rate, SAMPLE_RATE)
                audio = resample_poly(audio, SAMPLE_RATE // divisor,
                                      self._sample_rate // divisor).astype(np.float32)
            except Exception as exc:
                self.last_error = 'resample_unavailable:' + type(exc).__name__
                return np.empty(0, dtype=np.float32)
        return np.ascontiguousarray(audio[:int(self.max_seconds * SAMPLE_RATE)], dtype=np.float32)

    def cancel(self) -> None:
        self._close_stream()
        with self._lock:
            self._chunks.clear()
            self._samples = 0


class LocalASR:
    """Offline Mandarin-only recognition with mandatory neural speech detection.

    Use on a worker thread. warmup() prepares the local model before recording;
    otherwise it loads after speech is detected. Low-confidence output is rejected,
    never a partial phrase falsely presented as an accepted full utterance.
    """

    def __init__(self, model_path: Path | str = DEFAULT_MODEL, *, model=None,
                 speech_detector: Callable | None = None, compute_type: str = 'int8',
                 cpu_threads: int = 8):
        self.model_path = Path(model_path).expanduser()
        self._model = model
        self._speech_detector = speech_detector
        self._converter = None
        self.compute_type = compute_type
        self.cpu_threads = max(1, int(cpu_threads))
        # Loading, warmup and transcription share one reentrant lock: the startup
        # worker and a first recording must never create or use models together.
        self._lock = threading.RLock()
        self._warmed = False
        self._closed = False
        self._ort_prepared = False
        self._vad_cache_clear = None
        self.last_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _prepare_ort(self) -> None:
        if not self._ort_prepared:
            _use_cached_packages()
            import onnxruntime
            # This must precede both direct Whisper loading and the first VAD
            # session. It is a telemetry preference, not a native shutdown API.
            onnxruntime.disable_telemetry_events()
            self._ort_prepared = True

    def _ensure_vad(self) -> None:
        if self._speech_detector is None:
            self._prepare_ort()
            from faster_whisper.vad import get_speech_timestamps, get_vad_model, VadOptions
            self._vad_cache_clear = get_vad_model.cache_clear
            options = VadOptions(threshold=0.6, min_speech_duration_ms=250,
                min_silence_duration_ms=400, speech_pad_ms=120,
                max_speech_duration_s=MAX_SECONDS)
            self._speech_detector = lambda audio: get_speech_timestamps(
                audio, vad_options=options, sampling_rate=SAMPLE_RATE)

    def load(self) -> None:
        """Load existing weights only; raises if absent. Never downloads a model."""
        with self._lock:
            self._load_locked()

    def _load_locked(self) -> None:
        if self._closed:
            raise RuntimeError('ASR is closed')
        if self._model is not None:
            return
        for filename in ('model.bin', 'config.json', 'tokenizer.json'):
            if not (self.model_path / filename).is_file():
                raise FileNotFoundError('local ASR model incomplete')
        self._prepare_ort()
        from faster_whisper import WhisperModel
        self._model = WhisperModel(str(self.model_path), device='cpu',
            compute_type=self.compute_type, cpu_threads=self.cpu_threads,
            num_workers=1, local_files_only=True)
        try:
            from opencc import OpenCC
            self._converter = OpenCC('t2s')
        except ImportError:
            self._converter = None

    def warmup(self) -> bool:
        """Prepare VAD/ASR once using generated silence, never a recording device.

        This moves model loading and first-use kernels out of the first sentence.
        Only one decode token is requested; any synthetic output is discarded.
        Failure remains recoverable by the normal recognition/lip fallback path.
        """
        with self._lock:
            if self._closed:
                return False
            if self._warmed:
                return True
            self.last_error = None
            try:
                self._ensure_vad()
                synthetic = np.zeros(SAMPLE_RATE, dtype=np.float32)
                self._speech_detector(synthetic)
                self.load()
                segments, _ = self._model.transcribe(synthetic, language='zh',
                    task='transcribe', beam_size=1, temperature=0.0,
                    condition_on_previous_text=False, initial_prompt=None,
                    hotwords=None, vad_filter=False, word_timestamps=False,
                    max_new_tokens=1)
                # The transcribe API is lazy: consuming it initializes native work.
                for _ in segments:
                    pass
                self._warmed = True
                return True
            except Exception as exc:
                self.last_error = type(exc).__name__
                return False

    def close(self) -> None:
        """Release idle native resources before stopping the application's loop.

        Call after the UI has stopped accepting work and joined its workers.
        The shared lock also prevents releasing this instance during inference.
        faster-whisper keeps the ORT VAD session in a process-wide lru_cache;
        clearing that cache releases its idle reference before native teardown.
        This terminal operation is idempotent and never initializes a backend.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._speech_detector = None
            clear, self._vad_cache_clear = self._vad_cache_clear, None
            if clear is not None:
                try:
                    clear()
                except Exception:
                    pass  # Still release the remaining owned references.
            self._model = None
            self._converter = None
            self._warmed = False

    def transcribe(self, audio: np.ndarray) -> ASRResult:
        started = time.monotonic()
        duration = speech_seconds = 0.0
        metadata: tuple[dict, ...] = ()

        def result(reason, text=''):
            return ASRResult(bool(text), text, reason, metadata, duration,
                             speech_seconds, time.monotonic() - started)

        try:
            waveform = np.asarray(audio, dtype=np.float32)
        except (TypeError, ValueError):
            return result('invalid_audio')
        if waveform.ndim != 1 or not np.isfinite(waveform).all():
            return result('invalid_audio')
        waveform = np.ascontiguousarray(waveform[:int(MAX_SECONDS * SAMPLE_RATE)])
        duration = len(waveform) / SAMPLE_RATE
        if duration < 0.3:
            return result('no_audio')
        if np.max(np.abs(waveform)) > 1.001:
            return result('invalid_audio_range')
        if float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))) < 0.00025:
            return result('silent')
        with self._lock:
            if self._closed:
                return result('closed')
            self.last_error = None
            try:
                self._ensure_vad()
                spans = self._speech_detector(waveform)
                spans = [(max(0, int(x['start'])), min(len(waveform), int(x['end'])))
                         for x in spans]
                spans = [(a, b) for a, b in spans if b > a]
                speech_seconds = sum(b - a for a, b in spans) / SAMPLE_RATE
            except Exception as exc:
                self.last_error = type(exc).__name__
                return result('vad_unavailable')
            if speech_seconds < 0.25:
                return result('no_speech')
            try:
                self.load()
                # One recording is at most 20 seconds, below Whisper's 30-second
                # context. Separate VAD clips would each rerun its full encoder.
                # Keep the original waveform and internal pauses; VAD bounds only
                # trim outside the first/last speech, without splicing syllables.
                clips = [min(a for a, _ in spans) / SAMPLE_RATE,
                         max(b for _, b in spans) / SAMPLE_RATE]
                segments, _ = self._model.transcribe(waveform, language='zh',
                    task='transcribe', beam_size=5, temperature=0.0,
                    condition_on_previous_text=False, initial_prompt=None,
                    hotwords=None, vad_filter=False, clip_timestamps=clips,
                    word_timestamps=True, hallucination_silence_threshold=1.0,
                    no_speech_threshold=0.4, log_prob_threshold=-0.85,
                    compression_ratio_threshold=2.4)
                segments = list(segments)
                if not segments:
                    return result('empty_transcript')
                diagnostic = []
                texts = []
                for segment in segments:
                    raw = str(segment.text).strip()
                    if not raw:
                        continue
                    words = list(segment.words or [])
                    probability = float(np.mean([w.probability for w in words])) if words else 0.0
                    values = {'start': float(segment.start), 'end': float(segment.end),
                        'avg_logprob': float(segment.avg_logprob),
                        'no_speech_prob': float(segment.no_speech_prob),
                        'compression_ratio': float(segment.compression_ratio),
                        'mean_word_probability': probability, 'character_count': len(raw)}
                    diagnostic.append(values)
                    metadata = tuple(diagnostic)
                    if (not all(math.isfinite(v) for v in values.values())
                            or values['end'] <= values['start']
                            or values['avg_logprob'] < -0.85
                            or values['no_speech_prob'] > 0.4
                            or values['compression_ratio'] > 2.4
                            or probability < 0.45):
                        return result('low_confidence')
                    texts.append(raw)
                text = ''.join(texts).strip()
                if not text:
                    return result('empty_transcript')
                han = len(re.findall(r'[\u3400-\u4dbf\u4e00-\u9fff]', text))
                letters = sum(character.isalnum() for character in text)
                if not han or han / max(1, letters) < 0.35:
                    return result('not_mandarin_text')
                if re.search(r'(.{2,20})\1{3,}', text):
                    return result('repetitive_transcript')
                if self._converter is not None:
                    text = self._converter.convert(text)
                return result('accepted', text)
            except Exception as exc:
                self.last_error = type(exc).__name__
                return result('asr_unavailable')
