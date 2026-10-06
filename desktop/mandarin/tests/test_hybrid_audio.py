from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybrid_audio import AudioCapture, LocalASR, SAMPLE_RATE


def segment(text='今天天气很好。', **updates):
    values = dict(text=text, start=0.0, end=0.9, avg_logprob=-0.2,
                  no_speech_prob=0.02, compression_ratio=1.2,
                  words=[SimpleNamespace(probability=0.9)])
    values.update(updates)
    return SimpleNamespace(**values)


class Model:
    def __init__(self, segments=None):
        self.segments = [segment()] if segments is None else segments
        self.calls = []

    def transcribe(self, audio, **options):
        self.calls.append((audio.shape, options))
        return iter(self.segments), None


def reader(segments=None):
    model = Model(segments)
    return LocalASR(model=model, speech_detector=lambda a: [dict(start=0, end=len(a))]), model


@pytest.mark.parametrize('audio,reason', [
    (np.zeros(16000), 'silent'), (np.empty(0), 'no_audio'),
    (np.ones(100), 'no_audio'), (np.full(16000, np.nan), 'invalid_audio'),
    (np.ones((16000, 2)), 'invalid_audio'), (np.full(16000, 8), 'invalid_audio_range')])
def test_invalid_or_silent_never_runs_model(audio, reason):
    asr, model = reader()
    outcome = asr.transcribe(audio)
    assert not outcome.accepted and not outcome.text and outcome.reason == reason
    assert not model.calls


def test_noise_requires_neural_vad_even_when_loud():
    model = Model()
    asr = LocalASR(model=model, speech_detector=lambda a: [])
    noise = np.random.default_rng(8).uniform(-0.2, 0.2, 16000).astype(np.float32)
    assert asr.transcribe(noise).reason == 'no_speech'
    assert not model.calls


def test_vad_failure_falls_back_instead_of_using_energy():
    model = Model()
    def unavailable(audio):
        raise ImportError('no runtime')
    asr = LocalASR(model=model, speech_detector=unavailable)
    assert asr.transcribe(np.ones(16000) * 0.1).reason == 'vad_unavailable'
    assert not model.calls


@pytest.mark.parametrize('segments,reason', [
    ([], 'empty_transcript'),
    ([segment(text='')], 'empty_transcript'),
    ([segment(avg_logprob=-1.1)], 'low_confidence'),
    ([segment(no_speech_prob=0.9)], 'low_confidence'),
    ([segment(compression_ratio=4)], 'low_confidence'),
    ([segment(words=[])], 'low_confidence'),
    ([segment(words=[SimpleNamespace(probability=0.2)])], 'low_confidence'),
    ([segment(avg_logprob=float('nan'))], 'low_confidence'),
    ([segment(), segment(no_speech_prob=0.8)], 'low_confidence'),
    ([segment(text='This is English.')], 'not_mandarin_text'),
    ([segment(text='谢谢谢谢谢谢谢谢谢谢')], 'repetitive_transcript')])
def test_untrustworthy_result_never_returns_partial_text(segments, reason):
    asr, _ = reader(segments)
    outcome = asr.transcribe(np.ones(16000) * 0.1)
    assert outcome.reason == reason
    assert not outcome.accepted and outcome.text == ''


def test_accepted_is_chinese_isolated_and_metadata_does_not_contain_text():
    asr, model = reader()
    result = asr.transcribe(np.ones(16000) * 0.1)
    assert result.accepted and result.text == '今天天气很好。'
    assert result.text not in repr(result)
    assert all(isinstance(value, (float, int)) for part in result.segments for value in part.values())
    shape, options = model.calls[0]
    assert options['language'] == 'zh' and options['task'] == 'transcribe'
    assert options['initial_prompt'] is None and options['hotwords'] is None
    assert options['condition_on_previous_text'] is False
    assert options['clip_timestamps'] == [0.0, 1.0]


def test_input_clamped_to_twenty_seconds():
    asr, model = reader()
    result = asr.transcribe(np.ones(SAMPLE_RATE * 30) * 0.1)
    assert result.duration_seconds == 20
    assert model.calls[0][0] == (SAMPLE_RATE * 20,)


def test_internal_vad_pauses_use_one_encoder_window_without_splicing_audio():
    waveform = np.linspace(-0.2, 0.2, SAMPLE_RATE, dtype=np.float32)
    spans = [dict(start=1600, end=4800), dict(start=8000, end=12000)]
    model = Model()
    original = model.transcribe
    received = []
    def transcribe(audio, **options):
        received.append(audio.copy())
        return original(audio, **options)
    model.transcribe = transcribe
    asr = LocalASR(model=model, speech_detector=lambda a: spans)
    result = asr.transcribe(waveform)
    assert result.accepted and result.speech_seconds == pytest.approx(0.45)
    assert model.calls[0][1]['clip_timestamps'] == [0.1, 0.75]
    assert np.array_equal(received[0], waveform)
    assert model.calls[0][1]['word_timestamps'] is True


def test_vad_window_clamps_out_of_range_spans_to_captured_audio():
    model = Model()
    asr = LocalASR(model=model, speech_detector=lambda a: [
        dict(start=-1600, end=4000), dict(start=8000, end=32000)])
    result = asr.transcribe(np.ones(SAMPLE_RATE) * 0.1)
    assert result.accepted
    assert model.calls[0][1]['clip_timestamps'] == [0.0, 1.0]


def test_missing_model_does_not_download(tmp_path):
    asr = LocalASR(tmp_path / 'missing', speech_detector=lambda a: [dict(start=0, end=len(a))])
    result = asr.transcribe(np.ones(16000) * 0.1)
    assert result.reason == 'asr_unavailable'
    assert asr.last_error == 'FileNotFoundError' and not asr.loaded


def test_backend_failure_exposes_only_exception_class():
    asr, model = reader()
    def fail(*args, **kwargs):
        raise RuntimeError('sensitive transcript in native error')
    model.transcribe = fail
    result = asr.transcribe(np.ones(16000) * 0.1)
    assert not result.accepted and result.text == ''
    assert result.reason == 'asr_unavailable'
    assert asr.last_error == 'RuntimeError'


def test_warmup_is_synthetic_bounded_and_idempotent():
    asr, model = reader()
    assert asr.warmup() and asr.warmup()
    assert len(model.calls) == 1
    shape, options = model.calls[0]
    assert shape == (SAMPLE_RATE,)
    assert options['max_new_tokens'] == 1
    assert options['initial_prompt'] is None and options['hotwords'] is None
    # A real recognition still uses every existing reliability check.
    result = asr.transcribe(np.ones(SAMPLE_RATE) * 0.1)
    assert result.accepted
    assert model.calls[-1][1]['beam_size'] == 5
    assert model.calls[-1][1]['word_timestamps'] is True


def test_warmup_and_recognition_never_overlap():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    entered, release, attempted = threading.Event(), threading.Event(), threading.Event()
    model = Model()
    original = model.transcribe
    order = []
    def run(audio, **options):
        if options.get('max_new_tokens') == 1:
            assert np.count_nonzero(audio) == 0
            order.append('warmup')
            entered.set()
            assert release.wait(2)
        else:
            order.append('recognition')
        return original(audio, **options)
    model.transcribe = run
    asr = LocalASR(model=model, speech_detector=lambda a: [dict(start=0, end=len(a))])
    def recognize():
        attempted.set()
        return asr.transcribe(np.ones(SAMPLE_RATE) * 0.1)
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(asr.warmup)
        assert entered.wait(2)
        duplicate = pool.submit(asr.warmup)
        transcription = pool.submit(recognize)
        assert attempted.wait(2)
        assert order == ['warmup']
        release.set()
        assert first.result() and duplicate.result() and transcription.result().accepted
    assert order == ['warmup', 'recognition']


def test_warmup_failure_is_retryable_without_leaking_input():
    asr, model = reader()
    original = model.transcribe
    def fail(*args, **options):
        raise RuntimeError('PRIVATE DIAGNOSTIC')
    model.transcribe = fail
    assert not asr.warmup() and asr.last_error == 'RuntimeError'
    model.transcribe = original
    assert asr.warmup() and asr.last_error is None


class SoundDevice:
    class CallbackStop(Exception):
        pass

    def __init__(self, rate=16000, unavailable=False):
        self.rate = rate
        self.unavailable = unavailable
        self.streams = []

    def check_input_settings(self, **kw):
        if self.rate != 16000 or self.unavailable:
            raise ValueError('unsupported')

    def query_devices(self, *args):
        if self.unavailable:
            raise RuntimeError('not allowed')
        return {'default_samplerate': self.rate}

    def InputStream(self, **kw):
        obj = SimpleNamespace(**kw, started=False, stopped=False, closed=False)
        obj.start = lambda: setattr(obj, 'started', True)
        obj.stop = lambda: setattr(obj, 'stopped', True)
        obj.close = lambda: setattr(obj, 'closed', True)
        self.streams.append(obj)
        return obj


def test_capture_does_not_open_on_init_and_cancel_discards_everything():
    sd = SoundDevice()
    capture = AudioCapture(sounddevice_module=sd)
    assert not sd.streams
    assert capture.start() and not capture.start()
    capture._callback(np.full((1600, 1), 0.2), 1600, None, False)
    capture.cancel()
    assert sd.streams[0].closed and not capture.capturing
    assert capture.stop().size == 0
    assert capture.start()
    assert capture.stop().size == 0  # no stale samples from previous utterance


def test_cap_stops_input_and_stop_returns_16k_mono():
    sd = SoundDevice()
    capture = AudioCapture(max_seconds=0.5, sounddevice_module=sd)
    assert capture.start()
    with pytest.raises(SoundDevice.CallbackStop):
        capture._callback(np.full((16000, 1), 0.2), 16000, None, False)
    audio = capture.stop()
    assert audio.dtype == np.float32 and audio.shape == (8000,)
    assert sd.streams[0].closed
    assert capture.stop().size == 0


def test_native_48k_resampled_in_memory():
    capture = AudioCapture(sounddevice_module=SoundDevice(rate=48000))
    assert capture.start()
    waveform = np.sin(2 * np.pi * 200 * np.arange(48000) / 48000).astype(np.float32)
    capture._callback(waveform[:, None], 48000, None, False)
    audio = capture.stop()
    assert audio.shape == (16000,) and audio.dtype == np.float32
    assert np.max(np.abs(audio[100:-100])) > 0.9


def test_missing_mic_and_overflow_allow_lip_fallback():
    capture = AudioCapture(sounddevice_module=SoundDevice(unavailable=True))
    assert not capture.start() and 'microphone_unavailable' in capture.last_error
    assert capture.stop().size == 0
    capture = AudioCapture(sounddevice_module=SoundDevice())
    capture.start()
    capture._callback(np.ones((1600, 1)) * 0.1, 1600, None, True)
    assert capture.stop().size == 0 and capture.last_error == 'microphone_overflow'
