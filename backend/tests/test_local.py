"""Phase 2 tests — LocalRenderer.narrate wiring + narrate_only, driven by
injected fake TTS/ASR backends so nothing heavy downloads."""
import wave
from pathlib import Path

import numpy as np
import pytest

from myAIscene.events import EventEmitter
from myAIscene.local import LocalRenderer
from myAIscene.pipeline import narrate_only
from myAIscene.spec import load_spec

SPEC_PATH = Path(__file__).resolve().parents[2] / "specs" / "not_it_protocol.json"
SR = 24000


@pytest.fixture
def spec():
    return load_spec(SPEC_PATH)


class FakeTTS:
    """Produces silence whose *length* simulates ~0.33s/word (so the
    duration gate is meaningfully exercised)."""
    def __init__(self, secs_per_word=0.33):
        self.secs_per_word = secs_per_word

    def synth(self, text, voice):
        n = max(1, len(text.split()))
        samples = np.zeros(int(SR * self.secs_per_word * n), dtype="float32")
        return samples, SR


class FakeASR:
    """Returns a scripted transcript per wav file stem (== beat id)."""
    def __init__(self, transcripts):
        self.transcripts = transcripts

    def transcribe(self, wav_path):
        stem = Path(wav_path).stem
        return self.transcripts.get(stem, "")


def quiet():
    return EventEmitter(out=None)


def test_narrate_writes_real_wav_and_passes(spec, tmp_path):
    transcripts = {b.id: b.narration for b in spec.beats}
    r = LocalRenderer(out_dir=tmp_path, tts=FakeTTS(), asr=FakeASR(transcripts))
    m = narrate_only(spec, r, quiet(), limit=3)

    assert m.ok
    assert len(m.beats) == 3
    for br in m.beats:
        p = Path(br.audio_path)
        assert p.exists() and p.suffix == ".wav"
        with wave.open(str(p)) as w:                # it's a real, readable wav
            assert w.getframerate() == SR
            assert w.getnchannels() == 1
        assert br.duration_s > 0


def test_narration_gate_failure_is_reported_not_raised(spec, tmp_path):
    transcripts = {b.id: b.narration for b in spec.beats}
    transcripts["b02"] = "completely unrelated words the model misheard"
    r = LocalRenderer(out_dir=tmp_path, tts=FakeTTS(), asr=FakeASR(transcripts))
    m = narrate_only(spec, r, quiet(), limit=3)  # must not raise

    assert not m.ok
    b02 = next(b for b in m.beats if b.beat_id == "b02")
    assert not b02.narration_gate.passed
    assert b02.narration_gate.blocking


def test_duration_overflow_detected(spec, tmp_path):
    # 5s/word vastly overflows every window → duration gate fails
    transcripts = {b.id: b.narration for b in spec.beats}
    r = LocalRenderer(out_dir=tmp_path, tts=FakeTTS(secs_per_word=5.0), asr=FakeASR(transcripts))
    m = narrate_only(spec, r, quiet(), limit=2)
    assert all(not b.duration_gate.passed for b in m.beats)


def test_voice_override_threads_through(spec, tmp_path):
    seen = {}

    class SpyTTS(FakeTTS):
        def synth(self, text, voice):
            seen["voice"] = voice
            return super().synth(text, voice)

    transcripts = {b.id: b.narration for b in spec.beats}
    r = LocalRenderer(out_dir=tmp_path, tts=SpyTTS(), asr=FakeASR(transcripts), voice="bm_lewis")
    narrate_only(spec, r, quiet(), limit=1)
    assert seen["voice"] == "bm_lewis"
