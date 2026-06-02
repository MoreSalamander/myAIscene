"""Phase 4 — music generation + music_cue_verify gate (non-blocking)."""
import wave
from pathlib import Path

import numpy as np
import pytest

from myAIscene.events import EventEmitter
from myAIscene.local import LocalRenderer
from myAIscene.pipeline import music_only, run
from myAIscene.renderers import ScriptedRenderer
from myAIscene.spec import load_spec

SPEC_PATH = Path(__file__).resolve().parents[2] / "specs" / "not_it_protocol.json"
SR = 32000


@pytest.fixture
def spec():
    return load_spec(SPEC_PATH)


def quiet():
    return EventEmitter(out=None)


class FakeMusicEngine:
    def __init__(self, fails_on: set[str] | None = None):
        self._fails_on = fails_on or set()
        self.calls: list[str] = []

    def generate(self, prompt: str, duration_s: float, path: Path) -> None:
        self.calls.append(prompt)
        beat_id = path.stem.replace("_music", "")
        if beat_id in self._fails_on:
            raise RuntimeError("simulated MusicGen failure")
        # Write silence of the requested duration
        samples = np.zeros(int(SR * (duration_s + 0.5)), dtype="float32")
        from myAIscene.local import write_wav
        write_wav(path, samples, SR)


def test_music_writes_real_wav(spec, tmp_path):
    r = LocalRenderer(out_dir=tmp_path, music_engine=FakeMusicEngine())
    beat = spec.beats[0]
    out = r.music(beat, spec)
    assert out is not None
    assert Path(out.asset_path).exists()
    with wave.open(out.asset_path) as w:
        assert w.getframerate() == SR
        assert w.getnchannels() == 1
    assert out.duration_s > beat.duration_s  # slightly over (padded by fake)


def test_music_prompt_contains_keywords(spec, tmp_path):
    engine = FakeMusicEngine()
    r = LocalRenderer(out_dir=tmp_path, music_engine=engine)
    r.music(spec.beats[0], spec)
    assert len(engine.calls) == 1
    assert "documentary" in engine.calls[0].lower() or "nature" in engine.calls[0].lower()


def test_music_none_for_empty_music_spec(spec, tmp_path):
    from myAIscene.spec import Beat
    r = LocalRenderer(out_dir=tmp_path, music_engine=FakeMusicEngine())
    # Beat with empty music dict → should return None
    empty_beat = Beat(id="bX", t0=0, t1=5, narration="x", direction="d",
                      footage_prompt="f", music={})
    assert r.music(empty_beat, spec) is None


def test_music_non_blocking_gate_drops_gracefully(spec, tmp_path):
    r = ScriptedRenderer(fail_music_on={"b01", "b05"})
    m = run(spec, r, quiet())
    assert m.ok
    dropped = {b.beat_id for b in m.beats if b.music_dropped}
    assert dropped == {"b01", "b05"}


def test_music_only_runner_never_raises(spec, tmp_path):
    engine = FakeMusicEngine()
    r = LocalRenderer(out_dir=tmp_path, music_engine=engine)
    m = music_only(spec, r, quiet(), limit=3)
    assert m.summary()["beats"] == 3
    # all should pass since fake writes valid WAVs
    assert m.summary()["music_ok"] == 3
    assert m.summary()["music_dropped"] == 0


def test_music_path_stored_in_beat_result(spec):
    r = ScriptedRenderer()
    m = run(spec, r, quiet())
    assert m.ok
    for br in m.beats:
        # ScriptedRenderer.music always passes — path should be set
        assert br.music_path is not None
        assert br.music_dropped is False
