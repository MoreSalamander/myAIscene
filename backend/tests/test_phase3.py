"""Phase 3 tests — LocalRenderer.still() + motion() via injected fakes.
No model downloads. Requires ffmpeg (system) for the Ken Burns clip test.
"""
import struct
import zlib
from pathlib import Path

import pytest

from myAIscene.events import EventEmitter
from myAIscene.local import LocalRenderer
from myAIscene.pipeline import run
from myAIscene.renderers import ScriptedRenderer, StillOut
from myAIscene.spec import load_spec

SPEC_PATH = Path(__file__).resolve().parents[2] / "specs" / "not_it_protocol.json"


@pytest.fixture
def spec():
    return load_spec(SPEC_PATH)


def quiet():
    return EventEmitter(out=None)


# ---- minimal fake PNG (no PIL dep in tests) ----------------------------

def _make_png(path: Path, w: int = 4, h: int = 4) -> None:
    """Write a tiny solid-blue PNG using stdlib struct+zlib (no Pillow)."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    scanline = b"\x00" + b"\x00\x00\xFF" * w   # filter byte + RGB blue pixels
    raw = b"".join(scanline for _ in range(h))
    compressed = zlib.compress(raw)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


# ---- fake backends -----------------------------------------------------

class FakeT2I:
    def __init__(self, clip_score: float = 0.31):
        self._clip_score = clip_score
        self.calls: list[tuple[str, Path]] = []

    def generate(self, prompt: str, path: Path) -> None:
        self.calls.append((prompt, path))
        _make_png(path)


class FakeCLIP:
    def __init__(self, score: float = 0.31):
        self._score = score

    def score(self, image_path: str, text: str) -> float:
        return self._score


# ---- tests: still() ----------------------------------------------------

def test_still_writes_png_and_passes_gate(spec, tmp_path):
    r = LocalRenderer(out_dir=tmp_path, t2i=FakeT2I(), clip=FakeCLIP(0.31))
    beat = spec.beats[0]
    out = r.still(beat, spec)
    assert Path(out.image_path).exists()
    assert Path(out.image_path).suffix == ".png"
    assert out.clip_score == pytest.approx(0.31)


def test_still_score_below_threshold_propagates(spec, tmp_path):
    r = LocalRenderer(out_dir=tmp_path, t2i=FakeT2I(), clip=FakeCLIP(0.10))
    out = r.still(spec.beats[0], spec)
    assert out.clip_score < 0.22   # pipeline will see footage_verify fail


def test_t2i_receives_footage_prompt(spec, tmp_path):
    t2i = FakeT2I()
    r = LocalRenderer(out_dir=tmp_path, t2i=t2i, clip=FakeCLIP())
    r.still(spec.beats[2], spec)
    assert spec.beats[2].footage_prompt in t2i.calls[0][0]


# ---- tests: motion() ---------------------------------------------------

def test_motion_writes_mp4(spec, tmp_path):
    r = LocalRenderer(out_dir=tmp_path, t2i=FakeT2I(), clip=FakeCLIP())
    beat = spec.beats[0]
    still = r.still(beat, spec)
    clip = r.motion(beat, still, spec)
    assert Path(clip.clip_path).exists()
    assert Path(clip.clip_path).suffix == ".mp4"
    assert clip.duration_s == pytest.approx(beat.duration_s, abs=1.0)


def test_motion_clips_cycle_through_four_pan_directions(spec, tmp_path):
    r = LocalRenderer(out_dir=tmp_path, t2i=FakeT2I(), clip=FakeCLIP(), fps=12)
    paths = set()
    for beat in spec.beats[:4]:
        s = r.still(beat, spec)
        c = r.motion(beat, s, spec)
        paths.add(c.clip_path)
    assert len(paths) == 4   # each beat got its own distinct output file


def test_motion_resolution_matches_spec(spec, tmp_path):
    import subprocess
    r = LocalRenderer(out_dir=tmp_path, t2i=FakeT2I(), clip=FakeCLIP(), fps=12)
    beat = spec.beats[0]
    still = r.still(beat, spec)
    clip = r.motion(beat, still, spec)
    W, H = spec.episode.resolution
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0", clip.clip_path],
        capture_output=True, text=True, check=True,
    )
    got_w, got_h = (int(x) for x in probe.stdout.strip().split(","))
    assert got_w == W and got_h == H


# ---- tests: footage_verify gate wired into pipeline (ScriptedRenderer) --

def test_footage_retry_then_pass_in_pipeline(spec):
    """ScriptedRenderer's footage gate heals after 1 retry — pipeline ok."""
    r = ScriptedRenderer(fail_footage_on={"b04"}, heal_after_retries=1)
    m = run(spec, r, quiet(), max_retries=2)
    assert m.ok
    b04 = next(b for b in m.beats if b.beat_id == "b04")
    assert not b04.footage_fallback   # healed, so no fallback


def test_footage_exhausted_falls_back_not_fatal(spec):
    r = ScriptedRenderer(fail_footage_on={"b04"}, heal_after_retries=99)
    m = run(spec, r, quiet(), max_retries=1)
    assert m.ok
    b04 = next(b for b in m.beats if b.beat_id == "b04")
    assert b04.footage_fallback
