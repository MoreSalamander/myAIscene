"""Phase 5 — grade + assemble. All tests use real ffmpeg; no model inference."""
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np
import pytest

from myAIscene.events import EventEmitter
from myAIscene.local import LocalRenderer, write_wav
from myAIscene.luts import GRADES, ensure_luts, generate_cube
from myAIscene.pipeline import assemble_from_assets, run
from myAIscene.renderers import ScriptedRenderer
from myAIscene.spec import load_spec
from myAIscene.verifiers import assembly_verify

SPEC_PATH = Path(__file__).resolve().parents[2] / "specs" / "not_it_protocol.json"
SR_NARR = 24000
SR_MUSIC = 32000


@pytest.fixture
def spec():
    return load_spec(SPEC_PATH)


def quiet():
    return EventEmitter(out=None)


# ---- helpers to produce minimal real media assets ---------------------

def _make_png(path: Path, w=4, h=4) -> None:
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    row = b"\x00" + b"\x00\x80\x40" * w
    raw = b"".join(row for _ in range(h))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    path.write_bytes(png)


def _make_clip(path: Path, duration: float, W=1920, H=1080, fps=12) -> None:
    subprocess.run([
        "ffmpeg", "-f", "lavfi",
        "-i", f"color=c=0x004020:s={W}x{H}:r={fps}",
        "-t", str(duration), "-c:v", "libx264", "-preset", "ultrafast",
        "-y", str(path),
    ], check=True, capture_output=True)


def _make_narr_wav(path: Path, duration: float) -> None:
    silence = np.zeros(int(SR_NARR * duration), dtype="float32")
    write_wav(path, silence, SR_NARR)


def _make_music_wav(path: Path, duration: float) -> None:
    silence = np.zeros(int(SR_MUSIC * (duration + 0.5)), dtype="float32")
    write_wav(path, silence, SR_MUSIC)


def _make_fake_assets(spec, asset_dir: Path, fps=12) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    for beat in spec.beats:
        _make_clip(asset_dir / f"{beat.id}_clip.mp4", beat.duration_s, fps=fps)
        _make_narr_wav(asset_dir / f"{beat.id}.wav", min(beat.duration_s - 1, beat.duration_s))
        _make_music_wav(asset_dir / f"{beat.id}_music.wav", beat.duration_s)


# ---- LUT tests --------------------------------------------------------

def test_all_grades_generate_valid_cube():
    for name in GRADES:
        cube = generate_cube(name)
        lines = [l for l in cube.strip().splitlines() if not l.startswith("#")]
        assert lines[0] == "LUT_3D_SIZE 17"
        # 17³ = 4913 data lines + 1 header
        assert len(lines) == 17 ** 3 + 1


def test_ensure_luts_writes_files(tmp_path):
    paths = ensure_luts(tmp_path)
    assert len(paths) == len(GRADES)
    for name, p in paths.items():
        assert p.exists()
        assert p.suffix == ".cube"


def test_lut_cube_idempotent(tmp_path):
    paths1 = ensure_luts(tmp_path)
    paths2 = ensure_luts(tmp_path)
    for name in paths1:
        assert paths1[name].stat().st_mtime == paths2[name].stat().st_mtime


# ---- assemble tests ---------------------------------------------------

def test_assemble_produces_mp4(spec, tmp_path):
    _make_fake_assets(spec, tmp_path)
    r = LocalRenderer(out_dir=tmp_path, fps=12)
    out = str(tmp_path / "episode.mp4")
    gate = assemble_from_assets(spec, tmp_path, r, quiet(), out_path=out)
    assert gate.passed, gate.detail
    assert Path(out).exists()


def test_assemble_mp4_has_correct_resolution(spec, tmp_path):
    _make_fake_assets(spec, tmp_path)
    r = LocalRenderer(out_dir=tmp_path, fps=12)
    out = str(tmp_path / "episode.mp4")
    gate = assemble_from_assets(spec, tmp_path, r, quiet(), out_path=out)
    assert gate.passed
    W, H = spec.episode.resolution
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", out],
        capture_output=True, text=True, check=True,
    )
    w, h = (int(x) for x in probe.stdout.strip().split(","))
    assert w == W and h == H


def test_assemble_mp4_has_audio(spec, tmp_path):
    _make_fake_assets(spec, tmp_path)
    r = LocalRenderer(out_dir=tmp_path, fps=12)
    out = str(tmp_path / "episode.mp4")
    assemble_from_assets(spec, tmp_path, r, quiet(), out_path=out)
    audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type",
         "-of", "default=noprint_wrappers=1:nokey=1", out],
        capture_output=True, text=True,
    )
    assert audio.stdout.strip() == "audio"


def test_assemble_without_music_still_works(spec, tmp_path):
    # Only narration + clips, no music files
    _make_fake_assets(spec, tmp_path)
    for beat in spec.beats:
        (tmp_path / f"{beat.id}_music.wav").unlink(missing_ok=True)
    r = LocalRenderer(out_dir=tmp_path, fps=12)
    out = str(tmp_path / "episode_no_music.mp4")
    gate = assemble_from_assets(spec, tmp_path, r, quiet(), out_path=out)
    assert gate.passed, gate.detail


def test_assembly_verify_gate_wired_in_pipeline(spec):
    r = ScriptedRenderer()
    m = run(spec, r, quiet())
    assert m.ok
    assert m.assembly.passed


def test_assembly_verify_fails_loudly_on_wrong_duration(spec, tmp_path):
    gate = assembly_verify(
        exists=True, duration_s=10.0, expected_s=185.5,
        has_audio=True, resolution=(1920, 1080), expected_resolution=(1920, 1080),
    )
    assert not gate.passed
    assert "duration" in gate.detail
