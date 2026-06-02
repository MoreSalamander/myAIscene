import copy
import json
from pathlib import Path

import pytest

from myAIscene.spec import SpecError, load_spec, parse_spec

SPEC_PATH = Path(__file__).resolve().parents[2] / "specs" / "not_it_protocol.json"


@pytest.fixture
def raw():
    return json.loads(SPEC_PATH.read_text())


def test_reference_spec_loads():
    spec = load_spec(SPEC_PATH)
    assert spec.episode.title.startswith("The Not It Protocol")
    assert len(spec.beats) == 10
    assert spec.beats[0].t0 == 0
    assert spec.beats[-1].t1 == spec.episode.length_s == 183
    assert spec.episode.voice.engine == "kokoro"
    assert spec.audio.vo_db == -6


def test_beats_contiguous(raw):
    spec = parse_spec(raw)
    for a, b in zip(spec.beats, spec.beats[1:]):
        assert a.t1 == b.t0


def test_missing_required_field(raw):
    del raw["episode"]["title"]
    with pytest.raises(SpecError, match="title"):
        parse_spec(raw)


def test_non_contiguous_beats_rejected(raw):
    raw["beats"][2]["t0"] += 3  # punch a gap
    with pytest.raises(SpecError, match="contiguous"):
        parse_spec(raw)


def test_length_mismatch_rejected(raw):
    raw["episode"]["length_s"] = 999
    with pytest.raises(SpecError, match="length_s"):
        parse_spec(raw)


def test_duplicate_ids_rejected(raw):
    raw["beats"][1]["id"] = raw["beats"][0]["id"]
    with pytest.raises(SpecError, match="unique"):
        parse_spec(raw)


def test_first_beat_must_start_at_zero(raw):
    raw = copy.deepcopy(raw)
    raw["beats"][0]["t0"] = 1.0
    with pytest.raises(SpecError, match="t0=0"):
        parse_spec(raw)


def test_defaults_applied():
    minimal = {
        "episode": {"title": "T", "genre": "g", "tone": "t", "length_s": 5,
                    "voice": {"engine": "kokoro", "voice": "bm_george"}},
        "beats": [{"id": "b1", "t0": 0, "t1": 5, "narration": "hi",
                   "direction": "d", "footage_prompt": "f"}],
    }
    spec = parse_spec(minimal)
    assert spec.episode.font == "Playfair Display"
    assert spec.audio.crossfade_s == 1.75
    assert spec.episode.resolution == (1920, 1080)
