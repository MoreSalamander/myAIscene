"""SpecWriter tests — driven by ScriptedLLMEngine, zero Ollama calls."""
import json
from pathlib import Path

import pytest

from myAIscene.writer import (
    ScriptedLLMEngine,
    SpecWriteError,
    SpecWriter,
    _extract_json,
    _spec_to_dict,
)


# ---- minimal valid spec the fake LLM returns ---------------------------

VALID_SPEC = {
    "episode": {
        "title": "The Elevator Code",
        "genre": "mockumentary-comedy",
        "tone": "calm Attenborough, dry humor",
        "length_s": 30,
        "voice": {"engine": "kokoro", "voice": "bm_george", "desc": "British male"},
    },
    "beats": [
        {
            "id": "b01", "t0": 0, "t1": 15,
            "narration": "In the enclosed habitat of the elevator, unspoken laws govern all.",
            "direction": "Slow pan across elevator interior.",
            "footage_prompt": "empty office elevator interior, cinematic",
            "music": {"keywords": ["gentle documentary"], "mood": "curious"},
            "grade": {"lut": "warm_documentary", "note": "warm"},
        },
        {
            "id": "b02", "t0": 15, "t1": 30,
            "narration": "The cardinal rule: eyes forward. Always eyes forward.",
            "direction": "Close-up of elevator buttons.",
            "footage_prompt": "close-up of elevator floor number buttons, shallow focus",
            "music": {"keywords": ["building tension"], "mood": "tense"},
            "grade": {"lut": "cool_neutral", "note": "cool"},
        },
    ],
    "audio": {"vo_db": -6, "crossfade_s": 1.75, "ambient": True},
    "titlecard": {"text": "The Elevator Code", "subtitle": "A Social Contract", "fade_s": 2.0},
    "credits": {"year": 2026},
}

VALID_JSON = json.dumps(VALID_SPEC)


# ---- _extract_json -----------------------------------------------------

def test_extract_bare_json():
    assert _extract_json(VALID_JSON)["episode"]["title"] == "The Elevator Code"


def test_extract_strips_markdown_fences():
    wrapped = f"```json\n{VALID_JSON}\n```"
    assert _extract_json(wrapped)["episode"]["title"] == "The Elevator Code"


def test_extract_ignores_leading_prose():
    messy = f"Here is your spec:\n\n{VALID_JSON}\n\nHope that helps!"
    assert _extract_json(messy)["episode"]["title"] == "The Elevator Code"


def test_extract_raises_on_no_json():
    with pytest.raises(ValueError, match="no JSON"):
        _extract_json("sorry I cannot do that")


# ---- SpecWriter: happy path --------------------------------------------

def test_writer_returns_validated_spec():
    w = SpecWriter(ScriptedLLMEngine(VALID_JSON))
    result = w.write("elevator etiquette rules")
    assert result.spec.episode.title == "The Elevator Code"
    assert len(result.spec.beats) == 2
    assert result.attempts == 1


def test_writer_retries_on_bad_json_then_heals():
    bad_then_good = iter(["{not json}", VALID_JSON])
    class HealingLLM:
        def generate(self, system, prompt):
            return next(bad_then_good)
    w = SpecWriter(HealingLLM(), max_retries=2)
    result = w.write("elevator etiquette")
    assert result.spec is not None
    assert result.attempts == 2


def test_writer_raises_after_max_retries():
    w = SpecWriter(ScriptedLLMEngine("{bad json always}"), max_retries=2)
    with pytest.raises(SpecWriteError, match="failed after 2"):
        w.write("something")


def test_writer_emits_ndjson_events():
    from myAIscene.events import EventEmitter
    em = EventEmitter(out=None)
    w = SpecWriter(ScriptedLLMEngine(VALID_JSON))
    w.write("elevator", emitter=em)
    names = [e["event"] for e in em.collected]
    assert "step_start" in names
    assert "gate_pass" in names
    assert "step_complete" in names


def test_write_to_file(tmp_path):
    w = SpecWriter(ScriptedLLMEngine(VALID_JSON))
    out = tmp_path / "elevator.json"
    w.write_to_file("elevator etiquette", out)
    assert out.exists()
    # round-trip: the saved file should load back as a valid spec
    from myAIscene.spec import load_spec
    reloaded = load_spec(out)
    assert reloaded.episode.title == "The Elevator Code"


# ---- _spec_to_dict round-trip ------------------------------------------

def test_spec_to_dict_round_trips():
    from myAIscene.spec import parse_spec
    spec = parse_spec(VALID_SPEC)
    d = _spec_to_dict(spec)
    reloaded = parse_spec(d)
    assert reloaded.episode.title == spec.episode.title
    assert len(reloaded.beats) == len(spec.beats)
