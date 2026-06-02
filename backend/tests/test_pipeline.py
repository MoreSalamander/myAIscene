import io
from pathlib import Path

import pytest

from myAIscene.events import EventEmitter
from myAIscene.pipeline import PipelineError, run
from myAIscene.renderers import ScriptedRenderer
from myAIscene.spec import load_spec

SPEC_PATH = Path(__file__).resolve().parents[2] / "specs" / "not_it_protocol.json"


@pytest.fixture
def spec():
    return load_spec(SPEC_PATH)


def quiet_emitter():
    return EventEmitter(out=None)


def test_full_pipeline_offline_passes(spec):
    m = run(spec, ScriptedRenderer(), quiet_emitter())
    assert m.ok
    assert len(m.beats) == 10
    assert m.assembly.passed
    s = m.gate_summary()
    assert s["failed"] == 0


def test_narration_failure_is_blocking(spec):
    # never heals → exhausts retries → PipelineError
    r = ScriptedRenderer(fail_narration_on={"b03"}, heal_after_retries=99)
    with pytest.raises(PipelineError, match="narration_verify"):
        run(spec, r, quiet_emitter())


def test_narration_retry_then_pass(spec):
    # fails first attempt, heals on retry → episode still ok
    r = ScriptedRenderer(fail_narration_on={"b03"}, heal_after_retries=1)
    m = run(spec, r, quiet_emitter(), max_retries=2)
    assert m.ok


def test_footage_failure_falls_back_not_fatal(spec):
    r = ScriptedRenderer(fail_footage_on={"b05"}, heal_after_retries=99)
    m = run(spec, r, quiet_emitter())
    assert m.ok  # footage falls back to neutral clip, episode survives
    b05 = next(b for b in m.beats if b.beat_id == "b05")
    assert b05.footage_fallback


def test_music_drop_is_non_blocking(spec):
    r = ScriptedRenderer(fail_music_on={"b02", "b07"})
    m = run(spec, r, quiet_emitter())
    assert m.ok
    dropped = {b.beat_id for b in m.beats if b.music_dropped}
    assert dropped == {"b02", "b07"}


def test_event_stream_has_shared_vocabulary(spec):
    buf = io.StringIO()
    em = EventEmitter(out=buf)
    run(spec, ScriptedRenderer(), em)
    names = {e["event"] for e in em.collected}
    assert {"step_start", "step_complete", "gate_pass", "done"} <= names
    assert em.collected[-1]["event"] == "done"
