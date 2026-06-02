"""The named-stage orchestrator (ARCHITECTURE pipeline diagram).

Bounded retry → defined fallback (CONSTITUTION Article IV). Emits the
shared NDJSON vocabulary. Produces a manifest of every beat, gate, and
the final assembly. The model proposes; this scaffold disposes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import verifiers as V
from .events import EventEmitter
from .renderers import ClipOut, Renderer
from .spec import Beat, ProductionSpec

MAX_RETRIES = 2  # bounded; Article IV


class PipelineError(RuntimeError):
    """A blocking gate could not be satisfied within MAX_RETRIES."""


@dataclass
class BeatResult:
    beat_id: str
    gates: list[V.GateResult] = field(default_factory=list)
    footage_fallback: bool = False
    music_dropped: bool = False
    clip: ClipOut | None = None


@dataclass
class Manifest:
    title: str
    beats: list[BeatResult] = field(default_factory=list)
    assembly: V.GateResult | None = None
    ok: bool = False

    def gate_summary(self) -> dict[str, int]:
        passed = failed = 0
        for b in self.beats:
            for g in b.gates:
                if g.passed:
                    passed += 1
                else:
                    failed += 1
        if self.assembly:
            passed += int(self.assembly.passed)
            failed += int(not self.assembly.passed)
        return {"passed": passed, "failed": failed}


def _run_blocking(emitter, stage, beat_id, attempt_fn, max_retries):
    """Run a blocking synth+gate with bounded retry. attempt_fn() -> (out, GateResult).
    Returns the successful `out`, or raises PipelineError after retries."""
    last = None
    for attempt in range(max_retries + 1):
        out, gate = attempt_fn()
        last = gate
        if gate.passed:
            emitter.gate_pass(stage, beat=beat_id, **gate.metrics)
            return out, gate, attempt
        emitter.gate_fail(stage, beat=beat_id, detail=gate.detail, **gate.metrics)
        if attempt < max_retries:
            emitter.retry(stage, beat=beat_id, attempt=attempt + 1)
    raise PipelineError(f"{stage} failed for beat {beat_id} after {max_retries} retries: {last.detail}")


def run(
    spec: ProductionSpec,
    renderer: Renderer,
    emitter: EventEmitter | None = None,
    max_retries: int = MAX_RETRIES,
) -> Manifest:
    em = emitter or EventEmitter()
    em.step_start("spec_load", title=spec.episode.title, beats=len(spec.beats))
    em.step_complete("spec_load")

    manifest = Manifest(title=spec.episode.title)
    clips: list[ClipOut] = []

    for beat in spec.beats:
        br = BeatResult(beat_id=beat.id)

        # --- narration (blocking) + duration (blocking) ---
        em.step_start("narration_synth", beat=beat.id)
        narr, ngate, _ = _run_blocking(
            em, "narration_verify", beat.id,
            lambda: (lambda o: (o, V.narration_verify(beat, o.transcript)))(renderer.narrate(beat, spec)),
            max_retries,
        )
        br.gates.append(ngate)
        em.step_complete("narration_synth", beat=beat.id, audio=narr.audio_path)

        dgate = V.duration_verify(beat, narr.duration_s)
        br.gates.append(dgate)
        (em.gate_pass if dgate.passed else em.gate_fail)("duration_verify", beat=beat.id, **dgate.metrics)
        if not dgate.passed:
            raise PipelineError(f"duration_verify failed for beat {beat.id}: {dgate.detail}")

        # --- footage (blocking, retry → neutral fallback) ---
        em.step_start("footage_synth", beat=beat.id)
        try:
            still, fgate, _ = _run_blocking(
                em, "footage_verify", beat.id,
                lambda: (lambda s: (s, V.footage_verify(beat, s.clip_score)))(renderer.still(beat, spec)),
                max_retries,
            )
        except PipelineError:
            # defined fallback: a neutral clip rather than failing the episode
            br.footage_fallback = True
            em.fallback("footage_verify", beat=beat.id, to="neutral_clip")
            still = renderer.still(beat, spec)  # reused as neutral source
            br.gates.append(V.GateResult("footage_verify", passed=False, blocking=True,
                                         detail="fell back to neutral clip"))
        else:
            br.gates.append(fgate)
        clip = renderer.motion(beat, still, spec)
        br.clip = clip
        clips.append(clip)
        em.step_complete("footage_synth", beat=beat.id, clip=clip.clip_path)

        # --- music (NON-blocking) ---
        em.step_start("music_synth", beat=beat.id)
        music = renderer.music(beat, spec)
        mgate = V.music_cue_verify(beat, music.asset_path if music else None,
                                   music.duration_s if music else None)
        br.gates.append(mgate)
        if mgate.passed:
            em.gate_pass("music_cue_verify", beat=beat.id, **mgate.metrics)
        else:
            br.music_dropped = True
            em.skip("music_cue_verify", beat=beat.id, detail=mgate.detail)
        em.step_complete("music_synth", beat=beat.id)

        manifest.beats.append(br)

    # --- grade + assemble + verify (blocking) ---
    em.step_start("grade")
    em.step_complete("grade", luts=[b.grade.get("lut") for b in spec.beats])

    em.step_start("assemble")
    probe = renderer.assemble(clips, audio_path="/tmp/myAIscene/mix.wav", spec=spec)
    agate = V.assembly_verify(
        exists=probe.exists, duration_s=probe.duration_s, expected_s=spec.episode.length_s,
        has_audio=probe.has_audio, resolution=probe.resolution,
        expected_resolution=spec.episode.resolution,
    )
    manifest.assembly = agate
    if agate.passed:
        em.gate_pass("assembly_verify", **agate.metrics)
    else:
        em.gate_fail("assembly_verify", detail=agate.detail, **agate.metrics)
        raise PipelineError(f"assembly_verify failed: {agate.detail}")
    em.step_complete("assemble", path=probe.path)

    manifest.ok = True
    em.done(path=probe.path, **manifest.gate_summary())
    return manifest


# --- Phase 2: narration-only run (first real artifact = a timed VO track) ---

@dataclass
class NarrationBeatResult:
    beat_id: str
    audio_path: str
    duration_s: float
    narration_gate: V.GateResult
    duration_gate: V.GateResult
    transcript: str = ""

    @property
    def ok(self) -> bool:
        return self.narration_gate.passed and self.duration_gate.passed


@dataclass
class NarrationManifest:
    title: str
    beats: list[NarrationBeatResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(b.ok for b in self.beats)

    def summary(self) -> dict[str, Any]:
        return {
            "beats": len(self.beats),
            "narration_ok": sum(b.narration_gate.passed for b in self.beats),
            "duration_ok": sum(b.duration_gate.passed for b in self.beats),
            "total_audio_s": round(sum(b.duration_s for b in self.beats), 2),
        }


def narrate_only(
    spec: ProductionSpec,
    renderer: Renderer,
    emitter: EventEmitter | None = None,
    limit: int | None = None,
) -> NarrationManifest:
    """Run only narration_synth → narration_verify → duration_verify per
    beat, writing real VO and reporting both gates. Unlike run(), this does
    NOT raise on a content failure — its job is to produce the VO track and
    surface which beats need the script (human-owned) or window tightened.
    """
    em = emitter or EventEmitter()
    beats = spec.beats[:limit] if limit else spec.beats
    em.step_start("spec_load", title=spec.episode.title, beats=len(beats))
    em.step_complete("spec_load")

    manifest = NarrationManifest(title=spec.episode.title)
    for beat in beats:
        em.step_start("narration_synth", beat=beat.id)
        narr = renderer.narrate(beat, spec)
        em.step_complete("narration_synth", beat=beat.id,
                         audio=narr.audio_path, duration_s=round(narr.duration_s, 3))

        ng = V.narration_verify(beat, narr.transcript)
        (em.gate_pass if ng.passed else em.gate_fail)("narration_verify", beat=beat.id,
                                                       detail=ng.detail, **ng.metrics)
        dg = V.duration_verify(beat, narr.duration_s)
        (em.gate_pass if dg.passed else em.gate_fail)("duration_verify", beat=beat.id,
                                                      detail=dg.detail, **dg.metrics)

        manifest.beats.append(NarrationBeatResult(
            beat_id=beat.id, audio_path=narr.audio_path, duration_s=narr.duration_s,
            narration_gate=ng, duration_gate=dg, transcript=narr.transcript,
        ))

    em.done(**manifest.summary())
    return manifest
