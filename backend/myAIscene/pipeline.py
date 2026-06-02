"""The named-stage orchestrator (ARCHITECTURE pipeline diagram).

Bounded retry → defined fallback (CONSTITUTION Article IV). Emits the
shared NDJSON vocabulary. Produces a manifest of every beat, gate, and
the final assembly. The model proposes; this scaffold disposes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
    narr_path: str = ""           # set by run() — path to narration WAV
    music_path: str | None = None # set by run() — path to music WAV, or None if dropped
    window_s: float | None = None # override beat.duration_s for assembly (trim to narration)


@dataclass
class Manifest:
    title: str
    beats: list[BeatResult] = field(default_factory=list)
    assembly: V.GateResult | None = None
    out_path: str = ""
    ok: bool = False

    def gate_summary(self) -> dict[str, int]:
        passed = failed = 0
        for b in self.beats:
            for g in b.gates:
                (passed if g.passed else failed)
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
    raise PipelineError(
        f"{stage} failed for beat {beat_id} after {max_retries} retries: {last.detail}"
    )


def _expected_duration(spec: ProductionSpec) -> float:
    """Total expected output duration including title card."""
    title_s = (spec.titlecard.get("fade_s", 2.0) + 0.5) if spec.titlecard else 0.0
    return spec.episode.length_s + title_s


def run(
    spec: ProductionSpec,
    renderer: Renderer,
    emitter: EventEmitter | None = None,
    max_retries: int = MAX_RETRIES,
    out_path: str = "/tmp/myAIscene/episode.mp4",
) -> Manifest:
    em = emitter or EventEmitter()
    em.step_start("spec_load", title=spec.episode.title, beats=len(spec.beats))
    em.step_complete("spec_load")

    manifest = Manifest(title=spec.episode.title)

    for beat in spec.beats:
        br = BeatResult(beat_id=beat.id)

        # --- narration (blocking) + duration (blocking) ---
        em.step_start("narration_synth", beat=beat.id)
        narr, ngate, _ = _run_blocking(
            em, "narration_verify", beat.id,
            lambda b=beat: (
                lambda o: (o, V.narration_verify(b, o.transcript))
            )(renderer.narrate(b, spec)),
            max_retries,
        )
        br.gates.append(ngate)
        br.narr_path = narr.audio_path
        em.step_complete("narration_synth", beat=beat.id, audio=narr.audio_path)

        dgate = V.duration_verify(beat, narr.duration_s)
        br.gates.append(dgate)
        (em.gate_pass if dgate.passed else em.gate_fail)(
            "duration_verify", beat=beat.id, **dgate.metrics
        )
        if not dgate.passed:
            raise PipelineError(
                f"duration_verify failed for beat {beat.id}: {dgate.detail}"
            )

        # --- footage (blocking, retry → neutral fallback) ---
        em.step_start("footage_synth", beat=beat.id)
        try:
            still, fgate, _ = _run_blocking(
                em, "footage_verify", beat.id,
                lambda b=beat: (
                    lambda s: (s, V.footage_verify(b, s.clip_score))
                )(renderer.still(b, spec)),
                max_retries,
            )
        except PipelineError:
            br.footage_fallback = True
            em.fallback("footage_verify", beat=beat.id, to="neutral_clip")
            still = renderer.still(beat, spec)
            br.gates.append(
                V.GateResult("footage_verify", passed=False, blocking=True,
                             detail="fell back to neutral clip")
            )
        else:
            br.gates.append(fgate)
        clip = renderer.motion(beat, still, spec)
        br.clip = clip
        em.step_complete("footage_synth", beat=beat.id, clip=clip.clip_path)

        # --- music (NON-blocking) ---
        em.step_start("music_synth", beat=beat.id)
        music = renderer.music(beat, spec)
        mgate = V.music_cue_verify(
            beat,
            music.asset_path if music else None,
            music.duration_s if music else None,
        )
        br.gates.append(mgate)
        if mgate.passed:
            br.music_path = music.asset_path
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
    probe = renderer.assemble(spec.beats, manifest.beats, spec, out_path)
    expected_s = _expected_duration(spec)
    agate = V.assembly_verify(
        exists=probe.exists, duration_s=probe.duration_s, expected_s=expected_s,
        has_audio=probe.has_audio, resolution=probe.resolution,
        expected_resolution=spec.episode.resolution,
    )
    manifest.assembly = agate
    manifest.out_path = probe.path
    if agate.passed:
        em.gate_pass("assembly_verify", **agate.metrics)
    else:
        em.gate_fail("assembly_verify", detail=agate.detail, **agate.metrics)
        raise PipelineError(f"assembly_verify failed: {agate.detail}")
    em.step_complete("assemble", path=probe.path)

    manifest.ok = True
    em.done(path=probe.path, **manifest.gate_summary())
    return manifest


# ---- Phase-specific runners (report gates without raising) ----------------

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
    """Run only narration per beat — report gates, do not raise on content failure."""
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
        (em.gate_pass if ng.passed else em.gate_fail)(
            "narration_verify", beat=beat.id, detail=ng.detail, **ng.metrics
        )
        dg = V.duration_verify(beat, narr.duration_s)
        (em.gate_pass if dg.passed else em.gate_fail)(
            "duration_verify", beat=beat.id, detail=dg.detail, **dg.metrics
        )
        manifest.beats.append(NarrationBeatResult(
            beat_id=beat.id, audio_path=narr.audio_path, duration_s=narr.duration_s,
            narration_gate=ng, duration_gate=dg, transcript=narr.transcript,
        ))

    em.done(**manifest.summary())
    return manifest


@dataclass
class MusicBeatResult:
    beat_id: str
    asset_path: str | None
    duration_s: float | None
    gate: V.GateResult

    @property
    def ok(self) -> bool:
        return self.gate.passed


@dataclass
class MusicManifest:
    title: str
    beats: list[MusicBeatResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(b.ok for b in self.beats)

    def summary(self) -> dict[str, Any]:
        return {
            "beats": len(self.beats),
            "music_ok": sum(b.ok for b in self.beats),
            "music_dropped": sum(not b.ok for b in self.beats),
        }


def music_only(
    spec: ProductionSpec,
    renderer: Renderer,
    emitter: EventEmitter | None = None,
    limit: int | None = None,
) -> MusicManifest:
    """Generate music beds per beat; report non-blocking gate. Never raises."""
    em = emitter or EventEmitter()
    beats = spec.beats[:limit] if limit else spec.beats
    em.step_start("spec_load", title=spec.episode.title, beats=len(beats))
    em.step_complete("spec_load")

    manifest = MusicManifest(title=spec.episode.title)
    for beat in beats:
        em.step_start("music_synth", beat=beat.id)
        music = renderer.music(beat, spec)
        gate = V.music_cue_verify(
            beat,
            music.asset_path if music else None,
            music.duration_s if music else None,
        )
        if gate.passed:
            em.gate_pass("music_cue_verify", beat=beat.id, **gate.metrics)
        else:
            em.skip("music_cue_verify", beat=beat.id, detail=gate.detail)
        em.step_complete("music_synth", beat=beat.id)
        manifest.beats.append(MusicBeatResult(
            beat_id=beat.id,
            asset_path=music.asset_path if music else None,
            duration_s=music.duration_s if music else None,
            gate=gate,
        ))

    em.done(**manifest.summary())
    return manifest


def _wav_duration(path: Path) -> float | None:
    """Read duration of a WAV using stdlib wave — no ffprobe needed."""
    import wave as _wave
    try:
        with _wave.open(str(path), "r") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return None


def assemble_from_assets(
    spec: ProductionSpec,
    asset_dir: Path,
    renderer: Renderer,
    emitter: EventEmitter | None = None,
    out_path: str = "",
    narration_tail_s: float = 1.5,
) -> V.GateResult:
    """Assemble episode from pre-generated per-beat assets on disk.

    Expects in asset_dir:
        {beat.id}.wav          — narration WAV (required)
        {beat.id}_clip.mp4     — Ken Burns motion clip (required)
        {beat.id}_music.wav    — music bed WAV (optional)

    narration_tail_s: silence/visual breathing room appended after each
        narration ends (default 1.5s). Beats are trimmed to
        narr_dur + narration_tail_s rather than their full spec window,
        eliminating long empty gaps when narration is shorter than the window.

    Returns the assembly_verify GateResult.
    """
    from .renderers import ClipOut
    em = emitter or EventEmitter()
    out = out_path or str(asset_dir / "episode.mp4")

    em.step_start("assemble_from_assets")

    beat_results: list[BeatResult] = []
    for beat in spec.beats:
        br = BeatResult(beat_id=beat.id)
        narr_path = asset_dir / f"{beat.id}.wav"
        br.narr_path = str(narr_path)

        # Trim each beat to actual narration length + breathing room.
        # Prevents long silent gaps when narration is shorter than the spec window.
        narr_dur = _wav_duration(narr_path)
        if narr_dur is not None:
            br.window_s = min(narr_dur + narration_tail_s, beat.duration_s)
        # If WAV missing or unreadable, window_s stays None → assembler uses beat.duration_s

        clip_path = asset_dir / f"{beat.id}_clip.mp4"
        if clip_path.exists():
            br.clip = ClipOut(clip_path=str(clip_path), duration_s=beat.duration_s)
        music_path = asset_dir / f"{beat.id}_music.wav"
        br.music_path = str(music_path) if music_path.exists() else None
        br.music_dropped = br.music_path is None
        beat_results.append(br)

    probe = renderer.assemble(spec.beats, beat_results, spec, out, emitter=em)

    # Expected duration is the sum of actual windows used (not the spec declaration),
    # since we may have trimmed beats to narration length.
    title_s = (spec.titlecard.get("fade_s", 2.0) + 0.5) if spec.titlecard else 0.0
    expected_s = sum(
        (br.window_s if br.window_s is not None else beat.duration_s)
        for br, beat in zip(beat_results, spec.beats)
    ) + title_s

    gate = V.assembly_verify(
        exists=probe.exists, duration_s=probe.duration_s, expected_s=expected_s,
        has_audio=probe.has_audio, resolution=probe.resolution,
        expected_resolution=spec.episode.resolution,
    )
    (em.gate_pass if gate.passed else em.gate_fail)("assembly_verify", **gate.metrics)
    em.done(path=probe.path, passed=int(gate.passed), failed=int(not gate.passed))
    return gate
