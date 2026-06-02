"""The Renderer protocol + the offline ScriptedRenderer fake.

Per CONSTITUTION Article V, the whole pipeline is defined behind this
protocol so it can be proven offline with zero model downloads. The real
LocalRenderer (Phase 2+) implements the same contract over the
HuggingFace stack documented in ARCHITECTURE.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .spec import Beat, ProductionSpec


# ---- render outputs ----------------------------------------------------

@dataclass
class NarrationOut:
    audio_path: str
    transcript: str       # what the TTS model actually rendered (for narration_verify)
    duration_s: float


@dataclass
class StillOut:
    image_path: str
    clip_score: float     # CLIP cosine(prompt, image), for footage_verify


@dataclass
class ClipOut:
    clip_path: str        # ken-burns motion clip built from the still
    duration_s: float


@dataclass
class MusicOut:
    asset_path: str
    duration_s: float


@dataclass
class ProbeOut:
    path: str
    exists: bool
    duration_s: float
    has_audio: bool
    resolution: tuple[int, int]


class Renderer(Protocol):
    def narrate(self, beat: Beat, spec: ProductionSpec) -> NarrationOut: ...
    def still(self, beat: Beat, spec: ProductionSpec) -> StillOut: ...
    def motion(self, beat: Beat, still: StillOut, spec: ProductionSpec) -> ClipOut: ...
    def music(self, beat: Beat, spec: ProductionSpec) -> MusicOut | None: ...
    def assemble(self, beats: list, beat_results: list, spec: ProductionSpec, out_path: str, *, emitter=None) -> ProbeOut: ...


# ---- offline fake ------------------------------------------------------

@dataclass
class ScriptedRenderer:
    """Deterministic, gate-passing fake. Inject failures to exercise the
    pipeline's retry/fallback/skip logic in tests.

    fail_narration_on / fail_footage_on: sets of beat ids that should fail
      their gate (and self-heal after `heal_after_retries` retries, so we
      can also test the retry-then-pass path).
    fail_music_on: beat ids that produce no music (drives non-blocking skip).
    """
    fail_narration_on: set[str] = field(default_factory=set)
    fail_footage_on: set[str] = field(default_factory=set)
    fail_music_on: set[str] = field(default_factory=set)
    heal_after_retries: int = 0
    _attempts: dict[str, int] = field(default_factory=dict)

    def _attempt(self, key: str) -> int:
        self._attempts[key] = self._attempts.get(key, 0) + 1
        return self._attempts[key]

    def narrate(self, beat: Beat, spec: ProductionSpec) -> NarrationOut:
        n = self._attempt(f"narrate:{beat.id}")
        transcript = ("totally different words that will not match"
                      if beat.id in self.fail_narration_on and n <= self.heal_after_retries
                      else beat.narration)
        return NarrationOut(
            audio_path=f"/tmp/myAIscene/{beat.id}.wav",
            transcript=transcript,
            duration_s=min(beat.duration_s - 0.2, beat.duration_s),
        )

    def still(self, beat: Beat, spec: ProductionSpec) -> StillOut:
        n = self._attempt(f"still:{beat.id}")
        score = 0.05 if (beat.id in self.fail_footage_on and n <= self.heal_after_retries) else 0.31
        return StillOut(image_path=f"/tmp/myAIscene/{beat.id}.png", clip_score=score)

    def motion(self, beat: Beat, still: StillOut, spec: ProductionSpec) -> ClipOut:
        return ClipOut(clip_path=f"/tmp/myAIscene/{beat.id}.mp4", duration_s=beat.duration_s)

    def music(self, beat: Beat, spec: ProductionSpec) -> MusicOut | None:
        if beat.id in self.fail_music_on:
            return None
        return MusicOut(asset_path=f"/tmp/myAIscene/{beat.id}_music.wav",
                        duration_s=beat.duration_s + 0.5)

    def assemble(self, beats: list, beat_results: list, spec: ProductionSpec, out_path: str, *, emitter=None) -> ProbeOut:
        total = sum(b.t1 - b.t0 for b in beats)
        title_s = (spec.titlecard.get("fade_s", 2.0) + 0.5) if spec.titlecard else 0.0
        if emitter:
            for i, (beat, br) in enumerate(zip(beats, beat_results)):
                emitter.emit("step_start", "grade_mix", beat=beat.id, index=i+1, total=len(beats))
                emitter.emit("step_complete", "grade_mix", beat=beat.id, index=i+1, total=len(beats))
            for stage in ("concat", "titlecard", "grain"):
                emitter.emit("step_start", stage)
                emitter.emit("step_complete", stage)
        return ProbeOut(
            path=out_path, exists=True, duration_s=total + title_s,
            has_audio=True, resolution=spec.episode.resolution,
        )
