"""The five gates (CONSTITUTION Article II/III). Pure Python. No LLMs.

Each verifier returns a GateResult and never raises on a *content* failure
— the pipeline decides what to do with a failure (retry / fallback / skip /
stop). Verifiers raise only on programmer error (e.g. bad types).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from .spec import Beat

# thresholds (ARCHITECTURE table — tune here, not in callers)
NARRATION_MIN_RATIO = 0.80
DURATION_OVERFLOW_TOL_S = 0.5
FOOTAGE_MIN_CLIP = 0.22
ASSEMBLY_DUR_TOL_S = 1.0


@dataclass
class GateResult:
    gate: str
    passed: bool
    blocking: bool
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


def _normalize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def narration_verify(beat: Beat, transcript: str) -> GateResult:
    """Blocking. The spoken words must match the scripted narration."""
    want = _normalize(beat.narration)
    got = _normalize(transcript)
    ratio = SequenceMatcher(None, want, got).ratio() if want else 0.0
    passed = ratio >= NARRATION_MIN_RATIO
    return GateResult(
        gate="narration_verify", passed=passed, blocking=True,
        detail=f"token ratio {ratio:.3f} vs {NARRATION_MIN_RATIO}",
        metrics={"ratio": round(ratio, 4), "threshold": NARRATION_MIN_RATIO},
    )


def duration_verify(beat: Beat, audio_dur_s: float) -> GateResult:
    """Blocking. Narration must fit the beat's [t0,t1] window. Underflow is
    fine (padded with silence); overflow beyond tolerance fails."""
    window = beat.duration_s
    overflow = audio_dur_s - window
    passed = overflow <= DURATION_OVERFLOW_TOL_S
    return GateResult(
        gate="duration_verify", passed=passed, blocking=True,
        detail=f"audio {audio_dur_s:.2f}s vs window {window:.2f}s (overflow {overflow:+.2f}s)",
        metrics={"audio_s": round(audio_dur_s, 3), "window_s": round(window, 3),
                 "overflow_s": round(overflow, 3), "tol_s": DURATION_OVERFLOW_TOL_S},
    )


def footage_verify(beat: Beat, clip_score: float) -> GateResult:
    """Blocking (with retry→neutral fallback handled by the pipeline).
    The generated still must actually depict its footage_prompt."""
    passed = clip_score >= FOOTAGE_MIN_CLIP
    return GateResult(
        gate="footage_verify", passed=passed, blocking=True,
        detail=f"CLIP score {clip_score:.3f} vs {FOOTAGE_MIN_CLIP}",
        metrics={"clip_score": round(clip_score, 4), "threshold": FOOTAGE_MIN_CLIP},
    )


def music_cue_verify(beat: Beat, asset_path: str | None, asset_dur_s: float | None) -> GateResult:
    """NON-blocking (Article III). Music enhances; a missing or too-short
    bed is dropped, never fails the episode."""
    if not asset_path:
        return GateResult(gate="music_cue_verify", passed=False, blocking=False,
                          detail="no music asset produced")
    if asset_dur_s is None or asset_dur_s + 0.01 < beat.duration_s:
        return GateResult(gate="music_cue_verify", passed=False, blocking=False,
                          detail=f"music {asset_dur_s}s shorter than beat {beat.duration_s:.2f}s",
                          metrics={"asset_s": asset_dur_s, "beat_s": round(beat.duration_s, 3)})
    return GateResult(gate="music_cue_verify", passed=True, blocking=False,
                      detail="music covers beat",
                      metrics={"asset_s": round(asset_dur_s, 3)})


def assembly_verify(
    *, exists: bool, duration_s: float, expected_s: float,
    has_audio: bool, resolution: tuple[int, int], expected_resolution: tuple[int, int],
) -> GateResult:
    """Blocking. The commit boundary — final MP4 facts from ffprobe."""
    problems = []
    if not exists:
        problems.append("file missing")
    if abs(duration_s - expected_s) > ASSEMBLY_DUR_TOL_S:
        problems.append(f"duration {duration_s:.2f}s != expected {expected_s:.2f}s")
    if not has_audio:
        problems.append("no audio track")
    if tuple(resolution) != tuple(expected_resolution):
        problems.append(f"resolution {tuple(resolution)} != {tuple(expected_resolution)}")
    passed = not problems
    return GateResult(
        gate="assembly_verify", passed=passed, blocking=True,
        detail="ok" if passed else "; ".join(problems),
        metrics={"duration_s": round(duration_s, 3), "expected_s": round(expected_s, 3),
                 "has_audio": has_audio, "resolution": list(resolution)},
    )
