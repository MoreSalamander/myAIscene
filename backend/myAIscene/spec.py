"""ProductionSpec schema + loader + structural validation.

The spec is the doctrine (CONSTITUTION Article I). Structural problems are
authoring bugs and raise SpecError *before* the pipeline runs — they are
never a model failure, so they fail loud and early.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# tolerances (ARCHITECTURE: config, not magic numbers buried in logic)
LENGTH_TOL_S = 0.5
CONTIG_TOL_S = 0.01


class SpecError(ValueError):
    """A structural/authoring error in a ProductionSpec."""


@dataclass(frozen=True)
class Voice:
    engine: str
    voice: str
    desc: str = ""


@dataclass(frozen=True)
class Beat:
    id: str
    t0: float
    t1: float
    narration: str
    direction: str
    footage_prompt: str
    music: dict[str, Any] = field(default_factory=dict)
    grade: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.t1 - self.t0


@dataclass(frozen=True)
class Episode:
    title: str
    genre: str
    tone: str
    length_s: float
    voice: Voice
    platform: str = "youtube"
    aspect: str = "16:9"
    resolution: tuple[int, int] = (1920, 1080)
    font: str = "Playfair Display"
    grain: float = 0.07


@dataclass(frozen=True)
class Audio:
    vo_db: float = -6.0
    crossfade_s: float = 1.75
    ambient: bool = True


@dataclass(frozen=True)
class ProductionSpec:
    episode: Episode
    beats: list[Beat]
    audio: Audio
    titlecard: dict[str, Any] = field(default_factory=dict)
    credits: dict[str, Any] = field(default_factory=dict)


def _require(d: dict, keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise SpecError(f"{where}: missing required field(s): {', '.join(missing)}")


def load_spec(path: str | Path) -> ProductionSpec:
    """Load and structurally validate a ProductionSpec JSON file."""
    p = Path(path)
    try:
        raw = json.loads(p.read_text())
    except FileNotFoundError as e:
        raise SpecError(f"spec file not found: {p}") from e
    except json.JSONDecodeError as e:
        raise SpecError(f"spec is not valid JSON: {e}") from e
    return parse_spec(raw)


def parse_spec(raw: dict[str, Any]) -> ProductionSpec:
    _require(raw, ("episode", "beats"), "spec")
    ep_raw = raw["episode"]
    _require(ep_raw, ("title", "genre", "tone", "length_s", "voice"), "episode")
    _require(ep_raw["voice"], ("engine", "voice"), "episode.voice")

    res = ep_raw.get("resolution", [1920, 1080])
    episode = Episode(
        title=ep_raw["title"],
        genre=ep_raw["genre"],
        tone=ep_raw["tone"],
        length_s=float(ep_raw["length_s"]),
        voice=Voice(**{k: ep_raw["voice"][k] for k in ("engine", "voice", "desc") if k in ep_raw["voice"]}),
        platform=ep_raw.get("platform", "youtube"),
        aspect=ep_raw.get("aspect", "16:9"),
        resolution=(int(res[0]), int(res[1])),
        font=ep_raw.get("font", "Playfair Display"),
        grain=float(ep_raw.get("grain", 0.07)),
    )

    beats: list[Beat] = []
    for i, b in enumerate(raw["beats"]):
        _require(b, ("id", "t0", "t1", "narration", "direction", "footage_prompt"), f"beats[{i}]")
        beats.append(Beat(
            id=b["id"], t0=float(b["t0"]), t1=float(b["t1"]),
            narration=b["narration"], direction=b["direction"],
            footage_prompt=b["footage_prompt"],
            music=b.get("music", {}), grade=b.get("grade", {}),
        ))

    a = raw.get("audio", {})
    audio = Audio(
        vo_db=float(a.get("vo_db", -6.0)),
        crossfade_s=float(a.get("crossfade_s", 1.75)),
        ambient=bool(a.get("ambient", True)),
    )

    spec = ProductionSpec(
        episode=episode, beats=beats, audio=audio,
        titlecard=raw.get("titlecard", {}), credits=raw.get("credits", {}),
    )
    _validate_invariants(spec)
    return spec


def _validate_invariants(spec: ProductionSpec) -> None:
    """SPEC.md beat invariants — contiguity, ordering, ids, total length."""
    beats = spec.beats
    if not beats:
        raise SpecError("spec has no beats")

    ids = [b.id for b in beats]
    if len(ids) != len(set(ids)):
        raise SpecError("beat ids are not unique")

    if abs(beats[0].t0) > CONTIG_TOL_S:
        raise SpecError(f"first beat must start at t0=0, got {beats[0].t0}")

    for i, b in enumerate(beats):
        if b.t1 <= b.t0:
            raise SpecError(f"beat {b.id}: t1 ({b.t1}) must be > t0 ({b.t0})")
        if i + 1 < len(beats):
            nxt = beats[i + 1]
            if abs(b.t1 - nxt.t0) > CONTIG_TOL_S:
                raise SpecError(
                    f"beats not contiguous: {b.id}.t1={b.t1} != {nxt.id}.t0={nxt.t0}"
                )

    total = beats[-1].t1
    if abs(total - spec.episode.length_s) > LENGTH_TOL_S:
        raise SpecError(
            f"episode.length_s ({spec.episode.length_s}) != last beat t1 ({total})"
        )
