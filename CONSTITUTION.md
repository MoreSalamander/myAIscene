# my-AI-scene — CONSTITUTION

The non-negotiable laws of the project. Edit this *before* changing any
dependent code. (Doctrine-first, per the MoreSalamander thesis.)

---

## Article I — The spec is the doctrine, not a suggestion

A **ProductionSpec** is authored by a human before any model runs. It is
the generalization of a *Journal* production guide (narration, scene,
soundtrack, color, title) into a machine-readable, machine-checkable
document. The five sections of a production guide map to spec fields:

| Production-guide section | Spec field |
|---|---|
| §1 Voiceover script & scene guide | `beats[].narration`, `beats[].direction`, `beats[].footage_prompt`, timecodes |
| §2 Soundtrack recommendations | `beats[].music` (keywords, mood) |
| §3 Color grading & visual style | `beats[].grade`, `episode.font`, `episode.grain`, transitions |
| §4 Workflow checklist | the pipeline itself (encoded, not a TODO list) |
| §5 Title card & credits | `episode.titlecard`, `episode.credits` |

The model fills volume **inside** the spec. It never edits the spec.

## Article II — The model proposes; Python disposes

Every model output crosses a deterministic gate before it is trusted,
scored, retried, or committed. **No verifier is ever an LLM** — the
grader cannot be the thing it grades (the MyMaestro lesson). Verifiers
are pure Python, ffprobe, or a CLIP cosine score with a fixed
threshold. Nothing more clever.

## Article III — Continuity is the premise; sound is enhancement

Gates split into two classes, exactly as in my-AI-story:

- **Blocking** (the premise): `narration_verify`, `duration_verify`,
  `footage_verify`, `assembly_verify`. Failure → bounded retry → then
  fallback or hard stop. A wrong word spoken, or a clip that doesn't
  depict its beat, fails the beat.
- **Non-blocking** (enhancement): `music_cue_verify`. A music asset that
  can't be produced or doesn't fit is *dropped*, never fails the
  episode. Sound enhances; continuity is the premise.

## Article IV — Bounded retry, then a defined fallback

No unbounded loops. Each synthesis stage retries up to `MAX_RETRIES`,
then takes a *named* fallback: a neutral clip for footage, silence for
music, a hard stop only for narration/assembly. Every retry and
fallback emits an event. Thrashing is a bug, not a strategy.

## Article V — Offline-provable before online-expensive

Every stage is defined behind a `Renderer` protocol with a
`ScriptedRenderer` fake. The entire pipeline must run, pass its gates,
and produce a manifest with **zero model downloads and zero GPU** (the
`--dry-run` path). Real models are an implementation of the protocol,
not the definition of the system.

## Article VI — Local and free

All synthesis runs on local HuggingFace weights on this machine
(Apple M4 Pro / 24 GB / MPS). No paid APIs in the render path.
License class is recorded per model; non-commercial weights (e.g.
MusicGen) are flagged in `ARCHITECTURE.md` and acceptable for portfolio
output only.

## Article VII — One observable pipeline

The pipeline emits the studio's shared NDJSON event vocabulary
(`step_start` / `step_complete` / `token` / `done` / `error`), one
stream, named stages, observable end-to-end. The same discipline as
my-AI-stro's three pipelines and my-AI-story's renderer.
