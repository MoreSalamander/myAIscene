# my-AI-scene — ARCHITECTURE

How the doctrine in `CONSTITUTION.md` is realized. Read the Constitution
first.

---

## The pipeline (named stages, shared NDJSON vocabulary)

```
spec_load
  → narration_synth → narration_verify ─┐ (blocking)
  → duration_verify ─────────────────────┤ (blocking)
  → footage_synth   → footage_verify ────┤ (blocking, retry→neutral fallback)
  → music_synth     → music_cue_verify ──┘ (NON-blocking)
  → grade → assemble → assembly_verify     (blocking)
  → done
```

Each stage has one responsibility, an explicit name, and a visible data
connection to the next — the Build It Publisher lineage, rendered as
events. Events: `step_start`, `step_complete`, `token`, `done`, `error`,
plus `gate_pass` / `gate_fail` / `retry` / `fallback` / `skip`.

## Modules

| Module | Responsibility |
|---|---|
| `spec.py` | `ProductionSpec` / `Episode` / `Beat` dataclasses + JSON loader + structural validation |
| `events.py` | NDJSON event emitter + the event-name vocabulary |
| `verifiers.py` | the five pure-Python gates; each returns a `GateResult`, never raises on a content failure |
| `renderers.py` | `Renderer` protocol (`narrate`, `still`, `motion`, `music`, `assemble`) + `ScriptedRenderer` offline fake |
| `pipeline.py` | the named-stage orchestrator: bounded retry, fallback, event emission, manifest assembly |
| `cli.py` | `--spec`, `--dry-run`, `--out`, `--max-retries` |

## The render contract (`Renderer` protocol)

```python
class Renderer(Protocol):
    def narrate(self, beat) -> NarrationOut: ...   # audio path + transcript + duration_s
    def still(self, beat) -> StillOut: ...          # image path + clip_score vs prompt
    def motion(self, beat, still) -> ClipOut: ...   # ken-burns clip path + duration_s
    def music(self, beat) -> MusicOut | None: ...   # asset path + duration_s, or None
    def assemble(self, beat_clips, audio) -> ProbeOut: ...  # final mp4 + ffprobe facts
```

- **`ScriptedRenderer`** (Phase 1) returns deterministic, gate-passing
  fakes; can be seeded to fail any stage for tests.
- **`LocalRenderer`** (Phase 2+) implements the same protocol over the
  HuggingFace stack below.

## The HuggingFace local-free stack (M4 Pro / 24 GB / MPS)

| Stage | Model | License class | Notes |
|---|---|---|---|
| Narration | `hexgrad/Kokoro-82M` (British male `bm_george`) | permissive (Apache-2.0) | tiny, CPU-fast; Piper as studio-consistent fallback; Parler-TTS for describable voices |
| Stills | `stabilityai/sdxl-turbo` (default — fast, 4-step, fits 24 GB) | **non-commercial** (Stability NC) | gen at 1024×576; flagged NC like MusicGen |
| ↳ commercial tier | `stabilityai/stable-diffusion-xl-base-1.0` (25–30 step, slower on MPS) | OpenRAIL (commercial-ok) | swap via `--t2i-model` |
| ↳ quality tier | `black-forest-labs/FLUX.1-schnell` (12B — needs CPU offload on 24 GB, slow) | permissive (Apache-2.0) | not default: won't fit in VRAM without offload |
| Motion | **ffmpeg Ken Burns** (zoompan slow zoom/pan) over stills, scaled to `episode.resolution` | n/a | hybrid tier — no local T2V dependency; deterministic motion; duration ffprobe-verified |
| Music beds | `facebook/musicgen-medium` | **non-commercial (CC-BY-NC)** | portfolio-only; flagged |
| SFX / cues | `stabilityai/stable-audio-open-1.0` (≤47 s) | community (free) | non-blocking cues |
| Verify: words | `faster-whisper` (base) | permissive | transcript for `narration_verify` |
| Verify: visuals | `openai/clip-vit-large-patch14` | permissive | cosine score for `footage_verify` |
| Grade + assemble | **ffmpeg 8.0.1** | n/a | `lut3d` LUTs, `xfade`/`acrossfade`, `drawtext` titles, `ffprobe` facts |

**Visual tier = Hybrid stills + Ken Burns** (chosen for 24 GB). Local
text-to-video (LTX-Video / CogVideoX-2b) is *not* in the render path;
it can be added later as an alternate `motion()` impl behind the same
protocol if a cloud GPU becomes available. ffmpeg-driven slow pans,
soft zooms, and dissolves match the production guide's own direction
("slow pans, soft zooms, and dissolves only") and stay deterministic.

## Verification thresholds (config, not magic numbers in code)

| Gate | Metric | Default | Class |
|---|---|---|---|
| `narration_verify` | normalized token `SequenceMatcher` ratio | ≥ 0.80 | blocking |
| `duration_verify` | narration fits `[t0,t1]` window | overflow tol 0.5 s | blocking |
| `footage_verify` | CLIP cosine(prompt, image) | ≥ 0.22 | blocking → neutral fallback |
| `music_cue_verify` | asset exists & covers beat length | — | NON-blocking |
| `assembly_verify` | ffprobe: exists, dur ≈ Σ, audio track, WxH | dur tol 1.0 s | blocking |

## Audio/visual globals (from production-guide §2/§3)

VO ducked to `audio.vo_db` (-6 default) under beds; `audio.crossfade_s`
(1.75) between beats via `acrossfade`/`xfade`; per-beat LUT via
`grade.lut`; `episode.grain` film grain; serif title card
(`episode.font`, default Playfair Display) white-on-black `fade_s`.
