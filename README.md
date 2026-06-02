# my-AI-scene

> *Same thesis, third medium.*

A local-first, verification-driven **video production engine** for
[*The Journal of Informal Human Protocols*](https://github.com/) — the
MoreSalamander Productions mockumentary series. A human authors a
**ProductionSpec** (the doctrine); local HuggingFace models synthesize
the narration, footage, and music inside it; pure-Python verifiers
decide what survives to the final cut.

The video sibling of [my-AI-story](../myAIstory). Where my-AI-story
turns a series bible into a multi-voice audio episode, **my-AI-scene
turns a production guide into a graded, scored, assembled MP4** — under
the same [Deterministic Scaffold](../README.md) thesis:

> *A well-fenced model inside a deterministic scaffold becomes reliable
> as a system.* Human-owned constraints → AI synthesis → deterministic
> verification at every boundary.

The naming trio: **my-AI-stro** (maestro — the conductor),
**my-AI-story** (the written episode), **my-AI-scene** (the visual cut).

---

## Status

- **Phase 0 — docs:** ✅ `CONSTITUTION.md` / `ARCHITECTURE.md` / `SPEC.md`
- **Phase 1 — schema + verifiers (offline, scripted fakes):** ✅
- **Phase 2 — narration (Kokoro `bm_george` + Whisper gate):** ✅ real VO rendered, both gates green
- Phase 3 — visuals (FLUX stills + Ken Burns + CLIP gate): pending
- Phase 4 — music (MusicGen beds + Stable Audio cues, non-blocking): pending
- Phase 5 — grade + assemble (ffmpeg LUT/xfade/title → first MP4): pending
- Phase 6 — FastAPI + NDJSON web UI: pending

## Run it

```bash
cd backend
# Phase 1 — prove the scaffold offline (no models, deterministic):
./.venv/bin/python -m myAIscene.cli --spec ../specs/not_it_protocol.json --dry-run
./.venv/bin/python -m pytest -q

# Phase 2 — render real narration + Whisper-verified gates:
./.venv/bin/python -m myAIscene.cli --spec ../specs/not_it_protocol.json \
    --narrate --out-dir ../out/not_it/narration [--limit 3] [--voice bm_george]
```

`--dry-run` drives the full pipeline with `ScriptedRenderer` — no model
downloads, no GPU. The whole scaffold is provable before a weight loads.

### Setup (Phase 2 deps)

The venv **must be Python 3.12** (torch / ctranslate2 / kokoro have no
3.14 wheels yet):

```bash
python3.12 -m venv backend/.venv
backend/.venv/bin/pip install pytest numpy kokoro faster-whisper
brew install espeak-ng                                   # Kokoro British G2P
backend/.venv/bin/python -m spacy download en_core_web_sm  # avoids a runtime stall
```

`local.py` auto-points espeak at the Homebrew dylib. First synth also
lazily fetches `en_core_web_sm` if missing — pre-installing it avoids a
one-time network hang.

---

*A MoreSalamander StudioLabs production. Scientia Ludusque.*
