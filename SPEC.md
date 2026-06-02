# my-AI-scene — SPEC (ProductionSpec schema)

The authoring contract. A spec is a JSON file (zero-dep loading; YAML
authoring may be layered later). Loaded and structurally validated by
`spec.py` before the pipeline starts. Structural failures are a hard
error *here* — they are an authoring bug, not a model failure.

---

## Top level

```jsonc
{
  "episode": { ... },   // metadata + global style
  "beats":   [ ... ],   // ordered scenes, one per production-guide timecode row
  "audio":   { ... },   // global mix
  "titlecard": { ... },
  "credits": { ... }
}
```

## `episode`

| Field | Type | Required | Meaning |
|---|---|---|---|
| `title` | str | ✓ | episode title |
| `genre` | str | ✓ | e.g. `mockumentary-comedy` |
| `tone` | str | ✓ | narration direction, e.g. "calm Attenborough, dry humor" |
| `length_s` | number | ✓ | target total seconds (must ≈ last beat `t1`) |
| `platform` | str | | `youtube` |
| `aspect` | str | | `16:9` (default) |
| `resolution` | [w,h] | | `[1920,1080]` (default) |
| `font` | str | | title font, default `Playfair Display` |
| `grain` | number | | film grain 0–1, default `0.07` |
| `voice` | object | ✓ | `{engine, voice, desc}` |

## `beats[]` (the heart — one per timecode row)

| Field | Type | Required | Maps to |
|---|---|---|---|
| `id` | str | ✓ | stable beat id, e.g. `b01` |
| `t0`, `t1` | number | ✓ | start/end seconds; must be contiguous & increasing |
| `narration` | str | ✓ | §1 voiceover line (verified against spoken audio) |
| `direction` | str | ✓ | §1 scene direction (human note; not synthesized) |
| `footage_prompt` | str | ✓ | image prompt (verified via CLIP) |
| `music` | object | | `{keywords:[], mood}` — §2 |
| `grade` | object | | `{lut, note}` — §3 per-scene grade |

**Invariants (validated at load):** beats ordered; `t0[0]==0`;
`t1[i]==t0[i+1]` (contiguous); each `t1>t0`; `length_s ≈ t1[-1]`
(tol 0.5 s); ids unique.

## `audio`

| Field | Default | Meaning |
|---|---|---|
| `vo_db` | -6 | voiceover gain under music |
| `crossfade_s` | 1.75 | between-beat crossfade |
| `ambient` | true | subtle ambient bed |

## `titlecard` / `credits`

`titlecard`: `{text, subtitle, fade_s}`. `credits`: free-form key/values
rendered at the end (narration, script, year, channel).

---

## Reference spec

`specs/not_it_protocol.json` — the production guide in this document set,
ported field-for-field. It is the canonical example and the Phase-1
test fixture. Authoring a new *Journal* episode = writing a new spec
file; the engine is unchanged.
