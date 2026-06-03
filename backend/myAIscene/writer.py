"""SpecWriter — turn a free-form brief into a validated ProductionSpec.

The "Explain" step automated: an LLM drafts the spec from a plain-text
brief; the existing pure-Python spec validator gates it; bounded retry
if the output is malformed. The grader is never the LLM — spec.py
disposes, as always.

Swappable backends: OllamaEngine (default, local-free) or any class
that implements LLMEngine. Claude API drops in as a one-line swap.

    writer = SpecWriter(OllamaEngine("llama3.1:8b"))
    spec   = writer.write("A 60-second doc about elevator etiquette")
    # → validated ProductionSpec, ready for my-AI-scene
"""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Protocol

from .spec import ProductionSpec, SpecError, parse_spec

MAX_RETRIES = 3


# ---- backend protocol (swap Ollama for Claude with one line) -----------

class LLMEngine(Protocol):
    def generate(self, system: str, prompt: str) -> str: ...


class OllamaEngine:
    """Local Ollama backend — free, private, runs on your machine.
    Uses JSON mode when the server supports it."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def generate(self, system: str, prompt: str) -> str:
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "format": "json",         # ask Ollama for JSON-constrained output
            "options": {"temperature": self.temperature},
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read())["response"]


class ScriptedLLMEngine:
    """Deterministic fake for tests — returns a pre-baked response."""
    def __init__(self, response: str):
        self._response = response
    def generate(self, system: str, prompt: str) -> str:
        return self._response


# ---- system prompt + schema --------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
You are a video production script writer. Your job is to take a user's
brief and output a complete ProductionSpec as a single JSON object.

The ProductionSpec format:

{
  "episode": {
    "title":     string   — episode title
    "genre":     string   — e.g. "mockumentary-comedy", "explainer", "documentary"
    "tone":      string   — narration style, e.g. "calm Attenborough, dry humor"
    "length_s":  number   — total seconds (must equal last beat's t1)
    "voice":     {"engine": "kokoro", "voice": "bm_george", "desc": "British male narrator"}
    "font":      string   — title font, default "Playfair Display"
    "grain":     number   — film grain 0.0–1.0, default 0.07
    "platform":  "youtube"
  },
  "beats": [
    {
      "id":             "b01"          — sequential, b01 b02 b03 …
      "t0":             number         — start seconds (first beat = 0)
      "t1":             number         — end seconds (each t1 == next t0)
      "narration":      string         — the spoken voiceover line
      "direction":      string         — scene direction (not spoken)
      "footage_prompt": string         — image generation prompt for this scene
      "music": {
        "keywords": [string, …]        — soundtrack search keywords
        "mood":     string
      },
      "grade": {
        "lut":  string    — one of: warm_documentary | cool_neutral | high_contrast |
                            desaturated | cool_vignette | clean_sterile | warm_orange |
                            low_contrast | bright_vivid | sepia_fade
        "note": string    — human-readable grade note
      }
    }
    … one object per scene …
  ],
  "audio": {"vo_db": -6, "crossfade_s": 1.75, "ambient": true},
  "titlecard": {"text": string, "subtitle": string, "fade_s": 2.0},
  "credits": {"script": string, "year": 2026}
}

Rules:
- beats must be contiguous: each beat's t1 equals the next beat's t0
- first beat t0 must be 0
- episode.length_s must equal the last beat's t1
- each beat should be 8–25 seconds (narration + breathing room)
- narration should be vivid, specific, and match the tone
- footage_prompt should describe a single cinematic still image
- output ONLY the JSON object — no markdown, no explanation, nothing else
""")


def _build_prompt(brief: str, target_duration_s: int = 120) -> str:
    return textwrap.dedent(f"""\
Brief: {brief}

Target duration: approximately {target_duration_s} seconds.

Write a complete ProductionSpec JSON for this brief. Choose the number of
beats that naturally fits the story — there is no fixed count. Each beat
is one scene. Output only the JSON, nothing else.
""")


# ---- writer ------------------------------------------------------------

class WriteResult:
    def __init__(self, spec: ProductionSpec, raw: str, attempts: int):
        self.spec = spec
        self.raw = raw
        self.attempts = attempts


class SpecWriteError(RuntimeError):
    pass


class SpecWriter:
    """Turns a plain-text brief into a validated ProductionSpec.

    Bounded retry: if the LLM output fails spec validation, the error
    is fed back as context for the next attempt. After max_retries the
    writer raises SpecWriteError — never silently accepts a bad spec.
    """

    def __init__(self, llm: LLMEngine, max_retries: int = MAX_RETRIES):
        self.llm = llm
        self.max_retries = max_retries

    def write(
        self,
        brief: str,
        target_duration_s: int = 120,
        emitter=None,
    ) -> WriteResult:
        def _emit(event, stage, **kw):
            if emitter:
                emitter.emit(event, stage, **kw)

        _emit("step_start", "spec_write", brief=brief[:80])

        last_error: str = ""
        for attempt in range(1, self.max_retries + 1):
            _emit("step_start", "llm_generate", attempt=attempt, model=getattr(self.llm, "model", "?"))

            prompt = _build_prompt(brief, target_duration_s)
            if last_error:
                prompt += f"\n\nPrevious attempt failed validation:\n{last_error}\n\nFix those issues and output only the corrected JSON."

            raw = self.llm.generate(_SYSTEM_PROMPT, prompt)
            _emit("step_complete", "llm_generate", attempt=attempt, chars=len(raw))

            _emit("step_start", "spec_validate", attempt=attempt)
            try:
                parsed = _extract_json(raw)
                spec = parse_spec(parsed)
                _emit("gate_pass", "spec_validate",
                      beats=len(spec.beats), length_s=spec.episode.length_s)
                _emit("step_complete", "spec_write", beats=len(spec.beats),
                      title=spec.episode.title)
                return WriteResult(spec=spec, raw=raw, attempts=attempt)

            except (SpecError, ValueError, json.JSONDecodeError) as e:
                last_error = str(e)
                _emit("gate_fail", "spec_validate", detail=last_error, attempt=attempt)
                if attempt < self.max_retries:
                    _emit("retry", "spec_write", attempt=attempt, detail=last_error)

        raise SpecWriteError(
            f"spec generation failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def write_to_file(
        self,
        brief: str,
        out_path: Path,
        target_duration_s: int = 120,
        emitter=None,
    ) -> WriteResult:
        result = self.write(brief, target_duration_s=target_duration_s, emitter=emitter)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Serialize the validated spec back to clean JSON
        out_path.write_text(
            json.dumps(_spec_to_dict(result.spec), indent=2, ensure_ascii=False)
        )
        return result


# ---- helpers -----------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Extract the first JSON object from LLM output.
    Handles markdown fences and leading/trailing prose."""
    text = text.strip()
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()
    # Find the outermost { … }
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in LLM output")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON object in LLM output")


def _spec_to_dict(spec: ProductionSpec) -> dict:
    """Convert a ProductionSpec back to a serialisable dict."""
    return {
        "episode": {
            "title":    spec.episode.title,
            "genre":    spec.episode.genre,
            "tone":     spec.episode.tone,
            "length_s": spec.episode.length_s,
            "platform": spec.episode.platform,
            "resolution": list(spec.episode.resolution),
            "font":     spec.episode.font,
            "grain":    spec.episode.grain,
            "voice": {
                "engine": spec.episode.voice.engine,
                "voice":  spec.episode.voice.voice,
                "desc":   spec.episode.voice.desc,
            },
        },
        "beats": [
            {
                "id":             b.id,
                "t0":             b.t0,
                "t1":             b.t1,
                "narration":      b.narration,
                "direction":      b.direction,
                "footage_prompt": b.footage_prompt,
                "music":          b.music,
                "grade":          b.grade,
            }
            for b in spec.beats
        ],
        "audio": {
            "vo_db":       spec.audio.vo_db,
            "crossfade_s": spec.audio.crossfade_s,
            "ambient":     spec.audio.ambient,
        },
        "titlecard": spec.titlecard,
        "credits":   spec.credits,
    }
