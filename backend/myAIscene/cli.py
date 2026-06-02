"""my-AI-scene CLI.

    # Phase 1 — prove the scaffold offline (no models):
    python -m myAIscene.cli --spec ../specs/not_it_protocol.json --dry-run

    # Phase 2 — render real narration (Kokoro) + Whisper-verified gates:
    python -m myAIscene.cli --spec ../specs/not_it_protocol.json --narrate \
        --out-dir ../out/not_it/narration [--voice bm_george] [--limit 2]
"""
from __future__ import annotations

import argparse
import sys

from .events import EventEmitter
from .pipeline import MAX_RETRIES, PipelineError, narrate_only, run
from .renderers import ScriptedRenderer
from .spec import SpecError, load_spec


def _run_dry(spec, max_retries: int) -> int:
    em = EventEmitter()
    try:
        manifest = run(spec, ScriptedRenderer(), em, max_retries=max_retries)
    except PipelineError as e:
        print(f"pipeline halted (blocking gate): {e}", file=sys.stderr)
        return 1
    s = manifest.gate_summary()
    print(f"\nOK — {manifest.title}", file=sys.stderr)
    print(f"  beats: {len(manifest.beats)}  gates passed: {s['passed']}  failed: {s['failed']}",
          file=sys.stderr)
    return 0


def _run_narrate(spec, out_dir: str, voice: str | None, whisper_model: str, limit: int | None) -> int:
    from .local import LocalRenderer, WhisperASR  # lazy: heavy imports only on this path
    renderer = LocalRenderer(out_dir=out_dir, voice=voice, asr=WhisperASR(model_size=whisper_model))
    em = EventEmitter()
    manifest = narrate_only(spec, renderer, em, limit=limit)

    s = manifest.summary()
    print(f"\n{'OK' if manifest.ok else 'FAIL'} — narration: {manifest.title}", file=sys.stderr)
    print(f"  beats: {s['beats']}  narration_ok: {s['narration_ok']}  "
          f"duration_ok: {s['duration_ok']}  total VO: {s['total_audio_s']}s", file=sys.stderr)
    for b in manifest.beats:
        if not b.ok:
            why = []
            if not b.narration_gate.passed:
                why.append(f"narration({b.narration_gate.detail})")
            if not b.duration_gate.passed:
                why.append(f"duration({b.duration_gate.detail})")
            print(f"  - {b.beat_id}: {'; '.join(why)}", file=sys.stderr)
    return 0 if manifest.ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="myAIscene", description="local-first video production engine")
    ap.add_argument("--spec", required=True, help="path to a ProductionSpec JSON file")
    ap.add_argument("--dry-run", action="store_true",
                    help="full pipeline with the offline ScriptedRenderer (no models)")
    ap.add_argument("--narrate", action="store_true",
                    help="Phase 2: render real narration (Kokoro) + Whisper-verified gates")
    ap.add_argument("--out-dir", default="/tmp/myAIscene/narration", help="narration output dir")
    ap.add_argument("--voice", default=None, help="override episode voice (e.g. bm_george)")
    ap.add_argument("--whisper-model", default="base", help="faster-whisper model size")
    ap.add_argument("--limit", type=int, default=None, help="only the first N beats (smoke test)")
    ap.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    args = ap.parse_args(argv)

    try:
        spec = load_spec(args.spec)
    except SpecError as e:
        print(f"spec error: {e}", file=sys.stderr)
        return 2

    if args.narrate:
        return _run_narrate(spec, args.out_dir, args.voice, args.whisper_model, args.limit)
    if args.dry_run:
        return _run_dry(spec, args.max_retries)

    print("choose a mode: --dry-run (offline scaffold) or --narrate (Phase 2 real VO).",
          file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
