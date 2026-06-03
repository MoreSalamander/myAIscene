"""my-AI-scene CLI — renders a ProductionSpec into video.

To generate a spec from a brief, use my-AI-script (../myAIscript).

    # Phase 1 — prove the scaffold offline (no models):
    python -m myAIscene.cli --spec ../specs/not_it_protocol.json --dry-run

    # Phase 2 — render narration (Kokoro) + Whisper-verified gates:
    python -m myAIscene.cli --spec ../specs/not_it_protocol.json \\
        --narrate --out-dir ../out/not_it

    # Phase 3 — generate SDXL stills + Ken Burns clips:
    python -m myAIscene.cli --spec ../specs/not_it_protocol.json \\
        --visual --out-dir ../out/not_it

    # Phase 4 — generate music beds (MusicGen):
    python -m myAIscene.cli --spec ../specs/not_it_protocol.json \\
        --music --out-dir ../out/not_it

    # Phase 5 — assemble episode from pre-generated per-beat assets:
    python -m myAIscene.cli --spec ../specs/not_it_protocol.json \\
        --assemble --in-dir ../out/not_it [--out ../out/not_it/episode.mp4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .events import EventEmitter
from .pipeline import (MAX_RETRIES, PipelineError, assemble_from_assets,
                        music_only, narrate_only, run, visual_only)
from .renderers import ScriptedRenderer
from .spec import SpecError, load_spec


def _quiet():
    return EventEmitter()


def _run_dry(spec, max_retries: int) -> int:
    em = _quiet()
    try:
        m = run(spec, ScriptedRenderer(), em, max_retries=max_retries)
    except PipelineError as e:
        print(f"pipeline halted: {e}", file=sys.stderr)
        return 1
    s = m.gate_summary()
    print(f"\nOK — {m.title}", file=sys.stderr)
    print(f"  beats:{len(m.beats)}  passed:{s['passed']}  failed:{s['failed']}", file=sys.stderr)
    return 0


def _run_narrate(spec, out_dir, voice, whisper_model, limit) -> int:
    from .local import LocalRenderer, WhisperASR
    r = LocalRenderer(out_dir=out_dir, voice=voice, asr=WhisperASR(model_size=whisper_model))
    m = narrate_only(spec, r, _quiet(), limit=limit)
    s = m.summary()
    print(f"\n{'OK' if m.ok else 'FAIL'} — narration: {m.title}", file=sys.stderr)
    print(f"  beats:{s['beats']}  narr_ok:{s['narration_ok']}  dur_ok:{s['duration_ok']}"
          f"  total_vo:{s['total_audio_s']}s", file=sys.stderr)
    for b in m.beats:
        if not b.ok:
            issues = (
                ([f"narration({b.narration_gate.detail})"] if not b.narration_gate.passed else []) +
                ([f"duration({b.duration_gate.detail})"] if not b.duration_gate.passed else [])
            )
            print(f"  - {b.beat_id}: {'; '.join(issues)}", file=sys.stderr)
    return 0 if m.ok else 1


def _run_visual(spec, out_dir, limit) -> int:
    from .local import LocalRenderer
    r = LocalRenderer(out_dir=out_dir)
    m = visual_only(spec, r, _quiet(), limit=limit)
    s = m.summary()
    print(f"\n{'OK' if m.ok else 'PARTIAL'} — visuals: {m.title}", file=sys.stderr)
    print(f"  beats:{s['beats']}  footage_ok:{s['footage_ok']}  fallbacks:{s['fallbacks']}", file=sys.stderr)
    return 0 if m.ok else 1


def _run_music(spec, out_dir, model_id, limit) -> int:
    from .local import LocalRenderer, MusicGenEngine
    r = LocalRenderer(out_dir=out_dir, music_engine=MusicGenEngine(model_id=model_id))
    m = music_only(spec, r, _quiet(), limit=limit)
    s = m.summary()
    print(f"\nMusic: {m.title}", file=sys.stderr)
    print(f"  beats:{s['beats']}  ok:{s['music_ok']}  dropped:{s['music_dropped']}", file=sys.stderr)
    return 0


def _run_assemble(spec, in_dir, out_file) -> int:
    from .local import LocalRenderer
    r = LocalRenderer(out_dir=in_dir)
    gate = assemble_from_assets(spec, Path(in_dir), r, _quiet(), out_path=out_file)
    status = "OK" if gate.passed else "FAIL"
    print(f"\n{status} — assembly: {gate.detail}", file=sys.stderr)
    if out_file:
        print(f"  -> {out_file}", file=sys.stderr)
    return 0 if gate.passed else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="myAIscene",
        description="Render a ProductionSpec into video. Use my-AI-script to generate specs.",
    )
    ap.add_argument("--spec", required=True, help="path to a ProductionSpec JSON")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--narrate", action="store_true")
    ap.add_argument("--visual", action="store_true")
    ap.add_argument("--music", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--out-dir", default="/tmp/myAIscene")
    ap.add_argument("--in-dir", default="", help="asset dir for --assemble")
    ap.add_argument("--out", default="", help="output MP4 path for --assemble")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--whisper-model", default="base")
    ap.add_argument("--music-model", default="facebook/musicgen-small")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    args = ap.parse_args(argv)

    try:
        spec = load_spec(args.spec)
    except SpecError as e:
        print(f"spec error: {e}", file=sys.stderr)
        return 2

    if args.narrate:
        return _run_narrate(spec, args.out_dir, args.voice, args.whisper_model, args.limit)
    if args.visual:
        return _run_visual(spec, args.out_dir, args.limit)
    if args.music:
        return _run_music(spec, args.out_dir, args.music_model, args.limit)
    if args.assemble:
        in_dir = args.in_dir or args.out_dir
        return _run_assemble(spec, in_dir, args.out)
    if args.dry_run:
        return _run_dry(spec, args.max_retries)

    print("choose: --dry-run | --narrate | --visual | --music | --assemble", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
