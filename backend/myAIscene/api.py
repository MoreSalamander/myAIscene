"""FastAPI app — streams the pipeline as NDJSON (ARCHITECTURE Phase 6).

The pipeline is synchronous and emits through EventEmitter. A run executes
on a worker thread, its events land on a queue, and the HTTP response
streams them as NDJSON — same vocabulary the CLI prints, now observable
live in a browser.
"""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Iterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from myAIscene.events import EventEmitter
from myAIscene.spec import SpecError, load_spec

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
SPECS_DIR    = Path(__file__).resolve().parent.parent.parent / "specs"
OUT_DIR      = Path(__file__).resolve().parent.parent.parent / "out"

app = FastAPI(title="my-AI-scene",
              description="A MoreSalamander StudioLabs production.")


# ---- request model ----------------------------------------------------------

class RunRequest(BaseModel):
    spec_name: str
    mode: str = "assemble"        # "narrate" | "music" | "assemble"
    voice: Optional[str] = None
    limit: Optional[int] = None
    music_model: str = "facebook/musicgen-small"


class WriteRequest(BaseModel):
    brief: str
    duration_s: int = 120
    model: str = "llama3.1:8b"
    out_name: Optional[str] = None   # stem for the output spec file


# ---- streaming run ----------------------------------------------------------

def _ndjson_run(req: RunRequest) -> Iterator[str]:
    """Run pipeline on a worker thread; yield each event as NDJSON."""
    spec_path = SPECS_DIR / f"{req.spec_name}.json"
    if not spec_path.exists():
        yield json.dumps({"event": "error", "stage": "server",
                          "message": f"spec not found: {req.spec_name}"}) + "\n"
        return
    try:
        spec = load_spec(spec_path)
    except SpecError as e:
        yield json.dumps({"event": "error", "stage": "spec_load",
                          "message": str(e)}) + "\n"
        return

    out_dir = OUT_DIR / req.spec_name
    out_dir.mkdir(parents=True, exist_ok=True)

    q: queue.Queue[dict | None] = queue.Queue()
    em = EventEmitter(out=None, sink=q.put)

    def _worker() -> None:
        try:
            from myAIscene.local import LocalRenderer
            r = LocalRenderer(out_dir=out_dir, voice=req.voice)

            if req.mode == "narrate":
                from myAIscene.pipeline import narrate_only
                narrate_only(spec, r, em, limit=req.limit)

            elif req.mode == "music":
                from myAIscene.pipeline import music_only
                music_only(spec, r, em, limit=req.limit)

            elif req.mode == "assemble":
                from myAIscene.pipeline import assemble_from_assets
                assemble_from_assets(spec, out_dir, r, em,
                                     out_path=str(out_dir / "episode.mp4"))
            else:
                em.error("server", message=f"unknown mode: {req.mode}")

        except Exception as exc:
            em.error("server", message=f"{type(exc).__name__}: {exc}")
        finally:
            q.put(None)  # sentinel

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        ev = q.get()
        if ev is None:
            break
        yield json.dumps(ev, ensure_ascii=False) + "\n"


def _ndjson_write(req: WriteRequest) -> Iterator[str]:
    """Generate a ProductionSpec from a brief; stream events, end with the spec."""
    import re
    q: queue.Queue[dict | None] = queue.Queue()
    em = EventEmitter(out=None, sink=q.put)

    # derive a filesystem-safe spec name
    name = req.out_name or re.sub(r"[^a-z0-9]+", "_", req.brief[:40].lower()).strip("_")
    out_path = SPECS_DIR / f"{name}.json"

    def _worker() -> None:
        try:
            from myAIscene.writer import OllamaEngine, SpecWriteError, SpecWriter
            writer = SpecWriter(OllamaEngine(model=req.model), max_retries=3)
            result = writer.write_to_file(
                req.brief, out_path,
                target_duration_s=req.duration_s,
                emitter=em,
            )
            # Final event carries the spec name so the UI can switch to it
            em.emit("spec_ready", "spec_write",
                    spec_name=name, title=result.spec.episode.title,
                    beats=len(result.spec.beats))
        except Exception as exc:
            em.error("server", message=f"{type(exc).__name__}: {exc}")
        finally:
            q.put(None)

    threading.Thread(target=_worker, daemon=True).start()
    while True:
        ev = q.get()
        if ev is None:
            break
        yield json.dumps(ev, ensure_ascii=False) + "\n"


@app.post("/api/write")
def write_spec(req: WriteRequest) -> StreamingResponse:
    return StreamingResponse(_ndjson_write(req), media_type="application/x-ndjson")


@app.post("/api/run")
def run_pipeline(req: RunRequest) -> StreamingResponse:
    return StreamingResponse(_ndjson_run(req), media_type="application/x-ndjson")


# ---- spec endpoints ---------------------------------------------------------

@app.get("/api/specs")
def list_specs() -> list[dict]:
    out = []
    for p in sorted(SPECS_DIR.glob("*.json")):
        try:
            spec = load_spec(p)
            out.append({
                "name": p.stem,
                "title": spec.episode.title,
                "beats": len(spec.beats),
                "length_s": spec.episode.length_s,
                "genre": spec.episode.genre,
            })
        except Exception:
            pass
    return out


@app.get("/api/specs/{name}")
def get_spec(name: str) -> dict:
    p = SPECS_DIR / f"{name}.json"
    if not p.exists():
        raise HTTPException(404, "spec not found")
    return json.loads(p.read_text())


# ---- output file serving ----------------------------------------------------

@app.get("/api/output/{spec_name}/{filename:path}")
def get_output(spec_name: str, filename: str) -> FileResponse:
    path = OUT_DIR / spec_name / filename
    if not path.exists():
        raise HTTPException(404, "file not found")
    if filename.endswith(".mp4"):
        mt = "video/mp4"
    elif filename.endswith(".wav"):
        mt = "audio/wav"
    else:
        mt = "application/octet-stream"
    return FileResponse(str(path), media_type=mt)


@app.get("/api/output/{spec_name}")
def list_outputs(spec_name: str) -> dict:
    d = OUT_DIR / spec_name
    if not d.exists():
        return {"files": []}
    files = sorted(str(p.relative_to(d)) for p in d.iterdir()
                   if p.suffix in {".mp4", ".wav"} and not p.name.startswith("titlecard"))
    return {"files": files}


# ---- frontend ---------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = FRONTEND_DIR / "index.html"
    if not html.exists():
        return HTMLResponse("<h1>my-AI-scene</h1><p>frontend not found — "
                            "create frontend/index.html</p>")
    return HTMLResponse(html.read_text(encoding="utf-8"))
