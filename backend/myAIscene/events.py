"""NDJSON event vocabulary — the studio's shared pipeline-observability
contract (CONSTITUTION Article VII). Same event names as my-AI-stro's
pipelines and my-AI-story's renderer, so the eventual web UI can light
the same node graph.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TextIO

# the shared vocabulary
STEP_START = "step_start"
STEP_COMPLETE = "step_complete"
TOKEN = "token"
DONE = "done"
ERROR = "error"
# gate / control events
GATE_PASS = "gate_pass"
GATE_FAIL = "gate_fail"
RETRY = "retry"
FALLBACK = "fallback"
SKIP = "skip"


@dataclass
class EventEmitter:
    """Emits NDJSON events. Defaults to stdout; collects them too so a
    caller (CLI, tests, future API) can inspect the full stream."""
    out: TextIO | None = field(default_factory=lambda: sys.stdout)
    sink: Callable[[dict[str, Any]], None] | None = None
    collected: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, event: str, stage: str = "", **data: Any) -> dict[str, Any]:
        rec = {"event": event, "stage": stage, "ts": round(time.time(), 3), **data}
        self.collected.append(rec)
        if self.out is not None:
            self.out.write(json.dumps(rec) + "\n")
            self.out.flush()
        if self.sink is not None:
            self.sink(rec)
        return rec

    # convenience wrappers
    def step_start(self, stage: str, **d: Any): return self.emit(STEP_START, stage, **d)
    def step_complete(self, stage: str, **d: Any): return self.emit(STEP_COMPLETE, stage, **d)
    def gate_pass(self, stage: str, **d: Any): return self.emit(GATE_PASS, stage, **d)
    def gate_fail(self, stage: str, **d: Any): return self.emit(GATE_FAIL, stage, **d)
    def retry(self, stage: str, **d: Any): return self.emit(RETRY, stage, **d)
    def fallback(self, stage: str, **d: Any): return self.emit(FALLBACK, stage, **d)
    def skip(self, stage: str, **d: Any): return self.emit(SKIP, stage, **d)
    def error(self, stage: str, **d: Any): return self.emit(ERROR, stage, **d)
    def done(self, **d: Any): return self.emit(DONE, "pipeline", **d)
