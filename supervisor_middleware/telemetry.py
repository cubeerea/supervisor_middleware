"""Append-only JSONL event log shared by executor and supervisor processes.

LangSmith holds the distributed trace; this file holds the same timestamps locally so TIR and
interception latency can be computed with no network (see metrics.py)."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .messages import now_ns

_lock = threading.Lock()


class EventLog:
    def __init__(self, path: Path | str | None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        rec = {"event": event, "ts_ns": now_ns(), "pid": os.getpid(), **fields}
        if self.path:
            line = json.dumps(rec, default=str) + "\n"
            with _lock:
                # O_APPEND single write: lines from several processes don't interleave
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
                try:
                    os.write(fd, line.encode())
                finally:
                    os.close(fd)
        return rec


def read_events(path: Path | str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out
