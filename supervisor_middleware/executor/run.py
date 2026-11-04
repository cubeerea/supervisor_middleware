"""Run executor episodes (one LangGraph invocation each). CLI: `executor-run` / `python -m ...executor.run`."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..redis_bus import AsyncRedisBus
from ..telemetry import EventLog
from ..tracing import Tracer
from .graph import ExecutorDeps, build_graph
from .interception import InterruptListener
from .prompts import MIGRATION_TASK
from .scripted import SCENARIOS, ScriptedSource
from .sources import AnthropicSource, StreamSource

log = logging.getLogger("executor")


def resolve_backend(settings: Settings, backend: str | None = None) -> str:
    b = (backend or settings.executor_backend or "auto").lower()
    if b == "auto":
        return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "scripted"
    if b not in ("anthropic", "scripted"):
        raise ValueError(f"unknown executor backend {b!r}")
    return b


def make_source(backend: str, scenario: str = "explicit", seed: int = 0) -> StreamSource:
    return AnthropicSource() if backend == "anthropic" else ScriptedSource(scenario, seed=seed)


@dataclass
class EpisodeResult:
    thread_id: str
    outcome: str
    steps: int
    halts: list[dict[str, Any]]
    trajectory: list[dict[str, Any]]
    verify: dict[str, Any]
    workdir: str
    escalation: dict[str, Any] | None = None


async def run_episode(settings: Settings, *, source: StreamSource, bus: AsyncRedisBus, events: EventLog,
                      workdir: Path, task: str = MIGRATION_TASK, thread_id: str | None = None,
                      max_halts_per_step: int = 3, tracer: Tracer | None = None,
                      session_id: str | None = None) -> EpisodeResult:
    from ..sandbox_runner import run_command
    from ..sandbox_verify import verify_db

    thread_id = thread_id or f"ep-{uuid.uuid4().hex[:10]}"
    listener = InterruptListener(bus, settings.interrupt_channel, settings.ack_channel, events)
    deps = ExecutorDeps(settings=settings, source=source, bus=bus, listener=listener, events=events,
                        workdir=workdir, run_command=run_command, verify=verify_db,
                        max_halts_per_step=max_halts_per_step, tracer=tracer or Tracer())
    graph = build_graph(deps)
    events.emit("episode_started", thread_id=thread_id, backend=source.name,
                scenario=getattr(source, "scenario", None), model=settings.executor_model)
    init = {"task": task, "thread_id": thread_id, "session_id": session_id,
            "messages": [{"role": "user", "content": task}],
            "step": 0, "attempt": 0, "halts": [], "trajectory": []}
    tr = deps.tracer
    with tr.attributes(trace_name="executor_episode", session_id=session_id,
                       tags=[source.name] + ([source.scenario] if hasattr(source, "scenario") else [])), \
            tr.observation("executor_episode", as_type="agent", input={"task": task},
                        trace_context={"trace_id": Tracer.trace_id_for(thread_id)} if tr.enabled else None,
                        metadata={"thread_id": thread_id, "backend": source.name,
                                  "scenario": getattr(source, "scenario", None),
                                  "executor_model": settings.executor_model}) as ep:
        try:
            final = await graph.ainvoke(init, config={"recursion_limit": 4 * settings.max_steps + 20,
                                                       "run_name": "executor_episode",
                                                       "metadata": {"thread_id": thread_id}})
        finally:
            await listener.stop()
            await bus.flush()
        v = verify_db(workdir / "app.db")
        tr.update(ep, output={"outcome": final.get("outcome"), "steps": final.get("step", 0),
                              "halts": len(final.get("halts", [])), "verify": v})
    outcome = final.get("outcome") or ("max_steps" if final.get("step", 0) >= settings.max_steps else "stopped")
    events.emit("episode_finished", thread_id=thread_id, outcome=outcome, steps=final.get("step", 0),
                halts=len(final.get("halts", [])), verify=v)
    return EpisodeResult(thread_id, outcome, final.get("step", 0), final.get("halts", []),
                         final.get("trajectory", []), v, str(workdir), final.get("escalation"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one executor episode against the mock-ORM sandbox.")
    ap.add_argument("--backend", choices=["auto", "anthropic", "scripted"], default=None)
    ap.add_argument("--scenario", choices=SCENARIOS, default="explicit", help="scripted backend only")
    ap.add_argument("--redis-url", default=None)
    ap.add_argument("--workdir", type=Path, default=None, help="default: a fresh temp dir")
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    settings = get_settings()
    tracer = Tracer.from_settings(settings)

    async def _run() -> EpisodeResult:
        from ..sandbox_runner import prepare_workdir

        wd = a.workdir or Path(tempfile.mkdtemp(prefix="episode-"))
        prepare_workdir(wd, rows=a.rows)
        bus = AsyncRedisBus.from_url(a.redis_url or settings.redis_url)
        try:
            backend = resolve_backend(settings, a.backend)
            return await run_episode(settings, source=make_source(backend, a.scenario), bus=bus,
                                     events=EventLog(settings.events_path), workdir=wd, tracer=tracer)
        finally:
            await bus.aclose()
            tracer.flush()

    r = asyncio.run(_run())
    print(json.dumps({"thread_id": r.thread_id, "outcome": r.outcome, "steps": r.steps, "halts": r.halts,
                      "verify": r.verify, "workdir": r.workdir}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
