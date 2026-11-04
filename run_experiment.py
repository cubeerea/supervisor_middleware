#!/usr/bin/env python3
"""Phase 5: run N executor episodes against the mock-ORM trap and report TIR + interception latency.

    uv run python run_experiment.py                              # scripted executor, supervisor subprocess (Ollama)
    uv run python run_experiment.py --supervisor inline --judge rules --redis-url memory://   # no services at all
    uv run python run_experiment.py --backend anthropic --episodes 5   # real Claude (needs ANTHROPIC_API_KEY)

--supervisor spawn     start `supervisor_daemon.py` as a separate process (needs a real Redis)  [default]
--supervisor inline    run the daemon in this process (works with --redis-url memory://)
--supervisor external  a daemon is already running (docker compose / tmux pane 3)

Writes results/<experiment>/metrics.json and episodes.json; events go to the shared JSONL log.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

from supervisor_middleware.config import ROOT, get_settings
from supervisor_middleware.executor.prompts import MIGRATION_TASK, PRESSURE_SUFFIX
from supervisor_middleware.executor.run import make_source, resolve_backend, run_episode
from supervisor_middleware.executor.scripted import SCENARIOS
from supervisor_middleware.metrics import compute, format_report
from supervisor_middleware.redis_bus import AsyncRedisBus
from supervisor_middleware.sandbox_runner import prepare_workdir
from supervisor_middleware.telemetry import EventLog, read_events
from supervisor_middleware.tracing import Tracer


async def _redis_ok(url: str) -> bool:
    if url.startswith("memory://"):
        return True
    bus = AsyncRedisBus.from_url(url)
    try:
        return bool(await asyncio.wait_for(bus.client.ping(), 2))
    except Exception:  # noqa: BLE001
        return False
    finally:
        with contextlib.suppress(Exception):
            await bus.client.aclose()


@contextlib.asynccontextmanager
async def supervisor(mode: str, judge: str, settings, events: EventLog):
    if mode == "external":
        yield
        return
    if mode == "spawn":
        env = {**os.environ, "REDIS_URL": settings.redis_url, "EVENTS_PATH": str(settings.events_path),
               "STALL_DETECTION": "true" if settings.stall_detection else "false",
               "STALL_WINDOW": str(settings.stall_window), "SUPERVISOR_MODEL": settings.supervisor_model,
               "SUPERVISOR_CONFIRM_MODEL": settings.supervisor_confirm_model,
               "SUPERVISOR_JUDGE_MODE": settings.supervisor_judge_mode}
        proc = subprocess.Popen(  # noqa: ASYNC220 - Popen returns immediately; we never block on it here
            [sys.executable, str(ROOT / "supervisor_daemon.py"), "--backend", judge],
            env=env, cwd=ROOT)
        try:
            # wait until the daemon has subscribed to executor_stream
            bus = AsyncRedisBus.from_url(settings.redis_url)
            deadline = time.monotonic() + 120  # first Ollama load can take a while
            while time.monotonic() < deadline:
                subs = await bus.client.pubsub_numsub(settings.executor_channel)
                if subs and subs[0][1] > 0:
                    break
                if proc.poll() is not None:
                    raise SystemExit(f"supervisor daemon exited with {proc.returncode}")
                await asyncio.sleep(0.25)
            else:
                raise SystemExit("supervisor daemon did not subscribe within 120s")
            await bus.client.aclose()
            yield
        finally:
            proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(10)
            if proc.poll() is None:
                proc.kill()
        return
    # inline
    from supervisor_middleware.supervisor.daemon import SupervisorDaemon
    from supervisor_middleware.supervisor.judge import make_judge

    j = make_judge(settings, judge)
    if hasattr(j, "warmup"):
        await j.warmup()
    bus = AsyncRedisBus.from_url(settings.redis_url)
    daemon = SupervisorDaemon(bus, j, events, settings)
    stop = asyncio.Event()
    task = asyncio.create_task(daemon.run(stop))
    await asyncio.wait({task, asyncio.create_task(daemon.ready.wait())}, return_when=asyncio.FIRST_COMPLETED)
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            await daemon.drain(5)
        stop.set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        if hasattr(j, "aclose"):
            await j.aclose()
        await bus.aclose()


async def main_async(a: argparse.Namespace) -> int:
    settings = get_settings()
    shared_events = settings.events_path
    exp = a.name or time.strftime("exp-%Y%m%d-%H%M%S")
    out = ROOT / "results" / exp
    out.mkdir(parents=True, exist_ok=True)
    overrides = {"events_path": out / "events.jsonl"}
    if a.redis_url:
        overrides["redis_url"] = a.redis_url
    if a.grace is not None:
        overrides["verdict_grace_s"] = a.grace
    if a.model:
        overrides["executor_model"] = a.model
    if a.no_stall:
        overrides["stall_detection"] = False
    if a.no_command_check:
        overrides["command_check"] = False
    if a.supervisor_model:
        overrides["supervisor_model"] = a.supervisor_model
    if a.confirm_model:
        overrides["supervisor_confirm_model"] = a.confirm_model
    if a.judge_mode:
        overrides["supervisor_judge_mode"] = a.judge_mode
    settings = replace(settings, **overrides)
    if a.supervisor == "spawn" and settings.redis_url.startswith("memory://"):
        raise SystemExit("--supervisor spawn needs a real Redis; use --supervisor inline with memory://")
    if not await _redis_ok(settings.redis_url):
        raise SystemExit(f"Redis not reachable at {settings.redis_url}: `docker compose up -d redis`, "
                         "or use --redis-url memory:// --supervisor inline")
    backend = resolve_backend(settings, a.backend)
    scenarios = a.scenarios.split(",") if a.scenarios else list(SCENARIOS)
    events = EventLog(settings.events_path)
    print(f"[{exp}] executor={backend} ({settings.executor_model if backend == 'anthropic' else 'scripted'}) "
          f"supervisor={a.supervisor}/{a.judge} redis={settings.redis_url} episodes={a.episodes} "
          f"stall_detection={'on (window ' + str(settings.stall_window) + ')' if settings.stall_detection else 'off'}"
          + (" (external daemon: set STALL_DETECTION there)" if a.supervisor == "external" else "")
          + f" command_check={'on' if settings.command_check else 'off'}", flush=True)

    tracer = Tracer.from_settings(settings)
    print(f"tracing: {'langfuse @ ' + settings.langfuse_host if tracer.enabled else 'off (no LANGFUSE_* keys)'}",
          flush=True)
    results = []
    async with supervisor(a.supervisor, a.judge, settings, events):
        for i in range(a.episodes):
            scenario = scenarios[i % len(scenarios)]
            wd = Path(tempfile.mkdtemp(prefix=f"{exp}-{i:03d}-"))
            prepare_workdir(wd, rows=a.rows)
            bus = AsyncRedisBus.from_url(settings.redis_url)
            t0 = time.monotonic()
            try:
                r = await run_episode(settings, source=make_source(backend, scenario, seed=i), bus=bus,
                                      events=events, workdir=wd, thread_id=f"{exp}-{i:03d}", tracer=tracer,
                                      task=MIGRATION_TASK + (PRESSURE_SUFFIX if a.pressure else ""),
                                      session_id=exp)
            finally:
                await bus.aclose()
            results.append(r)
            label = scenario if backend == "scripted" else "claude"
            esc = f" escalated@step{r.escalation['step']}" if r.escalation else ""
            print(f"  episode {i + 1}/{a.episodes} [{label}] outcome={r.outcome} steps={r.steps} "
                  f"halts={len(r.halts)}{esc} migrated={r.verify.get('migrated')}/{r.verify.get('total')} "
                  f"({time.monotonic() - t0:.1f}s)", flush=True)
        await asyncio.sleep(0.5)  # let trailing verdict events land
    tracer.flush()

    evs = read_events(settings.events_path)
    if a.supervisor == "external" and shared_events != settings.events_path:
        evs += read_events(shared_events)  # an external daemon logs to the shared EVENTS_PATH
    m = compute(evs, {r.thread_id for r in results})
    gen = [e for e in evs if e["event"] == "generation_completed" and e.get("usage")
           and e.get("thread_id") in {r.thread_id for r in results}]
    tok = {k: sum((e["usage"].get(k) or 0) for e in gen) for k in
           ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
    m["executor_tokens"] = tok
    m["executor_cache_hit_rate"] = cache_hit_rate(tok)
    m["executor_cost_usd"] = estimate_cost(settings.executor_model, tok) if backend == "anthropic" else None
    m["refusals"] = sum(r.outcome == "refusal" for r in results)
    m.update(pressure=a.pressure, stall_detection=settings.stall_detection, command_check=settings.command_check,
             max_steps=settings.max_steps)
    m.update(experiment=exp, executor_backend=backend, judge=a.judge, supervisor_mode=a.supervisor,
             executor_model=settings.executor_model if backend == "anthropic" else "scripted",
             supervisor_model=settings.supervisor_model if a.judge == "ollama" else a.judge,
             judge_mode=settings.supervisor_judge_mode if a.judge == "ollama" else None,
             confirm_model=settings.supervisor_confirm_model
             if a.judge == "ollama" and settings.supervisor_judge_mode == "cascade" else None)
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    (out / "episodes.json").write_text(json.dumps([asdict(r) for r in results], indent=2, default=str))
    print("\n" + format_report(m))
    if any(tok.values()):
        print(f"executor tokens: {tok}")
        print(f"executor cache hit rate: {m['executor_cache_hit_rate']:.1%}  est. cost: "
              f"{_fmt_cost(m['executor_cost_usd'])}")
    print(f"\nwritten: {out / 'metrics.json'}")
    return 0


# USD per million tokens: (input, output, cache read). Cache writes (5-minute TTL) cost 1.25x input.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0, 0.20), "claude-sonnet-5": (2.0, 10.0, 0.20)}


def cache_hit_rate(tok: dict[str, int]) -> float:
    """Share of prompt tokens served from cache (input_tokens excludes cached reads and writes)."""
    total = sum(tok.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    return (tok.get("cache_read_input_tokens") or 0) / total if total else 0.0


def _fmt_cost(c: float | None) -> str:
    return "n/a" if c is None else f"${c:.2f}"


def estimate_cost(model: str, tok: dict[str, int]) -> float | None:
    if model not in PRICES:
        return None
    inp, out, read = PRICES[model]
    return ((tok.get("input_tokens") or 0) * inp + (tok.get("output_tokens") or 0) * out
            + (tok.get("cache_read_input_tokens") or 0) * read
            + (tok.get("cache_creation_input_tokens") or 0) * inp * 1.25) / 1e6


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--backend", choices=["auto", "anthropic", "scripted"], default=None)
    ap.add_argument("--scenarios", help=f"comma list for the scripted backend (default: all of {','.join(SCENARIOS)})")
    ap.add_argument("--supervisor", choices=["spawn", "inline", "external"], default="spawn")
    ap.add_argument("--judge", choices=["ollama", "rules"], default="ollama")
    ap.add_argument("--judge-mode", choices=["verdict", "checklist", "cascade"],
                    help="Ollama judge output mode (default SUPERVISOR_JUDGE_MODE)")
    ap.add_argument("--supervisor-model", help="Ollama model for the judge (default SUPERVISOR_MODEL)")
    ap.add_argument("--confirm-model", help="Ollama model that re-checks halts in cascade mode "
                                           "(default SUPERVISOR_CONFIRM_MODEL)")
    ap.add_argument("--redis-url")
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--grace", type=float, help="late-halt grace window in seconds (default VERDICT_GRACE_S)")
    ap.add_argument("--model", help="executor model id (default EXECUTOR_MODEL)")
    ap.add_argument("--name", help="experiment name (default exp-<timestamp>)")
    ap.add_argument("--pressure", action="store_true",
                    help="tell the executor no human is available and it must finish (experimental arm)")
    ap.add_argument("--no-stall", action="store_true",
                    help="disable the supervisor's stall detector / escalation (default: STALL_DETECTION, on)")
    ap.add_argument("--no-command-check", action="store_true",
                    help="let the executor run commands that use sqlite3/raw SQL (default: COMMAND_CHECK, on); "
                         "use it to measure the supervisor alone")
    a = ap.parse_args(argv)
    return asyncio.run(main_async(a))


if __name__ == "__main__":
    raise SystemExit(main())
