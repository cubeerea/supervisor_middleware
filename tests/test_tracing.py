"""Distributed tracing (Langfuse): supervisor observations are children of the executor's generation, in the
episode's trace, and halts are recorded as scores. Offline: in-memory span exporter + mock HTTP."""
import asyncio
from dataclasses import replace

import fakeredis
import pytest

from supervisor_middleware.config import Settings
from supervisor_middleware.executor.run import run_episode
from supervisor_middleware.executor.scripted import ScriptedSource
from supervisor_middleware.redis_bus import AsyncRedisBus
from supervisor_middleware.sandbox_runner import prepare_workdir
from supervisor_middleware.supervisor.daemon import SupervisorDaemon
from supervisor_middleware.supervisor.judge import RuleJudge
from supervisor_middleware.telemetry import EventLog
from supervisor_middleware.tracing import Tracer
from tests.tracing_fakes import FakeLangfuse


@pytest.fixture
def lf():
    f = FakeLangfuse()
    yield f
    f.close()


def _attr(span, key):
    return (span.attributes or {}).get(key)


async def test_supervisor_observation_is_child_of_executor_generation(tmp_path, lf):
    settings = replace(Settings(), events_path=tmp_path / "ev.jsonl", max_steps=8)
    events = EventLog(settings.events_path)
    server = fakeredis.FakeServer()

    def bus():
        return AsyncRedisBus(fakeredis.FakeAsyncRedis(server=server, decode_responses=True))

    sup_bus, ex_bus = bus(), bus()
    daemon = SupervisorDaemon(sup_bus, RuleJudge(), events, settings, tracer=lf.tracer)
    task = asyncio.create_task(daemon.run())
    await asyncio.wait_for(daemon.ready.wait(), 5)
    wd = prepare_workdir(tmp_path / "wd", rows=60)
    r = await run_episode(settings, source=ScriptedSource("explicit", chars_per_s=500, first_token_s=0.01),
                          bus=ex_bus, events=events, workdir=wd, thread_id="ep-trace", tracer=lf.tracer)
    await daemon.drain(5)
    task.cancel()
    await sup_bus.aclose()
    await ex_bus.aclose()
    assert r.halts and r.outcome == "finished"

    spans = lf.spans()
    by_name: dict[str, list] = {}
    for s in spans:
        by_name.setdefault(s.name, []).append(s)
    (episode,) = by_name["executor_episode"]
    gens = {s.context.span_id: s for s in by_name["executor_generate"]}
    evals = by_name["supervisor_evaluate"]
    assert gens and evals and by_name.get("judge") and by_name.get("tool_execute")

    trace_id = format(episode.context.trace_id, "032x")
    assert trace_id == Tracer.trace_id_for("ep-trace")  # deterministic per episode
    assert all(s.context.trace_id == episode.context.trace_id for s in spans)
    for e in evals:
        assert e.parent is not None and e.parent.span_id in gens, "evaluate must hang under executor_generate"
    for g in gens.values():
        assert g.parent.span_id == episode.context.span_id
    halted = [g for g in gens.values() if _attr(g, "langfuse.observation.metadata.intercepted") in (True, "true")]
    assert len(halted) == 1
    assert _attr(halted[0], "langfuse.observation.level") == "WARNING"

    scores = lf.scores()
    halts = [s for s in scores if s["name"] == "supervisor_halt" and s["value"] == 1]
    assert len(halts) == 1 and halts[0]["observationId"] == format(halted[0].context.span_id, "016x")
    assert {s["name"] for s in scores} >= {"supervisor_halt", "intercepted", "cancel_latency_ms", "violation"}


async def test_tracer_disabled_is_noop():
    t = Tracer(None)
    with t.observation("x") as obs:
        assert obs is None
    assert t.context(None) is None
    t.score("s", 1)
    t.flush()


async def test_tracer_propagates_body_exceptions(lf):
    with pytest.raises(ValueError), lf.tracer.observation("boom"):
        raise ValueError("x")
    with pytest.raises(asyncio.CancelledError), lf.tracer.observation("cancel"):
        raise asyncio.CancelledError()
    assert {s.name for s in lf.spans()} >= {"boom", "cancel"}
