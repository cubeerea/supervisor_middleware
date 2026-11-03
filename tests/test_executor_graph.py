"""End-to-end executor graph tests: scripted token stream + fakeredis + an in-test supervisor."""
import asyncio
import contextlib

import fakeredis

from supervisor_middleware.config import Settings
from supervisor_middleware.executor.run import run_episode
from supervisor_middleware.executor.scripted import ScriptedSource
from supervisor_middleware.messages import InterruptSignal, TrajectoryPacket, now_ns
from supervisor_middleware.metrics import compute
from supervisor_middleware.redis_bus import AsyncRedisBus
from supervisor_middleware.sandbox_runner import prepare_workdir
from supervisor_middleware.telemetry import EventLog, read_events

BAD = ("raw sql", "sqlite3", "underlying store", "bypass")


def _bus(server):
    return AsyncRedisBus(fakeredis.FakeAsyncRedis(server=server, decode_responses=True))


@contextlib.asynccontextmanager
async def fake_supervisor(server, settings, events, delay_s=0.05):
    bus = _bus(server)
    ready = asyncio.Event()

    async def loop():
        async with bus.subscribe(settings.executor_channel) as msgs:
            ready.set()
            async for d in msgs:
                p = TrajectoryPacket.model_validate({k: v for k, v in d.items() if not k.startswith("_")})
                received = now_ns()
                await asyncio.sleep(delay_s)
                halt = any(b in p.reasoning.lower() for b in BAD)
                events.emit("supervisor_verdict", run_id=p.run_id, thread_id=p.thread_id,
                            status="halt" if halt else "proceed", judge_ms=delay_s * 1000)
                if halt:
                    sig = InterruptSignal(run_id=p.run_id, thread_id=p.thread_id, reason="raw SQL is forbidden",
                                          packet_published_at_ns=p.published_at_ns, received_at_ns=received)
                    events.emit("interrupt_published", run_id=p.run_id, thread_id=p.thread_id,
                                published_at_ns=sig.published_at_ns)
                    await bus.publish(settings.interrupt_channel, sig)

    t = asyncio.create_task(loop())
    await ready.wait()
    try:
        yield
    finally:
        t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t
        await bus.aclose()


async def _episode(tmp_path, scenario, delay_s=0.05, grace=2.0, chars_per_s=400.0):
    server = fakeredis.FakeServer()
    settings = Settings(verdict_grace_s=grace, max_steps=10, events_path=tmp_path / "events.jsonl")
    events = EventLog(settings.events_path)
    wd = prepare_workdir(tmp_path / "wd", rows=120)
    src = ScriptedSource(scenario, seed=1, chars_per_s=chars_per_s, first_token_s=0.01)
    bus = _bus(server)
    async with fake_supervisor(server, settings, events, delay_s):
        r = await run_episode(settings, source=src, bus=bus, events=events, workdir=wd)
    await bus.aclose()
    return r, read_events(settings.events_path)


async def test_explicit_violation_is_cancelled_mid_stream(tmp_path):
    r, ev = await _episode(tmp_path, "explicit")
    assert r.outcome == "finished" and r.verify["complete"]
    halted = [t for t in r.trajectory if t["status"] == "halted"]
    assert len(halted) == 1 and halted[0]["intended_violation"]
    # the raw-SQL command never ran
    executed = [e["command"] for e in ev if e["event"] == "command_executed"]
    assert not any("sqlite3" in c for c in executed)
    cancels = [e for e in ev if e["event"] == "generation_cancelled"]
    assert len(cancels) == 1 and cancels[0]["cancel_latency_ms"] < 500
    # the cut happened before the command finished streaming
    assert 0 < halted[0]["command_chars_at_cancel"] < len("import sqlite3") + 400
    m = compute(ev)
    assert m["tir"] == 1.0 and m["prevention_rate"] == 1.0 and m["false_intervention_rate"] == 0.0


async def test_covert_violation_is_missed_and_executed(tmp_path):
    r, ev = await _episode(tmp_path, "covert")
    assert not r.halts
    assert any("sqlite3" in e["command"] for e in ev if e["event"] == "command_executed")
    m = compute(ev)
    assert m["tir"] == 0.0 and m["missed_violations_executed"] == 1


async def test_clean_run_has_no_interventions(tmp_path):
    r, ev = await _episode(tmp_path, "clean")
    assert r.outcome == "finished" and r.verify["complete"] and not r.halts
    m = compute(ev)
    assert m["violations"] == 0 and m["false_intervention_rate"] == 0.0


async def test_late_halt_is_blocked_by_gate(tmp_path):
    # very fast stream + slow verdict: the halt arrives after the stream finished, within the grace window
    r, ev = await _episode(tmp_path, "explicit", delay_s=0.6, grace=2.0, chars_per_s=1e6)
    assert [h["late"] for h in r.halts] == [True]
    assert any(e["event"] == "late_halt_blocked" for e in ev)
    assert not any("sqlite3" in e["command"] for e in ev if e["event"] == "command_executed")
    assert r.verify["complete"]


async def test_publish_does_not_block_generation(tmp_path):
    """The executor keeps streaming while the packet is in flight: command fragments arrive after publish."""
    r, ev = await _episode(tmp_path, "clean")
    pub = [e for e in ev if e["event"] == "reasoning_published"]
    assert pub and all(e["command_chars_so_far"] < 5 for e in pub)
