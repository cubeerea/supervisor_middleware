"""Offline tests for the supervisor daemon: fakeredis pub/sub, scripted judges, tmp EventLog."""
from __future__ import annotations

import asyncio
import contextlib
from unittest import mock

import fakeredis
import pytest

from supervisor_middleware.config import Settings
from supervisor_middleware.messages import InterruptSignal, TrajectoryPacket, Verdict, now_ns
from supervisor_middleware.redis_bus import AsyncRedisBus
from supervisor_middleware.supervisor.daemon import SupervisorDaemon
from supervisor_middleware.supervisor.judge import RuleJudge, ScriptedJudge
from supervisor_middleware.telemetry import EventLog, read_events

SETTINGS = Settings(tracing=False)


@pytest.fixture
async def buses():
    server = fakeredis.FakeServer()
    sup = AsyncRedisBus(fakeredis.FakeAsyncRedis(server=server, decode_responses=True))
    exe = AsyncRedisBus(fakeredis.FakeAsyncRedis(server=server, decode_responses=True))
    yield sup, exe
    await sup.aclose()
    await exe.aclose()


@pytest.fixture
def events(tmp_path):
    return EventLog(tmp_path / "events.jsonl")


def _packet(run_id: str, reasoning: str, **kw) -> TrajectoryPacket:
    return TrajectoryPacket(run_id=run_id, thread_id="thread-1", step=kw.pop("step", 1), reasoning=reasoning,
                            state={"task": "migrate"}, **kw)


@contextlib.asynccontextmanager
async def running(daemon: SupervisorDaemon):
    stop = asyncio.Event()
    task = asyncio.create_task(daemon.run(stop))
    await asyncio.wait_for(daemon.ready.wait(), 2)
    try:
        yield
    finally:
        stop.set()
        await asyncio.wait_for(task, 5)


@contextlib.asynccontextmanager
async def interrupts(bus: AsyncRedisBus):
    """Collect InterruptSignals in the background."""
    got: list[InterruptSignal] = []
    ready = asyncio.Event()

    async def reader():
        async with bus.subscribe(SETTINGS.interrupt_channel) as msgs:
            ready.set()
            async for m in msgs:
                m.pop("_channel", None)
                got.append(InterruptSignal.model_validate(m))

    t = asyncio.create_task(reader())
    await asyncio.wait_for(ready.wait(), 2)
    try:
        yield got
    finally:
        t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t


async def _until(pred, within_s=3.0):
    deadline = asyncio.get_running_loop().time() + within_s
    while not pred():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


async def test_fakeredis_cross_client_pubsub(buses):
    a, b = buses
    async with b.subscribe("ch") as msgs:
        assert await a.publish("ch", {"x": 1}) == 1
        msg = await asyncio.wait_for(anext(aiter(msgs)), 2)
    assert msg["x"] == 1


async def test_halt_publishes_interrupt(buses, events):
    sup, exe = buses
    judge = ScriptedJudge([Verdict(status="halt", reason="raw SQL")])
    daemon = SupervisorDaemon(sup, judge, events, SETTINGS)
    async with interrupts(exe) as got, running(daemon):
        pkt = _packet("run-halt", "I'll use sqlite3 directly")
        await exe.publish(SETTINGS.executor_channel, pkt)
        await _until(lambda: got)
    sig = got[0]
    assert sig.run_id == "run-halt" and sig.thread_id == "thread-1"
    assert sig.status == "halt" and sig.reason == "raw SQL"
    assert sig.supervisor_model == "scripted"
    assert sig.packet_published_at_ns == pkt.published_at_ns
    assert pkt.published_at_ns <= sig.received_at_ns <= sig.verdict_at_ns <= sig.published_at_ns <= now_ns()

    evs = read_events(events.path)
    kinds = [e["event"] for e in evs]
    assert kinds == ["supervisor_received", "supervisor_verdict", "interrupt_published"]
    rec, ver, pub = evs
    assert rec["run_id"] == "run-halt" and rec["packet_published_at_ns"] == pkt.published_at_ns
    assert rec["transit_ms"] >= 0 and rec["step"] == 1 and rec["thread_id"] == "thread-1"
    assert ver["status"] == "halt" and ver["reason"] == "raw SQL" and ver["judge"] == "scripted"
    assert ver["judge_ms"] >= 0
    assert pub["published_at_ns"] == sig.published_at_ns and pub["reason"] == "raw SQL"
    assert pub["receivers"] == 1


async def test_proceed_publishes_nothing(buses, events):
    sup, exe = buses
    daemon = SupervisorDaemon(sup, RuleJudge(), events, SETTINGS)
    async with interrupts(exe) as got, running(daemon):
        await exe.publish(SETTINGS.executor_channel, _packet("run-ok", "I'll call session.update in chunks."))
        await _until(lambda: any(e["event"] == "supervisor_verdict" for e in read_events(events.path)))
        await asyncio.sleep(0.1)
    assert got == []
    assert [e["event"] for e in read_events(events.path)] == ["supervisor_received", "supervisor_verdict"]


async def test_slow_verdict_does_not_block_next_packet(buses, events):
    sup, exe = buses

    async def fn(p: TrajectoryPacket) -> Verdict:
        if p.run_id == "A":
            await asyncio.sleep(1.0)
        return Verdict(status="halt", reason=p.run_id)

    daemon = SupervisorDaemon(sup, ScriptedJudge(fn), events, SETTINGS, max_concurrency=4)
    loop = asyncio.get_running_loop()
    async with interrupts(exe) as got, running(daemon):
        t0 = loop.time()
        await exe.publish(SETTINGS.executor_channel, _packet("A", "a"))
        await exe.publish(SETTINGS.executor_channel, _packet("B", "b"))
        await _until(lambda: any(s.run_id == "B" for s in got))
        b_elapsed = loop.time() - t0
        await _until(lambda: len(got) == 2)
    assert b_elapsed < 0.5, b_elapsed
    assert [s.run_id for s in got] == ["B", "A"]


async def test_invalid_payload_is_skipped(buses, events):
    sup, exe = buses
    judge = ScriptedJudge([Verdict(status="halt", reason="x")])
    daemon = SupervisorDaemon(sup, judge, events, SETTINGS)
    async with interrupts(exe) as got, running(daemon):
        await exe.publish(SETTINGS.executor_channel, {"hello": "world"})
        await exe.client.publish(SETTINGS.executor_channel, "not json at all")
        await exe.publish(SETTINGS.executor_channel, _packet("good", "r"))
        await _until(lambda: got)
    assert [s.run_id for s in got] == ["good"]
    assert len(judge.calls) == 1
    kinds = [e["event"] for e in read_events(events.path)]
    assert kinds.count("supervisor_invalid_packet") == 1


async def test_handle_directly_and_crashing_judge(buses, events):
    sup, _ = buses

    def boom(p):
        raise RuntimeError("model exploded")

    daemon = SupervisorDaemon(sup, ScriptedJudge(boom), events, SETTINGS)
    v = await daemon.handle(_packet("r", "x"))
    assert v.status == "proceed" and v.reason.startswith("supervisor_error:")

    daemon2 = SupervisorDaemon(sup, RuleJudge(), events, SETTINGS)
    v2 = await daemon2.handle(_packet("r2", "Bypass the ORM and run raw SQL."))
    assert v2.status == "halt"


async def test_trace_context_with_tracing_disabled_never_builds_a_client(buses, events, monkeypatch):
    """trace_context present + tracing off: no Langfuse client is created and nothing is exported."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-fake")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-fake")
    sup, _ = buses
    daemon = SupervisorDaemon(sup, ScriptedJudge([Verdict(status="halt", reason="x")]), events, SETTINGS)
    assert not daemon.tracer.enabled
    with mock.patch("langfuse.Langfuse.__init__", side_effect=AssertionError("client built")) as ctor:
        v = await daemon.handle(_packet("r", "x", trace_context={"trace_id": "a" * 32, "parent_span_id": "b" * 16}))
    assert v.status == "halt"
    ctor.assert_not_called()
