"""InterruptListener tool gate: a proceed ack short-circuits the late-halt grace window."""
import asyncio
import time

import fakeredis

from supervisor_middleware.executor.interception import InterruptListener
from supervisor_middleware.messages import VerdictAck, now_ns
from supervisor_middleware.redis_bus import AsyncRedisBus

INTERRUPTS, ACKS = "test_interrupts", "test_acks"


def _bus(server) -> AsyncRedisBus:
    return AsyncRedisBus(fakeredis.FakeAsyncRedis(server=server, decode_responses=True))


async def test_wait_for_late_halt_returns_early_on_proceed_ack():
    server = fakeredis.FakeServer()
    bus, sup = _bus(server), _bus(server)
    grace_s = 3.0
    async with InterruptListener(bus, INTERRUPTS, ACKS) as listener:
        async def ack_soon():
            await asyncio.sleep(0.05)
            await sup.publish(ACKS, VerdictAck(run_id="r1"))

        acker = asyncio.create_task(ack_soon())
        t0 = time.monotonic()
        sig = await listener.wait_for_late_halt("r1", now_ns() + int(grace_s * 1e9))
        elapsed = time.monotonic() - t0
        await acker
    await bus.aclose()
    await sup.aclose()
    assert sig is None
    assert elapsed < 1.0 < grace_s


async def test_wait_for_late_halt_waits_full_window_without_ack():
    server = fakeredis.FakeServer()
    bus, sup = _bus(server), _bus(server)
    grace_s = 0.4
    async with InterruptListener(bus, INTERRUPTS, ACKS) as listener:
        # an ack for a different run must not release this run's gate
        await sup.publish(ACKS, VerdictAck(run_id="someone-else"))
        t0 = time.monotonic()
        sig = await listener.wait_for_late_halt("r1", now_ns() + int(grace_s * 1e9))
        elapsed = time.monotonic() - t0
    await bus.aclose()
    await sup.aclose()
    assert sig is None
    assert elapsed >= grace_s - 0.05


async def test_outer_cancel_during_halt_cleanup_is_not_swallowed():
    """A halt cancels the stream; while the stream is still closing (HTTP teardown), the caller itself is
    cancelled (e.g. an episode timeout). That outer cancellation must propagate, not become an InterceptionHalt."""
    from supervisor_middleware.executor.interception import InterceptionHalt
    from supervisor_middleware.messages import InterruptSignal

    listener = InterruptListener(_bus(fakeredis.FakeServer()), INTERRUPTS)  # _on_signal driven directly
    started = asyncio.Event()

    async def stream():
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await asyncio.sleep(0.2)  # closing the HTTP response takes a moment
            raise

    caller = asyncio.create_task(listener.run_guarded("r1", stream()))
    await started.wait()
    listener._on_signal(InterruptSignal(run_id="r1", thread_id="t", reason="halt"))
    await asyncio.sleep(0.05)  # the stream is now in its cleanup
    caller.cancel()
    result = (await asyncio.gather(caller, return_exceptions=True))[0]
    assert not isinstance(result, InterceptionHalt), "outer cancellation was misreported as a supervisor halt"
    assert isinstance(result, asyncio.CancelledError)
