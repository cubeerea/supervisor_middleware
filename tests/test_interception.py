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
