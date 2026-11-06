"""Phase 4: the interception hook and kill switch.

`InterruptListener` is a background asyncio.Task subscribed to `supervisor_interrupts`. The executor
registers the run_id of each generation with `guard(run_id, task)`; when a halt for that run_id
arrives while the generation task is still running, the listener cancels that task, which severs
the Anthropic HTTP stream (the `async with client.messages.stream(...)` exits and closes the
connection). `run_guarded` turns that cancellation into an `InterceptionHalt` for the caller.

Halts that arrive after the stream already finished are kept (`late_halt`) so the tool node can
refuse to execute the command.

Escalations (`kind="escalate"`, from the supervisor's stall detector) never cancel anything: they are recorded per
thread (`escalation(thread_id)`) and the graph ends the episode before executing its next command.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import TypeVar

from ..messages import InterruptSignal, now_ns
from ..redis_bus import AsyncRedisBus
from ..telemetry import EventLog

log = logging.getLogger(__name__)
T = TypeVar("T")


class InterceptionHalt(Exception):
    """Raised in the executor when the supervisor severed the current generation."""

    def __init__(self, signal: InterruptSignal, partial: object = None, cancelled_at_ns: int = 0):
        super().__init__(signal.reason)
        self.signal = signal
        self.partial = partial  # whatever the stream had produced (see executor.streaming.PartialTurn)
        self.cancelled_at_ns = cancelled_at_ns


@dataclass
class _Guard:
    task: asyncio.Task
    signal: InterruptSignal | None = None
    received_at_ns: int = 0


@dataclass
class InterruptListener:
    bus: AsyncRedisBus
    channel: str
    ack_channel: str | None = None  # proceed verdicts; optional (without it the tool gate waits out the grace)
    events: EventLog = field(default_factory=lambda: EventLog(None))
    _guards: dict[str, _Guard] = field(default_factory=dict)
    _late: dict[str, InterruptSignal] = field(default_factory=dict)
    _acked: set[str] = field(default_factory=set)
    _escalations: dict[str, InterruptSignal] = field(default_factory=dict)
    _waiters: dict[str, asyncio.Event] = field(default_factory=dict)
    _task: asyncio.Task | None = None
    _ready: asyncio.Event = field(default_factory=asyncio.Event)

    async def start(self) -> InterruptListener:
        if self._task is None:
            self._task = asyncio.create_task(self._listen(), name="interrupt-listener")
            ready = asyncio.create_task(self._ready.wait())
            done, _ = await asyncio.wait({ready, self._task}, return_when=asyncio.FIRST_COMPLETED)
            if self._task in done:  # listener died before subscribing (e.g. Redis down)
                ready.cancel()
                self._task.result()
        return self

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None

    async def __aenter__(self) -> InterruptListener:
        return await self.start()

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    async def _listen(self) -> None:
        channels = [self.channel] + ([self.ack_channel] if self.ack_channel else [])
        async with self.bus.subscribe(*channels) as messages:
            self._ready.set()
            async for data in messages:
                if self.ack_channel and data.get("_channel") == self.ack_channel:
                    if rid := data.get("run_id"):
                        self._acked.add(rid)
                        if ev := self._waiters.get(rid):
                            ev.set()
                    continue
                try:
                    sig = InterruptSignal.model_validate({k: v for k, v in data.items() if not k.startswith("_")})
                except ValueError:
                    log.warning("ignoring malformed interrupt: %r", data)
                    continue
                self._on_signal(sig)

    def _on_signal(self, sig: InterruptSignal) -> None:
        received = now_ns()
        if sig.kind == "escalate":
            if sig.thread_id and sig.thread_id not in self._escalations:
                self._escalations[sig.thread_id] = sig
                self.events.emit("escalation_received", run_id=sig.run_id, thread_id=sig.thread_id, step=sig.step,
                                 reason=sig.reason, redis_transit_ms=(received - sig.published_at_ns) / 1e6)
            if ev := self._waiters.get(sig.run_id):  # wake the tool gate: it checks escalation() next
                ev.set()
            return
        guard = self._guards.get(sig.run_id)
        live = guard is not None and not guard.task.done()
        self.events.emit("interrupt_received", run_id=sig.run_id, thread_id=sig.thread_id, live=live,
                         interrupt_published_at_ns=sig.published_at_ns,
                         redis_transit_ms=(received - sig.published_at_ns) / 1e6)
        if live:
            guard.signal, guard.received_at_ns = sig, received
            guard.task.cancel(f"supervisor halt: {sig.reason}")
        else:
            self._late[sig.run_id] = sig  # stream already done (or not ours): the tool gate checks this
        if ev := self._waiters.get(sig.run_id):
            ev.set()

    # ------------------------------------------------------------------ executor-facing API

    async def run_guarded(self, run_id: str, coro: Awaitable[T], partial_ref: dict | None = None) -> T:
        """Run `coro` as a task that a matching halt can cancel. Raises InterceptionHalt on halt; an outer
        cancellation of the caller still propagates as CancelledError."""
        task = asyncio.ensure_future(coro)
        guard = _Guard(task)
        self._guards[run_id] = guard
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if guard.signal is None:  # we were cancelled from outside, not by the supervisor
                task.cancel()
                raise
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task  # let the stream's context manager close the HTTP connection
            me = asyncio.current_task()
            if me is not None and me.cancelling():  # caller cancelled from outside meanwhile: don't swallow it
                raise
            cancelled = now_ns()
            sig = guard.signal
            self.events.emit(
                "generation_cancelled", run_id=run_id, thread_id=sig.thread_id, reason=sig.reason,
                interrupt_published_at_ns=sig.published_at_ns, packet_published_at_ns=sig.packet_published_at_ns,
                cancel_latency_ms=(cancelled - sig.published_at_ns) / 1e6,
                end_to_end_ms=(cancelled - sig.packet_published_at_ns) / 1e6 if sig.packet_published_at_ns else None,
            )
            raise InterceptionHalt(sig, (partial_ref or {}).get("partial"), cancelled) from None
        finally:
            self._guards.pop(run_id, None)

    def escalation(self, thread_id: str) -> InterruptSignal | None:
        """The stall escalation for this episode, if the supervisor sent one."""
        return self._escalations.get(thread_id)

    def late_halt(self, run_id: str) -> InterruptSignal | None:
        return self._late.get(run_id)

    async def wait_for_verdict(self, run_id: str, timeout_s: float) -> tuple[InterruptSignal | None, bool]:
        """Action-mode gate: block until the supervisor answers for `run_id` (halt or proceed ack) or `timeout_s`
        passes. Returns (halt signal or None, answered). A halt or ack that arrived before the call counts."""
        if sig := self._late.get(run_id):
            return sig, True
        if run_id in self._acked:
            self._acked.discard(run_id)
            return None, True
        ev = self._waiters.setdefault(run_id, asyncio.Event())
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout_s)
        except TimeoutError:
            pass
        finally:
            self._waiters.pop(run_id, None)
        answered = run_id in self._acked or run_id in self._late
        self._acked.discard(run_id)
        return self._late.get(run_id), answered

    async def wait_for_late_halt(self, run_id: str, until_ns: int) -> InterruptSignal | None:
        """Tool gate: wait until `until_ns` (wall clock) for a halt on a stream that already finished. Returns early
        (None) once the supervisor has acknowledged a proceed verdict for this run."""
        if sig := self._late.get(run_id):
            return sig
        if run_id in self._acked:
            return None
        remaining = (until_ns - now_ns()) / 1e9
        if remaining <= 0:
            return None
        ev = self._waiters.setdefault(run_id, asyncio.Event())
        try:
            await asyncio.wait_for(ev.wait(), timeout=remaining)
        except TimeoutError:
            pass
        finally:
            self._waiters.pop(run_id, None)
        self._acked.discard(run_id)
        return self._late.get(run_id)
