"""Supervisor daemon: reads TrajectoryPackets off `executor_stream`, judges each reasoning trace concurrently and
publishes an InterruptSignal on `supervisor_interrupts` for every halt.

Each packet is handled in its own task (bounded by a semaphore), so a slow verdict never delays reading the next
packet. Every evaluation is a LangSmith `supervisor_evaluate` run stitched under the executor's run via the
packet's `trace_headers` (distributed tracing); when tracing is off nothing touches LangSmith at all.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import logging
import signal
import time
from typing import Any

from langsmith import run_helpers
from pydantic import ValidationError

from ..config import Settings, get_settings
from ..messages import InterruptSignal, TrajectoryPacket, Verdict, VerdictAck, now_ns
from ..redis_bus import AsyncRedisBus
from ..telemetry import EventLog
from .judge import Judge, make_judge

log = logging.getLogger("supervisor")


def _judge_model(judge: Judge) -> str:
    return str(getattr(judge, "model", None) or judge.name)


async def _run_judge(judge: Judge, packet: TrajectoryPacket) -> tuple[Verdict, float]:
    """Evaluate and time one packet. Never raises: a crashing judge yields a proceed verdict."""
    t0 = time.perf_counter()
    try:
        verdict = await judge.evaluate(packet)
        if not isinstance(verdict, Verdict):
            verdict = Verdict.model_validate(verdict)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 - the daemon must survive any judge failure
        log.exception("judge %s crashed on run %s", judge.name, packet.run_id)
        verdict = Verdict(status="proceed", reason=f"supervisor_error: judge crashed: {e!r}")
    return verdict, (time.perf_counter() - t0) * 1e3


def _trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    p: TrajectoryPacket = inputs["packet"]
    return {"run_id": p.run_id, "thread_id": p.thread_id, "step": p.step, "attempt": p.attempt,
            "reasoning": p.reasoning, "task": p.state.get("task")}


@run_helpers.traceable(name="supervisor_evaluate", run_type="chain", process_inputs=_trace_inputs)
async def supervisor_evaluate(judge: Judge, packet: TrajectoryPacket, holder: dict[str, Any]) -> dict[str, Any]:
    verdict, judge_ms = await _run_judge(judge, packet)
    holder["result"] = (verdict, judge_ms)
    rt = run_helpers.get_current_run_tree()
    if rt is not None:
        rt.add_metadata({"status": verdict.status, "reason": verdict.reason, "judge_ms": round(judge_ms, 3)})
        rt.add_tags([f"status:{verdict.status}"])
    return {"status": verdict.status, "reason": verdict.reason, "judge_ms": round(judge_ms, 3)}


class SupervisorDaemon:
    def __init__(self, bus: AsyncRedisBus, judge: Judge, events: EventLog, settings: Settings, *,
                 max_concurrency: int = 4, tracing: bool | None = None, ls_client: Any = None):
        self.bus = bus
        self.judge = judge
        self.events = events
        self.settings = settings
        self.tracing = settings.tracing if tracing is None else tracing
        self.ls_client = ls_client  # langsmith.Client; None -> the ambient/cached default client
        self.ready = asyncio.Event()
        self._sem = asyncio.Semaphore(max_concurrency)
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ evaluation (+ LangSmith)

    async def _evaluate(self, packet: TrajectoryPacket, transit_ms: float) -> tuple[Verdict, float]:
        if not self.tracing:
            # Plain call: no RunTree, no client, no network. NB: passing `parent` headers to a traceable would
            # force tracing on even inside tracing_context(enabled=False) (langsmith 0.14 `_setup_run`), so the
            # disabled path must not go through the traceable at all.
            with run_helpers.tracing_context(enabled=False):
                return await _run_judge(self.judge, packet)

        metadata = {
            "executor_run_id": packet.run_id,
            "thread_id": packet.thread_id,
            "step": packet.step,
            "attempt": packet.attempt,
            "transit_ms": round(transit_ms, 3),
            "supervisor_model": _judge_model(self.judge),
            "judge": self.judge.name,
        }
        extra: dict[str, Any] = {"metadata": metadata}
        if packet.trace_headers:
            # Mapping parent -> RunTree.from_headers(): this run becomes a child of the executor's run.
            extra["parent"] = packet.trace_headers
        # tracing_context() resets every field it is not given, so carry the ambient client through explicitly.
        client = self.ls_client or run_helpers.get_tracing_context().get("client")
        if client is not None:
            extra["client"] = client
        holder: dict[str, Any] = {}
        try:
            with run_helpers.tracing_context(enabled=True, client=client):
                await supervisor_evaluate(self.judge, packet, holder, langsmith_extra=extra)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - tracing problems must never cost us a verdict
            log.warning("traced evaluation failed for run %s", packet.run_id, exc_info=True)
        if "result" in holder:
            return holder["result"]
        return await _run_judge(self.judge, packet)

    # ------------------------------------------------------------------ one packet

    async def handle(self, packet: TrajectoryPacket, received_at_ns: int | None = None) -> Verdict:
        received_at_ns = received_at_ns or now_ns()
        transit_ms = (received_at_ns - packet.published_at_ns) / 1e6
        self.events.emit("supervisor_received", run_id=packet.run_id, thread_id=packet.thread_id, step=packet.step,
                         attempt=packet.attempt, packet_published_at_ns=packet.published_at_ns,
                         received_at_ns=received_at_ns, transit_ms=round(transit_ms, 3))

        verdict, judge_ms = await self._evaluate(packet, transit_ms)
        verdict_at_ns = now_ns()
        model = _judge_model(self.judge)
        self.events.emit("supervisor_verdict", run_id=packet.run_id, thread_id=packet.thread_id, step=packet.step,
                         attempt=packet.attempt, status=verdict.status, reason=verdict.reason,
                         judge_ms=round(judge_ms, 3), judge=self.judge.name, supervisor_model=model,
                         verdict_at_ns=verdict_at_ns)
        log.info("%s %-7s %7.1fms  %s", packet.run_id[:8], verdict.status, judge_ms, verdict.reason)

        if verdict.status == "halt":
            sig = InterruptSignal(run_id=packet.run_id, thread_id=packet.thread_id, reason=verdict.reason,
                                  supervisor_model=model, packet_published_at_ns=packet.published_at_ns,
                                  received_at_ns=received_at_ns, verdict_at_ns=verdict_at_ns)
            try:
                receivers = await self.bus.publish(self.settings.interrupt_channel, sig)
            except Exception:  # noqa: BLE001
                log.exception("failed to publish interrupt for run %s", packet.run_id)
                self.events.emit("interrupt_publish_failed", run_id=packet.run_id, reason=verdict.reason)
            else:
                self.events.emit("interrupt_published", run_id=packet.run_id, thread_id=packet.thread_id,
                                 step=packet.step, attempt=packet.attempt, reason=verdict.reason,
                                 published_at_ns=sig.published_at_ns, packet_published_at_ns=packet.published_at_ns,
                                 verdict_at_ns=verdict_at_ns, receivers=receivers)
        else:
            self.bus.publish_nowait(self.settings.ack_channel, VerdictAck(run_id=packet.run_id))
        return verdict

    # ------------------------------------------------------------------ loop

    def _dispatch(self, msg: dict[str, Any]) -> asyncio.Task | None:
        received_at_ns = now_ns()
        msg = {k: v for k, v in msg.items() if k != "_channel"}
        try:
            packet = TrajectoryPacket.model_validate(msg)
        except ValidationError as e:
            log.warning("skipping invalid packet: %s", e.errors(include_url=False)[:3])
            self.events.emit("supervisor_invalid_packet", error=str(e)[:500], run_id=msg.get("run_id"))
            return None
        task = asyncio.get_running_loop().create_task(self._guarded(packet, received_at_ns),
                                                      name=f"judge:{packet.run_id[:8]}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _guarded(self, packet: TrajectoryPacket, received_at_ns: int) -> Verdict | None:
        async with self._sem:
            try:
                return await self.handle(packet, received_at_ns)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("handling run %s failed", packet.run_id)
                return None

    async def run(self, stop: asyncio.Event | None = None) -> None:
        stop = stop or asyncio.Event()
        stop_task = asyncio.ensure_future(stop.wait())
        nxt: asyncio.Future | None = None
        try:
            async with self.bus.subscribe(self.settings.executor_channel) as messages:
                self.ready.set()
                log.info("subscribed to %s (judge=%s)", self.settings.executor_channel, self.judge.name)
                it = aiter(messages)
                while not stop.is_set():
                    nxt = asyncio.ensure_future(anext(it))
                    done, _ = await asyncio.wait({nxt, stop_task}, return_when=asyncio.FIRST_COMPLETED)
                    if nxt not in done:
                        nxt.cancel()
                        with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                            await nxt
                        break
                    try:
                        msg = nxt.result()
                    except StopAsyncIteration:
                        break
                    self._dispatch(msg)
        finally:
            stop_task.cancel()
            if nxt is not None and not nxt.done():
                nxt.cancel()
                with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration, Exception):
                    await nxt
            self.ready.clear()
            await self.drain()

    async def drain(self, wait_s: float | None = None) -> None:
        """Wait for in-flight evaluations (their interrupts still matter after a stop request)."""
        if self._tasks:
            await asyncio.wait(list(self._tasks), timeout=wait_s)


# ---------------------------------------------------------------------------------------------------- CLI


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="supervisor_daemon", description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=["ollama", "rules"], default=None,
                    help="verdict backend (default: SUPERVISOR_BACKEND or 'ollama')")
    ap.add_argument("--redis-url", default=None, help="override REDIS_URL")
    ap.add_argument("--no-warmup", action="store_true", help="skip loading the model before subscribing")
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args(argv)


async def _amain(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.redis_url:
        settings = dataclasses.replace(settings, redis_url=args.redis_url)
    judge = make_judge(settings, args.backend)
    bus = AsyncRedisBus.from_url(settings.redis_url)
    events = EventLog(settings.events_path)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)
    try:
        if not args.no_warmup and hasattr(judge, "warmup"):
            try:
                ms = await judge.warmup()
                log.info("warmed up %s in %.0fms", judge.name, ms)
            except Exception as e:  # noqa: BLE001
                log.warning("warmup of %s failed (%r); continuing", judge.name, e)
        daemon = SupervisorDaemon(bus, judge, events, settings, max_concurrency=args.max_concurrency)
        log.info("supervisor up: redis=%s judge=%s tracing=%s events=%s", settings.redis_url, judge.name,
                 settings.tracing, settings.events_path)
        await daemon.run(stop)
    finally:
        if hasattr(judge, "aclose"):
            await judge.aclose()
        await bus.aclose()
        if settings.tracing:
            with contextlib.suppress(Exception):
                from langsmith.run_trees import get_cached_client

                await asyncio.to_thread(get_cached_client().flush, 10.0)
    log.info("supervisor stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
