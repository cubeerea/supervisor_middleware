#!/usr/bin/env python3
"""Watch both interception channels and print one or two lines per message.

    uv run python scripts/redis_monitor.py [--redis-url redis://localhost:6379/0] [--width 100]

Shows time, channel, run_id[:8], step, truncated reasoning (executor_stream) or halt reason
(supervisor_interrupts), and the publish->receive delay in ms from the payload's `published_at_ns`.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
from datetime import datetime

import redis.asyncio as aioredis

try:
    from supervisor_middleware.config import EXECUTOR_STREAM, SUPERVISOR_INTERRUPTS
except ImportError:  # running outside the project venv
    EXECUTOR_STREAM, SUPERVISOR_INTERRUPTS = "executor_stream", "supervisor_interrupts"

USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


DIM, BOLD, CYAN, RED, YELLOW, GREEN = "2", "1", "36", "1;31", "33", "32"


def _short(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: max(width - 1, 0)] + "…"


def _delay_ms(payload: dict, recv_ns: int) -> str:
    sent = payload.get("published_at_ns")
    if not isinstance(sent, int) or sent <= 0:
        return "    ?ms"
    return f"{(recv_ns - sent) / 1e6:7.1f}ms"


def format_message(channel: str, raw: str | bytes, recv_ns: int, width: int = 100) -> str:
    """Render one pub/sub message. Pure function (no I/O) so it is easy to eyeball and test."""
    ts = _c(DIM, datetime.fromtimestamp(recv_ns / 1e9).strftime("%H:%M:%S.%f")[:-3])
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("payload is not an object")
    except (ValueError, TypeError):
        return f"{ts} {_c(YELLOW, f'{channel:<21}')} (unparsed) {_short(raw, width)}"

    run = str(payload.get("run_id", "?"))[:8]
    delay = _delay_ms(payload, recv_ns)
    if channel == SUPERVISOR_INTERRUPTS:
        tag = _c(RED, "HALT".ljust(21))
        model = payload.get("supervisor_model") or ""
        line1 = f"{ts} {tag} run={_c(BOLD, run)} {'':>7} {_c(RED, delay)} {_c(DIM, model)}"
        timing = ""
        pkt, rcv, vrd = (payload.get(k, 0) for k in ("packet_published_at_ns", "received_at_ns", "verdict_at_ns"))
        if all(isinstance(v, int) and v > 0 for v in (pkt, rcv, vrd)):
            timing = _c(DIM, f" [bus {(rcv - pkt) / 1e6:.1f}ms, judge {(vrd - rcv) / 1e6:.1f}ms]")
        return f"{line1}\n{'':>13}reason: {_short(payload.get('reason', ''), width)}{timing}"

    tag = _c(CYAN, f"{channel:<21}")
    step = f"step={payload.get('step', '?')}"
    if payload.get("attempt"):
        step += f".{payload['attempt']}"
    line1 = f"{ts} {tag} run={_c(BOLD, run)} {step:<7} {_c(GREEN, delay)}"
    return f"{line1}\n{'':>13}{_short(payload.get('reasoning', ''), width)}"


async def monitor(redis_url: str, width: int) -> None:
    client = aioredis.from_url(redis_url)
    pubsub = client.pubsub()
    try:
        await pubsub.subscribe(EXECUTOR_STREAM, SUPERVISOR_INTERRUPTS)
        print(_c(DIM, f"listening on {redis_url} :: {EXECUTOR_STREAM}, {SUPERVISOR_INTERRUPTS} (Ctrl-C to quit)"),
              flush=True)
        async for msg in pubsub.listen():
            if msg.get("type") != "message":
                continue
            recv_ns = time.time_ns()
            channel = msg["channel"].decode() if isinstance(msg["channel"], bytes) else msg["channel"]
            data = msg["data"].decode(errors="replace") if isinstance(msg["data"], bytes) else msg["data"]
            print(format_message(channel, data, recv_ns, width), flush=True)
    finally:
        with contextlib.suppress(Exception):
            await pubsub.aclose()
        with contextlib.suppress(Exception):
            await client.aclose()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--redis-url", default=os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    ap.add_argument("--width", type=int, default=100, help="truncate reasoning/reason to this many chars")
    args = ap.parse_args(argv)
    try:
        asyncio.run(monitor(args.redis_url, args.width))
    except KeyboardInterrupt:
        return 0
    except (aioredis.ConnectionError, OSError) as exc:
        print(f"redis_monitor: cannot reach {args.redis_url}: {exc}", file=sys.stderr)
        print("hint: docker compose up -d redis", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
