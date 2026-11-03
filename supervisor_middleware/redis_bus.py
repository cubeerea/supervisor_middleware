"""Non-blocking Redis pub/sub manager.

`publish_nowait` schedules the publish on the running loop and returns immediately, so the executor
can keep consuming tokens while the packet goes out. `subscribe` yields decoded JSON messages and
signals `ready` once Redis has confirmed the subscription (avoids missing an early interrupt).
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as aioredis
from pydantic import BaseModel

log = logging.getLogger(__name__)
_MEMORY_SERVER = None


def _encode(payload: BaseModel | dict[str, Any]) -> str:
    return payload.model_dump_json() if isinstance(payload, BaseModel) else json.dumps(payload)


class AsyncRedisBus:
    def __init__(self, client: aioredis.Redis):
        self.client = client
        self._pending: set[asyncio.Task] = set()  # strong refs so fire-and-forget tasks aren't GC'd

    @classmethod
    def from_url(cls, url: str) -> AsyncRedisBus:
        """`redis://...` for a real server; `memory://` for an in-process fakeredis server shared by every bus
        created in this process (single-process demos and tests; needs the dev dependency `fakeredis`)."""
        if url.startswith("memory://"):
            import fakeredis  # dev dependency, imported lazily

            global _MEMORY_SERVER
            if _MEMORY_SERVER is None:
                _MEMORY_SERVER = fakeredis.FakeServer()
            return cls(fakeredis.FakeAsyncRedis(server=_MEMORY_SERVER, decode_responses=True))
        return cls(aioredis.from_url(url, decode_responses=True))

    async def publish(self, channel: str, payload: BaseModel | dict[str, Any]) -> int:
        """Awaitable publish. Returns the number of subscribers that received it."""
        return await self.client.publish(channel, _encode(payload))

    def publish_nowait(self, channel: str, payload: BaseModel | dict[str, Any]) -> asyncio.Task:
        """Fire-and-forget publish: never blocks the caller. Errors are logged, not raised."""
        data = _encode(payload)  # serialise now: the caller may mutate state afterwards

        async def _send() -> int:
            try:
                return await self.client.publish(channel, data)
            except Exception:  # noqa: BLE001 - a lost packet must never crash the executor
                log.exception("publish to %s failed", channel)
                return 0

        task = asyncio.get_running_loop().create_task(_send(), name=f"publish:{channel}")
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return task

    async def flush(self) -> None:
        """Wait for in-flight fire-and-forget publishes (call at shutdown)."""
        if self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    @asynccontextmanager
    async def subscribe(self, *channels: str) -> AsyncIterator[AsyncIterator[dict[str, Any]]]:
        """`async with bus.subscribe("ch") as messages: async for msg in messages: ...`

        The subscription is confirmed by Redis before the context body runs."""
        pubsub = self.client.pubsub(ignore_subscribe_messages=True)
        await pubsub.subscribe(*channels)

        closed = False

        async def _iter() -> AsyncIterator[dict[str, Any]]:
            while not closed:
                try:
                    msg = await pubsub.get_message(timeout=1.0)
                except RuntimeError:  # pubsub closed underneath us (shutdown): end the iteration
                    if closed:
                        return
                    raise
                if msg is None:
                    continue
                if msg.get("type") != "message":
                    continue
                try:
                    data = json.loads(msg["data"])
                except (TypeError, ValueError):
                    log.warning("dropping non-JSON message on %s", msg.get("channel"))
                    continue
                if isinstance(data, dict):
                    data.setdefault("_channel", msg.get("channel"))
                    yield data

        try:
            yield _iter()
        finally:
            closed = True
            try:
                await pubsub.unsubscribe(*channels)
                await pubsub.aclose()
            except Exception:  # noqa: BLE001
                log.debug("pubsub close failed", exc_info=True)

    async def aclose(self) -> None:
        await self.flush()
        await self.client.aclose()
