"""Langfuse tracing, behind a wrapper that is a strict no-op when tracing is off and never lets a tracing failure
break the executor or the supervisor.

Trace layout (one Langfuse trace per executor episode, ids are deterministic from the thread id):

    executor_episode            (agent,      executor process)
    ├── executor_generate       (generation, executor process)  one per ProposeAction stream; halted ones carry
    │   │                                                       intercepted / cancel_latency_ms / end_to_end_ms
    │   └── supervisor_evaluate (evaluator,  supervisor process) stitched in via the packet's trace_context
    │       └── judge           (generation, supervisor process) the Qwen call
    └── tool_execute            (tool,       executor process)

Scores (filterable / chartable in the Langfuse UI): `supervisor_halt` (0/1) on each executor_generate written by
the supervisor, `intercepted` (0/1) and `cancel_latency_ms` written by the executor, `violation` (0/1, ground
truth) when known.
"""
from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

from .config import Settings

log = logging.getLogger(__name__)


class Tracer:
    def __init__(self, client: Any = None):
        self.client = client

    @classmethod
    def from_settings(cls, settings: Settings, **client_kwargs: Any) -> Tracer:
        if not settings.tracing:
            return cls(None)
        try:
            from langfuse import Langfuse

            return cls(Langfuse(public_key=settings.langfuse_public_key, secret_key=settings.langfuse_secret_key,
                                host=settings.langfuse_host, environment=settings.langfuse_environment,
                                **client_kwargs))
        except Exception:  # noqa: BLE001
            log.warning("Langfuse client could not be created; tracing disabled", exc_info=True)
            return cls(None)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    @staticmethod
    def trace_id_for(seed: str) -> str:
        """Deterministic 32-hex trace id (the same algorithm as Langfuse.create_trace_id(seed=...))."""
        from langfuse import Langfuse

        return Langfuse.create_trace_id(seed=seed)

    @contextlib.contextmanager
    def observation(self, name: str, **kwargs: Any) -> Iterator[Any]:
        """`with tracer.observation("x", as_type="span", trace_context=..., metadata=...) as obs:` yields the
        Langfuse observation, or None when tracing is off or the observation could not be started. Exceptions from
        the body always propagate unchanged."""
        cm = obs = None
        if self.client is not None:
            try:
                cm = self.client.start_as_current_observation(name=name, **kwargs)
                obs = cm.__enter__()
            except Exception:  # noqa: BLE001
                log.debug("could not start observation %s", name, exc_info=True)
                cm = None
        if cm is None:
            yield None
            return
        try:
            yield obs
        except BaseException as e:
            with contextlib.suppress(Exception):
                cm.__exit__(type(e), e, e.__traceback__)
            raise
        else:
            with contextlib.suppress(Exception):
                cm.__exit__(None, None, None)

    @staticmethod
    def context(obs: Any) -> dict[str, str] | None:
        """The TraceContext a remote process needs to attach a child to `obs`."""
        if obs is None:
            return None
        try:
            return {"trace_id": obs.trace_id, "parent_span_id": obs.id}
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def update(obs: Any, **kwargs: Any) -> None:
        if obs is not None:
            with contextlib.suppress(Exception):
                obs.update(**kwargs)

    def score(self, name: str, value: float | str, **kwargs: Any) -> None:
        if self.client is not None:
            with contextlib.suppress(Exception):
                self.client.create_score(name=name, value=value, **kwargs)

    def flush(self) -> None:
        if self.client is not None:
            with contextlib.suppress(Exception):
                self.client.flush()
