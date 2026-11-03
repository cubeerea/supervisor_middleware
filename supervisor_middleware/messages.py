"""Wire formats for the two Redis channels. Both sides import these, so they are the contract."""
from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field


def now_ns() -> int:
    """Wall-clock nanoseconds. Executor and supervisor share a host (or a Docker VM clock), so deltas
    between processes are meaningful at millisecond resolution."""
    return time.time_ns()


class TrajectoryPacket(BaseModel):
    """executor_stream: published the moment the `reasoning` field of ProposeAction closes."""

    run_id: str  # the executor generation's run id (LangSmith run id when tracing is on)
    thread_id: str  # one executor episode (many steps)
    step: int
    attempt: int = 0  # retries of the same step after a halt
    reasoning: str
    state: dict[str, Any] = Field(default_factory=dict)  # task + compact recent history
    trace_headers: dict[str, str] | None = None  # RunTree.to_headers() for distributed tracing
    parent_run_id: str | None = None
    published_at_ns: int = Field(default_factory=now_ns)


class Verdict(BaseModel):
    status: Literal["proceed", "halt"]
    reason: str = ""


class VerdictAck(BaseModel):
    """supervisor_acks: a proceed verdict. Not part of the kill path; it only tells the executor's tool gate
    that it need not wait out the grace window for a late halt."""

    run_id: str
    status: Literal["proceed"] = "proceed"
    published_at_ns: int = Field(default_factory=now_ns)


class InterruptSignal(BaseModel):
    """supervisor_interrupts: published only for halt verdicts."""

    run_id: str
    thread_id: str = ""
    status: Literal["halt"] = "halt"
    reason: str
    supervisor_model: str = ""
    packet_published_at_ns: int = 0  # copied from the TrajectoryPacket
    received_at_ns: int = 0  # when the supervisor read the packet
    verdict_at_ns: int = 0  # when the model's verdict came back
    published_at_ns: int = Field(default_factory=now_ns)
