"""Wire formats for the two Redis channels. Both sides import these, so they are the contract."""
from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field


def now_ns() -> int:
    """Wall-clock nanoseconds. Executor and supervisor share a host (or a Docker VM clock), so deltas
    between processes are meaningful at millisecond resolution."""
    return time.time_ns()


SupervisionMode = Literal["reasoning", "action", "trajectory"]


class TrajectoryPacket(BaseModel):
    """executor_stream: in `reasoning` mode, published the moment the `reasoning` field of ProposeAction closes; in
    `action` / `trajectory` mode, published once the call is complete and carrying the finished `command` (plus, in
    `trajectory` mode, the episode's earlier steps in `history`)."""

    run_id: str  # the executor generation's run id
    thread_id: str  # one executor episode (many steps)
    step: int
    attempt: int = 0  # retries of the same step after a halt
    reasoning: str
    mode: SupervisionMode = "reasoning"
    command: str | None = None  # action modes: the command the tool gate is holding
    # trajectory mode: earlier steps, oldest first: {step, attempt, status, reasoning, command, observation}
    history: list[dict[str, Any]] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)  # task + compact recent history
    trace_context: dict[str, str] | None = None  # Langfuse {"trace_id", "parent_span_id"} of executor_generate
    parent_run_id: str | None = None
    published_at_ns: int = Field(default_factory=now_ns)


class Verdict(BaseModel):
    status: Literal["proceed", "halt"]
    reason: str = ""
    # Judge-side accounting, not part of the model's output: {prompt_tokens, completion_tokens, cost_usd, provider}
    usage: dict[str, Any] | None = None


class VerdictAck(BaseModel):
    """supervisor_acks: a proceed verdict. Not part of the kill path; it only tells the executor's tool gate
    that it need not wait out the grace window for a late halt."""

    run_id: str
    status: Literal["proceed"] = "proceed"
    published_at_ns: int = Field(default_factory=now_ns)


class InterruptSignal(BaseModel):
    """supervisor_interrupts: published for halt verdicts (`kind="halt"`: cancel this run_id's generation) and for
    stall escalations (`kind="escalate"`: end this thread_id's episode before its next command; never cancels a
    stream). `kind` defaults to "halt", so older producers and consumers keep working."""

    run_id: str
    thread_id: str = ""
    status: Literal["halt"] = "halt"
    kind: Literal["halt", "escalate"] = "halt"
    reason: str
    step: int | None = None  # escalate: the packet's step
    signals: dict[str, Any] = Field(default_factory=dict)  # escalate: the stall detector's signals
    supervisor_model: str = ""
    packet_published_at_ns: int = 0  # copied from the TrajectoryPacket
    received_at_ns: int = 0  # when the supervisor read the packet
    verdict_at_ns: int = 0  # when the model's verdict came back
    published_at_ns: int = Field(default_factory=now_ns)
