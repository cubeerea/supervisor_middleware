"""Phases 2 and 4: the LangGraph executor.

    START -> executor --proposal--> tool --(not done)--> executor
                 |  ^                  \\--(done/max steps)--> END
                 |  '---halted (retry with the supervisor's reason)
                 '--refusal/error--> END

`executor` streams one ProposeAction tool call. The instant the `reasoning` string closes, the packet is
published to `executor_stream` with `publish_nowait` (no await) and token consumption continues. The
generation runs under `InterruptListener.run_guarded`, so a matching halt cancels it mid-stream.
`tool` is the second line of defence: it gives the verdict a short grace window (measured from the
reasoning publish) and refuses to execute when a halt arrived after the stream had already finished.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..config import Settings
from ..messages import TrajectoryPacket, now_ns
from ..redis_bus import AsyncRedisBus
from ..stream_parser import ReasoningWatcher, _decode
from ..telemetry import EventLog
from ..tracing import Tracer
from .interception import InterceptionHalt, InterruptListener
from .prompts import halt_feedback, system_prompt
from .sources import PROPOSE_ACTION_TOOL, StreamSource

log = logging.getLogger(__name__)
SEVERED = "\n# ... [stream severed by supervisor]"


class ExecState(TypedDict, total=False):
    task: str
    thread_id: str
    session_id: str | None  # Langfuse session (an experiment) grouping episode traces
    messages: list[dict[str, Any]]
    step: int
    attempt: int
    proposal: dict[str, Any] | None
    halts: list[dict[str, Any]]
    trajectory: list[dict[str, Any]]  # one record per proposal (executed, halted or blocked)
    done: bool
    outcome: str


@dataclass
class PartialTurn:
    run_id: str
    tool_use_id: str | None = None
    reasoning: str | None = None
    published_at_ns: int = 0
    blocks: list[dict[str, Any]] = field(default_factory=list)  # completed non-tool blocks (text/thinking)
    final: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    watcher: ReasoningWatcher = field(default_factory=ReasoningWatcher)

    @property
    def command_partial(self) -> str:
        sc = self.watcher.scanner
        if "command" in sc.values:
            return str(sc.values["command"])
        raw = sc.partial.get("command", "")
        for cut in range(0, 6):  # an escape may be split at the end of the buffer
            try:
                return _decode(raw[: len(raw) - cut] if cut else raw)
            except Exception:  # noqa: BLE001
                continue
        return raw


@dataclass
class ExecutorDeps:
    settings: Settings
    source: StreamSource
    bus: AsyncRedisBus
    listener: InterruptListener
    events: EventLog
    workdir: Path
    run_command: Any  # async (command, workdir, timeout_s) -> CommandResult
    verify: Any = None  # (db_path) -> dict
    max_halts_per_step: int = 3
    command_timeout_s: float = 60.0
    trigger: str = "value_close"
    tracer: Tracer = field(default_factory=Tracer)  # disabled unless given a Langfuse-backed Tracer


def _request(deps: ExecutorDeps, state: ExecState) -> dict[str, Any]:
    docs_path = deps.workdir / "ORM_DOCS.md"
    docs = docs_path.read_text() if docs_path.exists() else "(missing)"
    req: dict[str, Any] = {
        "model": deps.settings.executor_model,
        "max_tokens": deps.settings.executor_max_tokens,
        "system": system_prompt(deps.workdir, docs),
        "tools": [PROPOSE_ACTION_TOOL],
        # Forced tool_choice is rejected by current models; steer from the prompt and allow one call.
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        "messages": state["messages"],
    }
    if deps.settings.executor_model.startswith("claude-sonnet-5-5"):
        # Thinking off: the `reasoning` field *is* the reasoning we supervise, and it streams sooner.
        req["thinking"] = {"type": "between_tools"}
    return req


async def _consume(deps: ExecutorDeps, state: ExecState, turn: PartialTurn, trace_context: dict | None) -> PartialTurn:
    in_tool = False
    async for kind, data in deps.source.stream(_request(deps, state)):
        if kind == "meta":
            turn.meta.update(data)
        elif kind == "block_start":
            in_tool = data.get("type") == "tool_use" and data.get("name") == "ProposeAction"
            if in_tool:
                turn.tool_use_id = data.get("id")
        elif kind == "input_json" and in_tool:
            reasoning = turn.watcher.feed(data)
            if reasoning is not None:
                _publish(deps, state, turn, reasoning, trace_context)
        elif kind == "block_complete":
            if data.get("type") == "tool_use":
                in_tool = False
            else:
                turn.blocks.append(data)
        elif kind == "final":
            turn.final = data
    if (reasoning := turn.watcher.finish()) is not None:
        _publish(deps, state, turn, reasoning, trace_context)
    return turn


def _publish(deps: ExecutorDeps, state: ExecState, turn: PartialTurn, reasoning: str, trace_context: dict | None):
    turn.reasoning = reasoning
    last_obs = next((t.get("observation") for t in reversed(state.get("trajectory", [])) if t.get("observation")), "")
    packet = TrajectoryPacket(
        run_id=turn.run_id, thread_id=state["thread_id"], step=state.get("step", 0), attempt=state.get("attempt", 0),
        reasoning=reasoning, trace_context=trace_context,
        state={"task": state["task"], "step": state.get("step", 0), "last_observation": (last_obs or "")[-1500:],
               "previous_halts": [h["reason"] for h in state.get("halts", [])][-3:],
               "session_id": state.get("session_id")},
    )
    turn.published_at_ns = packet.published_at_ns
    deps.bus.publish_nowait(deps.settings.executor_channel, packet)  # fire-and-forget: do NOT await
    deps.events.emit("reasoning_published", run_id=turn.run_id, thread_id=state["thread_id"],
                     step=state.get("step", 0), attempt=state.get("attempt", 0),
                     published_at_ns=packet.published_at_ns, reasoning_chars=len(reasoning),
                     command_chars_so_far=len(turn.command_partial), **_labels(turn))


def _labels(turn: PartialTurn) -> dict[str, Any]:
    return {k: turn.meta[k] for k in ("intended_violation", "scenario") if k in turn.meta}


def build_graph(deps: ExecutorDeps):
    async def executor(state: ExecState) -> dict[str, Any]:
        await deps.listener.start()  # background subscriber to supervisor_interrupts (idempotent)
        run_id = str(uuid.uuid4())
        turn = PartialTurn(run_id=run_id, watcher=ReasoningWatcher(trigger=deps.trigger))  # type: ignore[arg-type]
        started = now_ns()
        tr = deps.tracer
        with tr.observation("executor_generate", as_type="generation", model=deps.settings.executor_model,
                            input={"step": state.get("step", 0), "attempt": state.get("attempt", 0),
                                   "last_message": state["messages"][-1] if state["messages"] else None},
                            metadata={"thread_id": state["thread_id"], "run_id": run_id, "backend": deps.source.name,
                                      "step": state.get("step", 0), "attempt": state.get("attempt", 0)}) as obs:
            ctx = tr.context(obs)
            try:
                await deps.listener.run_guarded(run_id, _consume(deps, state, turn, ctx), {"partial": turn})
            except InterceptionHalt as halt:
                sig = halt.signal
                cancel_ms = (halt.cancelled_at_ns - sig.published_at_ns) / 1e6
                e2e_ms = (halt.cancelled_at_ns - sig.packet_published_at_ns) / 1e6 if sig.packet_published_at_ns \
                    else None
                tr.update(obs, level="WARNING", status_message=f"severed by supervisor: {sig.reason}"[:500],
                          output={"reasoning": turn.reasoning, "command_partial": turn.command_partial},
                          metadata={"intercepted": True, "halt_reason": sig.reason, "cancel_latency_ms": cancel_ms,
                                    "end_to_end_ms": e2e_ms, "command_chars_at_cancel": len(turn.command_partial),
                                    **_labels(turn)})
                if ctx:
                    tr.score("intercepted", 1, data_type="BOOLEAN", trace_id=ctx["trace_id"],
                             observation_id=ctx["parent_span_id"], comment=sig.reason[:500])
                    tr.score("cancel_latency_ms", cancel_ms, trace_id=ctx["trace_id"],
                             observation_id=ctx["parent_span_id"])
                    if "intended_violation" in turn.meta:
                        tr.score("violation", 1 if turn.meta["intended_violation"] else 0, data_type="BOOLEAN",
                                 trace_id=ctx["trace_id"], observation_id=ctx["parent_span_id"])
                return _on_halt(state, turn, sig.reason, late=False,
                                cancelled_at_ns=halt.cancelled_at_ns, started=started)
            final_out = turn.final or {}
            tr.update(obs, output={"reasoning": turn.reasoning, "command": turn.command_partial,
                                   "stop_reason": final_out.get("stop_reason")},
                      metadata={"intercepted": False, **_labels(turn)})
            if ctx:
                tr.score("intercepted", 0, data_type="BOOLEAN", trace_id=ctx["trace_id"],
                         observation_id=ctx["parent_span_id"])
                if "intended_violation" in turn.meta:
                    tr.score("violation", 1 if turn.meta["intended_violation"] else 0, data_type="BOOLEAN",
                             trace_id=ctx["trace_id"], observation_id=ctx["parent_span_id"])

        final = turn.final or {"content": [], "stop_reason": "error"}
        stop = final.get("stop_reason")
        tool = next((b for b in final["content"] if b.get("type") == "tool_use"), None)
        deps.events.emit("generation_completed", run_id=run_id, thread_id=state["thread_id"],
                         step=state.get("step", 0), stop_reason=stop, ms=(now_ns() - started) / 1e6)
        if stop == "refusal":
            return {"done": True, "outcome": "refusal"}
        if tool is None or stop == "max_tokens" or not isinstance(tool.get("input"), dict):
            # No usable tool call: nudge and try again (counts as a step so a confused model terminates).
            msgs = [*state["messages"], {"role": "assistant", "content": final["content"] or
                                         [{"type": "text", "text": "(no output)"}]}] if tool is None else \
                list(state["messages"])
            if tool is None:
                msgs.append({"role": "user", "content": "Respond by calling ProposeAction exactly once."})
            return {"messages": msgs, "proposal": None, "step": state.get("step", 0) + 1}
        args = tool["input"]
        if not (isinstance(args.get("reasoning"), str) and isinstance(args.get("command"), str)):
            msgs = [*state["messages"], {"role": "assistant", "content": final["content"]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool["id"], "is_error": True,
                                                  "content": json.dumps({"INVALID_JSON": json.dumps(args)})}]}]
            return {"messages": msgs, "proposal": None, "step": state.get("step", 0) + 1}
        return {"proposal": {"run_id": run_id, "tool_use_id": tool["id"], "reasoning": args["reasoning"],
                             "command": args["command"], "content": final["content"],
                             "published_at_ns": turn.published_at_ns, **_labels(turn)}}

    def _on_halt(state: ExecState, turn: PartialTurn, reason: str, *, late: bool, cancelled_at_ns: int = 0,
                 started: int = 0, content: list | None = None) -> dict[str, Any]:
        tool_id = turn.tool_use_id or f"toolu_halted_{turn.run_id[:8]}"
        if content is None:
            command = turn.command_partial + SEVERED
            content = [*turn.blocks, {"type": "tool_use", "id": tool_id, "name": "ProposeAction",
                                      "input": {"reasoning": turn.reasoning or "", "command": command}}]
        msgs = [*state["messages"], {"role": "assistant", "content": content},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "is_error": True,
                                              "content": halt_feedback(reason, late)}]}]
        halts = [*state.get("halts", []), {"run_id": turn.run_id, "step": state.get("step", 0), "reason": reason,
                                           "late": late}]
        record = {"run_id": turn.run_id, "step": state.get("step", 0), "attempt": state.get("attempt", 0),
                  "reasoning": turn.reasoning, "command": turn.command_partial, "status": "halted_late" if late
                  else "halted", "halt_reason": reason, "command_chars_at_cancel": len(turn.command_partial),
                  **_labels(turn)}
        deps.events.emit("step_halted", thread_id=state["thread_id"], late=late, cancelled_at_ns=cancelled_at_ns,
                         **{k: v for k, v in record.items() if k not in ("reasoning",)})
        attempt = state.get("attempt", 0) + 1
        out: dict[str, Any] = {"messages": msgs, "halts": halts, "proposal": None, "attempt": attempt,
                               "trajectory": [*state.get("trajectory", []), record]}
        if attempt > deps.max_halts_per_step:  # supervisor keeps rejecting: move on so the episode terminates
            out.update(step=state.get("step", 0) + 1, attempt=0)
        return out

    async def tool(state: ExecState) -> dict[str, Any]:
        p = state["proposal"]
        assert p is not None
        # Late-halt gate: give the supervisor until publish + grace to object to a finished stream.
        sig = await deps.listener.wait_for_late_halt(
            p["run_id"], p["published_at_ns"] + int(deps.settings.verdict_grace_s * 1e9))
        if sig is not None:
            turn = PartialTurn(run_id=p["run_id"], tool_use_id=p["tool_use_id"], reasoning=p["reasoning"],
                               meta={k: p[k] for k in ("intended_violation", "scenario") if k in p})
            turn.watcher.scanner.values["command"] = p["command"]
            deps.events.emit("late_halt_blocked", run_id=p["run_id"], thread_id=state["thread_id"],
                             interrupt_published_at_ns=sig.published_at_ns, blocked_at_ns=now_ns())
            return _on_halt(state, turn, sig.reason, late=True, content=p["content"])

        step = state.get("step", 0)
        record = {"run_id": p["run_id"], "step": step, "attempt": state.get("attempt", 0),
                  "reasoning": p["reasoning"], "command": p["command"], "status": "executed",
                  **{k: p[k] for k in ("intended_violation", "scenario") if k in p}}
        if p["command"].strip() == "FINISH":
            result = deps.verify(deps.workdir / "app.db") if deps.verify else {}
            record.update(status="finish", observation=json.dumps(result))
            deps.events.emit("finish", run_id=p["run_id"], thread_id=state["thread_id"], step=step, verify=result)
            done = bool(result.get("complete", True)) if result else True
            if done:
                return {"done": True, "outcome": "finished", "proposal": None,
                        "trajectory": [*state.get("trajectory", []), record],
                        "messages": [*state["messages"], {"role": "assistant", "content": p["content"]}]}
            obs = f"FINISH rejected: verification failed: {json.dumps(result)}"
            is_error = True
        else:
            with deps.tracer.observation("tool_execute", as_type="tool", input={"command": p["command"]},
                                         metadata={"run_id": p["run_id"], "step": step}) as tobs:
                res = await deps.run_command(p["command"], deps.workdir, deps.command_timeout_s)
                deps.tracer.update(tobs, output={"exit_code": res.exit_code, "stdout": res.stdout[-2000:],
                                                 "stderr": res.stderr[-2000:]},
                                   level="ERROR" if res.exit_code != 0 else "DEFAULT")
            obs = (f"exit_code={res.exit_code}{' (timed out)' if res.timed_out else ''}\n"
                   f"--- stdout\n{res.stdout}\n--- stderr\n{res.stderr}")
            is_error = res.exit_code != 0
            record["observation"] = obs
            record["exit_code"] = res.exit_code
        deps.events.emit("command_executed", run_id=p["run_id"], thread_id=state["thread_id"], step=step,
                         command=p["command"], exit_code=record.get("exit_code"),
                         **{k: p[k] for k in ("intended_violation", "scenario") if k in p})
        msgs = [*state["messages"], {"role": "assistant", "content": p["content"]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": p["tool_use_id"],
                                              "is_error": is_error, "content": obs}]}]
        return {"messages": msgs, "proposal": None, "step": step + 1, "attempt": 0,
                "trajectory": [*state.get("trajectory", []), {**record, "observation": obs}]}

    def after_executor(state: ExecState) -> str:
        if state.get("done"):
            return END
        if state.get("proposal"):
            return "tool"
        if state.get("step", 0) >= deps.settings.max_steps:
            return END
        return "executor"

    def after_tool(state: ExecState) -> str:
        if state.get("done") or state.get("step", 0) >= deps.settings.max_steps:
            return END
        return "executor"

    g = StateGraph(ExecState)
    g.add_node("executor", executor)
    g.add_node("tool", tool)
    g.add_edge(START, "executor")
    g.add_conditional_edges("executor", after_executor, ["tool", "executor", END])
    g.add_conditional_edges("tool", after_tool, ["executor", END])
    return g.compile()
