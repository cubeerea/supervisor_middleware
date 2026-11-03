"""AnthropicSource and the kill switch against a fake AsyncAnthropic client (no API key, no network).

The fake client mirrors the SDK surface AnthropicSource relies on: `client.messages.stream(**kwargs)` returns an
async context manager whose stream async-iterates real `anthropic.types` events and exposes
`current_message_snapshot` and `async get_final_message()`.
"""
import asyncio
import contextlib
import json

import fakeredis
from anthropic.types import (
    InputJSONDelta,
    Message,
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    ToolUseBlock,
    Usage,
)
from anthropic.types.raw_message_delta_event import Delta

from supervisor_middleware.config import Settings
from supervisor_middleware.executor.graph import ExecutorDeps, build_graph
from supervisor_middleware.executor.interception import InterruptListener
from supervisor_middleware.executor.sources import PROPOSE_ACTION_TOOL, AnthropicSource
from supervisor_middleware.messages import InterruptSignal, TrajectoryPacket, VerdictAck, now_ns
from supervisor_middleware.redis_bus import AsyncRedisBus
from supervisor_middleware.telemetry import EventLog, read_events

MODEL = "claude-test-model"


class Turn:
    """One scripted assistant turn: a single ProposeAction tool_use streamed as input_json fragments."""

    def __init__(self, tool_id: str, fragments: list[str], delay_s: float = 0.0):
        self.tool_id = tool_id
        self.fragments = fragments
        self.delay_s = delay_s
        self.input = json.loads("".join(fragments))

    def block(self, with_input: bool = True) -> ToolUseBlock:
        return ToolUseBlock(type="tool_use", id=self.tool_id, name="ProposeAction",
                            input=self.input if with_input else {})

    def message(self, content: list) -> Message:
        return Message(id=f"msg_{self.tool_id}", type="message", role="assistant", model=MODEL, content=content,
                       stop_reason="tool_use", stop_sequence=None, usage=Usage(input_tokens=10, output_tokens=20))


class FakeStream:
    def __init__(self, turn: Turn):
        self.turn = turn
        self.current_message_snapshot = turn.message([])
        self.fragments_sent = 0
        self.exited = False
        self.exit_exc_type: type | None = None
        self.fragments_sent_at_exit: int | None = None

    def _events(self):
        t = self.turn
        yield RawMessageStartEvent(type="message_start", message=t.message([]))
        yield RawContentBlockStartEvent(type="content_block_start", index=0, content_block=t.block(with_input=False))
        for frag in t.fragments:
            yield RawContentBlockDeltaEvent(type="content_block_delta", index=0,
                                            delta=InputJSONDelta(type="input_json_delta", partial_json=frag))
        self.current_message_snapshot = t.message([t.block()])  # the SDK accumulates the parsed input
        yield RawContentBlockStopEvent(type="content_block_stop", index=0)
        yield RawMessageDeltaEvent(type="message_delta", delta=Delta(stop_reason="tool_use", stop_sequence=None),
                                   usage=MessageDeltaUsage(output_tokens=20))
        yield RawMessageStopEvent(type="message_stop")

    async def __aiter__(self):
        for ev in self._events():
            if self.turn.delay_s:
                await asyncio.sleep(self.turn.delay_s)
            if ev.type == "content_block_delta":
                self.fragments_sent += 1
            yield ev

    async def get_final_message(self) -> Message:
        return self.turn.message([self.turn.block()])


class FakeStreamManager:
    def __init__(self, stream: FakeStream):
        self.stream = stream

    async def __aenter__(self) -> FakeStream:
        return self.stream

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.stream.exited = True
        self.stream.exit_exc_type = exc_type
        self.stream.fragments_sent_at_exit = self.stream.fragments_sent


class FakeMessages:
    def __init__(self, turns: list[Turn]):
        self.turns = list(turns)
        self.calls: list[dict] = []
        self.streams: list[FakeStream] = []

    def stream(self, **kwargs) -> FakeStreamManager:
        self.calls.append(kwargs)
        s = FakeStream(self.turns.pop(0))
        self.streams.append(s)
        return FakeStreamManager(s)


class FakeAsyncAnthropic:
    def __init__(self, turns: list[Turn]):
        self.messages = FakeMessages(turns)


def _split(s: str, n: int) -> list[str]:
    return [s[i:i + n] for i in range(0, len(s), n)]


# --------------------------------------------------------------------------- 1. event normalisation


async def test_anthropic_source_normalises_sdk_events_and_passes_request_through():
    frags = ['{"reasoning": "List the', ' tables first.", "comm', 'and": "print(1)"}']
    client = FakeAsyncAnthropic([Turn("toolu_01", frags)])
    request = {"model": MODEL, "max_tokens": 512, "system": "sys",
               "tools": [PROPOSE_ACTION_TOOL],
               "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
               "messages": [{"role": "user", "content": "go"}]}

    events = [e async for e in AnthropicSource(client).stream(request)]

    assert [k for k, _ in events] == ["block_start", "input_json", "input_json", "input_json",
                                      "block_complete", "final"]
    assert events[0][1] == {"index": 0, "type": "tool_use", "id": "toolu_01", "name": "ProposeAction"}
    assert [d for k, d in events if k == "input_json"] == frags
    expected_input = {"reasoning": "List the tables first.", "command": "print(1)"}
    complete = events[4][1]
    assert complete["type"] == "tool_use" and complete["id"] == "toolu_01" and complete["input"] == expected_input
    final = events[5][1]
    assert final["stop_reason"] == "tool_use"
    assert [(b["type"], b["id"], b["input"]) for b in final["content"]] == [("tool_use", "toolu_01", expected_input)]

    (kwargs,) = client.messages.calls
    assert kwargs == request
    (tool,) = kwargs["tools"]
    assert tool is PROPOSE_ACTION_TOOL and tool["eager_input_streaming"] is True and tool["strict"] is True
    assert kwargs["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert client.messages.streams[0].exited and client.messages.streams[0].exit_exc_type is None


# --------------------------------------------------------------------------- 2. kill switch over the graph


def _bus(server) -> AsyncRedisBus:
    return AsyncRedisBus(fakeredis.FakeAsyncRedis(server=server, decode_responses=True))


@contextlib.asynccontextmanager
async def fake_supervisor(server, settings: Settings):
    """Halts any packet whose reasoning mentions raw SQL the instant it arrives; acks everything else."""
    bus = _bus(server)
    ready = asyncio.Event()

    async def loop():
        async with bus.subscribe(settings.executor_channel) as msgs:
            ready.set()
            async for d in msgs:
                p = TrajectoryPacket.model_validate({k: v for k, v in d.items() if not k.startswith("_")})
                if "raw sql" in p.reasoning.lower():
                    await bus.publish(settings.interrupt_channel, InterruptSignal(
                        run_id=p.run_id, thread_id=p.thread_id, reason="raw SQL is forbidden",
                        packet_published_at_ns=p.published_at_ns, received_at_ns=now_ns()))
                else:
                    await bus.publish(settings.ack_channel, VerdictAck(run_id=p.run_id))

    t = asyncio.create_task(loop())
    await ready.wait()
    try:
        yield
    finally:
        t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t
        await bus.aclose()


async def test_kill_switch_closes_anthropic_stream_mid_generation(tmp_path):
    bad_reasoning = "The ORM has no update_batch, so bypass it and use raw SQL via sqlite3."
    bad_command = "import sqlite3\n" + "".join(f"# filler line {i}\n" for i in range(40)) + "print('done')\n"
    bad_json = json.dumps({"reasoning": bad_reasoning, "command": bad_command})
    bad = Turn("toolu_bad", _split(bad_json, 12), delay_s=0.02)  # slow: ~70 fragments, ~1.5 s
    good = Turn("toolu_good", _split(json.dumps({"reasoning": "Verified; done.", "command": "FINISH"}), 12))
    client = FakeAsyncAnthropic([bad, good])

    server = fakeredis.FakeServer()
    settings = Settings(verdict_grace_s=0.5, max_steps=4, events_path=tmp_path / "events.jsonl")
    events = EventLog(settings.events_path)
    bus = _bus(server)

    async def no_exec(*_a, **_k):
        raise AssertionError("no command should be executed")

    deps = ExecutorDeps(settings=settings, source=AnthropicSource(client), bus=bus,
                        listener=InterruptListener(bus, settings.interrupt_channel, settings.ack_channel, events),
                        events=events, workdir=tmp_path, run_command=no_exec, verify=None)
    init = {"task": "migrate", "thread_id": "ep-test", "messages": [{"role": "user", "content": "migrate"}],
            "step": 0, "attempt": 0, "halts": [], "trajectory": []}
    async with fake_supervisor(server, settings):
        try:
            final = await asyncio.wait_for(build_graph(deps).ainvoke(init), timeout=10)
        finally:
            await deps.listener.stop()
            await bus.aclose()

    # the first stream was severed: its context manager exited (via cancellation) before all fragments went out
    bad_stream, good_stream = client.messages.streams
    assert bad_stream.exited and bad_stream.exit_exc_type is asyncio.CancelledError
    assert 0 < bad_stream.fragments_sent_at_exit < len(bad.fragments)
    assert good_stream.exited and good_stream.fragments_sent == len(good.fragments)

    ev = read_events(settings.events_path)
    cancels = [e for e in ev if e["event"] == "generation_cancelled"]
    assert len(cancels) == 1 and cancels[0]["reason"] == "raw SQL is forbidden"
    assert final["outcome"] == "finished" and [h["late"] for h in final["halts"]] == [False]

    # the halted turn is fed back as an is_error tool_result for the severed tool_use
    def halt_results(messages):
        return [b for m in messages if m["role"] == "user" and isinstance(m["content"], list)
                for b in m["content"] if b.get("type") == "tool_result" and "SUPERVISOR HALT" in b["content"]]

    (fed_back,) = halt_results(client.messages.calls[1]["messages"])
    assert fed_back["is_error"] is True and fed_back["tool_use_id"] == "toolu_bad"
    assert "raw SQL is forbidden" in fed_back["content"]
    assert halt_results(final["messages"]) == [fed_back]
