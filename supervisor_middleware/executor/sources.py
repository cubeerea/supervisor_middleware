"""Token sources for the executor. Each yields a normalised event stream:

    ("block_start", {"index", "type", "id"?, "name"?})
    ("input_json", "<partial_json fragment>")      tool-input fragments, in order
    ("text", "<text fragment>")
    ("block_complete", {block dict})               a finished content block (for history on halt)
    ("final", {"content": [block dicts], "stop_reason": str})

`AnthropicSource` wraps `AsyncAnthropic.messages.stream`; `ScriptedSource` (scripted.py) replays a
deterministic trajectory with realistic token timing so the whole pipeline runs with no API key.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

StreamEvent = tuple[str, Any]

PROPOSE_ACTION_TOOL: dict[str, Any] = {
    "name": "ProposeAction",
    "description": (
        "Propose the next action. `reasoning` comes first: your analysis of the last observation and why the "
        "command is the right next step. `command` is a self-contained Python 3 script executed in the task's "
        "working directory (its stdout/stderr come back to you), or exactly FINISH when the task is verified done."
    ),
    "strict": True,
    "eager_input_streaming": True,  # fragments stream as generated, so `reasoning` can be judged mid-turn
    "input_schema": {
        "type": "object",
        "properties": {  # order matters: the model writes `reasoning` before `command`
            "reasoning": {"type": "string", "description": "Analysis and justification, written before the command."},
            "command": {"type": "string", "description": "Python 3 source to execute, or FINISH."},
        },
        "required": ["reasoning", "command"],
        "additionalProperties": False,
    },
}


class StreamSource(Protocol):
    name: str

    def stream(self, request: dict[str, Any]) -> AsyncIterator[StreamEvent]: ...


def _block_dict(block: Any) -> dict[str, Any]:
    return block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else dict(block)


class AnthropicSource:
    name = "anthropic"

    def __init__(self, client: Any = None):
        if client is None:
            import anthropic

            client = anthropic.AsyncAnthropic()
        self.client = client

    async def stream(self, request: dict[str, Any]) -> AsyncIterator[StreamEvent]:
        # Cancelling the task that iterates this generator exits the `async with`, which closes the
        # HTTP response: that is the kill switch severing the stream.
        async with self.client.messages.stream(**request) as stream:
            async for event in stream:
                et = event.type
                if et == "content_block_start":
                    cb = event.content_block
                    yield "block_start", {"index": event.index, "type": cb.type, "id": getattr(cb, "id", None),
                                          "name": getattr(cb, "name", None)}
                elif et == "content_block_delta":
                    d = event.delta
                    if d.type == "input_json_delta":
                        yield "input_json", d.partial_json
                    elif d.type == "text_delta":
                        yield "text", d.text
                elif et == "content_block_stop":
                    snap = stream.current_message_snapshot
                    if event.index < len(snap.content):
                        yield "block_complete", _block_dict(snap.content[event.index])
            msg = await stream.get_final_message()
        yield "final", {"content": [_block_dict(b) for b in msg.content], "stop_reason": msg.stop_reason}
