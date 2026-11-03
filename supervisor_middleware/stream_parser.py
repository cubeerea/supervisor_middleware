"""Incremental parser for a streamed tool-input JSON object.

Anthropic streams tool input as `input_json_delta.partial_json` fragments, split at arbitrary byte
positions (mid-key, mid-escape). This scanner consumes fragments one at a time and reports, as soon
as it happens:

  * ``field_closed``  - a top-level string field's closing quote was read (value fully known)
  * ``key_started``   - a new top-level key was fully read (e.g. ``"command"`` after ``reasoning``)

`ReasoningWatcher` turns that into a single trigger: fire once the `reasoning` value is complete.
`trigger="value_close"` fires on the closing quote (earliest possible moment); `trigger="next_key"`
fires when the parser hits the following key (the literal rule in the project plan). Either way the
caller gets the complete, unescaped reasoning string.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ParseEvent:
    kind: Literal["field_closed", "key_started"]
    key: str
    value: object = None


@dataclass
class IncrementalObjectScanner:
    """Character-level scanner for one JSON object. Tracks only what it needs: top-level keys and
    top-level string values (nested values are skipped but bracket-balanced)."""

    _depth: int = 0
    _in_string: bool = False
    _escape: bool = False
    _expect: Literal["key", "colon", "value", "comma"] = "key"
    _buf: list[str] = field(default_factory=list)
    _current_key: str | None = None
    _string_role: Literal["key", "value", "nested"] | None = None
    values: dict[str, object] = field(default_factory=dict)
    partial: dict[str, str] = field(default_factory=dict)  # raw (still-escaped) text of open string values
    raw: list[str] = field(default_factory=list)

    def feed(self, chunk: str) -> list[ParseEvent]:
        self.raw.append(chunk)
        events: list[ParseEvent] = []
        for ch in chunk:
            ev = self._step(ch)
            if ev:
                events.append(ev)
        return events

    @property
    def text(self) -> str:
        return "".join(self.raw)

    def _step(self, ch: str) -> ParseEvent | None:
        if self._in_string:
            if self._escape:
                self._escape = False
                self._buf.append(ch)
            elif ch == "\\":
                self._escape = True
                self._buf.append(ch)
            elif ch == '"':
                self._in_string = False
                return self._close_string()
            else:
                self._buf.append(ch)
            if self._string_role == "value" and self._current_key is not None:
                self.partial[self._current_key] = "".join(self._buf)
            return None

        if ch.isspace():
            return None
        if ch == '"':
            self._in_string = True
            self._buf = []
            if self._depth == 1 and self._expect == "key":
                self._string_role = "key"
            elif self._depth == 1 and self._expect == "value":
                self._string_role = "value"
            else:
                self._string_role = "nested"
            return None
        if ch in "{[":
            self._depth += 1
            if self._depth == 1:
                self._expect = "key"
            return None
        if ch in "}]":
            self._depth -= 1
            if self._depth == 1:
                self._expect = "comma"  # a nested value just closed
            return None
        if self._depth != 1:
            return None
        if ch == ":" and self._expect == "colon":
            self._expect = "value"
        elif ch == "," and self._expect in ("comma", "value"):
            self._expect = "key"
        elif self._expect == "value":
            self._expect = "comma"  # start of a number / true / false / null: skip until ','
        return None

    def _close_string(self) -> ParseEvent | None:
        raw = "".join(self._buf)
        role = self._string_role
        self._string_role = None
        if role == "key":
            self._current_key = _decode(raw)
            self._expect = "colon"
            return ParseEvent("key_started", self._current_key)
        if role == "value" and self._current_key is not None:
            value = _decode(raw)
            self.values[self._current_key] = value
            self.partial.pop(self._current_key, None)
            self._expect = "comma"
            return ParseEvent("field_closed", self._current_key, value)
        return None


def _decode(raw: str) -> str:
    try:
        return json.loads(f'"{raw}"')
    except ValueError:
        return raw  # malformed escape from the model: keep the raw text rather than lose it


class ReasoningWatcher:
    """Feed tool-input fragments; `feed()` returns the full reasoning string exactly once, at the
    configured trigger point, and None otherwise."""

    def __init__(self, field_name: str = "reasoning", trigger: Literal["value_close", "next_key"] = "value_close"):
        self.field_name = field_name
        self.trigger = trigger
        self.scanner = IncrementalObjectScanner()
        self.fired = False
        self.reasoning: str | None = None
        self.next_key: str | None = None

    def feed(self, chunk: str) -> str | None:
        for ev in self.scanner.feed(chunk):
            if self.fired:
                continue
            if ev.kind == "field_closed" and ev.key == self.field_name:
                self.reasoning = str(ev.value)
                if self.trigger == "value_close":
                    self.fired = True
                    return self.reasoning
            elif ev.kind == "key_started" and self.reasoning is not None and ev.key != self.field_name:
                self.next_key = ev.key
                self.fired = True
                return self.reasoning
        return None

    def finish(self) -> str | None:
        """End of stream: if the trigger never fired (e.g. reasoning was the last field under
        `next_key`), fire now so the supervisor still sees it."""
        if not self.fired and self.reasoning is not None:
            self.fired = True
            return self.reasoning
        return None
