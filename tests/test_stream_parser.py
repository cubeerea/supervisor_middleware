import json
import random

import pytest

from supervisor_middleware.stream_parser import IncrementalObjectScanner, ReasoningWatcher

OBJ = {"reasoning": 'He said "use raw SQL"\n\\ path {x: [1]} ✓ é', "command": "print('a, \"b\"')\n"}


@pytest.mark.parametrize("trigger", ["value_close", "next_key"])
@pytest.mark.parametrize("seed", range(40))
def test_random_fragmentation(trigger, seed):
    s = json.dumps(OBJ, ensure_ascii=seed % 2 == 0)
    rng = random.Random(seed)
    w = ReasoningWatcher(trigger=trigger)
    fired_at, i, fired = None, 0, []
    while i < len(s):
        n = rng.randint(1, 6)
        r = w.feed(s[i:i + n])
        i += n
        if r is not None:
            fired.append(r)
            fired_at = i
    assert fired == [OBJ["reasoning"]]
    assert w.scanner.values["command"] == OBJ["command"]
    # fires before the command value is complete
    assert fired_at < len(s) - len(json.dumps(OBJ["command"])) + 8


def test_value_close_fires_at_closing_quote():
    w = ReasoningWatcher(trigger="value_close")
    assert w.feed('{"reasoning": "abc') is None
    assert w.feed('"') == "abc"
    assert w.feed(', "command": "x"}') is None  # only once


def test_next_key_waits_for_command_key():
    w = ReasoningWatcher(trigger="next_key")
    assert w.feed('{"reasoning": "abc", "comm') is None
    assert w.feed('and"') == "abc"
    assert w.next_key == "command"


def test_reasoning_last_fires_on_finish():
    w = ReasoningWatcher(trigger="next_key")
    assert w.feed('{"command": "x", "reasoning": "late"}') is None
    assert w.finish() == "late"
    assert w.finish() is None


def test_nested_values_are_skipped_and_partial_tracked():
    sc = IncrementalObjectScanner()
    sc.feed('{"meta": {"reasoning": "nested"}, "n": 12, "ok": true, "reasoning": "top", "command": "pri')
    assert sc.values["reasoning"] == "top"
    assert "meta" not in sc.values
    assert sc.partial["command"] == "pri"
