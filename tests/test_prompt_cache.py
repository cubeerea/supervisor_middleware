"""Executor request carries prompt-cache breakpoints; the cost/hit-rate helpers do the arithmetic right."""
from __future__ import annotations

from types import SimpleNamespace

from run_experiment import cache_hit_rate, estimate_cost
from supervisor_middleware.config import Settings
from supervisor_middleware.executor.graph import _request


def _deps(tmp_path, **kw):
    (tmp_path / "ORM_DOCS.md").write_text("# docs")
    return SimpleNamespace(settings=Settings(executor_model="claude-sonnet-5-5", **kw), workdir=tmp_path)


def _state():
    return {"messages": [{"role": "user", "content": "migrate"}]}


def test_request_has_system_and_automatic_breakpoints(tmp_path):
    req = _request(_deps(tmp_path), _state())
    (block,) = req["system"]
    assert block["type"] == "text" and "# docs" in block["text"]
    assert block["cache_control"] == {"type": "ephemeral"}
    assert req["cache_control"] == {"type": "ephemeral"}


def test_prompt_cache_can_be_disabled(tmp_path):
    req = _request(_deps(tmp_path, executor_prompt_cache=False), _state())
    assert isinstance(req["system"], str) and "cache_control" not in req


def test_history_is_not_copied_or_edited(tmp_path):
    state = _state()
    assert _request(_deps(tmp_path), state)["messages"] is state["messages"]


def test_cache_hit_rate_and_cost():
    tok = {"input_tokens": 1_000, "output_tokens": 100_000, "cache_read_input_tokens": 8_000,
           "cache_creation_input_tokens": 1_000}
    assert cache_hit_rate(tok) == 0.8
    assert cache_hit_rate({}) == 0.0
    # 1k*2 + 100k*10 + 8k*0.2 + 1k*2.5 = 2000 + 1_000_000 + 1600 + 2500 per 1e6
    assert abs(estimate_cost("claude-sonnet-5-5", tok) - 1.0061) < 1e-9
    assert estimate_cost("unknown-model", tok) is None
