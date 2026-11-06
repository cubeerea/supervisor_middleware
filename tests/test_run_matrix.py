"""run_matrix: cell commands keep everything but context/supervisor fixed; the summary reads each cell's metrics."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_matrix  # noqa: E402


def _args(**kw):
    a = run_matrix.main.__globals__["argparse"].Namespace(
        name="m", episodes=4, backend="scripted", scenarios=None, grace=None, num_ctx=16384,
        openrouter_model="qwen/qwen3.8-max-0902", reasoning_effort=None, extra=[])
    vars(a).update(kw)
    return a


def test_cell_commands_isolate_the_judge():
    q = run_matrix.cell_command(_args(), "trajectory", "qwen")
    o = run_matrix.cell_command(_args(reasoning_effort="low", extra=["--pressure"]), "action", "openrouter")
    for cmd in (q, o):
        assert {"--no-command-check", "--no-stall"} <= set(cmd)
        assert cmd[cmd.index("--redis-url") + 1] == "memory://" and cmd[cmd.index("--supervisor") + 1] == "inline"
    assert q[q.index("--supervision-mode") + 1] == "trajectory" and q[q.index("--judge") + 1] == "ollama"
    assert q[q.index("--num-ctx") + 1] == "16384" and q[q.index("--name") + 1] == "m-trajectory-qwen"
    assert o[o.index("--judge") + 1] == "openrouter"
    assert o[o.index("--supervisor-model") + 1] == "qwen/qwen3.8-max-0902"
    assert o[-3:] == ["--reasoning-effort", "low", "--pressure"]


def test_summarize(tmp_path, monkeypatch):
    monkeypatch.setattr(run_matrix, "ROOT", tmp_path)
    cell = tmp_path / "results" / "m-action-openrouter"
    cell.mkdir(parents=True)
    cell.joinpath("metrics.json").write_text(json.dumps({
        "supervisor_model": "qwen/x", "episodes": 5, "tir": 0.5, "prevention_rate": 0.5,
        "false_intervention_rate": 0.1, "episodes_violation_executed": 2, "episodes_bypass_executed": 1,
        "by_scenario": {"split": {"violations": 3, "tir": 0.0, "episodes_bypass_executed": 1},
                        "split_clean": {"violations": 0, "tir": None, "false_interventions": 1,
                                        "clean_proposals": 8, "episodes_migrated": 1},
                        "clean": {"violations": 0, "tir": None}},
        "post_halt_resubmits": {"after_false_halt": 2, "resubmitted_after_false_halt": 1},
        "latency_ms": {"judge": {"p50": 812.0}, "action_gate_wait": {"p50": 830.0}},
        "verdict_timeouts": 0, "supervisor_cost_usd": 0.0123, "executor_cost_usd": None}))
    rows, table = run_matrix.summarize("m", [("action", "openrouter"), ("trajectory", "openrouter")])
    assert rows[0]["tir_by_scenario"] == {"split": 0.0} and rows[1] == {"context": "trajectory",
                                                                       "supervisor": "openrouter", "missing": True}
    assert "| action | openrouter (qwen/x) | 50% | 0% | 50% | 10% | 1/5 | 1 | 1/8 | 1/2 | 812 | $0.0123 |" in table
    assert "(not run)" in table
