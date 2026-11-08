#!/usr/bin/env python3
"""Supervisor matrix: what the supervisor sees (context), for a hosted supervisor model.

    context     reasoning   rationale + last observation, published when `reasoning` closes; may cut the stream
                action      rationale + finished command; the tool gate blocks until the verdict
                trajectory  action + every earlier step of the episode
    supervisor  openrouter  one hosted model (--openrouter-model); the default
                qwen        local Ollama cascade (qwen2.5:3b screen -> qwen2.5-coder:7b confirm), opt-in with
                            --supervisors qwen,openrouter. Off by default: the v1/v2 runs already cover it

Every cell runs the same executor and scenarios with the command check and the stall detector OFF, so the numbers
measure the judge alone. Each cell is a `run_experiment.py` subprocess (inline daemon, in-memory Redis); results land
in results/<name>-<context>-<supervisor>/ and the comparison in results/<name>-matrix.{json,md}.

    uv run --env-file .env python run_matrix.py --name m1 --episodes 10 --openrouter-model qwen/qwen3.8-max-0902
    uv run --env-file .env python run_matrix.py --name m1 --backend anthropic --episodes 4 --resume
    uv run python run_matrix.py --name dry --dry-run          # print the commands only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTEXTS = ("reasoning", "action", "trajectory")
SUPERVISORS = ("qwen", "openrouter")
DEFAULT_SUPERVISORS = ("openrouter",)


def cell_name(name: str, context: str, supervisor: str) -> str:
    return f"{name}-{context}-{supervisor}"


def cell_command(a: argparse.Namespace, context: str, supervisor: str) -> list[str]:
    cmd = [sys.executable, str(ROOT / "run_experiment.py"), "--name", cell_name(a.name, context, supervisor),
           "--episodes", str(a.episodes), "--supervision-mode", context, "--supervisor", "inline",
           "--redis-url", "memory://", "--no-command-check", "--no-stall"]
    if a.backend:
        cmd += ["--backend", a.backend]
    if a.scenarios:
        cmd += ["--scenarios", a.scenarios]
    # one verdict budget for every cell: the reasoning cells' late-halt grace equals the action gate's timeout, so
    # cells differ only in what the judge sees, not in how long it may take (hosted verdicts reached 130 s at p99)
    cmd += ["--grace", str(a.grace if a.grace is not None else a.verdict_budget)]
    if supervisor == "qwen":
        # same context window in every qwen cell, so only what the judge is shown differs
        cmd += ["--judge", "ollama", "--judge-mode", "cascade", "--num-ctx", str(a.num_ctx)]
    else:
        cmd += ["--judge", "openrouter", "--supervisor-model", a.openrouter_model]
        if a.reasoning_effort is not None:
            cmd += ["--reasoning-effort", a.reasoning_effort]
    return cmd + a.extra


def _f(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def summarize(name: str, cells: list[tuple[str, str]]) -> tuple[list[dict], str]:
    rows = []
    for ctx, sup in cells:
        path = ROOT / "results" / cell_name(name, ctx, sup) / "metrics.json"
        if not path.exists():
            rows.append({"context": ctx, "supervisor": sup, "missing": True})
            continue
        m = json.loads(path.read_text())
        sc = m.get("by_scenario", {})
        rows.append({
            "context": ctx, "supervisor": sup, "supervisor_model": m.get("supervisor_model"),
            "episodes": m["episodes"], "tir": m.get("tir_raw_sql", m["tir"]), "prevention_rate": m["prevention_rate"],
            "premature_finishes": m.get("premature_finishes") or {},
            "false_intervention_rate": m["false_intervention_rate"],
            "episodes_violation_executed": m.get("episodes_violation_executed"),
            "episodes_bypass_executed": m.get("episodes_bypass_executed"),
            "episodes_migrated": m.get("episodes_migrated_cleanly"),
            "tir_by_scenario": {k: v["tir"] for k, v in sc.items() if v["violations"]},
            "split_episodes_bypassed": (sc.get("split") or {}).get("episodes_bypass_executed"),
            "split_clean_false_halts": (sc.get("split_clean") or {}).get("false_interventions"),
            "split_clean_proposals": (sc.get("split_clean") or {}).get("clean_proposals"),
            "split_clean_migrated": (sc.get("split_clean") or {}).get("episodes_migrated"),
            "resubmits_after_false_halt": (m.get("post_halt_resubmits") or {}).get("resubmitted_after_false_halt"),
            "false_halts_followed": (m.get("post_halt_resubmits") or {}).get("after_false_halt"),
            "judge_p50_ms": m["latency_ms"]["judge"]["p50"],
            "gate_wait_p50_ms": (m["latency_ms"].get("action_gate_wait") or {}).get("p50"),
            "verdict_timeouts": m.get("verdict_timeouts", 0),
            "supervisor_cost_usd": m.get("supervisor_cost_usd"),
            "executor_cost_usd": m.get("executor_cost_usd"),
        })
    scen = sorted({k for r in rows for k in r.get("tir_by_scenario", {})})
    head = ["context", "supervisor", "TIR (raw SQL)", "premature FINISH halted", *[f"TIR {s}" for s in scen],
            "prevented", "false halts", "migrated",
            "episodes raw SQL hit DB", "split: hit DB", "split_clean: false halts", "resubmit after false halt",
            "judge p50 ms", "sup. cost"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        if r.get("missing"):
            lines.append(f"| {r['context']} | {r['supervisor']} | (not run) |" + " |" * (len(head) - 3))
            continue
        ts = r["tir_by_scenario"]
        lines.append("| " + " | ".join([
            r["context"], f"{r['supervisor']} ({r['supervisor_model']})", _f(r["tir"]),
            f"{r['premature_finishes'].get('halted', 0)}/{r['premature_finishes'].get('proposed', 0)}",
            *[_f(ts.get(s)) for s in scen], _f(r["prevention_rate"]), _f(r["false_intervention_rate"]),
            "n/a" if r["episodes_migrated"] is None else f"{r['episodes_migrated']}/{r['episodes']}",
            f"{r['episodes_bypass_executed']}/{r['episodes']}", str(r["split_episodes_bypassed"]),
            "n/a" if r["split_clean_proposals"] is None else
            f"{r['split_clean_false_halts']}/{r['split_clean_proposals']}",
            f"{r['resubmits_after_false_halt']}/{r['false_halts_followed']}",
            "n/a" if r["judge_p50_ms"] is None else f"{r['judge_p50_ms']:.0f}",
            "n/a" if r["supervisor_cost_usd"] is None else f"${r['supervisor_cost_usd']:.4f}",
        ]) + " |")
    return rows, "\n".join(lines)


def rescore(cell: str) -> bool:
    """Recompute a finished cell's metrics with the current metrics code, keeping run metadata (cost, models)."""
    from supervisor_middleware.metrics import compute
    from supervisor_middleware.telemetry import read_events

    d = ROOT / "results" / cell
    if not (d / "metrics.json").exists():
        return False
    old = json.loads((d / "metrics.json").read_text())
    threads = {e["thread_id"] for e in json.loads((d / "episodes.json").read_text())}
    old.update(compute(read_events(d / "events.jsonl"), threads))
    (d / "metrics.json").write_text(json.dumps(old, indent=2))
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", default=time.strftime("matrix-%Y%m%d-%H%M%S"))
    ap.add_argument("--episodes", type=int, default=10, help="per cell (scripted: cycles through the scenarios)")
    ap.add_argument("--backend", choices=["scripted", "anthropic"], default="scripted")
    ap.add_argument("--scenarios", help="comma list for the scripted backend (default: all)")
    ap.add_argument("--contexts", default=",".join(CONTEXTS))
    ap.add_argument("--supervisors", default=",".join(DEFAULT_SUPERVISORS),
                    help=f"comma list of {','.join(SUPERVISORS)} (default: openrouter only)")
    ap.add_argument("--openrouter-model", default="qwen/qwen3.8-max-0902")
    ap.add_argument("--reasoning-effort", default=None, help="OpenRouter reasoning effort ('' = provider default)")
    ap.add_argument("--num-ctx", type=int, default=16384, help="Ollama context window for the qwen cells")
    ap.add_argument("--verdict-budget", type=float, default=200.0,
                    help="seconds a verdict may take in every cell: action-gate timeout, judge HTTP timeout and the "
                         "reasoning cells' late-halt grace (default 200)")
    ap.add_argument("--grace", type=float, help="override the reasoning cells' grace only (default: --verdict-budget)")
    ap.add_argument("--resume", action="store_true", help="skip cells that already have metrics.json")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute every finished cell's metrics from its events.jsonl (no runs, no API cost)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="after --: passed to every run_experiment.py")
    a = ap.parse_args(argv)
    a.extra = a.extra[1:] if a.extra[:1] == ["--"] else a.extra
    contexts = [c for c in a.contexts.split(",") if c]
    sups = [s for s in a.supervisors.split(",") if s]
    bad = [c for c in contexts if c not in CONTEXTS] + [s for s in sups if s not in SUPERVISORS]
    if bad:
        raise SystemExit(f"unknown context/supervisor: {bad}")
    cells = [(c, s) for s in sups for c in contexts]
    if a.rescore:
        for ctx, sup in cells:
            print(f"rescored {cell_name(a.name, ctx, sup)}" if rescore(cell_name(a.name, ctx, sup))
                  else f"skipped {cell_name(a.name, ctx, sup)} (not run)")
        a.dry_run = False
        cells_to_run: list[tuple[str, str]] = []
    else:
        cells_to_run = cells
    failed = []
    for ctx, sup in cells_to_run:
        cmd = cell_command(a, ctx, sup)
        done = (ROOT / "results" / cell_name(a.name, ctx, sup) / "metrics.json").exists()
        print(f"\n=== {ctx} x {sup}{' (done, skipped)' if done and a.resume else ''}\n$ {' '.join(cmd)}", flush=True)
        if a.dry_run or (done and a.resume):
            continue
        env = {**os.environ, "ACTION_VERDICT_TIMEOUT_S": str(a.verdict_budget),
               "SUPERVISOR_TIMEOUT_S": str(max(a.verdict_budget - 5, 30))}
        if subprocess.run(cmd, cwd=ROOT, env=env).returncode != 0:  # noqa: S603 - our own script, fixed argv
            failed.append((ctx, sup))
    if a.dry_run:
        return 0
    rows, table = summarize(a.name, cells)
    out = ROOT / "results"
    (out / f"{a.name}-matrix.json").write_text(json.dumps(rows, indent=2))
    (out / f"{a.name}-matrix.md").write_text(table + "\n")
    print("\n" + table)
    if failed:
        print(f"\nfailed cells: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
