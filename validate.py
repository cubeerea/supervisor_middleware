#!/usr/bin/env python3
"""The inspector: run every check CI runs, locally, and save what failed.

    python3 validate.py              # all checks
    python3 validate.py --only python

Checks:
    python    uv run ruff check . · uv run pytest -q
    infra     docker compose config -q (skipped if docker is missing)

Writes (both git-ignored):
    .agent/validation.json   every check: ok, seconds, command, output tail
    .agent/feedback.log      only the failures, stderr first: what orchestrator.py hands back to the agent

Exit code 0 = everything passed, 1 = something failed. Standard library only.
Tests run offline: ANTHROPIC_API_KEY and the LANGFUSE_* keys are blanked, tracing is off, and the tests use
scripted LLM streams, fakeredis and an in-memory Langfuse exporter, so nothing touches the real API, Ollama,
Langfuse or Redis.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / ".agent"
TAIL_LINES = 150  # per failing check; enough for a traceback, small enough for a prompt
TIMEOUT_S = 900


@dataclass
class Check:
    name: str
    lang: str
    cmd: list[str]
    cwd: str = "."


CHECKS = [
    Check("ruff", "python", ["uv", "run", "ruff", "check", "."]),
    Check("pytest", "python", ["uv", "run", "pytest", "-q"]),
    Check("compose config", "infra", ["docker", "compose", "config", "-q"]),
]
LANGS = ("python", "infra")


def tail(text: str, n: int = TAIL_LINES) -> str:
    lines = text.rstrip().splitlines()
    cut = len(lines) - n
    return ("\n".join([f"... ({cut} earlier lines cut)"] + lines[-n:])) if cut > 0 else "\n".join(lines)


def run(check: Check) -> dict:
    env = {**os.environ, "ANTHROPIC_API_KEY": "", "LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": "",
           "LANGFUSE_TRACING": "false", "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false", "CI": "1"}  # offline, like CI; never the real key
    start = time.monotonic()
    try:
        r = subprocess.run(check.cmd, check=False, cwd=ROOT / check.cwd, env=env, capture_output=True, text=True,
                           timeout=TIMEOUT_S)
        code, out, err = r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        code, out, err = 127, "", f"{check.cmd[0]}: command not found (install it or fix PATH)"
    except subprocess.TimeoutExpired as e:
        code = 124
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = f"timed out after {TIMEOUT_S}s"
    return {
        "name": check.name, "lang": check.lang, "ok": code == 0, "exit_code": code,
        "seconds": round(time.monotonic() - start, 1),
        "command": " ".join(check.cmd) + (f"   (in {check.cwd}/)" if check.cwd != "." else ""),
        # stderr first: compiler and tool errors land there; pytest and ruff report on stdout
        "stderr_tail": tail(err), "stdout_tail": tail(out),
    }


def feedback(results: list[dict]) -> str:
    parts = []
    for r in results:
        if r["ok"]:
            continue
        parts.append(f"=== FAILED [{r['lang']}] {r['name']}  (exit {r['exit_code']})\n$ {r['command']}")
        if r["stderr_tail"]:
            parts.append(f"--- stderr\n{r['stderr_tail']}")
        if r["stdout_tail"]:
            parts.append(f"--- stdout\n{r['stdout_tail']}")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--all", action="store_true", help="accepted for compatibility; every check always runs")
    ap.add_argument("--only", help=f"comma-separated subset of {','.join(LANGS)}")
    a = ap.parse_args()

    langs = set(a.only.split(",")) if a.only else set(LANGS)
    if unknown := langs - set(LANGS):
        ap.error(f"unknown: {','.join(sorted(unknown))}")
    if shutil.which("docker") is None:
        langs.discard("infra")

    results = []
    for check in CHECKS:
        if check.lang not in langs:
            continue
        print(f"→ {check.name} …", end=" ", flush=True)
        r = run(check)
        print(("ok" if r["ok"] else "FAILED") + f" ({r['seconds']}s)", flush=True)
        results.append(r)

    passed = all(r["ok"] for r in results)
    OUT.mkdir(exist_ok=True)
    report = {"passed": passed, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "skipped": sorted(set(LANGS) - langs),
              "checks": results}
    (OUT / "validation.json").write_text(json.dumps(report, indent=2))
    (OUT / "feedback.log").write_text(feedback(results))
    if report["skipped"]:
        print(f"skipped: {', '.join(report['skipped'])}")
    print("PASSED" if passed else f"FAILED, details in {OUT / 'feedback.log'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
