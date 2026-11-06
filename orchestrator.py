#!/usr/bin/env python3
"""The driver: an implementor agent does one task, validate.py and an evaluator agent judge it, repeat.

    python3 orchestrator.py new "Add per-app persona rules" --details "..." [--max-iterations 8]
    python3 orchestrator.py run            # work the task until it passes, blocks or hits max iterations
    python3 orchestrator.py run --watch    # always-on: after that, wait for the next `new` task
    python3 orchestrator.py run --pr       # on pass: push the branch and open a PR (never merges)
    python3 orchestrator.py run --watch --pr --routine [--routine-every 6]
                                           # also queue routine work (validation, tests, bugs) when idle
    python3 orchestrator.py status | stop

Loop, per iteration:
    1. read progress.json → build the execution context (task, iteration, last feedback, git state, rules)
    2. IMPLEMENTOR:  claude -p <context> --output-format json  (permissions: see AGENT_ARGS)
    3. guard: the agent must not touch the inspector, this driver, CI or secrets
    4. run validate.py; failures (stderr first) become `last_feedback` for the next iteration
    5. EVALUATOR (read-only, sees only the task and the diff, not the implementor's own account):
       `VERDICT: APPROVE` → "passed"; `VERDICT: REJECT` + reasons → `last_feedback`, next iteration
    6. otherwise loop until max_iterations → "blocked"
An implementor that finds nothing worth changing (routine work) answers `NOTHING TO DO: ...`; that
ends the task as passed with no commits and no PR.

Routines (`--routine`, needs `--watch`): whenever the loop is idle and `--routine-every` hours have
passed since the last one, the next kind in ROUTINE_ORDER is queued on its own `agent/routine-*`
branch. None is queued while ROUTINE_MAX_OPEN_PRS routine PRs are still open (a human merges them).

State lives in progress.json (git-ignored, created by `new`; replaced tasks are archived to
.agent/history/). Logs: .agent/logs/. To stop between steps: `python3 orchestrator.py stop`
(or create .agent/STOP). Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROGRESS = ROOT / "progress.json"
AGENT_DIR = ROOT / ".agent"
LOGS = AGENT_DIR / "logs"
ARCHIVE = AGENT_DIR / "history"
ROUTINE_STATE = AGENT_DIR / "routine.json"
STOP_FILE = AGENT_DIR / "STOP"

AGENT_TIMEOUT_S = 45 * 60
EVAL_TIMEOUT_S = 15 * 60
WATCH_POLL_S = 30
FEEDBACK_MAX_CHARS = 12_000  # keep the injected log from swamping the prompt
EVAL_DIFF_MAX_CHARS = 40_000  # the evaluator reads the rest from disk if it needs to
MAX_AGENT_ERRORS = 3  # consecutive crashes/timeouts (rate limits, auth) before giving up
BASE = os.environ.get("AGENT_BASE", "origin/main")  # task branches are cut from this
ACTIVE = ("pending", "running", "validating", "evaluating", "failed")  # a task still being worked

# Headless agent. acceptEdits + an allowlist: it can edit files and run the checks and local git,
# nothing else runs without a prompt (and -p can't prompt, so it is refused). No push/PR/merge:
# the orchestrator does that only after validation passes.
SECRETS_DENY = ["Read(./.env)", "Read(./.env.*)", "Read(./results/**)", "Read(./**/*.pem)"]
AGENT_ARGS = [
    "--output-format", "json",
    "--permission-mode", "acceptEdits",
    "--allowedTools",
    "Read", "Edit", "Write", "Glob", "Grep",
    "Bash(python3 validate.py*)",
    "Bash(uv run ruff *)", "Bash(uv run pytest *)", "Bash(uv run python *)", "Bash(uv sync*)",
    "Bash(docker compose config*)",
    "Bash(git status*)", "Bash(git diff*)", "Bash(git log*)", "Bash(git show*)",
    "Bash(git add *)", "Bash(git commit *)", "Bash(git restore *)",
    "Bash(ls *)", "Bash(cat *)", "Bash(grep *)", "Bash(rg *)",
    "--disallowedTools",
    "Bash(git push*)", "Bash(gh *)", "Bash(git reset --hard*)", "Bash(git checkout*)", "Bash(git switch*)",
    "Bash(rm -rf*)", "Bash(curl *)", "Bash(security *)",
    *SECRETS_DENY,
    "Edit(./validate.py)", "Edit(./orchestrator.py)", "Edit(./progress.json)", "Edit(./.github/**)",
    "Write(./validate.py)", "Write(./orchestrator.py)", "Write(./progress.json)", "Write(./.github/**)",
]

# Evaluator: read-only. It may read code, inspect git and run tests, never edit or commit.
EVAL_ARGS = [
    "--output-format", "json",
    "--permission-mode", "default",
    "--allowedTools",
    "Read", "Glob", "Grep",
    "Bash(git status*)", "Bash(git diff*)", "Bash(git log*)", "Bash(git show*)",
    "Bash(uv run pytest *)", "Bash(ls *)", "Bash(cat *)", "Bash(grep *)", "Bash(rg *)",
    "--disallowedTools",
    "Edit", "Write", "NotebookEdit",
    "Bash(git add*)", "Bash(git commit*)", "Bash(git restore*)", "Bash(git reset*)", "Bash(git stash*)",
    "Bash(git push*)", "Bash(gh *)", "Bash(git checkout*)", "Bash(git switch*)",
    "Bash(rm *)", "Bash(curl *)", "Bash(security *)",
    *SECRETS_DENY,
]

# The agent may not change how it is judged, nor touch secrets. Checked after every agent run.
PROTECTED = ("validate.py", "orchestrator.py", ".github/", "sandbox/ORM_DOCS.md")
FORBIDDEN = (".env", "results/")

NOOP_MARK = "NOTHING TO DO:"
VERDICT_RE = re.compile(r"^\W*VERDICT:\s*(APPROVE|REJECT)\b", re.IGNORECASE | re.MULTILINE)

TEMPLATE = {
    "task": "",
    "details": "",
    "branch": "",
    "status": "idle",  # idle | pending | running | validating | evaluating | failed | passed | blocked | stopped
    "iteration": 0,
    "max_iterations": 8,
    "budget_usd_per_iteration": 5.0,
    "evaluate": True,  # run the evaluator agent after validation passes
    "eval_budget_usd": 2.0,
    "routine": "",  # routine kind when queued by --routine, else ""
    "last_feedback": "",
    "pr_url": "",
    "total_cost_usd": 0.0,
    "history": [],  # one entry per iteration: {iteration, result, seconds, cost_usd, at}
    "updated_at": "",
}

# ---------------------------------------------------------------- routines

ROUTINE_ORDER = ("validation", "tests", "bugs")
ROUTINE_MAX_OPEN_PRS = 2  # don't pile up unreviewed routine PRs
ROUTINE_PREFIX = "agent/routine-"
ROUTINE_LIMITS = {"max_iterations": 4, "budget_usd_per_iteration": 3.0, "eval_budget_usd": 1.5}
ROUTINES = {
    "validation": (
        "Routine: validation sweep",
        (
            "Run `python3 validate.py --all`. Fix every failure at its root cause. "
            "Then read the ruff and pytest output for warnings that come from THIS repo's code "
            "(DeprecationWarning, ResourceWarning, never-awaited coroutines, unused-variable or type warnings) and fix those; ignore warnings raised inside third-party packages. "
            f"If everything passes and no warning comes from our code, change nothing and start your final "
            f"message with `{NOOP_MARK} <what you checked>`."
        ),
    ),
    "tests": (
        "Routine: strengthen tests",
        (
            "Find ONE important behaviour in supervisor_middleware/ or sandbox/ that no test covers: the "
            "stream parser, Redis bus, interception hook, supervisor daemon, executor graph or metrics code whose "
            "behaviour is never asserted in tests/. Look at `git log --oneline -20` and prefer an area recent routine commits did not touch. "
            "Add focused tests that assert real behaviour and would fail if it broke. Tests run offline: scripted LLM "
            "streams, fakeredis and temp SQLite (see tests/conftest.py); never the real Anthropic API, Ollama, "
            "Langfuse or a live Redis. If a new test exposes a bug, fix it minimally and say so in the commit message. Do not "
            "rewrite existing tests unless they are wrong. Keep the diff small: one area per run. "
            f"If every important behaviour you checked is already covered, start your final message with "
            f"`{NOOP_MARK} <areas checked>`."
        ),
    ),
    "bugs": (
        "Routine: hunt for one real bug",
        (
            "Find ONE real, user-visible defect in supervisor_middleware/ or sandbox/: a crash, wrong result, data loss, race, "
            "resource leak or security issue. Look at `git log --oneline -20` and pick an area recent routine "
            "commits did not touch. Prove it before fixing: write a test that fails because of the bug, then fix "
            "it with the smallest change and keep the test. No style refactors, no speculative "
            "hardening, no 'might be a problem' fixes without a reproduction. If you cannot find a bug you can "
            f"reproduce, change nothing and start your final message with `{NOOP_MARK} <areas checked>`."
        ),
    ),
}


# ---------------------------------------------------------------- state

def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def load() -> dict:
    if not PROGRESS.exists():
        return dict(TEMPLATE)
    return {**TEMPLATE, **json.loads(PROGRESS.read_text())}


def save(p: dict, **changes) -> dict:
    p.update(changes, updated_at=now())
    tmp = PROGRESS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(p, indent=2) + "\n")
    tmp.replace(PROGRESS)  # atomic: `status` in another pane never reads half a file
    return p


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def queue_task(task: str, details: str = "", branch: str = "", **settings) -> dict:
    """Replace progress.json with a new pending task; the old one (if any) is archived first."""
    old = load()
    if old["status"] in ("running", "validating", "evaluating"):
        raise RuntimeError(f"a task is {old['status']}: {old['task']!r}. `stop` it first.")
    if old["task"]:
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        stamp = re.sub(r"[^0-9T]", "", old["updated_at"])[:15] or time.strftime("%Y%m%dT%H%M%S")
        (ARCHIVE / f"{stamp}-{slug(old['task'])}.json").write_text(json.dumps(old, indent=2) + "\n")
    p = dict(TEMPLATE, history=[])
    save(p, task=task, details=details, branch=branch or f"agent/{slug(task)}", status="pending", **settings)
    STOP_FILE.unlink(missing_ok=True)
    return p


# ---------------------------------------------------------------- git

def git(*args: str, check: bool = False) -> str:
    r = subprocess.run(["git", *args], check=False, cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "task"


def ensure_branch(p: dict) -> None:
    """Work on the task's own branch, cut from BASE (origin/main). Never on main."""
    branch = p["branch"]
    if branch in ("main", "master", ""):
        raise RuntimeError(f"refusing to work on branch {branch!r}")
    if git("branch", "--show-current") == branch:
        return
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("working tree has uncommitted changes; commit or stash them first")
    if git("rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"):
        git("switch", branch, check=True)
    else:
        if BASE.startswith("origin/"):
            git("fetch", "origin", BASE.removeprefix("origin/"), check=True)
        git("switch", "-c", branch, BASE, check=True)  # works inside a worktree, unlike `switch main`
    log(f"on branch {branch}")


def base_commit() -> str:
    return git("merge-base", "HEAD", BASE) or "HEAD"


def commits_ahead() -> int:
    out = git("rev-list", "--count", f"{base_commit()}..HEAD")
    return int(out) if out.isdigit() else 0


def changed_paths() -> list[str]:
    """Everything that differs from BASE: commits on the branch plus the working tree."""
    paths = set(git("diff", "--name-only", base_commit()).splitlines())
    paths |= {line[3:].split(" -> ")[-1] for line in git("status", "--porcelain").splitlines()}
    return sorted(x for x in paths if x)


def guard_violations() -> list[str]:
    out = []
    for path in changed_paths():
        if any(path == x or path.startswith(x) for x in PROTECTED):
            out.append(f"{path} is protected (it decides whether your work passes); revert your change to it")
        if any(path == x or path.startswith(x) for x in FORBIDDEN):
            out.append(f"{path} holds user data or secrets; remove it from your changes and never stage it")
    return out


# ---------------------------------------------------------------- context

def context(p: dict) -> str:
    """The execution context handed to the implementor each iteration."""
    feedback = p["last_feedback"].strip()
    if len(feedback) > FEEDBACK_MAX_CHARS:
        feedback = "... (earlier output cut)\n" + feedback[-FEEDBACK_MAX_CHARS:]
    recent = [f"  #{h['iteration']}: {h['result']}" for h in p["history"][-5:]]
    return "\n".join([
        "You are the IMPLEMENTOR, an unattended coding agent working in the supervisor_middleware repo. No human will",
        "answer questions during this run. Follow CLAUDE.md, except: do NOT push, open PRs or merge (the",
        "orchestrator does). After validate.py passes, an independent EVALUATOR agent reviews your diff",
        "against the task; if it rejects the work, its reasons arrive below as feedback.",
        "",
        f"TASK: {p['task']}",
        f"DETAILS: {p['details'] or '(none)'}",
        f"ITERATION: {p['iteration']} of {p['max_iterations']}",
        f"BRANCH: {p['branch']} (already checked out; stay on it)",
        "",
        "GIT STATE:",
        git("status", "--short") or "  (clean)",
        "Commits on this branch:",
        git("log", "--oneline", f"{BASE}..HEAD") or "  (none yet)",
        "",
        "PREVIOUS ITERATIONS:" if recent else "PREVIOUS ITERATIONS: none, this is the first attempt.",
        *recent,
        "",
        "LAST FEEDBACK (fix all of this first; it is the real output of validate.py or the evaluator):"
        if feedback else "LAST FEEDBACK: none.",
        feedback,
        "",
        "RULES:",
        "1. Make the smallest change that completes the task. Add or update tests for it.",
        "2. Before finishing, run `python3 validate.py` and make it pass.",
        "3. Commit with `git add <specific paths>` then `git commit -m ...` (never `git add -A`).",
        "4. Never edit validate.py, orchestrator.py, progress.json, .github/ or sandbox/ORM_DOCS.md. Never",
        "   read .env or results/, and never call the real Anthropic API, Langfuse or a live Redis from tests.",
        "5. Do not weaken, skip or delete tests to make validation pass.",
        "6. If the task is impossible, unsafe or too ambiguous to do without a human, stop and make the",
        "   first line of your final message `BLOCKED: <one-sentence reason>`.",
        "7. If the task asks you to look for problems and you find none worth changing, change nothing and",
        f"   make the first line of your final message `{NOOP_MARK} <what you checked>`.",
        "8. End with a 1-3 line summary of what you changed.",
    ])


def eval_context(p: dict) -> str:
    """What the evaluator sees: the task and the diff only, never the implementor's own account."""
    base = base_commit()
    diff = git("diff", f"{base}..HEAD")
    if len(diff) > EVAL_DIFF_MAX_CHARS:
        diff = diff[:EVAL_DIFF_MAX_CHARS] + "\n... (diff cut; read the files or `git diff` for the rest)"
    return "\n".join([
        "You are the EVALUATOR in an implementor/evaluator loop for the supervisor_middleware repo. Another agent made the",
        "change below for this task. Automated checks (ruff, pytest, docker compose config)",
        "already PASSED. Your job is an independent review: should this change be opened as a pull request?",
        "You cannot edit files or commit; do not try. Read surrounding code and run tests if you need to.",
        "",
        f"TASK: {p['task']}",
        f"DETAILS: {p['details'] or '(none)'}",
        "",
        f"COMMITS (base {base[:9]}):",
        git("log", "--format=%h %s%n%b", f"{base}..HEAD") or "(none)",
        "",
        f"DIFF (`git diff {base[:9]}..HEAD`):",
        diff or "(empty)",
        "",
        "CHECK:",
        "1. Does the change actually accomplish the task and its details? Is anything required missing?",
        "2. Correctness: bugs, unhandled edge cases, broken callers, regressions.",
        "3. Tests: do new or changed tests assert the real behaviour (would they fail without the change)?",
        "   Were any tests weakened, skipped or deleted?",
        "4. Scope: unrelated edits, needless churn, new dependencies, secrets or data/ files.",
        "Approve sound work. Do not reject for style preferences or for things outside the task.",
        "",
        "REPLY: the first line is exactly `VERDICT: APPROVE` or `VERDICT: REJECT`. For REJECT, follow with a",
        "numbered list of concrete problems (file:line, what is wrong, what to change). Under 300 words.",
    ])


def parse_verdict(text: str) -> tuple[bool | None, str]:
    """(approved, reasons) from the evaluator's reply; approved is None when there is no verdict."""
    m = VERDICT_RE.search(text)
    if not m:
        return None, text.strip()
    return m.group(1).upper() == "APPROVE", text[m.end():].strip()


# ---------------------------------------------------------------- steps

def run_claude(prompt: str, args: list[str], budget: float, label: str, timeout: int) -> dict:
    """Run one headless Claude turn. Returns {ok, result, cost_usd, error}."""
    base = shlex.split(os.environ.get("AGENT_CMD", "claude -p"))  # override: tests / other agents
    cmd = [*base, prompt, *args]
    if budget:
        cmd += ["--max-budget-usd", str(budget)]
    LOGS.mkdir(parents=True, exist_ok=True)
    stem = LOGS / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}"
    stem.with_suffix(".prompt.txt").write_text(prompt)
    try:
        r = subprocess.run(cmd, check=False, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "result": "", "cost_usd": 0.0, "error": f"{label} timed out after {timeout}s"}
    except FileNotFoundError:
        return {"ok": False, "result": "", "cost_usd": 0.0, "error": f"{base[0]}: command not found"}
    stem.with_suffix(".out.json").write_text(r.stdout)
    if r.stderr:
        stem.with_suffix(".err.txt").write_text(r.stderr)
    try:
        out = json.loads(r.stdout)
    except json.JSONDecodeError:
        out = {"result": r.stdout, "is_error": r.returncode != 0}
    ok = r.returncode == 0 and not out.get("is_error")
    return {"ok": ok, "result": str(out.get("result") or ""), "cost_usd": float(out.get("total_cost_usd") or 0),
            "error": "" if ok else (r.stderr.strip()[-2000:] or str(out.get("result"))[-2000:] or f"exit {r.returncode}")}


def run_agent(p: dict, prompt: str) -> dict:
    """The implementor."""
    return run_claude(prompt, AGENT_ARGS, p.get("budget_usd_per_iteration") or 0, f"iter{p['iteration']}-impl",
                      AGENT_TIMEOUT_S)


def run_evaluator(p: dict) -> dict:
    """The evaluator. Adds `approved` (True/False/None) and `reasons`. Must leave the tree untouched."""
    before = (git("rev-parse", "HEAD"), git("status", "--porcelain"))
    a = run_claude(eval_context(p), EVAL_ARGS, p.get("eval_budget_usd") or 0, f"iter{p['iteration']}-eval",
                   EVAL_TIMEOUT_S)
    if (git("rev-parse", "HEAD"), git("status", "--porcelain")) != before:
        raise RuntimeError("the evaluator changed the working tree or HEAD; inspect the branch by hand")
    a["approved"], a["reasons"] = parse_verdict(a["result"]) if a["ok"] else (None, "")
    return a


def run_validate() -> tuple[bool, str]:
    r = subprocess.run([sys.executable, str(ROOT / "validate.py")], check=False, cwd=ROOT, text=True)
    fb = AGENT_DIR / "feedback.log"
    return r.returncode == 0, fb.read_text() if fb.exists() else ""


def open_pr(p: dict) -> None:
    git("push", "-u", "origin", p["branch"], check=True)
    r = subprocess.run(["gh", "pr", "create", "--fill", "--head", p["branch"]], check=False, cwd=ROOT, capture_output=True,
                       text=True)
    url = (r.stdout.strip().splitlines() or [""])[-1] if r.returncode == 0 else ""
    save(p, pr_url=url)
    log(f"PR: {url}" if url else f"PR not created: {r.stderr.strip()}")


def stop_requested() -> bool:
    return STOP_FILE.exists()


def record(p: dict, result: str, started: float, cost: float = 0.0) -> None:
    p["history"].append({"iteration": p["iteration"], "result": result, "seconds": round(time.monotonic() - started),
                         "cost_usd": round(cost, 4), "at": now()})
    p["total_cost_usd"] = round(p["total_cost_usd"] + cost, 4)


def work(p: dict, open_pr_on_pass: bool) -> None:
    """Iterate on the current task until it passes, blocks, stops or runs out of iterations."""
    ensure_branch(p)
    errors = {"implementor": 0, "evaluator": 0}  # consecutive, per role

    def crashed(what: str, error: str, started: float, cost: float) -> bool:
        """Record a crash; True when it is time to give up."""
        errors[what] += 1
        record(p, f"{what} error: {error[:200]}", started, cost)
        save(p, status="failed", last_feedback=f"The previous {what} run crashed: {error}")
        log(f"{what} error ({errors[what]}/{MAX_AGENT_ERRORS}): {error[:300]}")
        if errors[what] >= MAX_AGENT_ERRORS:
            save(p, status="blocked")
            log(f"blocked: {what} keeps failing (auth? rate limit? budget?)")
            return True
        time.sleep(60 * errors[what])  # back off before retrying
        return False

    while True:
        if stop_requested():
            save(p, status="stopped")
            log("stopped (STOP file)")
            return
        if p["iteration"] >= p["max_iterations"]:
            save(p, status="blocked", last_feedback=f"Hit max_iterations ({p['max_iterations']}). "
                                                    + p["last_feedback"][-2000:])
            log("blocked: max iterations reached; see last_feedback")
            return

        save(p, iteration=p["iteration"] + 1, status="running")
        started = time.monotonic()
        log(f"iteration {p['iteration']}/{p['max_iterations']}: implementor working…")
        a = run_agent(p, context(p))

        if not a["ok"]:
            if crashed("implementor", a["error"], started, a["cost_usd"]):
                return
            continue
        errors["implementor"] = 0
        first_line = (a["result"].strip().splitlines() or [""])[0]

        if first_line.startswith("BLOCKED:"):
            record(p, first_line, started, a["cost_usd"])
            save(p, status="blocked", last_feedback=first_line)
            log(first_line)
            return

        if violations := guard_violations():
            record(p, "guard: touched protected files", started, a["cost_usd"])
            save(p, status="failed", last_feedback="GUARD FAILED, nothing was validated:\n- " + "\n- ".join(violations))
            log("guard failed: " + "; ".join(violations))
            continue

        if first_line.startswith(NOOP_MARK) and not commits_ahead() and not git("status", "--porcelain"):
            record(p, first_line, started, a["cost_usd"])
            save(p, status="passed", last_feedback=first_line)
            log(f"nothing to do: {first_line.removeprefix(NOOP_MARK).strip()}")
            return

        save(p, status="validating")
        log("validating…")
        passed, fb = run_validate()
        if passed and git("status", "--porcelain", "--untracked-files=normal"):
            passed, fb = False, "Validation passed but the work is not committed. Commit it:\n" + git("status", "--short")
        if passed and not commits_ahead():
            passed, fb = False, (f"Validation passed but there are no commits on {p['branch']}. Commit your work, "
                                 f"or if there is genuinely nothing to change, start your reply with `{NOOP_MARK}`.")
        if not passed:
            record(p, "validation failed", started, a["cost_usd"])
            save(p, status="failed", last_feedback=fb)
            log("validation failed; feeding the log back")
            continue

        cost = a["cost_usd"]
        if p.get("evaluate", True):
            save(p, status="evaluating")
            log("evaluating…")
            ev = run_evaluator(p)
            cost += ev["cost_usd"]
            if not ev["ok"] or ev["approved"] is None:
                error = ev["error"] or "evaluator gave no VERDICT line: " + ev["result"][-500:]
                if crashed("evaluator", error, started, cost):
                    return
                continue
            errors["evaluator"] = 0
            if not ev["approved"]:
                record(p, "evaluator rejected", started, cost)
                save(p, status="failed", last_feedback="EVALUATOR REJECTED (validation passed, but an independent "
                                                       "review found problems; fix them):\n" + ev["reasons"])
                log("evaluator rejected; feeding its review back")
                continue

        record(p, "passed", started, cost)
        save(p, status="passed", last_feedback="")
        log(f"PASSED after {p['iteration']} iteration(s), ${p['total_cost_usd']:.2f} total")
        if open_pr_on_pass:
            open_pr(p)
        return


# ---------------------------------------------------------------- routines

def load_routine() -> dict:
    try:
        return {"next": 0, "last_queued_at": 0.0, **json.loads(ROUTINE_STATE.read_text())}
    except (FileNotFoundError, json.JSONDecodeError):
        return {"next": 0, "last_queued_at": 0.0}


def open_routine_prs() -> int | None:
    """Open PRs from routine branches, or None when gh can't tell (then don't queue)."""
    r = subprocess.run(["gh", "pr", "list", "--state", "open", "--json", "headRefName", "--limit", "100"],
                       check=False, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return sum(1 for pr in json.loads(r.stdout) if pr["headRefName"].startswith(ROUTINE_PREFIX))
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def maybe_queue_routine(every_s: float, kinds: tuple[str, ...] = ROUTINE_ORDER) -> bool:
    """Queue the next routine task if the loop is idle and one is due. True when one was queued."""
    if load()["status"] in ACTIVE or stop_requested():
        return False
    state = load_routine()
    if time.time() - state["last_queued_at"] < every_s:
        return False
    open_prs = open_routine_prs()
    if open_prs is None or open_prs >= ROUTINE_MAX_OPEN_PRS:
        why = "can't list PRs (gh)" if open_prs is None else f"{open_prs} routine PRs still open"
        log(f"routine due but skipped: {why}; retrying in an hour")
        state["last_queued_at"] = time.time() - every_s + 3600
        ROUTINE_STATE.write_text(json.dumps(state) + "\n")
        return False
    kind = kinds[state["next"] % len(kinds)]
    task, details = ROUTINES[kind]
    queue_task(task, details, branch=f"{ROUTINE_PREFIX}{kind}-{time.strftime('%Y%m%d-%H%M')}", routine=kind,
               **ROUTINE_LIMITS)
    AGENT_DIR.mkdir(exist_ok=True)
    ROUTINE_STATE.write_text(json.dumps({"next": state["next"] + 1, "last_queued_at": time.time()}) + "\n")
    log(f"routine queued: {task}")
    return True


# ---------------------------------------------------------------- commands

def cmd_new(a: argparse.Namespace) -> None:
    try:
        p = queue_task(a.task, a.details or "", a.branch or "", max_iterations=a.max_iterations,
                       budget_usd_per_iteration=a.budget, evaluate=not a.no_evaluate)
    except RuntimeError as e:
        sys.exit(str(e))
    print(f"queued: {a.task!r} on {p['branch']} (max {a.max_iterations} iterations). Next: python3 orchestrator.py run")


def cmd_run(a: argparse.Namespace) -> None:
    if a.routine and not a.watch:
        sys.exit("--routine needs --watch")
    kinds = tuple(k.strip() for k in a.routine_kinds.split(",") if k.strip())
    if unknown := set(kinds) - set(ROUTINES):
        sys.exit(f"unknown routine kind(s): {', '.join(sorted(unknown))}")
    every_s = a.routine_every * 3600
    STOP_FILE.unlink(missing_ok=True)
    while True:
        p = load()
        if p["status"] in ACTIVE:  # running/validating/evaluating = resumed after a crash
            try:
                work(p, a.pr)
            except RuntimeError as e:
                save(p, status="blocked", last_feedback=str(e))
                log(f"blocked: {e}")
        if not a.watch or stop_requested():
            return
        log(f"idle ({load()['status']}); waiting for `orchestrator.py new`"
            + (f" or the next routine (every {a.routine_every:g}h)" if a.routine else "") + " …")
        while load()["status"] != "pending" and not stop_requested():
            if a.routine and maybe_queue_routine(every_s, kinds):
                break
            time.sleep(WATCH_POLL_S)
        if stop_requested():
            log("stopped")
            return


def cmd_status(_: argparse.Namespace) -> None:
    p = load()
    print(f"{p['status'].upper()}  {p['task'] or '(no task)'}" + (f"  [routine: {p['routine']}]" if p["routine"] else ""))
    print(f"branch {p['branch'] or '-'} · iteration {p['iteration']}/{p['max_iterations']} · "
          f"${p['total_cost_usd']:.2f} · evaluator {'on' if p['evaluate'] else 'off'} · updated {p['updated_at'] or '-'}")
    if p["pr_url"]:
        print(f"PR {p['pr_url']}")
    if ROUTINE_STATE.exists():
        r = load_routine()
        print(f"next routine: {ROUTINE_ORDER[r['next'] % len(ROUTINE_ORDER)]} · last queued "
              f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(r['last_queued_at']))}")
    if p["last_feedback"]:
        print("last feedback:\n  " + "\n  ".join(p["last_feedback"].strip().splitlines()[:15]))


def cmd_stop(_: argparse.Namespace) -> None:
    AGENT_DIR.mkdir(exist_ok=True)
    STOP_FILE.touch()
    print("stop requested: the loop exits after the current step (an agent run in progress finishes first)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new", help="queue a task (replaces the previous one, which is archived)")
    n.add_argument("task")
    n.add_argument("--details", help="acceptance criteria, files to look at, constraints")
    n.add_argument("--branch", help="default: agent/<task-slug>")
    n.add_argument("--max-iterations", type=int, default=TEMPLATE["max_iterations"])
    n.add_argument("--budget", type=float, default=TEMPLATE["budget_usd_per_iteration"],
                   help="max USD per implementor run (claude --max-budget-usd)")
    n.add_argument("--no-evaluate", action="store_true", help="skip the evaluator agent (validate.py only)")
    r = sub.add_parser("run", help="work the queued task")
    r.add_argument("--watch", action="store_true", help="always-on: then wait for the next task")
    r.add_argument("--pr", action="store_true", help="on pass: push and open a PR (never merges)")
    r.add_argument("--routine", action="store_true", help="with --watch: queue routine work when idle")
    r.add_argument("--routine-every", type=float, default=6.0, help="hours between routine tasks (default 6)")
    r.add_argument("--routine-kinds", default=",".join(ROUTINE_ORDER),
                   help=f"comma-separated rotation (default {','.join(ROUTINE_ORDER)})")
    sub.add_parser("status")
    sub.add_parser("stop")
    a = ap.parse_args()
    {"new": cmd_new, "run": cmd_run, "status": cmd_status, "stop": cmd_stop}[a.cmd](a)


if __name__ == "__main__":
    main()
