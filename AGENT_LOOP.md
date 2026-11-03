# Always-on agent loop: setup guide

> **Status in `supervisor_middleware`:** installed. `orchestrator.py`/`validate.py` are adapted (checks: ruff,
> pytest, `docker compose config`). The agent worktree is `../supervisor_middleware-agent`, and the loop runs in tmux
> session `agent` (`run --watch`, no `--pr`/`--routine`). There is no GitHub remote yet: `origin` is the local bare
> repo `../supervisor_middleware.git`, so branches stay local and are merged by hand. To enable PRs and routines,
> add a GitHub `origin` and restart with `run --watch --pr --routine`.

This guide sets up an unattended coding loop that works through tasks you queue, checks its own
work, and opens pull requests for you to review. It was built for Polytone but is meant to be
copied into any git project hosted on GitHub. Everything specific to one project lives in two
files, and the [porting checklist](#3-adapt-it-to-the-project) lists every line to change.

- [How it works](#how-it-works)
- [What you need](#what-you-need)
- [Setup](#setup)
- [Daily use](#daily-use)
- [Writing good tasks](#writing-good-tasks)
- [Routines](#routines)
- [Safety model](#safety-model)
- [Cost control](#cost-control)
- [Reference](#reference)
- [Troubleshooting](#troubleshooting)

## How it works

```
 you: orchestrator.py new "task" ──▶ progress.json (status: pending)
                                          │
        ┌─────────────────────────────────▼─────────────────────────────────┐
        │ 1. IMPLEMENTOR   claude -p <task + feedback>   edits + commits     │
        │ 2. GUARD         did it touch protected files or secrets? ──fail──┐│
        │ 3. VALIDATE      validate.py (the same checks as CI)      ──fail──┤│
        │ 4. EVALUATOR     read-only claude reviews the diff only   ──reject┤│
        │                                                                   ││
        │      pass ──▶ push branch + open PR (never merges)                ││
        │      fail ──▶ its output becomes feedback for the next iteration ◀┘│
        └───────────────────────────────────────────────────────────────────┘
                 repeats until passed · blocked · max_iterations
 idle + --routine ──▶ queues maintenance work every N hours (validation → tests → bugs)
```

- The **implementor** is a headless Claude Code run that can edit files, run the checks and
  commit, and nothing else.
- **`validate.py`** runs the project's real checks and writes the failures to a log. That log is
  fed back word for word, so the agent fixes the actual error rather than guessing.
- The **evaluator** is a second, read-only Claude Code run. It sees only the task and the diff,
  never the implementor's account of what it did, so it reviews independently.
- **You** review and merge the PRs. The loop never merges.

The loop runs in its own git worktree inside tmux. It keeps working while you use your main
checkout, and picks up where it left off after a crash or restart.

## What you need

| Tool | Why | Check |
|---|---|---|
| [Claude Code CLI](https://docs.claude.com/en/docs/claude-code) | Runs the implementor and evaluator | `claude --version`, and you're logged in (`claude` then `/login`, or `ANTHROPIC_API_KEY` set) |
| [GitHub CLI](https://cli.github.com/) | Opens PRs and counts open routine PRs | `gh auth status` |
| git, with a GitHub remote named `origin` and a `main` branch | Each task branch is cut from `origin/main` | `git remote -v` |
| Python 3.9+ | Both scripts use the standard library only | `python3 --version` |
| tmux (recommended) | Keeps the loop running after you close the terminal | `tmux -V` |
| CI on pull requests (recommended) | A second gate before you merge | `.github/workflows/*.yml` |

## Setup

### 1. Copy the files

From the Polytone repo, copy these into the root of the new project:

| File | Copy as-is? |
|---|---|
| `orchestrator.py` | Yes, then make the edits in step 3 |
| `validate.py` | No, it's a template: replace its `CHECKS` list (step 3) |
| `backend/tests/test_orchestrator.py` | Optional, but it tests the loop offline with fake agents. Move it to wherever your tests live and fix the `sys.path` line. |

```bash
cp ~/Code/polytone/{orchestrator.py,validate.py} /path/to/new-project/
```

### 2. Ignore the runtime state

Add these lines to `.gitignore`:

```gitignore
# unattended agent loop (orchestrator.py): runtime state and logs
progress.json
progress.json.tmp
.agent/
```

`progress.json` holds the current task, and `.agent/` holds logs, validation reports, archived tasks
and routine state. Each checkout has its own copy, which is why you queue tasks in the agent's
worktree (step 6).

### 3. Adapt it to the project

**`validate.py`.** This is the only file most projects need to rethink. It should run exactly
what CI runs.

```python
CHECKS = [
    Check(name, lang, cmd, cwd="."),   # lang groups checks for --only and for skipping
    ...
]
LANGS = (...)                          # every lang used above
```

Example `CHECKS` for common stacks:

```python
# Node / TypeScript
Check("lint", "js", ["pnpm", "lint"]),
Check("typecheck", "js", ["pnpm", "tsc", "--noEmit"]),
Check("test", "js", ["pnpm", "test", "--", "--run"]),
Check("build", "js", ["pnpm", "build"]),

# Python (uv)
Check("ruff", "python", ["uv", "run", "ruff", "check", "."]),
Check("pytest", "python", ["uv", "run", "pytest", "-q"]),

# Go
Check("vet", "go", ["go", "vet", "./..."]),
Check("test", "go", ["go", "test", "./..."]),

# Rust
Check("clippy", "rust", ["cargo", "clippy", "--all-targets", "--locked", "--", "-D", "warnings"]),
Check("test", "rust", ["cargo", "test", "--locked"]),
```

Make the following changes in `validate.py` too:

- **Path-based skipping.** `rust_changed()` skips the slow Rust checks unless `src-tauri/`
  changed, matching a CI path filter. Point it at your own slow subtree, or delete it and the
  `langs.discard("rust")` line in `main()`.
- **Offline environment.** `run()` blanks `ANTHROPIC_API_KEY` and sets `CI=1`, so tests never call
  real services. Add your own secrets there, for example `"STRIPE_KEY": ""`.
- **Ruff target.** The ruff check above also lints `validate.py` and `orchestrator.py`. Keep that
  if the project uses Python.

**`orchestrator.py`.** Search for each item below and update it. Everything else is generic.

| What | Where | Change to |
|---|---|---|
| Env var names | `POLYTONE_AGENT_BASE`, `POLYTONE_AGENT_CMD` | Optional rename, e.g. `AGENT_BASE`, `AGENT_CMD` |
| Implementor's allowed commands | `AGENT_ARGS` → `--allowedTools` | Your toolchain: keep `Bash(python3 validate.py*)` and the git lines, and replace `uv`/`pnpm`/`cargo` with your build and test commands, e.g. `Bash(npm test*)`, `Bash(go test*)` |
| Evaluator's allowed commands | `EVAL_ARGS` → `--allowedTools` | Your test command, e.g. `Bash(npm test*)` instead of `Bash(uv run pytest *)` |
| Secret files | `SECRETS_DENY` and `FORBIDDEN` | Every path holding secrets or user data: `.env*`, `*.pem`, `secrets/`, local databases |
| Protected files | `PROTECTED`, plus the `Edit(...)`/`Write(...)` entries in `AGENT_ARGS` | Keep `validate.py`, `orchestrator.py` and `.github/`, and add anything else that decides whether work passes, such as a coverage config |
| Project name and checks in the prompts | `context()` and `eval_context()` | Replace "Polytone repo", the list "ruff, pytest, pnpm build, cargo", and rule 4's project-specific items (the Keychain, Neo4j port 7687) with your own never-touch list |
| Routine prompts | `ROUTINES` | Replace `backend/`, `src/`, `backend/tests/conftest.py`, `backend/daemon/server.py` and the cargo/pnpm mentions with your source and test directories |

To find every remaining Polytone-specific string:

```bash
grep -nE "Polytone|POLYTONE|backend/|src-tauri|src/|pnpm|cargo|uv run|Neo4j|7687|Keychain|conftest" orchestrator.py validate.py
```

### 4. Tell the agents the house rules

The implementor follows the project's `CLAUDE.md`. Add a section like this:

```markdown
## Git workflow (every change)
`main` is never committed to directly. For each change: branch from an up-to-date main → commit
(tests passing) → push → PR → wait for CI → squash-merge. Never merge a red PR.
Never commit `.env`, secrets, or local data.

## Unattended agent loop
`orchestrator.py` runs headless Claude Code on one task from `progress.json` (git-ignored) as an
implementor/evaluator pair: the implementor makes the change, `validate.py` (the same checks as CI)
must pass, then a read-only evaluator agent reviews the diff against the task. Failures are fed back
until it passes or hits `max_iterations`. Run it in its own worktree inside tmux.
The agent commits on `agent/<task>` only; pushing, PRs and merging stay with the orchestrator/human.
Agents must never edit `validate.py`, `orchestrator.py`, `progress.json` or `.github/`.
```

Put the project's testing rules in `CLAUDE.md` too, such as "tests must run offline" and "never use
the real database". The agent reads `CLAUDE.md` on every run.

### 5. Protect `main` (recommended)

In **GitHub → Settings → Branches**, add a rule for `main` that requires a pull request and passing
status checks. The loop never pushes to `main`, and this rule enforces that for everyone else too.

### 6. Create the agent's worktree and smoke-test it

```bash
cd /path/to/new-project
git add orchestrator.py validate.py .gitignore CLAUDE.md && git commit -m "Add agent loop"   # via a PR
python3 validate.py                      # must pass on a clean main first

git fetch origin
git worktree add --detach ../new-project-agent origin/main   # --detach: main is already checked out in your main checkout
cd ../new-project-agent
# install dependencies here too (node_modules, .venv, …); worktrees don't share them
python3 orchestrator.py new "Fix a typo in the README" --details "Any one typo. Smoke test of the loop." --max-iterations 2
python3 orchestrator.py run --pr
```

You should see `iteration 1/2: implementor working…`, then `validating…`, `evaluating…`, `PASSED`
and a PR URL. Merge or close the PR, then start the always-on loop.

> **Why a worktree?** The loop switches branches and needs a clean working tree. In your main
> checkout it would fight with your own edits. Another session switching that checkout to `main`
> partway through a task can also land commits on the wrong branch.

## Daily use

Start it once, in tmux:

```bash
cd ../new-project-agent
tmux new -s agent
python3 orchestrator.py run --watch --pr --routine     # Ctrl-b d to detach; `tmux attach -t agent` to return
```

From any other terminal, **in the agent's worktree**:

```bash
python3 orchestrator.py new "Add CSV export to reports" --details "Acceptance: …  Tests: …"
python3 orchestrator.py status         # state, iteration, cost, PR, last feedback
python3 orchestrator.py stop           # exits after the current step
```

| Situation | What to do |
|---|---|
| A task passed | Review the PR, then merge or close it. The loop waits for the next `new` task, or the next routine. |
| A task is **blocked** | Run `status` to see why: the agent said `BLOCKED:`, it hit max iterations, or crashed repeatedly. Requeue it with better details, or finish it by hand on its branch. |
| You want to queue another task | `new` replaces the current task, which is archived to `.agent/history/`. It refuses while a task is mid-run, so `stop` first. |
| The machine rebooted | Rerun the `run …` command. A task left `running` resumes from its last iteration. |
| You want to update the loop itself | Merge the change into `main`. When the loop is idle, `stop` it, then in the worktree run `git fetch origin && git switch --detach origin/main` and start `run` again. A running process keeps its old code until restarted. |

The worktree's branch changes with every task, so never edit files there by hand while the loop is running.

## Writing good tasks

The implementor sees only the task title, `--details`, `CLAUDE.md` and the repo, and nobody is
there to answer its questions. Write details the way you would for a contractor:

```bash
python3 orchestrator.py new "Rate-limit the login endpoint" --details "
Acceptance: 5 failed attempts per IP per 15 min → HTTP 429 with Retry-After. Successful login resets the count.
Where: api/auth.py (login handler), reuse the Redis client in api/cache.py.
Tests: unit tests for the counter and an API test that hits the limit. Must run without real Redis (use the fake in tests/fakes.py).
Out of scope: UI changes, other endpoints."
```

- **Acceptance criteria** let the evaluator decide approve or reject. Without them it can only
  judge whether the change looks reasonable.
- **Test requirements** stop the agent from shipping untested code.
- **"Out of scope"** keeps the diff reviewable.
- **One concern per task.** Big tasks run out of iterations, so split them.
- **`--max-iterations`** defaults to 8. Use 2–3 for small tasks, so a confused agent stops early.
- **`--no-evaluate`** skips the review step, for mechanical tasks like a dependency bump.

## Routines

With `--routine` (it needs `--watch`), an idle loop queues maintenance work every
`--routine-every` hours (default 6). It cycles through three kinds, each on its own
`agent/routine-<kind>-<timestamp>` branch:

| Kind | What the agent does |
|---|---|
| `validation` | Runs every check, fixes failures and the warnings that come from the project's own code |
| `tests` | Finds one important behaviour with no test and adds focused tests |
| `bugs` | Finds one real, reproducible bug, writes a failing test, then fixes it |

- **Nothing to fix.** The agent answers `NOTHING TO DO: …`. The task then passes with no commits
  and no PR.
- **Open PR limit.** No routine is queued while 2 routine PRs are open (`ROUTINE_MAX_OPEN_PRS`).
  Unmerged routine PRs therefore pause routines instead of piling up.
- **Tighter limits.** Routines get 4 iterations and smaller budgets (`ROUTINE_LIMITS`).
- **Choosing kinds.** Use `--routine-kinds tests,bugs` to change the rotation. To add a kind, add
  an entry to `ROUTINES` and `ROUTINE_ORDER`.

## Safety model

| Layer | What it prevents |
|---|---|
| `--allowedTools` allowlist plus headless mode | The agent can run only the listed commands. Headless runs can't ask permission, so anything else is refused. |
| `--disallowedTools` | No `git push`, `gh`, branch switching, `reset --hard`, `rm -rf`, `curl`, or reading secret files |
| Guard after every run | Changes to `PROTECTED` files (the checks that judge it) or `FORBIDDEN` paths (secrets, data) fail the iteration. The agent can't weaken its own checks. |
| `validate.py` | Nothing proceeds unless the same checks as CI pass |
| Commit checks | Uncommitted work, or no commits at all, counts as a failure. A no-op must be declared explicitly. |
| Evaluator | An independent, read-only review of the diff. The loop aborts if the evaluator changes the tree or HEAD. |
| Orchestrator-only push | Only the orchestrator pushes, and only after everything passed |
| Human merge | The loop opens PRs but never merges |
| Branch protection plus CI | A last gate on GitHub |

Prompt rules also tell the agent not to weaken or skip tests, and to answer `BLOCKED:` when a task
is unsafe or ambiguous. Prompt rules are only guidance, which is why the guard and the evaluator
exist.

## Cost control

| Setting | Default | Where |
|---|---|---|
| Max USD per implementor run | $5 | `new --budget`, or `budget_usd_per_iteration` |
| Max USD per evaluator run | $2 | `eval_budget_usd` in `TEMPLATE` |
| Max iterations per task | 8 | `new --max-iterations` |
| Implementor and evaluator timeouts | 45 min and 15 min | `AGENT_TIMEOUT_S`, `EVAL_TIMEOUT_S` |
| Routine limits | 4 iterations, $3 and $1.50 | `ROUTINE_LIMITS` |
| Routine frequency | every 6 h | `run --routine-every` |

The worst case per task is therefore about `max_iterations × (budget + eval budget)`, which is
$56 with the defaults. `status` shows the running total, and every prompt and output is kept in
`.agent/logs/`. On a Claude subscription instead of an API key, budgets cap usage rather than spend.

## Reference

### Commands

```
orchestrator.py new "<task>" [--details "…"] [--branch name] [--max-iterations N] [--budget USD] [--no-evaluate]
orchestrator.py run [--watch] [--pr] [--routine] [--routine-every HOURS] [--routine-kinds a,b,c]
orchestrator.py status
orchestrator.py stop                  # or: touch .agent/STOP
validate.py [--all] [--only lang1,lang2]
```

### Task states (`progress.json` → `status`)

| State | Meaning |
|---|---|
| `pending` | Queued; `run` picks it up |
| `running` / `validating` / `evaluating` | In progress. If the loop died here, `run` resumes it. |
| `failed` | The last iteration failed. The next one is about to start with the feedback. |
| `passed` | Done. A PR was opened if `--pr` was set. |
| `blocked` | Needs a human: max iterations, `BLOCKED:` from the agent, or 3 crashes in a row |
| `stopped` | Stopped by `stop` |

### Files

| Path | Contents |
|---|---|
| `progress.json` | The current task: status, iteration, `last_feedback`, cost, history, PR URL |
| `.agent/validation.json` | The last `validate.py` run: every check, its timing and output tail |
| `.agent/feedback.log` | Only the failures, stderr first. This is what the agent sees. |
| `.agent/logs/<time>-iterN-impl.*` | Each agent run's prompt, JSON output and stderr (`-eval` for the evaluator) |
| `.agent/history/` | Archived tasks |
| `.agent/routine.json` | Which routine is next and when the last one was queued |
| `.agent/STOP` | Present means "stop after the current step" |

### Environment

| Variable | Default | Meaning |
|---|---|---|
| `POLYTONE_AGENT_BASE` | `origin/main` | The branch task branches are cut from, and diffs are measured against |
| `POLYTONE_AGENT_CMD` | `claude -p` | The agent command. Point it at a fake script to test the loop, or at another CLI that accepts the same flags. |

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `working tree has uncommitted changes` | The worktree is dirty, from a leftover run or a hand edit. Commit, stash or `git restore` there, then rerun. |
| `refusing to work on branch 'main'` | The task's `--branch` was set to main. Requeue without `--branch`. |
| Blocked after "implementor error" 3 times | Usually auth, a rate limit or the budget. Run `claude -p "hi"` in the worktree, check `.agent/logs/*.err.txt`, then requeue. |
| "evaluator gave no VERDICT line" | The evaluator's reply didn't start with `VERDICT:`. Read its log. Repeated misses usually mean the diff is huge, so split the task. |
| Validation fails in the loop but passes for you | Dependencies are missing in the worktree (`node_modules`, `.venv`), or your shell exports env vars the loop doesn't have. Install them in the worktree, and put needed test env in `validate.py`. |
| PR not created | Check `gh auth status` in the worktree. The branch is already pushed, so you can run `gh pr create --head <branch>` by hand. |
| Routines never run | Two routine PRs are still open, or `gh` can't list PRs. The log says which. Merge or close them. |
| The agent keeps editing a protected file | The guard message is fed back to it. If it persists, the task probably needs that file changed, so do that part yourself. |
| Commits landed on the wrong branch | The loop was run in a shared checkout. Always run it in its own worktree. |
