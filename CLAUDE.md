# supervisor_middleware

Continuous interception middleware: a LangGraph **executor** (Claude, streaming a `ProposeAction(reasoning, command)`
tool call) publishes its reasoning to Redis the instant the `reasoning` field closes; a standalone **supervisor
daemon** (local Qwen2.5-3B via Ollama) judges it and, on `halt`, publishes a kill signal that cancels the executor's
in-flight Anthropic stream. Phase 5 measures True Intervention Rate and interception latency against a mock-ORM
"raw SQL" trap. See README.md for the architecture and PROJECT_PLAN.md for the original plan.

## Layout
- `supervisor_middleware/` – package. `messages.py` (wire contract), `redis_bus.py`, `stream_parser.py`,
  `telemetry.py` (JSONL events), `executor/` (graph, stream sources, interception), `supervisor/` (judge, daemon),
  `metrics.py`.
- `sandbox/` – mock ORM trap (`mock_orm.py`, `ORM_DOCS.md`, `seed_db.py`). `ORM_DOCS.md` intentionally documents
  `update_batch`, which `mock_orm.py` intentionally does NOT implement. Never "fix" that.
- `tests/` – pytest, offline only.

## Commands
- `uv sync` · `uv run pytest -q` · `uv run ruff check .` · `python3 validate.py` (= all of the above + compose config)

## Testing rules
- Tests run offline: scripted executor streams, `fakeredis`, rule/scripted judges, temp dirs. Never call the real
  Anthropic API, Ollama, Langfuse or a live Redis from tests (use `tests/tracing_fakes.py`).
- Never write into `results/` or `sandbox/*.db` from tests; use `tmp_path`.

## Git workflow (every change)
`main` is never committed to directly. For each change: branch from an up-to-date main → commit
(tests passing) → merge after review. Never commit `.env`, secrets, `results/` or `*.db`.

## Unattended agent loop
`orchestrator.py` runs headless Claude Code on one task from `progress.json` (git-ignored) as an
implementor/evaluator pair: the implementor makes the change, `validate.py` (the same checks as CI)
must pass, then a read-only evaluator agent reviews the diff against the task. Failures are fed back
until it passes or hits `max_iterations`. Run it in its own worktree inside tmux. See AGENT_LOOP.md.
The agent commits on `agent/<task>` only; pushing and merging stay with the orchestrator/human.
Agents must never edit `validate.py`, `orchestrator.py`, `progress.json`, `.github/` or `sandbox/ORM_DOCS.md`.
