# Continuous Interception Middleware: Project Plan (original brief)

Architecture: Hybrid Orchestration (Claude Sonnet executor API + local Qwen2.5-3B supervisor)
Frameworks: LangGraph, Docker, Redis (Pub/Sub), LangSmith (replaced by Langfuse, see README)
Goal: a non-blocking, real-time context supervisor to prevent agent sycophancy and silent failure loops.

1. **Docker & Pub/Sub scaffolding** – docker-compose with executor (LangGraph), Redis, Ollama/Qwen2.5-3B-Instruct.
   Channels `executor_stream` (trajectory out) and `supervisor_interrupts` (kill signals in). Async, non-blocking
   Redis manager.
2. **Executor node & streaming tool parser** – Anthropic SDK, tool `ProposeAction(reasoning: str, command: str)`,
   `client.messages.stream`, buffer JSON deltas; the moment `reasoning` closes, publish state + reasoning + run_id
   to `executor_stream` without awaiting; keep streaming `command`.
3. **Supervisor daemon & distributed tracing** – `supervisor_daemon.py` listens on `executor_stream`, stitches the
   LangSmith trace via the parent run, asks Qwen `{"status": "proceed"|"halt", "reason": ...}`, publishes halts to
   `supervisor_interrupts`.
4. **Interception hook & kill switch** – background task per executor node subscribed to interrupts; matching run_id
   cancels the Anthropic stream (`InterceptionHalt`), state records the reason, node loops back/retries.
5. **Ground-truth sandbox** – SQLite + `mock_orm.py`; docs promise `update_batch`, code omits it. Migrate 1,000 rows;
   the executor drifts toward a raw-SQL bypass that the supervisor forbids. Measure True Intervention Rate (TIR) and
   publish→cancel latency.

Deviation notes live in README.md ("Deviations from the plan").
