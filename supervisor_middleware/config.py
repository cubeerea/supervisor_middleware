"""Runtime settings, read once from the environment. Every value has an offline-safe default."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXECUTOR_STREAM = "executor_stream"  # executor -> supervisor: reasoning packets
SUPERVISOR_INTERRUPTS = "supervisor_interrupts"  # supervisor -> executor: halt signals
SUPERVISOR_ACKS = "supervisor_acks"  # supervisor -> executor: proceed verdicts (lets the tool gate skip the grace wait)


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    redis_url: str = field(default_factory=lambda: os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    executor_channel: str = field(default_factory=lambda: os.environ.get("EXECUTOR_CHANNEL", EXECUTOR_STREAM))
    interrupt_channel: str = field(
        default_factory=lambda: os.environ.get("INTERRUPT_CHANNEL", SUPERVISOR_INTERRUPTS))
    ack_channel: str = field(default_factory=lambda: os.environ.get("ACK_CHANNEL", SUPERVISOR_ACKS))

    # Executor (Anthropic). The plan named Claude 3.5 Sonnet, which is retired; the current Sonnet is the default.
    executor_model: str = field(default_factory=lambda: os.environ.get("EXECUTOR_MODEL", "claude-sonnet-5-5"))
    executor_max_tokens: int = field(default_factory=lambda: int(os.environ.get("EXECUTOR_MAX_TOKENS", "16000")))
    executor_effort: str = field(default_factory=lambda: os.environ.get("EXECUTOR_EFFORT", "medium"))
    # Prompt caching: explicit breakpoint on the static system prompt + automatic breakpoint on the growing history.
    executor_prompt_cache: bool = field(
        default_factory=lambda: os.environ.get("EXECUTOR_PROMPT_CACHE", "true").lower() not in ("0", "false", "no"))
    # "anthropic" (real API) or "scripted" (deterministic offline token stream, see executor/scripted.py)
    executor_backend: str = field(default_factory=lambda: os.environ.get("EXECUTOR_BACKEND", "auto"))
    max_steps: int = field(default_factory=lambda: int(os.environ.get("EXECUTOR_MAX_STEPS", "12")))
    # After the stream finishes, how long the tool node waits for a late verdict before running the command.
    verdict_grace_s: float = field(default_factory=lambda: float(os.environ.get("VERDICT_GRACE_S", "12.0")))

    # Supervisor (local Qwen through Ollama's HTTP API)
    supervisor_backend: str = field(default_factory=lambda: os.environ.get("SUPERVISOR_BACKEND", "ollama"))
    ollama_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    supervisor_model: str = field(default_factory=lambda: os.environ.get("SUPERVISOR_MODEL", "qwen2.5:3b-instruct"))
    # "verdict": the model writes {reason, status}. "checklist": it names the step's action and answers three yes/no
    # rule questions; the verdict is derived in code (fewer false halts on reasoning that rejects a forbidden option).
    # "cascade": SUPERVISOR_MODEL screens every step in verdict mode; only its halts are re-checked by
    # SUPERVISOR_CONFIRM_MODEL in checklist mode, and a halt needs both.
    supervisor_judge_mode: str = field(default_factory=lambda: os.environ.get("SUPERVISOR_JUDGE_MODE", "cascade"))
    supervisor_confirm_model: str = field(
        default_factory=lambda: os.environ.get("SUPERVISOR_CONFIRM_MODEL", "qwen2.5-coder:7b"))
    supervisor_timeout_s: float = field(default_factory=lambda: float(os.environ.get("SUPERVISOR_TIMEOUT_S", "30")))
    # Ollama context window for the judge (0 = Ollama's default, ~2-4k tokens, which silently truncates trajectory
    # prompts). Changing it reloads the model, so set it once per daemon.
    supervisor_num_ctx: int = field(default_factory=lambda: int(os.environ.get("SUPERVISOR_NUM_CTX", "0")))

    # What the supervisor judges, and when:
    #   "reasoning"   packet on the closing quote of `reasoning` (rationale + last observation); a halt can cut the
    #                 stream mid-command (the original design)
    #   "action"      packet once the ProposeAction call is complete (reasoning + command); the tool gate blocks
    #                 until the verdict arrives
    #   "trajectory"  "action" plus every earlier step of the episode (reasoning, command, output)
    supervision_mode: str = field(default_factory=lambda: os.environ.get("SUPERVISION_MODE", "reasoning"))
    # Action modes: longest the tool gate waits for a verdict. On timeout the command runs (fail-open, like the
    # judge's own error handling) and an `action_verdict_timeout` event is logged so it shows up in the metrics.
    action_verdict_timeout_s: float = field(
        default_factory=lambda: float(os.environ.get("ACTION_VERDICT_TIMEOUT_S", "90")))

    # OpenRouter supervisor (SUPERVISOR_BACKEND=openrouter): any model with structured-output support.
    openrouter_api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", ""))
    openrouter_url: str = field(default_factory=lambda: os.environ.get("OPENROUTER_URL",
                                                                       "https://openrouter.ai/api/v1"))
    openrouter_model: str = field(default_factory=lambda: os.environ.get("OPENROUTER_MODEL", "qwen/qwen3-coder-next"))
    # Reasoning effort for thinking models ("", "minimal", "low", "medium", "high"); "" = the provider's default.
    openrouter_reasoning_effort: str = field(default_factory=lambda: os.environ.get("OPENROUTER_REASONING_EFFORT", ""))
    # Command check (executor tool gate): refuse to run a finished command that ground_truth.explain flags as a
    # raw-SQL bypass, whatever its reasoning says. Catches the covert case the reasoning-only supervisor cannot see.
    command_check: bool = field(default_factory=lambda: _bool("COMMAND_CHECK", True))
    # Stall detection (supervisor/stall.py): escalate an episode that makes no progress over STALL_WINDOW steps.
    stall_detection: bool = field(default_factory=lambda: _bool("STALL_DETECTION", True))
    stall_window: int = field(default_factory=lambda: int(os.environ.get("STALL_WINDOW", "3")))

    # Telemetry
    events_path: Path = field(
        default_factory=lambda: Path(os.environ.get("EVENTS_PATH", str(ROOT / "results" / "events.jsonl"))))
    # Langfuse: on when both keys are set (LANGFUSE_TRACING=false turns it off anyway).
    langfuse_public_key: str = field(default_factory=lambda: os.environ.get("LANGFUSE_PUBLIC_KEY", ""))
    langfuse_secret_key: str = field(default_factory=lambda: os.environ.get("LANGFUSE_SECRET_KEY", ""))
    langfuse_host: str = field(default_factory=lambda: os.environ.get("LANGFUSE_HOST", "http://localhost:3000"))
    langfuse_environment: str = field(default_factory=lambda: os.environ.get("LANGFUSE_TRACING_ENVIRONMENT",
                                                                             "development"))
    tracing: bool = field(default_factory=lambda: _bool("LANGFUSE_TRACING", True) and bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY")) and bool(os.environ.get("LANGFUSE_SECRET_KEY")))

    # Sandbox
    sandbox_dir: Path = field(default_factory=lambda: Path(os.environ.get("SANDBOX_DIR", str(ROOT / "sandbox"))))


def get_settings() -> Settings:
    return Settings()
