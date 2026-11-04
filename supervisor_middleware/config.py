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
    verdict_grace_s: float = field(default_factory=lambda: float(os.environ.get("VERDICT_GRACE_S", "3.0")))

    # Supervisor (local Qwen through Ollama's HTTP API)
    supervisor_backend: str = field(default_factory=lambda: os.environ.get("SUPERVISOR_BACKEND", "ollama"))
    ollama_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    supervisor_model: str = field(default_factory=lambda: os.environ.get("SUPERVISOR_MODEL", "qwen2.5:3b-instruct"))
    supervisor_timeout_s: float = field(default_factory=lambda: float(os.environ.get("SUPERVISOR_TIMEOUT_S", "30")))

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
