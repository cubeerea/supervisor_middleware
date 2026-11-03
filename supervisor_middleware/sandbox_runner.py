"""Per-episode sandbox: a working dir with the mock ORM, its docs and a freshly seeded app.db, plus a
subprocess runner for the executor's proposed Python snippets."""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import os
import shutil
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .config import ROOT

SANDBOX_SRC = ROOT / "sandbox"
WORKDIR_FILES = ("mock_orm.py", "ORM_DOCS.md")  # what the executor gets to see; seed/verify stay outside
DB_NAME = "app.db"
MAX_OUTPUT_CHARS = 4000

_SECRET_PREFIXES = ("ANTHROPIC_", "LANGSMITH_", "LANGCHAIN_", "OPENAI_", "PYTHON", "VIRTUAL_ENV", "UV_")
_SECRET_MARKERS = ("API_KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL")


def load_sandbox_module(name: str, src_dir: Path = SANDBOX_SRC) -> ModuleType:
    """Import sandbox/<name>.py by path (sandbox/ is not a package). Cached per (dir, name)."""
    path = (Path(src_dir) / f"{name}.py").resolve()
    key = f"_sandbox_{name}_{abs(hash(str(path)))}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def prepare_workdir(dest: Path, rows: int = 1000, seed: int = 7) -> Path:
    """Create `dest` with mock_orm.py, ORM_DOCS.md and a fresh dest/app.db. Returns `dest`."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in WORKDIR_FILES:
        shutil.copy2(SANDBOX_SRC / name, dest / name)
    load_sandbox_module("seed_db").seed(dest / DB_NAME, n=rows, seed=seed)
    return dest


def verify_workdir(workdir: Path) -> dict[str, Any]:
    """Ground-truth migration score for the workdir's app.db (see sandbox/verify.py)."""
    return load_sandbox_module("verify").verify(Path(workdir) / DB_NAME)


@dataclass
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int | None
    duration_s: float
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"[... {len(text) - limit} chars truncated ...]\n" + text[-limit:]


def scrubbed_env() -> dict[str, str]:
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(_SECRET_PREFIXES) and not any(m in k.upper() for m in _SECRET_MARKERS)
    }
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        proc.kill()


def _write_script(workdir: Path, command: str) -> tuple[Path, Path]:
    workdir = workdir.resolve()
    script = workdir / f"_cmd_{uuid.uuid4().hex[:12]}.py"
    script.write_text(command, encoding="utf-8")
    return workdir, script


async def run_command(command: str, workdir: Path, timeout_s: float = 30) -> CommandResult:
    """Run a Python snippet in `workdir` with the current interpreter.

    The snippet is written to a temp file inside `workdir`, so the script's directory (sys.path[0]) is the
    workdir and `import mock_orm` resolves to the copy there. Not run with -I (that would drop sys.path[0]);
    PYTHON* variables are stripped from the env instead and -s disables the user site dir.
    """
    workdir, script = await asyncio.to_thread(_write_script, Path(workdir), command)
    start = time.monotonic()
    timed_out = False
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-s", "-B", script.name,
            cwd=workdir,
            env=scrubbed_env(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # own process group, so a timeout kills grandchildren too
        )
        out_task = asyncio.create_task(proc.stdout.read())
        err_task = asyncio.create_task(proc.stderr.read())
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout_s)
        except TimeoutError:
            timed_out = True
            _kill_group(proc)
            await proc.wait()
        except asyncio.CancelledError:
            _kill_group(proc)
            raise
        finally:
            # also reap anything the snippet left running in its group
            if proc.returncode is not None:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(proc.pid, signal.SIGKILL)
        try:
            out_b, err_b = await asyncio.wait_for(asyncio.gather(out_task, err_task), timeout=5)
        except TimeoutError:
            out_task.cancel()
            err_task.cancel()
            out_b, err_b = b"", b""
    finally:
        await asyncio.to_thread(script.unlink, missing_ok=True)
    stderr = err_b.decode("utf-8", "replace")
    if timed_out:
        stderr += f"\n[sandbox] command timed out after {timeout_s:g}s and was killed"
    return CommandResult(
        stdout=_truncate(out_b.decode("utf-8", "replace")),
        stderr=_truncate(stderr),
        exit_code=proc.returncode,
        duration_s=time.monotonic() - start,
        timed_out=timed_out,
    )
