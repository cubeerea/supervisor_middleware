"""Import shim: `sandbox/` is a plain directory of files copied into each episode workdir (not a package),
so load sandbox/verify.py by path."""
from __future__ import annotations

import importlib.util
from functools import cache
from pathlib import Path
from typing import Any

from .config import ROOT


@cache
def _module():
    spec = importlib.util.spec_from_file_location("_sandbox_verify", ROOT / "sandbox" / "verify.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_db(path: Path | str) -> dict[str, Any]:
    return _module().verify(Path(path))
