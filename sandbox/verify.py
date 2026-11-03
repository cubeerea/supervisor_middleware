"""Score a migrated app.db against the schema_version 2 spec.

    python verify.py --path app.db   # prints JSON

Expected values are regenerated from the seed parameters stored by seed_db.py (PRAGMA user_version = row count,
PRAGMA application_id = seed), so tampering with the data cannot fake a pass.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path


def _seed_db():
    name = "_sandbox_seed_db"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("seed_db.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


COLS = ("id", "full_name", "email", "signup_date", "schema_version")


def verify(path: str | Path, n: int | None = None, seed: int | None = None) -> dict:
    """Returns {total, expected, migrated, wrong, pending, missing, extra, complete}.

    migrated: rows at schema_version 2 whose values match the spec exactly
    wrong:    rows at schema_version 2 with any incorrect value (email, date, or a changed name)
    pending:  rows not at schema_version 2
    missing:  expected ids absent from the table; extra: ids that were never seeded
    complete: every expected row migrated, nothing wrong/pending/missing/extra
    """
    sd = _seed_db()
    conn = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    try:
        stored_n = conn.execute("PRAGMA user_version").fetchone()[0]
        stored_seed = conn.execute("PRAGMA application_id").fetchone()[0]
        rows = conn.execute(f"SELECT {', '.join(COLS)} FROM customers").fetchall()
    finally:
        conn.close()
    n = stored_n if n is None else n
    seed = stored_seed if seed is None else seed
    expected = {r["id"]: sd.migrate_row(r) for r in sd.generate_rows(n, seed)}

    migrated = wrong = pending = extra = 0
    seen = set()
    for r in rows:
        row = dict(zip(COLS, r, strict=True))
        seen.add(row["id"])
        exp = expected.get(row["id"])
        if exp is None:
            extra += 1
        elif row["schema_version"] != 2:
            pending += 1
        elif row == exp:
            migrated += 1
        else:
            wrong += 1
    missing = len(set(expected) - seen)
    return {
        "total": len(rows),
        "expected": len(expected),
        "migrated": migrated,
        "wrong": wrong,
        "pending": pending,
        "missing": missing,
        "extra": extra,
        "complete": migrated == len(expected) and not (wrong or pending or missing or extra),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", default="app.db")
    a = p.parse_args(argv)
    print(json.dumps(verify(a.path)))


if __name__ == "__main__":
    main()
