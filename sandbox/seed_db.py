"""Create a fresh app.db with deterministic, messy schema_version=1 customer rows.

    python seed_db.py --path app.db --rows 1000
"""
from __future__ import annotations

import argparse
import datetime as dt
import random
import sqlite3
from pathlib import Path

FIRST = ["Ada", "Grace", "Alan", "Linus", "Margaret", "Ken", "Barbara", "Dennis", "Radia", "Edsger",
         "Frances", "John", "Katherine", "Tim", "Hedy", "Donald", "Sophie", "Guido", "Anita", "Bjarne"]
LAST = ["Lovelace", "Hopper", "Turing", "Torvalds", "Hamilton", "Thompson", "Liskov", "Ritchie", "Perlman",
        "Dijkstra", "Allen", "McCarthy", "Johnson", "Berners-Lee", "Lamarr", "Knuth", "Wilson", "van Rossum",
        "Borg", "Stroustrup"]
DOMAINS = ["example.com", "mail.example.org", "corp.example.net", "example.io"]

SCHEMA = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL,
    signup_date TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1
)
"""


def _messy_case(rng: random.Random, s: str) -> str:
    mode = rng.randrange(4)
    if mode == 0:
        return s.upper()
    if mode == 1:
        return s.title()
    if mode == 2:
        return "".join(ch.upper() if rng.random() < 0.5 else ch.lower() for ch in s)
    return s  # already lowercase


def _pad(rng: random.Random, s: str) -> str:
    left = rng.choice(["", "", " ", "  ", "\t"])
    right = rng.choice(["", "", " ", "   ", "\n"])
    return f"{left}{s}{right}"


def generate_rows(n: int = 1000, seed: int = 7) -> list[dict]:
    """Deterministic schema_version=1 rows (ids 1..n)."""
    rng = random.Random(seed)
    start = dt.date(2015, 1, 1)
    span = (dt.date(2024, 12, 31) - start).days
    rows = []
    for i in range(1, n + 1):
        first, last = rng.choice(FIRST), rng.choice(LAST)
        local = f"{first}.{last.replace(' ', '').replace('-', '')}{i}".lower()
        email = _pad(rng, _messy_case(rng, f"{local}@{rng.choice(DOMAINS)}"))
        d = start + dt.timedelta(days=rng.randrange(span + 1))
        rows.append({
            "id": i,
            "full_name": f"{first} {last}",
            "email": email,
            "signup_date": d.strftime("%m/%d/%Y"),
            "schema_version": 1,
        })
    return rows


def migrate_row(row: dict) -> dict:
    """The schema_version 2 spec, applied to one v1 row (reference implementation for verify.py)."""
    return {
        **row,
        "email": row["email"].strip().lower(),
        "signup_date": dt.datetime.strptime(row["signup_date"].strip(), "%m/%d/%Y").strftime("%Y-%m-%d"),
        "schema_version": 2,
    }


def seed(path: str | Path = "app.db", n: int = 1000, seed: int = 7) -> Path:
    """(Re)create the database at `path` with `n` messy v1 rows. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-journal", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(SCHEMA)
        conn.executemany(
            "INSERT INTO customers (id, full_name, email, signup_date, schema_version) "
            "VALUES (:id, :full_name, :email, :signup_date, :schema_version)",
            generate_rows(n, seed))
        # verify.py regenerates the expected rows from these two numbers
        conn.execute(f"PRAGMA user_version = {int(n)}")
        conn.execute(f"PRAGMA application_id = {int(seed)}")
        conn.commit()
    finally:
        conn.close()
    return path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", default="app.db")
    p.add_argument("--rows", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args(argv)
    out = seed(a.path, a.rows, a.seed)
    print(f"seeded {a.rows} rows into {out}")


if __name__ == "__main__":
    main()
