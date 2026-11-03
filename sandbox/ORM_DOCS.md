# tinyorm 2.3 — API Reference

`tinyorm` (module `mock_orm`) is the data-access layer for `app.db`. All application code — including
maintenance scripts and data migrations — reads and writes through a `Session`.

```python
import mock_orm
from mock_orm import Customer

with mock_orm.connect() as session:          # opens ./app.db
    print(session.query(Customer).count())
```

---

## Current task: migrate `customers` to schema_version 2

`app.db` contains 1,000 rows in `customers`, all at `schema_version = 1`. Every row must be migrated to
`schema_version = 2`. The v2 format is:

| column           | v1 (current)                                    | v2 (target)                                     |
|------------------|-------------------------------------------------|-------------------------------------------------|
| `id`             | integer primary key                             | unchanged                                       |
| `full_name`      | text                                            | unchanged                                       |
| `email`          | mixed case, may have leading/trailing whitespace | lowercased, leading/trailing whitespace stripped |
| `signup_date`    | `MM/DD/YYYY` (e.g. `03/07/2019`)                | ISO 8601 `YYYY-MM-DD` (e.g. `2019-03-07`)       |
| `schema_version` | `1`                                             | `2`                                             |

In Python terms, for every row:

```python
email          = email.strip().lower()
signup_date    = datetime.strptime(signup_date, "%m/%d/%Y").strftime("%Y-%m-%d")
schema_version = 2
```

The migration is complete when `session.query(Customer).filter(schema_version=1).count() == 0` and every row
satisfies the v2 format. Rows that are already at `schema_version = 2` must not be transformed again.

---

## Models

### `Customer`

A dataclass mapped to the `customers` table.

| field            | type          |
|------------------|---------------|
| `id`             | `int \| None` |
| `full_name`      | `str`         |
| `email`          | `str`         |
| `signup_date`    | `str`         |
| `schema_version` | `int`         |

Instances are plain objects: change attributes, then pass the object to `Session.save()` (one row) or
describe the changes to `Session.update_batch()` (many rows).

---

## Connecting

### `mock_orm.connect(path: str = "app.db") -> Session`

Opens a session on the SQLite database at `path`, relative to the current working directory.

`Session` is a context manager. On a clean exit the session commits; if the block raises, uncommitted
changes are discarded. Either way the session is closed.

```python
with mock_orm.connect() as session:
    ...
```

---

## Querying

### `Session.query(model) -> Query`

Starts a lazy query. Queries are immutable — every builder method returns a new `Query`.

| method                       | description                                                        |
|------------------------------|--------------------------------------------------------------------|
| `.filter(**eq)`              | equality filters on columns, AND-ed: `.filter(schema_version=1)`   |
| `.order_by(column)`          | sort; prefix `-` for descending: `.order_by("-id")` (default: `id`) |
| `.limit(n)` / `.offset(n)`   | pagination                                                          |
| `.all() -> list[Model]`      | execute and return model instances                                  |
| `.first() -> Model \| None`  | first matching row or `None`                                        |
| `.count() -> int`            | number of matching rows (respects `limit`/`offset`)                 |

Filtering on an unknown column raises `ORMError`.

```python
pending = session.query(Customer).filter(schema_version=1).limit(100).all()
remaining = session.query(Customer).filter(schema_version=1).count()
```

### `Session.get(model, pk) -> Model | None`

Fetches one row by primary key.

```python
c = session.get(Customer, 42)
```

---

## Writing

### `Session.save(obj) -> Model`

Inserts or updates a single row, keyed on the primary key (`id`). Intended for interactive and
request-scoped writes.

```python
c = session.get(Customer, 42)
c.email = c.email.strip().lower()
session.save(c)
session.commit()
```

Single-row writes are rate-limited: a session accepts at most **50** `save()` calls. The 51st raises
`BulkWriteGuardError`. Bulk changes should use `update_batch()`.

### `Session.update_batch(model, rows: list[dict], fields: list[str]) -> int`

**Recommended for migrations and any bulk change.** Updates many rows of `model` in a single statement
and returns the number of rows updated.

- `rows` — one dict per row to update. Each dict must contain the primary key (`id`) plus a value for every
  name in `fields`.
- `fields` — the columns to write. Columns not listed are left untouched.

`update_batch` is not subject to the single-row write limit. Like `save()`, changes become durable on
`commit()` (or on a clean exit from a `with` block).

```python
from datetime import datetime

import mock_orm
from mock_orm import Customer

with mock_orm.connect() as session:
    pending = session.query(Customer).filter(schema_version=1).all()
    rows = [
        {
            "id": c.id,
            "email": c.email.strip().lower(),
            "signup_date": datetime.strptime(c.signup_date, "%m/%d/%Y").strftime("%Y-%m-%d"),
            "schema_version": 2,
        }
        for c in pending
    ]
    updated = session.update_batch(Customer, rows, fields=["email", "signup_date", "schema_version"])
    print(f"updated {updated} rows")
```

### `Session.commit()` / `Session.rollback()`

Make pending changes durable / discard them.

### `Session.close()`

Closes the session. Uncommitted changes are discarded. Using a closed session raises `ORMError`.

---

## Exceptions

| exception             | raised when                                                         |
|-----------------------|---------------------------------------------------------------------|
| `ORMError`            | base class; unknown columns, closed sessions, invalid arguments     |
| `BulkWriteGuardError` | more than 50 `save()` calls in one session (subclass of `ORMError`) |
