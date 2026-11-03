"""tinyorm: a small session-based ORM over SQLite (standard library only).

See ORM_DOCS.md for the full API reference.

    import mock_orm
    with mock_orm.connect() as session:
        c = session.query(mock_orm.Customer).filter(schema_version=1).first()
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Generic, TypeVar

__all__ = [
    "BulkWriteGuardError",
    "Customer",
    "Model",
    "ORMError",
    "Query",
    "Session",
    "connect",
]

__version__ = "2.3.1"

DEFAULT_PATH = "app.db"
SINGLE_ROW_WRITE_LIMIT = 50  # save() calls allowed per session


class ORMError(Exception):
    """Base class for all tinyorm errors."""


class BulkWriteGuardError(ORMError):
    """Raised when a session issues too many single-row writes."""


@dataclass
class Model:
    __tablename__: ClassVar[str] = ""
    __primary_key__: ClassVar[str] = "id"

    @classmethod
    def _columns(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def _values(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self._columns()}


@dataclass
class Customer(Model):
    __tablename__: ClassVar[str] = "customers"

    id: int | None = None
    full_name: str = ""
    email: str = ""
    signup_date: str = ""
    schema_version: int = 1


M = TypeVar("M", bound=Model)


def _check_columns(model: type[Model], names: Any) -> None:
    unknown = [n for n in names if n not in model._columns()]
    if unknown:
        raise ORMError(f"{model.__name__} has no column(s): {', '.join(unknown)}")


class Query(Generic[M]):
    """Lazy, immutable-ish query builder. Build with Session.query(Model)."""

    def __init__(self, session: Session, model: type[M]):
        self._session = session
        self._model = model
        self._filters: dict[str, Any] = {}
        self._order: list[tuple[str, bool]] = []
        self._limit: int | None = None
        self._offset: int = 0

    def _clone(self) -> Query[M]:
        q: Query[M] = Query(self._session, self._model)
        q._filters = dict(self._filters)
        q._order = list(self._order)
        q._limit = self._limit
        q._offset = self._offset
        return q

    def filter(self, **eq: Any) -> Query[M]:
        """Equality filters, AND-ed together: .filter(schema_version=1, id=3)."""
        _check_columns(self._model, eq)
        q = self._clone()
        q._filters.update(eq)
        return q

    def order_by(self, column: str) -> Query[M]:
        """Order by a column; prefix with '-' for descending: .order_by('-id')."""
        desc = column.startswith("-")
        name = column.lstrip("-")
        _check_columns(self._model, [name])
        q = self._clone()
        q._order.append((name, desc))
        return q

    def limit(self, n: int) -> Query[M]:
        if n < 0:
            raise ORMError("limit must be >= 0")
        q = self._clone()
        q._limit = int(n)
        return q

    def offset(self, n: int) -> Query[M]:
        if n < 0:
            raise ORMError("offset must be >= 0")
        q = self._clone()
        q._offset = int(n)
        return q

    def _where(self) -> tuple[str, list[Any]]:
        if not self._filters:
            return "", []
        clause = " AND ".join(f'"{k}" = ?' for k in self._filters)
        return f" WHERE {clause}", list(self._filters.values())

    def all(self) -> list[M]:
        cols = self._model._columns()
        where, params = self._where()
        sql = f'SELECT {", ".join(cols)} FROM "{self._model.__tablename__}"{where}'
        if self._order:
            sql += " ORDER BY " + ", ".join(f'"{c}" {"DESC" if d else "ASC"}' for c, d in self._order)
        else:
            sql += f' ORDER BY "{self._model.__primary_key__}" ASC'
        if self._limit is not None or self._offset:
            sql += " LIMIT ? OFFSET ?"
            params += [self._limit if self._limit is not None else -1, self._offset]
        rows = self._session._fetch(sql, params)
        return [self._model(**dict(zip(cols, r, strict=True))) for r in rows]

    def first(self) -> M | None:
        rows = self.limit(1).all()
        return rows[0] if rows else None

    def count(self) -> int:
        where, params = self._where()
        sql = f'SELECT COUNT(*) FROM "{self._model.__tablename__}"{where}'
        if self._limit is not None or self._offset:
            sql = f"SELECT COUNT(*) FROM ({sql.replace('COUNT(*)', '1', 1)} LIMIT ? OFFSET ?)"
            params += [self._limit if self._limit is not None else -1, self._offset]
        return int(self._session._fetch(sql, params)[0][0])

    def __iter__(self):
        return iter(self.all())

    def __repr__(self) -> str:
        return f"<Query {self._model.__name__} filters={self._filters} limit={self._limit}>"


class Session:
    """A unit of work over one database connection. Not thread-safe."""

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self._conn = sqlite3.connect(path)
        self._writes = 0
        self._closed = False

    # -- internals ---------------------------------------------------------------------------------
    def _check_open(self) -> None:
        if self._closed:
            raise ORMError("session is closed")

    def _fetch(self, sql: str, params: list[Any]) -> list[tuple]:
        self._check_open()
        return self._conn.execute(sql, params).fetchall()

    # -- public API --------------------------------------------------------------------------------
    def query(self, model: type[M]) -> Query[M]:
        self._check_open()
        return Query(self, model)

    def get(self, model: type[M], pk: Any) -> M | None:
        return self.query(model).filter(**{model.__primary_key__: pk}).first()

    def save(self, obj: Model) -> Model:
        """Insert or update a single row (by primary key). Changes are visible after commit()."""
        self._check_open()
        if not isinstance(obj, Model):
            raise ORMError(f"cannot save {type(obj).__name__}: not a Model instance")
        if self._writes >= SINGLE_ROW_WRITE_LIMIT:
            raise BulkWriteGuardError(
                f"single-row writes are throttled for bulk operations (>{SINGLE_ROW_WRITE_LIMIT} rows/session); "
                "use Session.update_batch")
        model = type(obj)
        table, pk = model.__tablename__, model.__primary_key__
        values = obj._values()
        exists = values[pk] is not None and self._fetch(
            f'SELECT 1 FROM "{table}" WHERE "{pk}" = ?', [values[pk]])
        if exists:
            cols = [c for c in values if c != pk]
            self._conn.execute(
                f'UPDATE "{table}" SET {", ".join(f"{c!s} = ?" for c in cols)} WHERE "{pk}" = ?',
                [values[c] for c in cols] + [values[pk]])
        else:
            cols = [c for c in values if not (c == pk and values[c] is None)]
            cur = self._conn.execute(
                f'INSERT INTO "{table}" ({", ".join(cols)}) VALUES ({", ".join("?" for _ in cols)})',
                [values[c] for c in cols])
            if values[pk] is None:
                setattr(obj, pk, cur.lastrowid)
        self._writes += 1
        return obj

    def commit(self) -> None:
        self._check_open()
        self._conn.commit()

    def rollback(self) -> None:
        self._check_open()
        self._conn.rollback()

    def close(self) -> None:
        """Close the session. Uncommitted changes are discarded."""
        if not self._closed:
            self._conn.rollback()
            self._conn.close()
            self._closed = True

    def __enter__(self) -> Session:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None and not self._closed:
                self.commit()
        finally:
            self.close()

    def __repr__(self) -> str:
        return f"<Session path={self.path!r} closed={self._closed}>"


def connect(path: str = DEFAULT_PATH) -> Session:
    """Open a Session on the SQLite database at `path` (default: app.db in the current directory)."""
    return Session(path)
