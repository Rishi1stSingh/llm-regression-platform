from __future__ import annotations

import re
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# Statements that are never allowed in evaluation mode
_DESTRUCTIVE_PATTERN = re.compile(
    r"^\s*(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|REPLACE|GRANT|REVOKE|ATTACH|DETACH)\b",
    re.IGNORECASE,
)


class SQLExecutor(ABC):
    """Abstraction for executing SQL against an evaluation database."""

    @abstractmethod
    def execute(self, sql: str) -> list[tuple[Any, ...]]:
        """Execute a read-only SQL statement and return the result rows."""
        raise NotImplementedError


class SQLiteExecutor(SQLExecutor):
    """Execute read-only SQL against a SQLite evaluation database.

    A new connection is opened for each execution so the connection is always
    created and used in the same thread. This avoids SQLite errors when the
    executor is called from worker threads (e.g. asyncio.to_thread).
    """

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)

    def execute(self, sql: str) -> list[tuple[Any, ...]]:
        self._validate_sql(sql)
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row
        try:
            cursor = connection.execute(sql)
            rows = cursor.fetchall()
            return [tuple(row) for row in rows]
        except sqlite3.Error as exc:
            raise ValueError(f"SQL execution failed: {exc}") from exc
        finally:
            connection.close()

    def _validate_sql(self, sql: str) -> None:
        stripped = sql.strip().rstrip(";").strip()
        if not stripped:
            raise ValueError("Empty SQL statement")
        if _DESTRUCTIVE_PATTERN.match(stripped):
            raise ValueError(f"Destructive SQL statement is not allowed: {sql.strip()[:100]}")
        # Only allow SELECT and WITH (CTE) statements
        if not re.match(r"^(SELECT|WITH)\b", stripped, re.IGNORECASE):
            raise ValueError(f"Only read-only SELECT/WITH statements are allowed: {sql.strip()[:100]}")

    def close(self) -> None:
        """No persistent connection to close; kept for API compatibility."""
        return

    def __enter__(self) -> SQLiteExecutor:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class FakeSQLExecutor(SQLExecutor):
    """In-memory executor for unit tests. Returns a fixed result set."""

    def __init__(self, result: list[tuple[Any, ...]] | None = None,
                 error: Exception | None = None) -> None:
        self.result = result if result is not None else []
        self.error = error
        self.executed_sql: list[str] = []

    def execute(self, sql: str) -> list[tuple[Any, ...]]:
        self.executed_sql.append(sql)
        if self.error is not None:
            raise self.error
        return self.result