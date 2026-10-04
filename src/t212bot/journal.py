"""SQLite journal of command runs and the orders they placed, kept under state_dir.

The dashboard reads it to show recent runs. Each call opens its own connection so the
journal can be shared between the CLI and the dashboard's request threads.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    command TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    exit_code INTEGER,
    environment TEXT NOT NULL,
    dry_run INTEGER NOT NULL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES runs(id),
    created_at TEXT NOT NULL,
    environment TEXT NOT NULL,
    dry_run INTEGER NOT NULL,
    kind TEXT NOT NULL,
    ticker TEXT,
    quantity REAL,
    payload TEXT NOT NULL,
    order_id TEXT,
    result TEXT
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"dry_run": True, **dataclasses.asdict(value)}
    return value


class Journal:
    def __init__(self, path: Path | str):
        self.path = Path(path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            conn.executescript(SCHEMA)
            with conn:
                yield conn
        finally:
            conn.close()

    def start_run(self, command: str, environment: str, dry_run: bool) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO runs (command, started_at, environment, dry_run) VALUES (?, ?, ?, ?)",
                (command, _now(), environment, int(dry_run)),
            )
            return int(cur.lastrowid or 0)

    def finish_run(self, run_id: int, exit_code: int, error: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET finished_at = ?, exit_code = ?, error = ? WHERE id = ?",
                (_now(), exit_code, error, run_id),
            )

    def record_order(
        self,
        run_id: int | None,
        environment: str,
        dry_run: bool,
        kind: str,
        payload: dict[str, Any],
        result: Any,
    ) -> None:
        result = _jsonable(result)
        order_id = result.get("id") if isinstance(result, dict) else None
        if kind == "cancel":
            order_id = payload.get("orderId")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO orders (run_id, created_at, environment, dry_run, kind, ticker,"
                " quantity, payload, order_id, result) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    _now(),
                    environment,
                    int(dry_run),
                    kind,
                    payload.get("ticker"),
                    payload.get("quantity"),
                    json.dumps(payload),
                    None if order_id is None else str(order_id),
                    json.dumps(result, default=str),
                ),
            )

    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in rows]

    def recent_orders(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, run_id, created_at, environment, dry_run, kind, ticker, quantity,"
                " order_id FROM orders ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in rows]
