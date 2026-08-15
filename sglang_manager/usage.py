"""Token usage recording & aggregation (SQLite).

Every proxied request is recorded once it finishes (for ``stream=true`` that
means when the SSE stream has been fully consumed / closed).  Token counts are
taken from the upstream response's OpenAI ``usage`` object:

- plain responses: parsed from the JSON body;
- streaming responses: captured from the final SSE ``data:`` frame (the one
  that carries ``usage``, sent before ``[DONE]``).

Storage is a single SQLite table; queries are trivial aggregates grouped by
model / hour / day.  Recording must never break the request path — all
failures are logged and swallowed.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    epoch REAL NOT NULL,
    model TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    stream INTEGER NOT NULL DEFAULT 0,
    status INTEGER,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    duration_ms REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_usage_epoch ON usage(epoch);
CREATE INDEX IF NOT EXISTS idx_usage_model ON usage(model);
"""

def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def extract_usage_from_json(body: bytes) -> dict | None:
    """Parse an OpenAI-style ``usage`` object out of a JSON response body."""

    if not body:
        return None
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    return usage if isinstance(usage, dict) else None


class SseUsageScanner:
    """Incrementally scan an SSE byte stream for the final ``usage`` frame.

    Frames are split on ``\\n\\n`` (which also matches ``\\r\\n\\r\\n``), so
    frames fragmented across TCP/ASGI chunks are handled; only the *last*
    ``data:`` frame carrying a ``usage`` object is kept.  Memory use is
    bounded (only the current partial frame is buffered).
    """

    _MAX_FRAME = 1 << 20  # 1 MiB safety cap for a single pathological frame

    def __init__(self) -> None:
        self._buf = b""
        self.usage: dict | None = None

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk
        while True:
            idx = self._buf.find(b"\n\n")
            if idx < 0:
                break
            frame, self._buf = self._buf[:idx], self._buf[idx + 2:]
            self._scan(frame)
        if len(self._buf) > self._MAX_FRAME:
            self._buf = self._buf[-self._MAX_FRAME:]

    def finish(self) -> dict | None:
        """Flush the trailing partial frame (stream ended mid-frame)."""

        if self._buf.strip():
            self._scan(self._buf)
            self._buf = b""
        return self.usage

    def _scan(self, frame: bytes) -> None:
        for line in frame.split(b"\n"):
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            if isinstance(obj, dict):
                u = obj.get("usage")
                if isinstance(u, dict):
                    self.usage = u


class UsageRecorder:
    """Single-connection SQLite writer + aggregator. Thread-safe via a lock."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        if db_path != ":memory:":
            parent = Path(db_path).parent
            if str(parent) not in ("", "."):
                parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def record(
        self,
        *,
        model: str,
        endpoint: str,
        stream: bool = False,
        status: int | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        duration_ms: float = 0.0,
        ts_epoch: float | None = None,
    ) -> None:
        epoch = ts_epoch if ts_epoch is not None else time.time()
        total = prompt_tokens + completion_tokens
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO usage (ts, epoch, model, endpoint, stream, status,"
                    " prompt_tokens, completion_tokens, total_tokens, duration_ms)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        _iso(epoch), epoch, model, endpoint, 1 if stream else 0,
                        status, int(prompt_tokens), int(completion_tokens), int(total),
                        float(duration_ms),
                    ),
                )
                self._conn.commit()
        except Exception:  # noqa: BLE001 - never break the request path
            logger.exception("failed to record usage for %s", model)

    # ------------------------------------------------------------ queries

    def query(
        self,
        *,
        model: str | None = None,
        since: float | None = None,
        until: float | None = None,
        group_by: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        where, params = [], []
        if model:
            where.append("model = ?")
            params.append(model)
        if since is not None:
            where.append("epoch >= ?")
            params.append(since)
        if until is not None:
            where.append("epoch <= ?")
            params.append(until)
        clause = (" WHERE " + " AND ".join(where)) if where else ""

        with self._lock:
            summary_row = self._conn.execute(
                f"SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0),"
                f" COALESCE(SUM(completion_tokens),0), COALESCE(SUM(total_tokens),0),"
                f" COALESCE(AVG(duration_ms),0) FROM usage{clause}",
                params,
            ).fetchone()

            by_model = self._conn.execute(
                f"SELECT model, COUNT(*), COALESCE(SUM(prompt_tokens),0),"
                f" COALESCE(SUM(completion_tokens),0), COALESCE(SUM(total_tokens),0)"
                f" FROM usage{clause} GROUP BY model ORDER BY 5 DESC",
                params,
            ).fetchall()

            series = []
            if group_by in ("hour", "day"):
                width = 3600 if group_by == "hour" else 86400
                rows = self._conn.execute(
                    f"SELECT CAST(epoch / {width} AS INTEGER) * {width}, COUNT(*),"
                    f" COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0)"
                    f" FROM usage{clause} GROUP BY 1 ORDER BY 1",
                    params,
                ).fetchall()
                series = [
                    {
                        "bucket_epoch": r[0],
                        "bucket": _iso(r[0]),
                        "requests": r[1],
                        "prompt_tokens": r[2],
                        "completion_tokens": r[3],
                    }
                    for r in rows
                ]

            recent = self._conn.execute(
                f"SELECT id, ts, model, endpoint, stream, status, prompt_tokens,"
                f" completion_tokens, total_tokens, duration_ms FROM usage{clause}"
                f" ORDER BY id DESC LIMIT ?",
                [*params, int(limit)],
            ).fetchall()

        return {
            "summary": {
                "requests": summary_row[0],
                "prompt_tokens": summary_row[1],
                "completion_tokens": summary_row[2],
                "total_tokens": summary_row[3],
                "avg_duration_ms": round(summary_row[4], 1),
            },
            "by_model": [
                {
                    "model": r[0],
                    "requests": r[1],
                    "prompt_tokens": r[2],
                    "completion_tokens": r[3],
                    "total_tokens": r[4],
                }
                for r in by_model
            ],
            "series": series,
            "recent": [
                {
                    "id": r[0],
                    "ts": r[1],
                    "model": r[2],
                    "endpoint": r[3],
                    "stream": bool(r[4]),
                    "status": r[5],
                    "prompt_tokens": r[6],
                    "completion_tokens": r[7],
                    "total_tokens": r[8],
                    "duration_ms": r[9],
                }
                for r in recent
            ],
        }
