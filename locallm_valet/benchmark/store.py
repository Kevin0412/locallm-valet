# -*- coding: utf-8 -*-
"""SQLite-backed benchmark result store.

The benchmark page previously globbed ``benchmark_results/*.jsonl`` files to
aggregate per-dataset/per-model scores.  This module replaces that with a
single queryable SQLite database so the benchmark is "maintained" (persisted,
queried, deduplicated) in one place:

* ``benchmark_result`` — one row per ``(model_name, dataset, item_id, thinking)``
  result, deduplicated on that key so a re-run of the same question in the same
  mode replaces the old row instead of multiplying it.
* ``benchmark_speed``  — per-model single-request throughput (prefill/decode).

New results from a benchmark job are written here (see ``job.py``).  Existing
``*.jsonl`` results are backfilled via :func:`backfill_jsonl` (exposed on the
CLI as ``benchmark backfill``).  The JSONL files remain as historical artifacts;
the SQLite database is the source of truth for the dashboard.

The database path mirrors ``usage.db``: it defaults to ``data/benchmark.db``
(relative to the process CWD, i.e. the project root when launched with
``python -m locallm_valet --config config.yaml``) and can be overridden via the
``benchmark.db_path`` config key.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("locallm_valet.benchmark.store")

# Default location, relative to the process CWD (project root when launched via
# ``python -m locallm_valet --config config.yaml``).  Matches ``usage.db``.
_DEFAULT_DB = "data/benchmark.db"


_RUN_SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmark_result (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    dataset TEXT NOT NULL,
    item_id TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    thinking INTEGER NOT NULL DEFAULT 0,
    question TEXT NOT NULL DEFAULT '',
    system TEXT NOT NULL DEFAULT '',
    ground_truth TEXT NOT NULL DEFAULT '',
    choices TEXT NOT NULL DEFAULT '[]',
    raw_response TEXT NOT NULL DEFAULT '',
    extracted_answer TEXT,
    is_correct INTEGER,
    score_detail TEXT NOT NULL DEFAULT '',
    ttft_ms REAL,
    decode_ms REAL,
    tps REAL,
    latency_ms REAL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    tool_calls TEXT,
    run_ts TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'job'
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_benchmark_result
    ON benchmark_result(model_name, dataset, item_id, thinking);
CREATE INDEX IF NOT EXISTS idx_benchmark_model ON benchmark_result(model_name);
CREATE INDEX IF NOT EXISTS idx_benchmark_dataset ON benchmark_result(dataset);
"""

_SPEED_SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmark_speed (
    model_name TEXT PRIMARY KEY,
    prefill_tps REAL,
    decode_tps REAL,
    samples INTEGER,
    steady INTEGER,
    tps REAL,
    ts TEXT
);
"""

_JOB_SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmark_job (
    id TEXT PRIMARY KEY,
    dataset TEXT NOT NULL,
    models TEXT NOT NULL DEFAULT '[]',
    sample INTEGER,
    concurrency INTEGER NOT NULL DEFAULT 1,
    max_tokens INTEGER NOT NULL DEFAULT 256,
    enable_thinking INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running',
    total_items INTEGER NOT NULL DEFAULT 0,
    done_items INTEGER NOT NULL DEFAULT 0,
    current_model TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS benchmark_job_item (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    item_id TEXT NOT NULL,
    ord INTEGER NOT NULL DEFAULT 0,
    category TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL DEFAULT '',
    system TEXT NOT NULL DEFAULT '',
    ground_truth TEXT NOT NULL DEFAULT '',
    choices TEXT NOT NULL DEFAULT '[]',
    meta TEXT NOT NULL DEFAULT '{}',
    thinking INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    is_correct INTEGER,
    raw_response TEXT NOT NULL DEFAULT '',
    extracted_answer TEXT,
    score_detail TEXT NOT NULL DEFAULT '',
    latency_ms REAL,
    tps REAL,
    ttft_ms REAL,
    decode_ms REAL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_job_item ON benchmark_job_item(job_id, model_name, status);
"""

# Columns shared by every INSERT into benchmark_result.
_RESULT_COLUMNS = (
    "model_name, dataset, item_id, category, thinking, question, system,"
    " ground_truth, choices, raw_response, extracted_answer, is_correct,"
    " score_detail, ttft_ms, decode_ms, tps, latency_ms, prompt_tokens,"
    " completion_tokens, reasoning_tokens, cached_tokens, tool_calls,"
    " run_ts, source"
)
_RESULT_PLACEHOLDERS = (
    ":model_name, :dataset, :item_id, :category, :thinking, :question,"
    " :system, :ground_truth, :choices, :raw_response, :extracted_answer,"
    " :is_correct, :score_detail, :ttft_ms, :decode_ms, :tps, :latency_ms,"
    " :prompt_tokens, :completion_tokens, :reasoning_tokens, :cached_tokens,"
    " :tool_calls, :run_ts, :source"
)
_RESULT_UPDATE = """
    ON CONFLICT(model_name, dataset, item_id, thinking) DO UPDATE SET
        category=excluded.category, question=excluded.question,
        system=excluded.system, ground_truth=excluded.ground_truth,
        choices=excluded.choices, raw_response=excluded.raw_response,
        extracted_answer=excluded.extracted_answer,
        is_correct=excluded.is_correct, score_detail=excluded.score_detail,
        ttft_ms=excluded.ttft_ms, decode_ms=excluded.decode_ms,
        tps=excluded.tps, latency_ms=excluded.latency_ms,
        prompt_tokens=excluded.prompt_tokens,
        completion_tokens=excluded.completion_tokens,
        reasoning_tokens=excluded.reasoning_tokens,
        cached_tokens=excluded.cached_tokens, tool_calls=excluded.tool_calls,
        run_ts=excluded.run_ts, source=excluded.source
"""


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _now_iso() -> str:
    return _iso(_time_time())


def _time_time() -> float:
    import time as _t
    return _t.time()


class BenchmarkStore:
    """Single-connection SQLite store for benchmark results + speeds.

    Thread-safe via a lock.  Mirrors :class:`locallm_valet.usage.UsageRecorder`
    so the two stores behave (and fail) the same way.
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path) if db_path is not None else _DEFAULT_DB
        self._lock = threading.Lock()
        if self.db_path != ":memory:":
            parent = Path(self.db_path).parent
            if str(parent) not in ("", "."):
                parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_RUN_SCHEMA)
        self._conn.executescript(_SPEED_SCHEMA)
        self._conn.executescript(_JOB_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ write

    def record_results(
        self,
        model_name: str,
        dataset: str,
        results: list[Any],
        *,
        source: str = "job",
        run_ts: Optional[str] = None,
    ) -> int:
        """Insert/upsert a batch of results for ``model_name`` on ``dataset``.

        ``results`` may be ``BenchmarkResult`` objects or JSON ``dict``s (the
        JSONL shape produced by ``BenchmarkResult.to_dict``).  Rows collide on
        ``(model_name, dataset, item_id, thinking)``; on collision the existing
        row is updated so a re-run replaces rather than duplicates.

        Returns the number of results processed.
        """
        if not results:
            return 0
        run_ts = run_ts or _now_iso()
        rows = [_row_from(r, dataset, source=source, run_ts=run_ts) for r in results]
        with self._lock:
            self._conn.executemany(
                _INSERT_SQL, rows,
            )
            self._conn.commit()
        return len(rows)

    def record_speed(self, model_name: str, stats: dict | None) -> None:
        """Persist a model's single-request throughput into SQLite."""
        if not stats:
            return
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO benchmark_speed
                    (model_name, prefill_tps, decode_tps, samples, steady,
                     tps, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_name) DO UPDATE SET
                    prefill_tps=excluded.prefill_tps,
                    decode_tps=excluded.decode_tps,
                    samples=excluded.samples,
                    steady=excluded.steady,
                    tps=excluded.tps,
                    ts=excluded.ts
                """,
                (
                    model_name,
                    stats.get("prefill_tps"),
                    stats.get("decode_tps"),
                    stats.get("samples"),
                    stats.get("steady"),
                    stats.get("tps"),
                    stats.get("ts") or _now_iso(),
                ),
            )
            self._conn.commit()

    # ------------------------------------------------------------------ read

    def load_speeds(self) -> dict[str, dict]:
        """Read persisted speeds: ``{model: {prefill_tps, decode_tps, ts}}``."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT model_name, prefill_tps, decode_tps, samples, steady,"
                "       tps, ts FROM benchmark_speed ORDER BY model_name"
            ).fetchall()
        out: dict[str, dict] = {}
        for model, pre, dec, samples, steady, tps, ts in rows:
            entry: dict[str, Any] = {}
            if pre is not None:
                entry["prefill_tps"] = pre
            if dec is not None:
                entry["decode_tps"] = dec
            if samples is not None:
                entry["samples"] = samples
                if steady is not None:
                    entry["steady"] = steady
            if tps is not None:
                entry["tps"] = tps
            if ts:
                entry["ts"] = ts
            out[model] = entry
        return out

    def query_aggregate(self) -> dict[str, dict[str, dict]]:
        """Return ``{dataset: {model: stats}}`` for the benchmark dashboard.

        ``stats`` is ``{"t", "c", "cat", "lat", "tps", "tok", "thinking"}``
        where ``cat`` is ``{category: [total, correct]}`` and ``thinking`` is
        one of ``"thinking"`` / ``"non-thinking"`` / ``"mixed"`` / ``"-"``.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT model_name, dataset, category, thinking, is_correct,"
                "       latency_ms, tps, completion_tokens"
                " FROM benchmark_result"
            ).fetchall()

        by_ds: dict[str, dict[str, dict]] = defaultdict(dict)
        for model, ds, cat, thinking, is_correct, lat, tps, tok in rows:
            s = by_ds[ds].setdefault(
                model,
                {"t": 0, "c": 0, "cat": defaultdict(lambda: [0, 0]),
                 "lat": [], "tps": [], "tok": [], "thinking": set()},
            )
            correct = None if is_correct is None else bool(is_correct)
            s["t"] += 1
            if correct is True:
                s["c"] += 1
            s["cat"][cat][0] += 1
            if correct is True:
                s["cat"][cat][1] += 1
            if lat:
                s["lat"].append(lat)
            if tps:
                s["tps"].append(tps)
            if tok:
                s["tok"].append(tok)
            s["thinking"].add(bool(thinking))

        # Normalise mutable containers into JSON-serialisable values.
        for models in by_ds.values():
            for s in models.values():
                s["cat"] = {k: list(v) for k, v in s["cat"].items()}
                th = s.pop("thinking")
                if th == {True}:
                    s["thinking"] = "thinking"
                elif th == {False}:
                    s["thinking"] = "non-thinking"
                elif th:
                    s["thinking"] = "mixed"
                else:
                    s["thinking"] = "-"
        return dict(by_ds)

    # ------------------------------------------------------- durable job queue

    def create_job(
        self,
        *,
        job_id: str,
        dataset: str,
        models: list[str],
        items: list[Any],
        sample: Optional[int] = None,
        concurrency: int = 1,
        max_tokens: int = 256,
        enable_thinking: bool = False,
    ) -> int:
        """Register a new job and persist its per-model item queue.

        Returns the number of queued item rows (``len(models) * len(items)``).
        The full question list is stored so a job can be resumed at any time
        purely from this table.
        """
        now = _now_iso()
        total = len(models) * len(items)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO benchmark_job (id, dataset, models,"
                " sample, concurrency, max_tokens, enable_thinking, status,"
                " total_items, done_items, current_model, error, created_at,"
                " updated_at) VALUES (?,?,?,?,?,?,?,?,?,0,'','',?,?)",
                (
                    job_id, dataset, json.dumps(models, ensure_ascii=False),
                    sample, concurrency, max_tokens, 1 if enable_thinking else 0,
                    "running", total, now, now,
                ),
            )
            for model_name in models:
                for ord_i, item in enumerate(items):
                    self._conn.execute(
                        "INSERT OR IGNORE INTO benchmark_job_item (job_id,"
                        " model_name, item_id, ord, category, question, system,"
                        " ground_truth, choices, meta, thinking, status)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?,?,'pending')",
                        _item_to_job_row(job_id, model_name, ord_i, item,
                                         enable_thinking),
                    )
            self._conn.commit()
        return total

    def get_job(self, job_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM benchmark_job WHERE id = ?", (job_id,)
            ).fetchone()
        return _job_row_to_dict(row) if row else None

    def list_jobs(self, limit: int = 50) -> list[dict]:
        """Return persisted jobs, newest first (used to resume after restart)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM benchmark_job ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [_job_row_to_dict(r) for r in rows]

    def update_job(self, job_id: str, **fields: Any) -> None:
        allowed = {"status", "done_items", "current_model", "error", "updated_at"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if "updated_at" not in updates:
            updates["updated_at"] = _now_iso()
        if not updates:
            return
        assignments = ", ".join(f"{k} = ?" for k in updates)
        with self._lock:
            self._conn.execute(
                f"UPDATE benchmark_job SET {assignments} WHERE id = ?",
                [*updates.values(), job_id],
            )
            self._conn.commit()

    def job_items(
        self, job_id: str, model_name: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        """Return job item rows (as dicts) filtered by job/model/status."""
        where = ["job_id = ?"]
        params: list[Any] = [job_id]
        if model_name is not None:
            where.append("model_name = ?")
            params.append(model_name)
        if status is not None:
            where.append("status = ?")
            params.append(status)
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM benchmark_job_item WHERE "
                + " AND ".join(where) + " ORDER BY ord",
                params,
            ).fetchall()
        return [_item_row_to_dict(r) for r in rows]

    def claim_items(
        self, job_id: str, model_name: str, limit: int = 1,
    ) -> list[dict]:
        """Atomically mark up to ``limit`` pending items as in-progress and return them.

        Claims follow ``ord`` order so a resume replays the original sequence.
        """
        now = _now_iso()
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM benchmark_job_item WHERE job_id=? AND"
                " model_name=? AND status='pending' ORDER BY ord LIMIT ?",
                (job_id, model_name, int(limit)),
            ).fetchall()
            ids = [r[0] for r in rows]
            if not ids:
                return []
            self._conn.executemany(
                "UPDATE benchmark_job_item SET status='in_progress',"
                " started_at=? WHERE id=?",
                [(now, i) for i in ids],
            )
            out = self._conn.execute(
                "SELECT * FROM benchmark_job_item WHERE id IN (%s)"
                % ",".join("?" for _ in ids),
                ids,
            ).fetchall()
            self._conn.commit()
        return [_item_row_to_dict(r) for r in out]

    def finish_item(self, job_id: str, job_item_id: int, result: dict) -> None:
        """Mark a job item completed, storing its scored result."""
        from .schema import BenchmarkResult
        now = _now_iso()

        def _int(v) -> int:
            return int(v) if v else 0

        def _num(v):
            try:
                return float(v) if v else None
            except (TypeError, ValueError):
                return None

        mode = result.get("model_name", "")
        with self._lock:
            self._conn.execute(
                "UPDATE benchmark_job_item SET status='done', is_correct=?,"
                " raw_response=?, extracted_answer=?, score_detail=?,"
                " latency_ms=?, tps=?, ttft_ms=?, decode_ms=?, prompt_tokens=?,"
                " completion_tokens=?, reasoning_tokens=?, finished_at=?"
                " WHERE job_id=? AND id=?",
                (
                    result.get("is_correct"), result.get("raw_response", ""),
                    result.get("extracted_answer"), result.get("score_detail", ""),
                    _num(result.get("latency_ms")), _num(result.get("tps")),
                    _num(result.get("ttft_ms")), _num(result.get("decode_ms")),
                    _int(result.get("prompt_tokens")),
                    _int(result.get("completion_tokens")),
                    _int(result.get("reasoning_tokens")), now, job_id, job_item_id,
                ),
            )
            self._conn.execute(
                "UPDATE benchmark_job SET done_items=done_items+1,"
                " updated_at=?, current_model=COALESCE(NULLIF(current_model,''),?)"
                " WHERE id=?",
                (now, mode, job_id),
            )
            self._conn.commit()

    def fail_item(self, job_id: str, job_item_id: int, error: str) -> None:
        now = _now_iso()
        with self._lock:
            self._conn.execute(
                "UPDATE benchmark_job_item SET status='failed',"
                " raw_response=?, finished_at=? WHERE job_id=? AND id=?",
                (f"[ERROR] {error}", now, job_id, job_item_id),
            )
            self._conn.commit()

    def reset_in_progress(self, job_id: str, model_name: str | None = None) -> int:
        """Return any in-progress items for a job back to pending.

        Used on (re)start / resume so items that were claimed by an interrupted
        run (server restart / stop) are re-claimed instead of skipped.
        Returns the number of items reset.
        """
        with self._lock:
            if model_name is None:
                cur = self._conn.execute(
                    "UPDATE benchmark_job_item SET status='pending', started_at=NULL"
                    " WHERE job_id=? AND status='in_progress'",
                    (job_id,),
                )
            else:
                cur = self._conn.execute(
                    "UPDATE benchmark_job_item SET status='pending', started_at=NULL"
                    " WHERE job_id=? AND model_name=? AND status='in_progress'",
                    (job_id, model_name),
                )
            self._conn.commit()
        return cur.rowcount


def _item_to_job_row(job_id: str, model_name: str, ord_i: int, item: Any,
                     enable_thinking: bool) -> tuple:
    """Convert a ``BenchmarkItem`` / dict into a ``benchmark_job_item`` tuple."""
    if hasattr(item, "to_dict"):
        d = item.to_dict()
        meta = getattr(item, "meta", {}) or {}
    else:
        d = item if isinstance(item, dict) else {}
        meta = d.get("meta") or {}
    choices = d.get("choices") or []
    choices_json = json.dumps(choices, ensure_ascii=False) if isinstance(choices, list) else "[]"
    meta_json = json.dumps(meta, ensure_ascii=False) if isinstance(meta, dict) else "{}"
    language = d.get("language") or ""
    return (
        job_id, model_name, str(d.get("item_id", "")), ord_i,
        str(d.get("category", "")), str(d.get("question", "")),
        str(d.get("system", "")), str(d.get("ground_truth", "")),
        choices_json, meta_json, 1 if enable_thinking else 0,
    )


_JOB_COLS = ("id", "dataset", "models", "sample", "concurrency", "max_tokens",
             "enable_thinking", "status", "total_items", "done_items",
             "current_model", "error", "created_at", "updated_at")


def _job_row_to_dict(row) -> dict:
    d = {name: val for name, val in zip(_JOB_COLS, row)}
    try:
        d["models"] = json.loads(d.get("models") or "[]")
    except ValueError:
        d["models"] = []
    d["enable_thinking"] = bool(d.get("enable_thinking"))
    return d


_ITEM_COLS = ("id", "job_id", "model_name", "item_id", "ord", "category",
              "question", "system", "ground_truth", "choices", "meta",
              "thinking", "status", "is_correct", "raw_response",
              "extracted_answer", "score_detail", "latency_ms", "tps",
              "ttft_ms", "decode_ms", "prompt_tokens", "completion_tokens",
              "reasoning_tokens", "started_at", "finished_at")


def _item_row_to_dict(row) -> dict:
    d = {name: val for name, val in zip(_ITEM_COLS, row)}
    for jk in ("choices", "meta"):
        try:
            d[jk] = json.loads(d.get(jk) or ("[]" if jk == "choices" else "{}"))
        except ValueError:
            d[jk] = {} if jk == "meta" else []
    d["thinking"] = bool(d.get("thinking"))
    return d


def item_from_row(d: dict) -> "BenchmarkItem":
    """Rebuild a ``BenchmarkItem`` from a ``benchmark_job_item`` row dict.

    Used by the scheduler to resume a job purely from persisted state.
    """
    from .schema import BenchmarkItem
    question = d.get("question", "")
    return BenchmarkItem(
        item_id=d.get("item_id", ""),
        category=d.get("category", ""),
        question=question,
        ground_truth=d.get("ground_truth", ""),
        choices=d.get("choices", []),
        system=d.get("system", ""),
        language="zh" if any("\u4e00" <= c <= "\u9fff" for c in question) else "en",
        meta=d.get("meta", {}) or {},
    )


def _row_from(r: Any, dataset: str, *, source: str, run_ts: str) -> dict:
    """Convert a ``BenchmarkResult`` or JSON dict into a ``benchmark_result`` row."""
    if hasattr(r, "to_dict"):
        d = r.to_dict()
        item = getattr(r, "item", None)
        system = getattr(item, "system", "") if item is not None else ""
    else:
        d = r if isinstance(r, dict) else {}
        system = d.get("system", "")

    def num(v: Any) -> Optional[float]:
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def intnum(v: Any) -> int:
        if v is None or v == "":
            return 0
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    is_correct = d.get("is_correct")
    if isinstance(is_correct, bool):
        correct_v: Optional[int] = 1 if is_correct else 0
    elif is_correct is None:
        correct_v = None
    else:
        correct_v = 1 if is_correct else 0

    choices = d.get("choices") or []
    choices_json = json.dumps(choices, ensure_ascii=False) if isinstance(choices, list) else "[]"

    tool_calls = d.get("tool_calls")
    tool_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None

    return {
        "model_name": str(d.get("model_name", "")),
        "dataset": dataset,
        "item_id": str(d.get("item_id", "")),
        "category": str(d.get("category", "")),
        "thinking": 1 if d.get("thinking") else 0,
        "question": str(d.get("question", "")),
        "system": str(system),
        "ground_truth": str(d.get("ground_truth", "")),
        "choices": choices_json,
        "raw_response": str(d.get("raw_response", "")),
        "extracted_answer": d.get("extracted_answer"),
        "is_correct": correct_v,
        "score_detail": str(d.get("score_detail", "")),
        "ttft_ms": num(d.get("ttft_ms")),
        "decode_ms": num(d.get("decode_ms")),
        "tps": num(d.get("tps")),
        "latency_ms": num(d.get("latency_ms")),
        "prompt_tokens": intnum(d.get("prompt_tokens")),
        "completion_tokens": intnum(d.get("completion_tokens")),
        "reasoning_tokens": intnum(d.get("reasoning_tokens")),
        "cached_tokens": intnum(d.get("cached_tokens")),
        "tool_calls": tool_json,
        "run_ts": run_ts,
        "source": source,
    }


_INSERT_SQL = (
    "INSERT INTO benchmark_result (" + _RESULT_COLUMNS + ") VALUES ("
    + _RESULT_PLACEHOLDERS + ")" + _RESULT_UPDATE
)


# ---------------------------------------------------------------------------
# JSONL backfill
# ---------------------------------------------------------------------------

def _parse_stem(stem: str, model_name: str) -> tuple[str, str]:
    """Parse ``{model}_{dataset}`` / ``{dataset}`` from a file stem.

    Returns ``(model, dataset)``.  Model names routinely contain underscores
    (e.g. ``Qwen3-1.7B-Q4_K_M``) so we strip the model prefix only when the
    stem actually starts with the record's own ``model_name``; otherwise the
    whole stem is treated as the dataset (legacy naming).
    """
    if stem.startswith(model_name):
        rest = stem[len(model_name):].lstrip("_")
        return model_name, rest
    return model_name, stem


def _first_model(f: Path) -> Optional[str]:
    """Read the first record's ``model_name`` (used only to parse the stem)."""
    try:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                return json.loads(line).get("model_name")
    except (OSError, ValueError):
        return None
    return None


def backfill_jsonl(
    store: BenchmarkStore,
    results_dir: str | Path = "benchmark_results",
) -> dict[str, int]:
    """Backfill every ``*_results.jsonl`` in ``results_dir`` into ``store``.

    Backfill rules (so a partial legacy run never inflates the dashboard):

    * A per-model file ``{model}_{dataset}_results.jsonl`` is the authoritative
      complete run for that ``(model, dataset)``; any legacy
      ``{dataset}_results.jsonl`` records for the same ``(model, dataset)`` are
      skipped because the per-model file supersedes them.
    * ``(model, dataset)`` pairs with no per-model file are taken from the
      legacy file.

    Returns ``{dataset: count}`` of rows written.
    """
    d = Path(results_dir)
    if not d.is_dir():
        return {}

    files = sorted(d.glob("*_results.jsonl"), key=lambda p: p.stat().st_mtime)

    # Split files into authoritative per-model files and legacy files.
    per_model_files: list[Path] = []
    authoritative: set[tuple[str, str]] = set()
    for f in files:
        stem = f.stem[: -len("_results")]
        model = _first_model(f) or ""
        if not model:
            continue
        parsed_model, dataset = _parse_stem(stem, model)
        # A per-model file has a dataset different from the plain stem
        # (i.e. there was a model_ prefix to strip).
        if dataset and dataset != stem:
            per_model_files.append(f)
            authoritative.add((parsed_model, dataset))

    written: dict[str, int] = defaultdict(int)
    for f in files:
        stem = f.stem[: -len("_results")]
        model = _first_model(f) or ""
        if not model:
            continue
        if f in per_model_files:
            parsed_model, dataset = _parse_stem(stem, model)
            skip: set[tuple[str, str]] = set()
        else:
            # Legacy file: the whole stem is the dataset; skip records whose
            # (model, dataset) has an authoritative per-model file.
            dataset = stem
            skip = authoritative
        n = _ingest_file(store, f, dataset, skip_pairs=skip)
        written[dataset] += n
        logger.info("backfilled %s -> %d rows (dataset=%s)", f.name, n, dataset)

    # Backfill per-model single-request throughputs (speeds.json).
    speeds_file = d / "speeds.json"
    if speeds_file.is_file():
        try:
            data = json.loads(speeds_file.read_text("utf-8"))
            for model, stats in data.items():
                if not isinstance(stats, dict):
                    continue
                speed_entry = dict(stats)
                speed_entry.setdefault("ts", _iso(speeds_file.stat().st_mtime))
                store.record_speed(model, speed_entry)
            logger.info("backfilled %d speed entries from %s", len(data), speeds_file.name)
        except Exception:  # noqa: BLE001
            logger.exception("failed to backfill speeds.json")

    return dict(written)


def _ingest_file(
    store: BenchmarkStore,
    f: Path,
    dataset: str,
    skip_pairs: set[tuple[str, str]],
) -> int:
    """Read one JSONL file, skipping records in ``skip_pairs``, and record them."""
    run_ts = _iso(f.stat().st_mtime)
    collected: list[dict] = []
    with open(f, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            model = d.get("model_name") or ""
            if not model:
                continue
            if (model, dataset) in skip_pairs:
                continue
            collected.append(_row_from(d, dataset, source="jsonl_backfill", run_ts=run_ts))
    if not collected:
        return 0
    with store._lock:
        store._conn.executemany(_INSERT_SQL, collected)
        store._conn.commit()
    return len(collected)


# Convenience accessor for callers that want the default store without config.
def open_default_store(db_path: str | Path | None = None) -> BenchmarkStore:
    return BenchmarkStore(db_path)
