"""
Case store (SQLite) and in-process event bus.

- CaseStore persists every completed case so the dashboard survives restarts
  (this is what interfaces.write_agent_log writes to).
- EventBus fans out live events to WebSocket subscribers
  (this is what interfaces.broadcast_ws publishes to).
"""
import os
import json
import sqlite3
import asyncio
import logging
import tempfile
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CaseStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS cases (
                   delivery_id TEXT PRIMARY KEY,
                   status TEXT,
                   completed_at REAL,
                   data TEXT NOT NULL
               )"""
        )
        self._conn.commit()

    def save(self, case: Dict[str, Any]) -> None:
        payload = json.dumps(case, default=str)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cases (delivery_id, status, completed_at, data) VALUES (?, ?, ?, ?)",
                (case.get("delivery_id"), case.get("status"), case.get("completed_at"), payload),
            )
            self._conn.commit()

    def get(self, delivery_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT data FROM cases WHERE delivery_id = ?", (delivery_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, limit: int = 500) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM cases ORDER BY completed_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def clear(self) -> int:
        with self._lock:
            count = self._conn.execute("DELETE FROM cases").rowcount
            self._conn.commit()
        return count

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def stats(self) -> Dict[str, Any]:
        """Aggregate KPIs across all stored cases."""
        cases = self.list(limit=100000)
        totals = {"cases": len(cases), "resolved": 0, "escalated": 0}
        savings = {"time_min": 0.0, "fuel_inr": 0.0, "cost_inr": 0.0}
        by_failure: Dict[str, int] = {}
        by_path: Dict[str, int] = {}
        by_outcome: Dict[str, int] = {}
        durations = []
        for case in cases:
            status = case.get("status")
            if status in totals:
                totals[status] += 1
            for key in savings:
                savings[key] += float((case.get("savings") or {}).get(key, 0) or 0)
            by_failure[case.get("failure_type") or "unknown"] = by_failure.get(case.get("failure_type") or "unknown", 0) + 1
            by_path[case.get("resolution_path") or "none"] = by_path.get(case.get("resolution_path") or "none", 0) + 1
            by_outcome[case.get("final_outcome") or "none"] = by_outcome.get(case.get("final_outcome") or "none", 0) + 1
            if case.get("duration_ms") is not None:
                durations.append(float(case["duration_ms"]))
        durations.sort()
        return {
            "totals": totals,
            "resolution_rate": round(totals["resolved"] / totals["cases"], 3) if totals["cases"] else 0.0,
            "savings": {k: round(v, 1) for k, v in savings.items()},
            "by_failure_type": by_failure,
            "by_resolution_path": by_path,
            "by_outcome": by_outcome,
            "avg_case_ms": round(sum(durations) / len(durations), 1) if durations else 0.0,
            "p95_case_ms": durations[min(len(durations) - 1, round(0.95 * (len(durations) - 1)))] if durations else 0.0,
        }


class EventBus:
    """Fan-out of live events to any number of async subscribers (WebSocket clients)."""

    def __init__(self, max_queue: int = 500):
        self._subscribers: List[asyncio.Queue] = []
        self._max_queue = max_queue

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    async def publish(self, event: Dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass  # a slow client drops events rather than stalling the agent


_store: Optional[CaseStore] = None
_bus: Optional[EventBus] = None
_singleton_lock = threading.Lock()


def running_on_vercel() -> bool:
    """True inside a Vercel function (its Python handler sets __VC_HANDLER_ENTRYPOINT even when
    the project does not expose Vercel's system environment variables)."""
    return bool(os.getenv("VERCEL") or os.getenv("__VC_HANDLER_ENTRYPOINT"))


def default_db_path() -> str:
    """venlix_agent.db locally; on Vercel only /tmp is writable (and it is per instance)."""
    if running_on_vercel():
        return os.path.join(tempfile.gettempdir(), "venlix_agent.db")
    return "venlix_agent.db"


def _open_store() -> CaseStore:
    path = os.getenv("VENLIX_DB_PATH") or default_db_path()
    try:
        return CaseStore(path)
    except sqlite3.OperationalError:
        # A read-only deployment directory (serverless hosts) cannot hold the database file.
        fallback = os.path.join(tempfile.gettempdir(), "venlix_agent.db")
        if os.path.abspath(path) == os.path.abspath(fallback):
            raise
        logger.warning("Cannot open case store at %s; using %s instead", path, fallback)
        return CaseStore(fallback)


def get_store() -> CaseStore:
    global _store
    with _singleton_lock:
        if _store is None:
            _store = _open_store()
        return _store


def get_event_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


def reset_singletons() -> None:
    """Used by tests to point the store at a fresh database."""
    global _store, _bus
    with _singleton_lock:
        if _store is not None:
            _store.close()
        _store = None
        _bus = None
