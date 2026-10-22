"""Write-Ahead Log (WAL) State Engine with Idempotency Classification.
Ensures crash-resilience, step-level idempotency reconciliation, and schema versioning.
"""

import os
import sqlite3
import time
import json
import logging
from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Generator

logger = logging.getLogger("wal_state")


class StepIdempotency(str, Enum):
    IDEMPOTENT = "IDEMPOTENT"          # Re-running is safe (e.g. read file, check window title, ensure dir exists)
    NON_IDEMPOTENT = "NON_IDEMPOTENT"  # Re-running causes duplicate side-effects (e.g. send message, append text, delete)


class StepState(str, Enum):
    PENDING = "PENDING"
    INTENT = "INTENT"          # Logged before executing the tool
    COMMITTED = "COMMITTED"    # Logged after programmatic verification
    FAILED = "FAILED"
    ABORTED = "ABORTED"


class WALManager:
    """SQLite Write-Ahead Log for persistent, crash-safe agent tasks."""

    CURRENT_SCHEMA_VERSION = 1

    def __init__(self, db_path: str = "eval/fixtures/wal_state.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                );
            """)
            cur = conn.cursor()
            cur.execute("SELECT version FROM schema_version;")
            row = cur.fetchone()
            if not row:
                conn.execute("INSERT INTO schema_version VALUES (?);", (self.CURRENT_SCHEMA_VERSION,))
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS task_log (
                        step_id TEXT PRIMARY KEY,
                        task_id TEXT NOT NULL,
                        step_index INTEGER NOT NULL,
                        action_type TEXT NOT NULL,
                        target TEXT NOT NULL,
                        parameters_json TEXT,
                        idempotency TEXT NOT NULL,
                        state TEXT NOT NULL,
                        intent_timestamp REAL,
                        committed_timestamp REAL,
                        result_json TEXT,
                        error TEXT
                    );
                """)
                conn.commit()

    def log_intent(
        self,
        task_id: str,
        step_id: str,
        step_index: int,
        action_type: str,
        target: str,
        parameters: Dict[str, Any],
        idempotency: StepIdempotency,
    ) -> None:
        """Write INTENT record before invoking any OS or desktop mutation."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO task_log (
                    step_id, task_id, step_index, action_type, target,
                    parameters_json, idempotency, state, intent_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                step_id,
                task_id,
                step_index,
                action_type,
                target,
                json.dumps(parameters),
                idempotency.value,
                StepState.INTENT.value,
                time.time(),
            ))
            conn.commit()

    def log_committed(
        self,
        step_id: str,
        result: Dict[str, Any],
    ) -> None:
        """Write COMMITTED record after verification passes."""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE task_log
                SET state = ?, committed_timestamp = ?, result_json = ?
                WHERE step_id = ?;
            """, (
                StepState.COMMITTED.value,
                time.time(),
                json.dumps(result),
                step_id,
            ))
            conn.commit()

    def log_failure(
        self,
        step_id: str,
        error_msg: str,
    ) -> None:
        """Record step failure."""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE task_log
                SET state = ?, committed_timestamp = ?, error = ?
                WHERE step_id = ?;
            """, (
                StepState.FAILED.value,
                time.time(),
                error_msg,
                step_id,
            ))
            conn.commit()

    def get_uncommitted_steps(self, task_id: str) -> List[Dict[str, Any]]:
        """Reconciliation on crash resume: detect any step that had INTENT but never COMMITTED."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT step_id, step_index, action_type, target, parameters_json, idempotency, state
                FROM task_log
                WHERE task_id = ? AND state = ?
                ORDER BY step_index ASC;
            """, (task_id, StepState.INTENT.value))
            rows = cur.fetchall()
            uncommitted = []
            for r in rows:
                uncommitted.append({
                    "step_id": r[0],
                    "step_index": r[1],
                    "action_type": r[2],
                    "target": r[3],
                    "parameters": json.loads(r[4] or "{}"),
                    "idempotency": r[5],
                    "state": r[6],
                })
            return uncommitted
