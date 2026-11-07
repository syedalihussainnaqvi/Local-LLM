"""SQLite storage layer for scraped web documents, chunks, and crawl history."""

import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from . import config

logger = logging.getLogger("storage")


class StorageManager:
    """Manages SQLite persistence for sources, chunks, and audit logs with robust connection management."""

    def __init__(self, db_path: Path | str = config.DB_PATH):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager guaranteeing connection commit on success and close in finally block."""
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        with self._lock, self._connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sources (
                    url TEXT PRIMARY KEY,
                    canonical_url TEXT,
                    title TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    http_status INTEGER DEFAULT 200,
                    char_count INTEGER DEFAULT 0,
                    version INTEGER DEFAULT 1,
                    raw_text TEXT
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    section TEXT,
                    content TEXT NOT NULL,
                    chunk_hash TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    FOREIGN KEY (source_url) REFERENCES sources(url) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS crawl_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT,
                    timestamp TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_url);
                CREATE INDEX IF NOT EXISTS idx_sources_hash ON sources(content_hash);
            """)

    def upsert_document(
        self,
        url: str,
        title: str,
        text: str,
        content_hash: str,
        retrieved_at: str,
        chunks: List[Dict[str, Any]],
        canonical_url: str = "",
        http_status: int = 200,
    ) -> Tuple[bool, str]:
        """
        Insert or update a document and its chunks.
        Returns (is_new_or_updated: bool, message: str).
        Skips re-indexing if content hash is identical (incremental update).
        """
        with self._lock, self._connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT content_hash, version, status FROM sources WHERE url = ?", (url,))
            existing = cur.fetchone()

            now_iso = retrieved_at or datetime.now(timezone.utc).isoformat()

            if existing:
                old_hash, old_version, _ = existing["content_hash"], existing["version"], existing["status"]
                if old_hash == content_hash:
                    cur.execute(
                        """
                        UPDATE sources
                        SET retrieved_at = ?, status = 'active', error = NULL, http_status = ?
                        WHERE url = ?
                        """,
                        (now_iso, http_status, url),
                    )
                    self._record_history_cursor(
                        cur, url, "refresh", "skipped", f"Document unchanged (version {old_version})", now_iso
                    )
                    return False, f"Document unchanged (version {old_version})"

                new_version = old_version + 1
                cur.execute(
                    """
                    UPDATE sources
                    SET canonical_url = ?, title = ?, content_hash = ?, retrieved_at = ?,
                        status = 'active', error = NULL, http_status = ?, char_count = ?,
                        version = ?, raw_text = ?
                    WHERE url = ?
                    """,
                    (canonical_url or url, title, content_hash, now_iso, http_status, len(text), new_version, text, url),
                )
                cur.execute("DELETE FROM chunks WHERE source_url = ?", (url,))
                msg = f"Document updated to version {new_version}"
            else:
                new_version = 1
                cur.execute(
                    """
                    INSERT INTO sources (url, canonical_url, title, content_hash, retrieved_at,
                                         status, error, http_status, char_count, version, raw_text)
                    VALUES (?, ?, ?, ?, ?, 'active', NULL, ?, ?, ?, ?)
                    """,
                    (url, canonical_url or url, title, content_hash, now_iso, http_status, len(text), new_version, text),
                )
                msg = "New document inserted (version 1)"

            for ch in chunks:
                cur.execute(
                    """
                    INSERT INTO chunks (chunk_id, source_url, title, section, content, chunk_hash, retrieved_at, char_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ch["chunk_id"],
                        url,
                        ch.get("title", title),
                        ch.get("section", ""),
                        ch["content"],
                        ch["chunk_hash"],
                        ch.get("retrieved_at", now_iso),
                        ch.get("char_count", len(ch["content"])),
                    ),
                )

            self._record_history_cursor(cur, url, "upsert", "success", msg, now_iso)
            return True, msg

    def record_error(self, url: str, error_message: str, http_status: int = 0) -> None:
        """Record an error for a URL without destroying previous data if it existed."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT url FROM sources WHERE url = ?", (url,))
            exists = cur.fetchone()
            if exists:
                cur.execute(
                    "UPDATE sources SET status = 'error', error = ?, http_status = ?, retrieved_at = ? WHERE url = ?",
                    (error_message, http_status, now_iso, url),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO sources (url, canonical_url, title, content_hash, retrieved_at,
                                         status, error, http_status, char_count, version, raw_text)
                    VALUES (?, ?, ?, '', ?, 'error', ?, ?, 0, 1, '')
                    """,
                    (url, url, url, now_iso, error_message, http_status),
                )
            self._record_history_cursor(cur, url, "scrape_error", "failure", error_message, now_iso)

    def _record_history_cursor(
        self, cur: sqlite3.Cursor, url: str, action: str, status: str, details: str, timestamp: str
    ) -> None:
        cur.execute(
            """
            INSERT INTO crawl_history (url, action, status, details, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (url, action, status, details, timestamp),
        )

    def record_history(self, url: str, action: str, status: str, details: str) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection() as conn:
            self._record_history_cursor(conn.cursor(), url, action, status, details, now_iso)

    def get_source(self, url: str) -> Optional[Dict[str, Any]]:
        """Retrieve single source metadata."""
        with self._lock, self._connection() as conn:
            row = conn.execute("SELECT * FROM sources WHERE url = ?", (url,)).fetchone()
            return dict(row) if row else None

    def get_all_sources(self) -> List[Dict[str, Any]]:
        """Retrieve all registered sources ordered by last retrieval."""
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                """
                SELECT s.url, s.canonical_url, s.title, s.content_hash, s.retrieved_at,
                       s.status, s.error, s.http_status, s.char_count, s.version,
                       COUNT(c.chunk_id) AS chunk_count
                FROM sources s
                LEFT JOIN chunks c ON s.url = c.source_url
                GROUP BY s.url
                ORDER BY s.retrieved_at DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Retrieve all active chunks across all sources."""
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                """
                SELECT c.chunk_id, c.source_url, c.title, c.section, c.content,
                       c.chunk_hash, c.retrieved_at, c.char_count
                FROM chunks c
                JOIN sources s ON c.source_url = s.url
                WHERE s.status = 'active'
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def get_chunks_for_url(self, url: str) -> List[Dict[str, Any]]:
        """Retrieve chunks belonging to a specific URL."""
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE source_url = ? ORDER BY chunk_id ASC", (url,)
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_source(self, url: str) -> bool:
        """Delete a source and cascade delete all its chunks."""
        with self._lock, self._connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM chunks WHERE source_url = ?", (url,))
            cur.execute("DELETE FROM sources WHERE url = ?", (url,))
            now_iso = datetime.now(timezone.utc).isoformat()
            self._record_history_cursor(cur, url, "delete", "success", "Deleted source and associated chunks", now_iso)
            return cur.rowcount > 0

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent crawl/scrape audit records."""
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM crawl_history ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_stats(self) -> Dict[str, Any]:
        """Summary statistics for knowledge base."""
        with self._lock, self._connection() as conn:
            cur = conn.cursor()
            source_count = cur.execute("SELECT COUNT(*) FROM sources WHERE status = 'active'").fetchone()[0]
            chunk_count = cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            error_count = cur.execute("SELECT COUNT(*) FROM sources WHERE status = 'error'").fetchone()[0]
            last_ts = cur.execute("SELECT MAX(retrieved_at) FROM sources").fetchone()[0]
            return {
                "active_sources": source_count,
                "total_chunks": chunk_count,
                "error_sources": error_count,
                "last_refresh": last_ts or "Never",
            }


# Singleton accessor
_GLOBAL_STORAGE: Optional[StorageManager] = None

def get_storage() -> StorageManager:
    global _GLOBAL_STORAGE
    if _GLOBAL_STORAGE is None:
        _GLOBAL_STORAGE = StorageManager()
    return _GLOBAL_STORAGE
