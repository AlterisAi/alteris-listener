"""Derived data store — separate SQLite DB for LLM-generated outputs.

All derived data (commitments, tasks, summaries, insights) lives in
~/.alteris/derived.db, separate from the immutable graph.db.
This can be blown away and rebuilt without affecting source data.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DERIVED_DB_PATH = Path.home() / ".alteris" / "derived.db"

DERIVED_SCHEMA = """
-- ══════════════════════════════════════════════════════════════════
-- Commitments (extracted from email threads and meetings)
-- ══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS commitments (
    id              TEXT PRIMARY KEY,           -- deterministic: sha256(thread_id + canonical fields)
    thread_id       TEXT,                       -- source thread (NULL for standalone msgs)
    source_node_ids TEXT NOT NULL,              -- JSON array of node IDs that produced this
    commitment_type TEXT NOT NULL,              -- inbound_request | user_commitment | deadline |
                                               -- waiting_on | follow_up | payment_due
    who             TEXT NOT NULL,              -- "user" or contact name/email
    what            TEXT NOT NULL,              -- what was committed/requested
    to_whom         TEXT,                       -- who is expecting this
    deadline        TEXT,                       -- ISO-8601 or NULL
    status          TEXT NOT NULL DEFAULT 'open',  -- open | done | overdue | cancelled
    priority        INTEGER NOT NULL DEFAULT 3,    -- 1-3
    confidence      REAL NOT NULL DEFAULT 0.5,     -- 0.0-1.0
    provenance      TEXT,                       -- user_committed | assigned_to_user |
                                               -- group_decision | system_detected
    note            TEXT,                       -- human-readable context (markdown)
    raw_extraction  TEXT,                       -- full JSON from LLM for debugging
    prompt_version  TEXT NOT NULL,              -- which prompt produced this
    model_used      TEXT NOT NULL,              -- gemini-flash-3 | qwen3:30b-a3b
    processed_at    INTEGER NOT NULL,           -- when extraction ran
    created_at      INTEGER NOT NULL            -- timestamp of source message
);

CREATE INDEX IF NOT EXISTS idx_commitments_thread ON commitments(thread_id);
CREATE INDEX IF NOT EXISTS idx_commitments_status ON commitments(status);
CREATE INDEX IF NOT EXISTS idx_commitments_type ON commitments(commitment_type);
CREATE INDEX IF NOT EXISTS idx_commitments_deadline ON commitments(deadline) WHERE deadline IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_commitments_who ON commitments(who);
CREATE INDEX IF NOT EXISTS idx_commitments_priority ON commitments(priority, status);

-- ══════════════════════════════════════════════════════════════════
-- Extraction runs (track what's been processed)
-- ══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS extraction_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id       TEXT,                       -- NULL for standalone
    node_ids        TEXT NOT NULL,              -- JSON array of node IDs processed
    prompt_version  TEXT NOT NULL,
    model_used      TEXT NOT NULL,
    status          TEXT NOT NULL,              -- success | error | skipped
    error_msg       TEXT,
    commitments_found INTEGER DEFAULT 0,
    processing_ms   INTEGER,
    created_at      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_thread ON extraction_runs(thread_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON extraction_runs(status);

-- ══════════════════════════════════════════════════════════════════
-- Briefings (synthesized daily/weekly summaries)
-- ══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS briefings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    briefing_type   TEXT NOT NULL,              -- daily | weekly | urgent
    content         TEXT NOT NULL,              -- markdown briefing
    commitment_ids  TEXT NOT NULL,              -- JSON array of commitment IDs included
    prompt_version  TEXT NOT NULL,
    model_used      TEXT NOT NULL,
    created_at      INTEGER NOT NULL
);
"""


class DerivedStore:
    """Manages the derived data SQLite database."""

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or DERIVED_DB_PATH
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(DERIVED_SCHEMA)
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def is_thread_processed(self, thread_id: str, prompt_version: str) -> bool:
        """Check if a thread has already been extracted with this prompt version."""
        row = self.conn.execute(
            """SELECT 1 FROM extraction_runs 
               WHERE thread_id = ? AND prompt_version = ? AND status = 'success'
               LIMIT 1""",
            (thread_id, prompt_version),
        ).fetchone()
        return row is not None

    def is_node_processed(self, node_id: str, prompt_version: str) -> bool:
        """Check if a standalone node has been extracted."""
        row = self.conn.execute(
            """SELECT 1 FROM extraction_runs
               WHERE node_ids LIKE ? AND prompt_version = ? AND status = 'success'
               LIMIT 1""",
            (f'%"{node_id}"%', prompt_version),
        ).fetchone()
        return row is not None

    def record_run(
        self,
        thread_id: str | None,
        node_ids: list[str],
        prompt_version: str,
        model_used: str,
        status: str,
        commitments_found: int = 0,
        error_msg: str | None = None,
        processing_ms: int | None = None,
    ) -> int:
        """Record an extraction run."""
        cur = self.conn.execute(
            """INSERT INTO extraction_runs 
               (thread_id, node_ids, prompt_version, model_used, status,
                error_msg, commitments_found, processing_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                thread_id,
                json.dumps(node_ids),
                prompt_version,
                model_used,
                status,
                error_msg,
                commitments_found,
                processing_ms,
                int(time.time()),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def upsert_commitment(self, commitment: dict) -> None:
        """Insert or update a commitment."""
        self.conn.execute(
            """INSERT OR REPLACE INTO commitments
               (id, thread_id, source_node_ids, commitment_type, who, what,
                to_whom, deadline, status, priority, confidence, provenance,
                note, raw_extraction, prompt_version, model_used,
                processed_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                commitment["id"],
                commitment.get("thread_id"),
                json.dumps(commitment["source_node_ids"]),
                commitment["commitment_type"],
                commitment["who"],
                commitment["what"],
                commitment.get("to_whom"),
                commitment.get("deadline"),
                commitment.get("status", "open"),
                commitment.get("priority", 3),
                commitment.get("confidence", 0.5),
                commitment.get("provenance"),
                commitment.get("note"),
                commitment.get("raw_extraction"),
                commitment["prompt_version"],
                commitment["model_used"],
                commitment.get("processed_at", int(time.time())),
                commitment["created_at"],
            ),
        )
        self.conn.commit()

    def get_open_commitments(self, limit: int = 50) -> list[dict]:
        """Get open commitments ordered by priority and deadline."""
        rows = self.conn.execute(
            """SELECT * FROM commitments
               WHERE status = 'open'
               ORDER BY priority ASC, 
                        CASE WHEN deadline IS NOT NULL THEN 0 ELSE 1 END,
                        deadline ASC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_overdue_commitments(self) -> list[dict]:
        """Get commitments past their deadline."""
        now = time.strftime("%Y-%m-%d")
        rows = self.conn.execute(
            """SELECT * FROM commitments
               WHERE status = 'open' AND deadline IS NOT NULL AND deadline < ?
               ORDER BY deadline ASC""",
            (now,),
        ).fetchall()
        return [dict(r) for r in rows]

    def reset(self) -> None:
        """Drop all derived data. Safe — source data in graph.db is untouched."""
        self.conn.executescript("""
            DELETE FROM commitments;
            DELETE FROM extraction_runs;
            DELETE FROM briefings;
        """)
        self.conn.commit()
        logger.info("Derived store reset.")
