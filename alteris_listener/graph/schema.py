"""SQLite schema for the Alteris knowledge graph.

All graph state lives in a single SQLite file at ~/.alteris/graph.db.
Tables:
  nodes          — immutable ground truth (emails, messages, meetings, contacts, etc.)
  derived_nodes  — LLM-generated (tasks, summaries, insights), versioned by prompt
  edges          — all relationships (structural + derived)
  exemplars      — HITL few-shot preference bank
  activity_signals — rolling activity counts per entity for cheap pertinence scoring
  contact_stats  — precomputed contact interaction matrix
  schema_mappings — learned source adapter mappings
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

GRAPH_DB_PATH = Path.home() / ".alteris" / "graph.db"

SCHEMA_SQL = """
-- ══════════════════════════════════════════════════════════════════
-- Immutable ground truth nodes
-- ══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,        -- deterministic: source_type:source_id
    node_type   TEXT NOT NULL,           -- email, message, meeting, contact, calendar_event, document
    source      TEXT NOT NULL,           -- mail, imessage, slack, calendar, granola
    timestamp   INTEGER,                -- unix epoch, NULL for atemporal (contacts)
    subject     TEXT,                    -- email subject, meeting title, etc.
    sender      TEXT,                    -- sender address/name
    recipients  TEXT,                    -- JSON array of recipient addresses
    body_hash   TEXT,                    -- SHA-256 of body for dedup
    body_preview TEXT,                   -- first 500 chars for display
    data        BLOB,                   -- msgpack'd full canonical fields
    embedding   BLOB,                   -- f16 vector (384 or 768 dim), NULL until embedded
    thread_id   TEXT,                    -- source-native thread/conversation ID
    heuristic_score REAL DEFAULT 0.0,   -- computed importance score (0-1)
    tier        TEXT DEFAULT 'cold',    -- hot | warm | cold | archived
    created_at  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nodes_type_time ON nodes(node_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_nodes_source_time ON nodes(source, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_nodes_thread ON nodes(thread_id) WHERE thread_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_nodes_sender ON nodes(sender) WHERE sender IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_nodes_tier ON nodes(tier);
CREATE INDEX IF NOT EXISTS idx_nodes_score ON nodes(heuristic_score DESC);
CREATE INDEX IF NOT EXISTS idx_nodes_body_hash ON nodes(body_hash) WHERE body_hash IS NOT NULL;

-- ══════════════════════════════════════════════════════════════════
-- Derived nodes (LLM-generated, versioned)
-- ══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS derived_nodes (
    id              TEXT PRIMARY KEY,
    node_type       TEXT NOT NULL,           -- task, insight, summary, priority_signal
    data            BLOB NOT NULL,           -- msgpack'd content
    embedding       BLOB,
    prompt_id       TEXT NOT NULL,           -- which prompt version produced this
    source_nodes    TEXT NOT NULL,           -- JSON array of node IDs
    confidence      REAL,
    created_at      INTEGER NOT NULL,
    superseded_by   TEXT                     -- points to newer version when prompt evolves
);

CREATE INDEX IF NOT EXISTS idx_derived_type ON derived_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_derived_prompt ON derived_nodes(prompt_id);

-- ══════════════════════════════════════════════════════════════════
-- Edges (all relationships)
-- ══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS edges (
    src         TEXT NOT NULL,
    dst         TEXT NOT NULL,
    edge_type   TEXT NOT NULL,           -- sent_to, cc_to, same_thread, attendee,
                                        -- mentions, caused_by, blocks, same_entity,
                                        -- reply_to, forwarded_from
    weight      REAL DEFAULT 1.0,        -- learned importance
    metadata    BLOB,                    -- msgpack'd edge-specific data
    created_at  INTEGER NOT NULL,
    PRIMARY KEY (src, dst, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_edges_src_type ON edges(src, edge_type);
CREATE INDEX IF NOT EXISTS idx_edges_dst_type ON edges(dst, edge_type);

-- ══════════════════════════════════════════════════════════════════
-- Contact stats (precomputed interaction matrix)
-- ══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS contact_stats (
    contact_id      TEXT PRIMARY KEY,        -- normalized email or phone
    display_name    TEXT,
    total_messages  INTEGER DEFAULT 0,
    sent_to_count   INTEGER DEFAULT 0,       -- user → contact
    recv_from_count INTEGER DEFAULT 0,       -- contact → user
    first_seen      INTEGER,                 -- unix epoch
    last_seen       INTEGER,                 -- unix epoch
    avg_body_length REAL DEFAULT 0.0,
    reply_ratio     REAL DEFAULT 0.0,        -- bidirectionality score
    importance_tier INTEGER DEFAULT 3,       -- 1=inner circle, 2=regular, 3=peripheral
    sources         TEXT DEFAULT '[]',       -- JSON array of sources seen in
    updated_at      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_contact_importance ON contact_stats(importance_tier, last_seen DESC);

-- ══════════════════════════════════════════════════════════════════
-- Activity signals (rolling counts for pertinence)
-- ══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS activity_signals (
    entity_id   TEXT NOT NULL,               -- contact_id, thread_id, topic cluster
    entity_type TEXT NOT NULL,               -- contact, thread, topic
    window      TEXT NOT NULL,               -- 1h, 24h, 7d
    event_count INTEGER NOT NULL DEFAULT 0,
    last_event  INTEGER NOT NULL,            -- unix epoch
    trend       REAL DEFAULT 0.0,            -- rate of change vs baseline
    PRIMARY KEY (entity_id, entity_type, window)
);

-- ══════════════════════════════════════════════════════════════════
-- Exemplars (HITL few-shot preference bank)
-- ══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS exemplars (
    id          TEXT PRIMARY KEY,
    query_type  TEXT NOT NULL,               -- todo_extractor, priority_reasoner, etc.
    input_hash  TEXT NOT NULL,               -- hash of input that produced this
    output      BLOB NOT NULL,               -- msgpack'd LLM output
    user_action TEXT,                        -- kept, dismissed, edited, escalated
    node_ids    TEXT NOT NULL,               -- JSON array of involved node IDs
    prompt_id   TEXT NOT NULL,               -- which prompt version
    fitness     REAL DEFAULT 0.0,
    created_at  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_exemplars_query ON exemplars(query_type, fitness DESC);

-- ══════════════════════════════════════════════════════════════════
-- Schema mappings (learned source adapters)
-- ══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS schema_mappings (
    source_id   TEXT PRIMARY KEY,            -- e.g. "mail.app", "slack.workspace_name"
    mapping     BLOB NOT NULL,               -- msgpack'd field mapping dict
    transforms  BLOB,                        -- msgpack'd normalization steps
    confidence  REAL DEFAULT 0.5,
    needs_review TEXT DEFAULT '[]',           -- JSON array of uncertain fields
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);

-- ══════════════════════════════════════════════════════════════════
-- Archived nodes (compressed, out of active graph)
-- ══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS archived_nodes (
    id          TEXT PRIMARY KEY,
    node_type   TEXT,
    source      TEXT,
    timestamp   INTEGER,
    data        BLOB,
    summary_id  TEXT,                        -- which summary node absorbed this
    archived_at INTEGER NOT NULL
);

-- ══════════════════════════════════════════════════════════════════
-- Bootstrap state tracking
-- ══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS bootstrap_state (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  INTEGER NOT NULL
);
"""


def init_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Initialize the graph database, creating tables if needed."""
    path = db_path or GRAPH_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
    # Migrate existing databases before applying schema (which may reference new columns)
    _migrate_triage_columns(conn)

    conn.executescript(SCHEMA_SQL)
    conn.commit()

    logger.info("Graph database initialized at %s", path)
    return conn


def _migrate_triage_columns(conn: sqlite3.Connection):
    """Add triage columns if they don't exist (for databases created before Pass 3)."""
    # Check if nodes table exists at all
    table_exists = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='nodes'"
    ).fetchone()[0]
    if not table_exists:
        return  # Fresh DB, CREATE TABLE will handle it

    cursor = conn.execute("PRAGMA table_info(nodes)")
    columns = {row[1] for row in cursor.fetchall()}

    if "triage_relevant" not in columns:
        conn.execute("ALTER TABLE nodes ADD COLUMN triage_relevant INTEGER")
        conn.execute("ALTER TABLE nodes ADD COLUMN triage_reason TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_triage ON nodes(triage_relevant) "
            "WHERE triage_relevant IS NOT NULL"
        )
        conn.commit()
        logger.info("Migrated: added triage columns to nodes table")

    if "triage_adjusted" not in columns:
        conn.execute("ALTER TABLE nodes ADD COLUMN triage_adjusted INTEGER")
        conn.commit()
        logger.info("Migrated: added triage_adjusted column to nodes table")

    if "domain" not in columns:
        conn.execute("ALTER TABLE nodes ADD COLUMN domain TEXT")       # work, personal, family, financial, health, legal, travel, shopping
        conn.execute("ALTER TABLE nodes ADD COLUMN topics TEXT")       # JSON array of topic tags
        conn.execute("ALTER TABLE nodes ADD COLUMN entities TEXT")     # JSON array of named entities (companies, people, projects)
        conn.execute("ALTER TABLE nodes ADD COLUMN pii_flags TEXT")    # JSON array: financial, medical, legal, credentials
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_domain ON nodes(domain) "
            "WHERE domain IS NOT NULL"
        )
        conn.commit()
        logger.info("Migrated: added classification columns (domain, topics, entities, pii_flags)")
