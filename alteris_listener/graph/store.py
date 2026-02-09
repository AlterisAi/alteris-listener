"""GraphStore — thread-safe wrapper around the graph SQLite database.

Provides typed methods for inserting/querying nodes, edges, contacts,
and activity signals. All heavy lifting goes through this class.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from pathlib import Path
from typing import Any

import msgpack

from alteris_listener.graph.schema import GRAPH_DB_PATH, init_db

logger = logging.getLogger(__name__)


class GraphStore:
    """Manages the knowledge graph SQLite database."""

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or GRAPH_DB_PATH
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = init_db(self._db_path)
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ══════════════════════════════════════════════════════════════
    # Node operations
    # ══════════════════════════════════════════════════════════════

    def upsert_node(
        self,
        node_id: str,
        node_type: str,
        source: str,
        timestamp: int | None,
        subject: str | None = None,
        sender: str | None = None,
        recipients: list[str] | None = None,
        body_hash: str | None = None,
        body_preview: str | None = None,
        data: dict | None = None,
        thread_id: str | None = None,
        heuristic_score: float = 0.0,
        tier: str = "cold",
    ) -> bool:
        """Insert or update a node. Returns True if inserted (new)."""
        now = int(time.time())
        packed_data = msgpack.packb(data or {}, use_bin_type=True)
        recipients_json = json.dumps(recipients or [])

        try:
            self.conn.execute(
                """INSERT INTO nodes
                   (id, node_type, source, timestamp, subject, sender, recipients,
                    body_hash, body_preview, data, thread_id, heuristic_score, tier, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     heuristic_score = MAX(nodes.heuristic_score, excluded.heuristic_score),
                     tier = excluded.tier,
                     data = excluded.data
                """,
                (
                    node_id, node_type, source, timestamp, subject, sender,
                    recipients_json, body_hash, body_preview, packed_data,
                    thread_id, heuristic_score, tier, now,
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_node(self, node_id: str) -> dict | None:
        """Fetch a single node by ID."""
        row = self.conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_node(row)

    def get_nodes_by_type(
        self,
        node_type: str,
        limit: int = 100,
        min_score: float = 0.0,
    ) -> list[dict]:
        """Fetch nodes by type, ordered by score."""
        rows = self.conn.execute(
            """SELECT * FROM nodes
               WHERE node_type = ? AND heuristic_score >= ?
               ORDER BY heuristic_score DESC LIMIT ?""",
            (node_type, min_score, limit),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_thread_nodes(self, thread_id: str) -> list[dict]:
        """Fetch all nodes in a thread, ordered chronologically."""
        rows = self.conn.execute(
            """SELECT * FROM nodes WHERE thread_id = ?
               ORDER BY timestamp ASC""",
            (thread_id,),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def count_nodes(self, node_type: str | None = None) -> int:
        if node_type:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE node_type = ?", (node_type,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
        return row[0]

    def update_node_embedding(self, node_id: str, embedding_blob: bytes):
        """Store embedding for a node."""
        self.conn.execute(
            "UPDATE nodes SET embedding = ? WHERE id = ?",
            (embedding_blob, node_id),
        )
        self.conn.commit()

    def update_node_score(self, node_id: str, score: float):
        """Update heuristic score."""
        self.conn.execute(
            "UPDATE nodes SET heuristic_score = ? WHERE id = ?",
            (score, node_id),
        )
        self.conn.commit()

    def update_node_tier(self, node_id: str, tier: str):
        """Promote/demote a node between tiers."""
        self.conn.execute(
            "UPDATE nodes SET tier = ? WHERE id = ?", (tier, node_id)
        )
        self.conn.commit()

    def set_triage_result(self, node_id: str, relevant: int, reason: str = ""):
        """Store the Pass 3 triage score for a node.

        Args:
            relevant: Score as int 0-10 (representing 0.0-1.0 in 0.1 increments).
            reason: Short explanation from LLM (max 200 chars).
        """
        self.conn.execute(
            "UPDATE nodes SET triage_relevant = ?, triage_reason = ? WHERE id = ?",
            (relevant, reason, node_id),
        )
        # Don't commit here — caller batches commits for performance

    def _row_to_node(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        if d.get("data"):
            d["data"] = msgpack.unpackb(d["data"], raw=False)
        if d.get("recipients"):
            d["recipients"] = json.loads(d["recipients"])
        return d

    # ══════════════════════════════════════════════════════════════
    # Edge operations
    # ══════════════════════════════════════════════════════════════

    def add_edge(
        self,
        src: str,
        dst: str,
        edge_type: str,
        weight: float = 1.0,
        metadata: dict | None = None,
    ) -> bool:
        """Insert an edge. Returns False if it already exists."""
        now = int(time.time())
        packed_meta = msgpack.packb(metadata or {}, use_bin_type=True) if metadata else None
        try:
            self.conn.execute(
                """INSERT OR IGNORE INTO edges (src, dst, edge_type, weight, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (src, dst, edge_type, weight, packed_meta, now),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_edges_from(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
    ) -> list[dict]:
        """Get outgoing edges from a node."""
        if edge_types:
            placeholders = ",".join("?" * len(edge_types))
            rows = self.conn.execute(
                f"SELECT * FROM edges WHERE src = ? AND edge_type IN ({placeholders})",
                [node_id] + edge_types,
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM edges WHERE src = ?", (node_id,)
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_edges_to(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
    ) -> list[dict]:
        """Get incoming edges to a node."""
        if edge_types:
            placeholders = ",".join("?" * len(edge_types))
            rows = self.conn.execute(
                f"SELECT * FROM edges WHERE dst = ? AND edge_type IN ({placeholders})",
                [node_id] + edge_types,
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM edges WHERE dst = ?", (node_id,)
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_neighbors(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
    ) -> list[dict]:
        """Get all neighbor node IDs (bidirectional)."""
        outgoing = self.get_edges_from(node_id, edge_types)
        incoming = self.get_edges_to(node_id, edge_types)

        neighbor_ids = set()
        edges = []
        for e in outgoing:
            neighbor_ids.add(e["dst"])
            edges.append(e)
        for e in incoming:
            neighbor_ids.add(e["src"])
            edges.append(e)

        return edges

    def count_edges(self, edge_type: str | None = None) -> int:
        if edge_type:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM edges WHERE edge_type = ?", (edge_type,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()
        return row[0]

    def _row_to_edge(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        if d.get("metadata"):
            d["metadata"] = msgpack.unpackb(d["metadata"], raw=False)
        return d

    # ══════════════════════════════════════════════════════════════
    # Contact stats
    # ══════════════════════════════════════════════════════════════

    def upsert_contact(
        self,
        contact_id: str,
        display_name: str | None = None,
        sent_to_delta: int = 0,
        recv_from_delta: int = 0,
        timestamp: int | None = None,
        avg_body_length: float = 0.0,
        source: str | None = None,
    ):
        """Incrementally update contact stats."""
        now = int(time.time())
        ts = timestamp or now

        existing = self.conn.execute(
            "SELECT * FROM contact_stats WHERE contact_id = ?", (contact_id,)
        ).fetchone()

        if existing:
            total = existing["total_messages"] + sent_to_delta + recv_from_delta
            sent = existing["sent_to_count"] + sent_to_delta
            recv = existing["recv_from_count"] + recv_from_delta
            reply_ratio = min(sent, recv) / max(total, 1)

            sources = json.loads(existing["sources"])
            if source and source not in sources:
                sources.append(source)

            self.conn.execute(
                """UPDATE contact_stats SET
                     display_name = COALESCE(?, display_name),
                     total_messages = ?,
                     sent_to_count = ?,
                     recv_from_count = ?,
                     first_seen = MIN(first_seen, ?),
                     last_seen = MAX(last_seen, ?),
                     avg_body_length = (avg_body_length * total_messages + ? * ?) / MAX(?, 1),
                     reply_ratio = ?,
                     sources = ?,
                     updated_at = ?
                   WHERE contact_id = ?""",
                (
                    display_name, total, sent, recv,
                    ts, ts,
                    avg_body_length, (sent_to_delta + recv_from_delta), total,
                    reply_ratio, json.dumps(sources), now,
                    contact_id,
                ),
            )
        else:
            total = sent_to_delta + recv_from_delta
            sent = sent_to_delta
            recv = recv_from_delta
            reply_ratio = min(sent, recv) / max(total, 1)
            sources = [source] if source else []

            self.conn.execute(
                """INSERT INTO contact_stats
                   (contact_id, display_name, total_messages, sent_to_count,
                    recv_from_count, first_seen, last_seen, avg_body_length,
                    reply_ratio, sources, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    contact_id, display_name, total, sent, recv,
                    ts, ts, avg_body_length, reply_ratio,
                    json.dumps(sources), now,
                ),
            )
        self.conn.commit()

    def get_contact(self, contact_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM contact_stats WHERE contact_id = ?", (contact_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["sources"] = json.loads(d["sources"])
        return d

    def get_top_contacts(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM contact_stats
               ORDER BY importance_tier ASC, total_messages DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def recompute_contact_tiers(self):
        """Assign importance tiers based on composite score ranking.

        Uses a composite of volume, bidirectionality, recency, and multi-source
        presence to rank contacts. Tier 1 is the top ~25 by composite score
        (true inner circle), not just anyone who crosses a flat threshold.
        """
        now = int(time.time())
        thirty_days = 30 * 86400
        ninety_days = 90 * 86400

        rows = self.conn.execute(
            """SELECT contact_id, total_messages, sent_to_count, recv_from_count,
                      reply_ratio, last_seen, sources
               FROM contact_stats"""
        ).fetchall()

        if not rows:
            return

        # Compute composite score for each contact
        scored: list[tuple[str, float]] = []
        for row in rows:
            total = row["total_messages"]
            reply = row["reply_ratio"]
            last_seen = row["last_seen"] or 0
            sources_raw = row["sources"]
            if isinstance(sources_raw, str):
                n_sources = len(json.loads(sources_raw))
            else:
                n_sources = len(sources_raw) if sources_raw else 0

            # Volume: log scale so 1000 msgs isn't 100x better than 10
            vol_score = math.log1p(total) / math.log1p(500)  # saturates around 500

            # Bidirectionality: true conversations vs one-way
            bidir_score = min(1.0, reply * 2.5)  # 0.4 reply_ratio → 1.0

            # Recency: hard cutoff — must have been seen in last 90 days to be tier 1
            age_days = (now - last_seen) / 86400 if last_seen else 9999
            if age_days <= 7:
                recency_score = 1.0
            elif age_days <= 30:
                recency_score = 0.8
            elif age_days <= 90:
                recency_score = 0.5
            else:
                recency_score = 0.1

            # Multi-source bonus: seen in both mail + imessage is strong signal
            source_bonus = min(0.2, (n_sources - 1) * 0.1)

            composite = (
                vol_score * 0.30
                + bidir_score * 0.35
                + recency_score * 0.25
                + source_bonus
            )
            scored.append((row["contact_id"], composite))

        # Sort by composite score descending
        scored.sort(key=lambda x: -x[1])

        # Tier 1: top 25 (hard cap) — but must have reply_ratio > 0.1 and seen in 90 days
        # Tier 2: next ~200 or composite > 0.3
        # Tier 3: everything else
        tier_1_limit = 25
        tier_1_count = 0

        # Build lookup of original rows for eligibility checks
        contact_data = {}
        for row in rows:
            contact_data[row["contact_id"]] = row

        tier_assignments: dict[str, int] = {}
        for contact_id, composite in scored:
            cd = contact_data[contact_id]
            last_seen = cd["last_seen"] or 0
            age_days = (now - last_seen) / 86400 if last_seen else 9999

            if (
                tier_1_count < tier_1_limit
                and cd["reply_ratio"] >= 0.1
                and age_days <= 90
                and cd["total_messages"] >= 10
                and composite >= 0.4
            ):
                tier_assignments[contact_id] = 1
                tier_1_count += 1
            elif composite >= 0.2 or (cd["total_messages"] >= 5 and age_days <= 90):
                tier_assignments[contact_id] = 2
            else:
                tier_assignments[contact_id] = 3

        # Batch update
        for contact_id, tier in tier_assignments.items():
            self.conn.execute(
                "UPDATE contact_stats SET importance_tier = ? WHERE contact_id = ?",
                (tier, contact_id),
            )

        # Everything not in scored (shouldn't happen, but safety)
        self.conn.execute(
            "UPDATE contact_stats SET importance_tier = 3 WHERE importance_tier NOT IN (1, 2, 3)"
        )
        self.conn.commit()
        logger.info(
            "Recomputed contact tiers: %d tier-1, %d tier-2, %d tier-3",
            sum(1 for t in tier_assignments.values() if t == 1),
            sum(1 for t in tier_assignments.values() if t == 2),
            sum(1 for t in tier_assignments.values() if t == 3),
        )

    # ══════════════════════════════════════════════════════════════
    # Activity signals
    # ══════════════════════════════════════════════════════════════

    def update_activity(
        self,
        entity_id: str,
        entity_type: str,
        timestamp: int | None = None,
    ):
        """Increment activity signal for an entity across all windows."""
        now = timestamp or int(time.time())

        for window in ("1h", "24h", "7d"):
            self.conn.execute(
                """INSERT INTO activity_signals
                     (entity_id, entity_type, window, event_count, last_event)
                   VALUES (?, ?, ?, 1, ?)
                   ON CONFLICT(entity_id, entity_type, window) DO UPDATE SET
                     event_count = event_count + 1,
                     last_event = MAX(last_event, ?)""",
                (entity_id, entity_type, window, now, now),
            )
        self.conn.commit()

    def get_activity(self, entity_id: str, entity_type: str) -> dict:
        """Get activity signals for an entity."""
        rows = self.conn.execute(
            """SELECT window, event_count, last_event, trend
               FROM activity_signals
               WHERE entity_id = ? AND entity_type = ?""",
            (entity_id, entity_type),
        ).fetchall()
        return {r["window"]: dict(r) for r in rows}

    # ══════════════════════════════════════════════════════════════
    # Bootstrap state
    # ══════════════════════════════════════════════════════════════

    def get_bootstrap_state(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM bootstrap_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_bootstrap_state(self, key: str, value: str):
        now = int(time.time())
        self.conn.execute(
            """INSERT INTO bootstrap_state (key, value, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?""",
            (key, value, now, value, now),
        )
        self.conn.commit()

    # ══════════════════════════════════════════════════════════════
    # Stats / diagnostics
    # ══════════════════════════════════════════════════════════════

    def graph_stats(self) -> dict[str, Any]:
        """Return summary statistics about the graph."""
        stats: dict[str, Any] = {}

        stats["total_nodes"] = self.count_nodes()
        stats["total_edges"] = self.count_edges()

        for ntype in ("email", "message", "meeting", "contact", "calendar_event"):
            stats[f"nodes_{ntype}"] = self.count_nodes(ntype)

        for etype in ("sent_to", "same_thread", "cc_to", "attendee", "reply_to", "same_entity"):
            stats[f"edges_{etype}"] = self.count_edges(etype)

        contact_count = self.conn.execute(
            "SELECT COUNT(*) FROM contact_stats"
        ).fetchone()[0]
        stats["contacts"] = contact_count

        for tier in (1, 2, 3):
            ct = self.conn.execute(
                "SELECT COUNT(*) FROM contact_stats WHERE importance_tier = ?",
                (tier,),
            ).fetchone()[0]
            stats[f"contacts_tier_{tier}"] = ct

        bootstrap = self.get_bootstrap_state("last_pass")
        stats["bootstrap_last_pass"] = bootstrap

        # Triage stats — use adjusted score if available
        # These columns only exist after triage migration runs
        try:
            triaged = self.conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE triage_relevant IS NOT NULL"
            ).fetchone()[0]
            stats["triaged_total"] = triaged

            for tier_name, lo, hi in [("ignore", 0, 2), ("lightweight", 3, 6), ("deep", 7, 10)]:
                count = self.conn.execute(
                    """SELECT COUNT(*) FROM nodes
                       WHERE triage_relevant IS NOT NULL
                       AND COALESCE(triage_adjusted, triage_relevant) BETWEEN ? AND ?""",
                    (lo, hi),
                ).fetchone()[0]
                stats[f"triage_{tier_name}"] = count

            adjusted_count = self.conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE triage_adjusted IS NOT NULL"
            ).fetchone()[0]
            stats["triage_adjusted_count"] = adjusted_count
        except sqlite3.OperationalError:
            stats["triaged_total"] = 0
            stats["triage_ignore"] = 0
            stats["triage_lightweight"] = 0
            stats["triage_deep"] = 0
            stats["triage_adjusted_count"] = 0

        return stats
