"""Progressive summarization — compress cold-tier nodes into durable summaries.

Implements graph contraction: groups of cold nodes collapse into single
summary nodes that preserve essential edges and knowledge. The originals
are archived but recoverable.

Grouping strategies:
  - By thread: all emails in a completed thread → one summary
  - By contact × time window: all interactions with Person X in month Y
  - By topic cluster: HDBSCAN clusters from embedding space
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone

import msgpack

from alteris_listener.graph.store import GraphStore

logger = logging.getLogger(__name__)


def find_thread_groups(
    store: GraphStore,
    max_age_days: int = 90,
    min_thread_size: int = 2,
) -> list[dict]:
    """Find email threads eligible for summarization.

    A thread is eligible if:
    - All messages are older than max_age_days
    - Thread has at least min_thread_size messages
    - Thread is not already summarized

    Returns list of thread groups with metadata.
    """
    now = int(time.time())
    cutoff = now - (max_age_days * 86400)

    rows = store.conn.execute(
        """SELECT thread_id, COUNT(*) as msg_count,
                  MIN(timestamp) as first_ts, MAX(timestamp) as last_ts,
                  GROUP_CONCAT(DISTINCT sender) as senders
           FROM nodes
           WHERE thread_id IS NOT NULL
             AND node_type = 'email'
             AND tier != 'archived'
             AND timestamp < ?
           GROUP BY thread_id
           HAVING msg_count >= ?
           ORDER BY last_ts DESC""",
        (cutoff, min_thread_size),
    ).fetchall()

    groups = []
    for row in rows:
        # Check if already summarized
        existing = store.conn.execute(
            """SELECT id FROM derived_nodes
               WHERE node_type = 'summary'
                 AND source_nodes LIKE ?""",
            (f'%"{row["thread_id"]}"%',),
        ).fetchone()

        if existing:
            continue

        groups.append({
            "thread_id": row["thread_id"],
            "message_count": row["msg_count"],
            "first_timestamp": row["first_ts"],
            "last_timestamp": row["last_ts"],
            "participants": row["senders"].split(",") if row["senders"] else [],
            "group_type": "thread",
        })

    return groups


def find_contact_period_groups(
    store: GraphStore,
    max_age_days: int = 90,
    period_days: int = 30,
    min_messages: int = 3,
) -> list[dict]:
    """Find contact × time period groups eligible for summarization.

    Groups interactions with each contact into monthly buckets.
    Only returns groups where all messages are older than max_age_days.
    """
    now = int(time.time())
    cutoff = now - (max_age_days * 86400)

    rows = store.conn.execute(
        """SELECT n.sender, n.timestamp, n.id
           FROM nodes n
           WHERE n.timestamp < ?
             AND n.sender IS NOT NULL
             AND n.sender != ''
             AND n.tier != 'archived'
           ORDER BY n.sender, n.timestamp""",
        (cutoff,),
    ).fetchall()

    # Group by contact × month
    contact_periods: dict[str, list] = defaultdict(list)
    for row in rows:
        period_key = row["timestamp"] // (period_days * 86400)
        key = f"{row['sender']}:{period_key}"
        contact_periods[key].append(row)

    groups = []
    for key, nodes in contact_periods.items():
        if len(nodes) < min_messages:
            continue

        sender = nodes[0]["sender"]
        groups.append({
            "contact": sender,
            "message_count": len(nodes),
            "first_timestamp": nodes[0]["timestamp"],
            "last_timestamp": nodes[-1]["timestamp"],
            "node_ids": [n["id"] for n in nodes],
            "group_type": "contact_period",
        })

    return groups


def find_recurring_patterns(store: GraphStore) -> list[dict]:
    """Detect recurring topic/thread patterns from historical data.

    Looks for clusters of emails that recur temporally:
    similar subject patterns, same participants, roughly periodic.
    """
    rows = store.conn.execute(
        """SELECT subject, sender, COUNT(*) as occurrences,
                  MIN(timestamp) as first_seen, MAX(timestamp) as last_seen,
                  GROUP_CONCAT(DISTINCT sender) as participants
           FROM nodes
           WHERE node_type = 'email'
             AND subject IS NOT NULL
             AND subject != ''
             AND tier != 'archived'
           GROUP BY subject
           HAVING occurrences >= 3
             AND (MAX(timestamp) - MIN(timestamp)) > 604800  -- at least 1 week span
           ORDER BY occurrences DESC
           LIMIT 100""",
    ).fetchall()

    patterns = []
    for row in rows:
        span_days = (row["last_seen"] - row["first_seen"]) / 86400
        avg_gap = span_days / max(row["occurrences"] - 1, 1)

        # Only keep if it looks periodic (gap between 7-120 days)
        if 7 <= avg_gap <= 120:
            patterns.append({
                "subject": row["subject"],
                "occurrences": row["occurrences"],
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
                "avg_gap_days": avg_gap,
                "participants": row["participants"].split(",") if row["participants"] else [],
                "group_type": "recurring_pattern",
            })

    return patterns


def archive_nodes(
    store: GraphStore,
    node_ids: list[str],
    summary_id: str,
):
    """Move nodes to the archive table and mark them as archived.

    The originals are preserved in archived_nodes for potential
    re-expansion if the summary proves inadequate.
    """
    now = int(time.time())

    for nid in node_ids:
        node = store.get_node(nid)
        if not node:
            continue

        # Copy to archive
        store.conn.execute(
            """INSERT OR IGNORE INTO archived_nodes
               (id, node_type, source, timestamp, data, summary_id, archived_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                nid,
                node.get("node_type"),
                node.get("source"),
                node.get("timestamp"),
                msgpack.packb(node.get("data", {}), use_bin_type=True),
                summary_id,
                now,
            ),
        )

        # Update tier on original
        store.update_node_tier(nid, "archived")

    store.conn.commit()
    logger.info("Archived %d nodes under summary %s", len(node_ids), summary_id)


def create_summary_node(
    store: GraphStore,
    summary_id: str,
    summary_data: dict,
    source_node_ids: list[str],
    prompt_id: str = "manual",
) -> str:
    """Create a derived summary node and transfer essential edges.

    The summary node inherits edges from the archived nodes:
    - Contact edges get aggregated with message counts as weights
    - Thread edges point to the summary
    - Topic/project edges are preserved

    Returns the summary_id.
    """
    now = int(time.time())
    packed = msgpack.packb(summary_data, use_bin_type=True)

    store.conn.execute(
        """INSERT OR REPLACE INTO derived_nodes
           (id, node_type, data, prompt_id, source_nodes, confidence, created_at)
           VALUES (?, 'summary', ?, ?, ?, ?, ?)""",
        (
            summary_id,
            packed,
            prompt_id,
            json.dumps(source_node_ids),
            summary_data.get("confidence", 0.5),
            now,
        ),
    )

    # Collect and aggregate edges from source nodes
    contact_edge_counts: dict[str, int] = defaultdict(int)
    other_edges: list[tuple] = []

    for nid in source_node_ids:
        for edge in store.get_edges_from(nid):
            if edge["edge_type"] in ("sent_to", "sent", "cc_to"):
                contact_edge_counts[edge["dst"]] += 1
            elif edge["edge_type"] not in ("same_thread",):
                other_edges.append((edge["dst"], edge["edge_type"]))

        for edge in store.get_edges_to(nid):
            if edge["edge_type"] in ("sent_to", "sent", "cc_to"):
                contact_edge_counts[edge["src"]] += 1

    # Create aggregated contact edges on the summary
    for contact_id, count in contact_edge_counts.items():
        store.add_edge(
            src=summary_id,
            dst=contact_id,
            edge_type="involved",
            weight=float(count),
            metadata={"message_count": count},
        )

    # Preserve other edge types
    seen = set()
    for dst, etype in other_edges:
        key = (dst, etype)
        if key not in seen:
            store.add_edge(src=summary_id, dst=dst, edge_type=etype)
            seen.add(key)

    store.conn.commit()
    return summary_id


def get_summarization_candidates(
    store: GraphStore,
    max_age_days: int = 90,
) -> dict:
    """Get all groups eligible for summarization.

    Returns a dict with thread groups, contact period groups,
    and recurring patterns.
    """
    return {
        "thread_groups": find_thread_groups(store, max_age_days=max_age_days),
        "contact_period_groups": find_contact_period_groups(store, max_age_days=max_age_days),
        "recurring_patterns": find_recurring_patterns(store),
    }
