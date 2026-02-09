"""Ingest messages into the knowledge graph.

Handles Pass 0 (structural extraction) and Pass 1 (heuristic scoring)
of the bootstrap pipeline. All operations are deterministic — no LLM calls.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import msgpack

from alteris_listener.graph.entities import (
    ExtractedEntities,
    extract_entities_from_message,
    parse_address,
)
from alteris_listener.graph.scoring import (
    assign_temporal_bucket,
    bucket_score_threshold,
    heuristic_importance,
)
from alteris_listener.graph.store import GraphStore
from alteris_listener.sources.base import Message

logger = logging.getLogger(__name__)


def _node_type_for_source(source: str) -> str:
    """Map source name to canonical node_type."""
    return {
        "mail": "email",
        "imessage": "message",
        "slack": "message",
        "granola": "meeting",
        "calendar": "calendar_event",
    }.get(source, "document")


def _ensure_contact_node(store: GraphStore, contact_id: str, display_name: str = ""):
    """Create a contact node if it doesn't exist."""
    existing = store.get_node(contact_id)
    if not existing:
        store.upsert_node(
            node_id=contact_id,
            node_type="contact",
            source="inferred",
            timestamp=None,
            subject=display_name,
            data={"email": contact_id.split(":", 1)[-1], "display_name": display_name},
            tier="warm",
        )


def ingest_message(
    store: GraphStore,
    msg: Message,
    thread_sizes: dict[str, int] | None = None,
    now: int | None = None,
) -> Optional[str]:
    """Ingest a single message into the graph.

    Performs:
    1. Entity extraction (deterministic)
    2. Node creation
    3. Edge creation (sender→node, node→recipient, thread edges)
    4. Contact stat updates
    5. Heuristic scoring
    6. Activity signal updates

    Returns the node_id if successfully ingested, None if skipped.
    """
    now = now or int(time.time())
    thread_sizes = thread_sizes or {}

    # Extract entities
    entities = extract_entities_from_message(msg)

    # Parse CC recipients from metadata (mail reader provides these)
    meta = msg.metadata or {}
    cc_raw = meta.get("cc", [])
    if cc_raw:
        from alteris_listener.graph.entities import parse_address
        for cc_str in cc_raw:
            cc_contact = parse_address(cc_str)
            if cc_contact:
                entities.cc_recipients.append(cc_contact)

    # Get timestamp as unix epoch
    ts = int(msg.timestamp.timestamp()) if msg.timestamp else now

    # Determine node type
    node_type = _node_type_for_source(msg.source)

    # Build data payload
    meta = msg.metadata or {}
    data = {
        "source": msg.source,
        "sender_raw": msg.sender,
        "recipient_raw": msg.recipient,
        "subject": msg.subject,
        "body": msg.body[:5000] if msg.body else "",
        "is_from_me": msg.is_from_me,
        "thread_id": msg.thread_id,
    }
    data.update(meta)

    # Look up sender contact stats for scoring
    sender_stats = {}
    sender_addr = ""
    if entities.sender and not entities.sender.is_automated:
        sender_addr = entities.sender.normalized_email
        cs = store.get_contact(entities.sender.contact_id)
        if cs:
            sender_stats = cs

    # Determine directionality
    is_direct = entities.is_to_user

    # Compute heuristic score
    thread_id = entities.thread_id or ""
    thread_size = thread_sizes.get(thread_id, 1)

    from alteris_listener.graph.entities import USER_EMAILS
    score = heuristic_importance(
        timestamp=ts,
        sender=sender_addr,
        subject=msg.subject,
        body_length=len(msg.body) if msg.body else 0,
        node_type=node_type,
        is_direct=is_direct,
        thread_size=thread_size,
        sender_total_messages=sender_stats.get("total_messages", 0),
        sender_reply_ratio=sender_stats.get("reply_ratio", 0.0),
        sender_importance_tier=sender_stats.get("importance_tier", 3),
        user_emails=USER_EMAILS,
        now=now,
    )

    # Assign tier based on temporal bucket
    bucket = assign_temporal_bucket(ts, now)
    threshold = bucket_score_threshold(bucket)

    if bucket <= 2:
        tier = "hot"
    elif bucket == 3:
        tier = "warm"
    else:
        tier = "cold"

    # Create node
    store.upsert_node(
        node_id=entities.node_id,
        node_type=node_type,
        source=msg.source,
        timestamp=ts,
        subject=msg.subject,
        sender=sender_addr,
        recipients=[r.normalized_email for r in entities.recipients],
        body_hash=entities.body_hash,
        body_preview=entities.body_preview,
        data=data,
        thread_id=thread_id or None,
        heuristic_score=score,
        tier=tier,
    )

    # ── Create edges ─────────────────────────────────────────────

    # Sender → node (sent_by)
    if entities.sender and not entities.sender.is_automated:
        _ensure_contact_node(store, entities.sender.contact_id, entities.sender.display_name)
        store.add_edge(
            src=entities.sender.contact_id,
            dst=entities.node_id,
            edge_type="sent",
            weight=1.0,
        )

    # Node → recipients (sent_to)
    for recip in entities.recipients:
        if not recip.is_automated:
            _ensure_contact_node(store, recip.contact_id, recip.display_name)
            store.add_edge(
                src=entities.node_id,
                dst=recip.contact_id,
                edge_type="sent_to",
                weight=1.0,
            )

    # CC recipients
    for recip in entities.cc_recipients:
        if not recip.is_automated:
            _ensure_contact_node(store, recip.contact_id, recip.display_name)
            store.add_edge(
                src=entities.node_id,
                dst=recip.contact_id,
                edge_type="cc_to",
                weight=0.3,
            )

    # Same-thread edges: link to other nodes with same thread_id
    if thread_id:
        existing_in_thread = store.get_thread_nodes(thread_id)
        for other in existing_in_thread:
            if other["id"] != entities.node_id:
                store.add_edge(
                    src=entities.node_id,
                    dst=other["id"],
                    edge_type="same_thread",
                    weight=0.8,
                )

    # ── Update contact stats ─────────────────────────────────────

    if entities.sender and not entities.sender.is_automated:
        if entities.is_from_user:
            # User sent this — update all recipients
            for recip in entities.recipients:
                if not recip.is_automated and not recip.is_user:
                    store.upsert_contact(
                        contact_id=recip.contact_id,
                        display_name=recip.display_name,
                        sent_to_delta=1,
                        timestamp=ts,
                        avg_body_length=len(msg.body) if msg.body else 0,
                        source=msg.source,
                    )
        else:
            # Someone sent to user
            store.upsert_contact(
                contact_id=entities.sender.contact_id,
                display_name=entities.sender.display_name,
                recv_from_delta=1,
                timestamp=ts,
                avg_body_length=len(msg.body) if msg.body else 0,
                source=msg.source,
            )

    # ── Update activity signals ──────────────────────────────────

    if entities.sender and not entities.sender.is_automated:
        store.update_activity(entities.sender.contact_id, "contact", ts)

    if thread_id:
        store.update_activity(thread_id, "thread", ts)

    return entities.node_id


def ingest_messages(
    store: GraphStore,
    messages: list[Message],
    now: int | None = None,
    contact_lookup: "ContactLookup | None" = None,
) -> dict[str, int]:
    """Ingest a batch of messages into the graph.

    Pre-computes thread sizes for better heuristic scoring.
    If contact_lookup is provided, resolves phone numbers to names.

    Returns dict with counts: {ingested, skipped, total}.
    """
    now = now or int(time.time())

    # Resolve phone numbers to names if we have Contacts data
    if contact_lookup:
        for msg in messages:
            if msg.sender and msg.sender.startswith("+"):
                name = contact_lookup.resolve_name(msg.sender)
                if name:
                    msg.sender = f"{name} <{msg.sender}>"
            if msg.recipient and msg.recipient.startswith("+"):
                name = contact_lookup.resolve_name(msg.recipient)
                if name:
                    msg.recipient = f"{name} <{msg.recipient}>"

    # Pre-compute thread sizes
    thread_sizes: dict[str, int] = {}
    for msg in messages:
        tid = msg.thread_id
        if tid:
            thread_sizes[tid] = thread_sizes.get(tid, 0) + 1

    ingested = 0
    skipped = 0

    for msg in messages:
        node_id = ingest_message(store, msg, thread_sizes=thread_sizes, now=now)
        if node_id:
            ingested += 1
        else:
            skipped += 1

    # Recompute contact tiers after batch
    if ingested > 0:
        store.recompute_contact_tiers()

    logger.info("Ingested %d messages (%d skipped)", ingested, skipped)
    return {"ingested": ingested, "skipped": skipped, "total": len(messages)}


def link_calendar_to_meetings(store: GraphStore) -> dict[str, int]:
    """Link calendar events to meeting notes (Granola) via title + time overlap.

    Also creates attendee edges from calendar events to contact nodes.

    Should run after both calendar and granola sources are ingested.
    Returns: {linked: N, attendee_edges: N}
    """
    # Get all calendar events
    cal_events = store.conn.execute("""
        SELECT id, subject, timestamp, data FROM nodes
        WHERE node_type = 'calendar_event'
    """).fetchall()

    # Get all meeting notes
    meetings = store.conn.execute("""
        SELECT id, subject, timestamp FROM nodes
        WHERE node_type = 'meeting'
    """).fetchall()

    linked = 0
    attendee_edges = 0

    for cal in cal_events:
        cal_title = (cal["subject"] or "").strip().lower()
        cal_ts = cal["timestamp"] or 0

        if not cal_title or not cal_ts:
            continue

        # Find matching meetings: similar title and within 2 hours
        for mtg in meetings:
            mtg_title = (mtg["subject"] or "").strip().lower()
            mtg_ts = mtg["timestamp"] or 0

            if not mtg_title or not mtg_ts:
                continue

            # Time must be within 2 hours
            if abs(cal_ts - mtg_ts) > 7200:
                continue

            # Title matching: exact, substring, or significant word overlap
            if cal_title == mtg_title:
                match = True
            elif cal_title in mtg_title or mtg_title in cal_title:
                match = True
            else:
                # Word overlap: at least 50% of words in shorter title
                cal_words = set(cal_title.split())
                mtg_words = set(mtg_title.split())
                # Remove common stopwords
                stopwords = {"the", "a", "an", "and", "or", "with", "for", "to", "in", "on", "at"}
                cal_words -= stopwords
                mtg_words -= stopwords
                if cal_words and mtg_words:
                    overlap = len(cal_words & mtg_words)
                    shorter = min(len(cal_words), len(mtg_words))
                    match = shorter > 0 and overlap / shorter >= 0.5
                else:
                    match = False

            if match:
                store.add_edge(
                    src=cal["id"],
                    dst=mtg["id"],
                    edge_type="same_event",
                    weight=1.0,
                )
                linked += 1

        # Create attendee edges from calendar event to contacts
        data_blob = cal["data"]
        if data_blob:
            try:
                data = msgpack.unpackb(data_blob, raw=False) if isinstance(data_blob, bytes) else {}
            except Exception:
                data = {}

            attendees = data.get("attendees", [])
            for att in attendees:
                email = (att.get("email") or "").lower().strip()
                if not email:
                    continue

                contact_id = f"contact:{email}"
                name = att.get("name", "")
                _ensure_contact_node(store, contact_id, name)

                store.add_edge(
                    src=cal["id"],
                    dst=contact_id,
                    edge_type="attendee",
                    weight=0.8,
                    metadata={"status": att.get("status", ""), "role": att.get("role", "")},
                )
                attendee_edges += 1

    store.conn.commit()
    logger.info("Calendar-meeting links: %d, attendee edges: %d", linked, attendee_edges)

    # Boost contacts who share a calendar with the user to tier 1
    # Look for calendar events with family_signal metadata
    family_events = store.conn.execute("""
        SELECT DISTINCT data FROM nodes
        WHERE node_type = 'calendar_event' AND data IS NOT NULL
    """).fetchall()

    family_names: set[str] = set()
    for row in family_events:
        data_blob = row["data"]
        if data_blob:
            try:
                data = msgpack.unpackb(data_blob, raw=False) if isinstance(data_blob, bytes) else {}
            except Exception:
                continue
            if data.get("family_signal"):
                name = data.get("shared_calendar_with", "")
                if name:
                    family_names.add(name.lower())

    if family_names:
        # Find contacts whose display_name starts with a family name
        contacts = store.conn.execute(
            "SELECT contact_id, display_name, importance_tier FROM contact_stats"
        ).fetchall()
        for contact in contacts:
            cname = (contact["display_name"] or "").lower()
            if any(cname.startswith(fn) for fn in family_names):
                if contact["importance_tier"] != 1:
                    store.conn.execute(
                        "UPDATE contact_stats SET importance_tier = 1 WHERE contact_id = ?",
                        (contact["contact_id"],),
                    )
                    logger.info(
                        "Tier 1 boost (shared calendar): %s → %s",
                        contact["contact_id"], contact["display_name"],
                    )
        store.conn.commit()

    return {"linked": linked, "attendee_edges": attendee_edges}
