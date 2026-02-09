"""Pass 4: Commitment extraction from email threads and meetings.

Architecture:
  - Groups triaged nodes into threads (chronological order)
  - Routes sensitive threads (PII: financial, medical, legal, credentials)
    to local qwen3:30b-a3b with chunked sequential processing
  - Routes non-sensitive threads to Gemini Flash via async for parallelism
  - Stores extracted commitments in derived.db

Local processing uses a "running state" pattern:
  Message 1 → LLM → state: {commitments: [...], questions: [...]}
  Message 2 + state → LLM → state updated
  Final state → commitments table
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from typing import Optional

from rich.console import Console

from alteris_listener.graph.derived_store import DerivedStore
from alteris_listener.graph.store import GraphStore

logger = logging.getLogger(__name__)
console = Console()

PROMPT_VERSION = "extract_v1"

# PII types that require local processing
SENSITIVE_PII = {"financial", "credentials", "medical", "legal"}

# Max concurrent cloud calls
MAX_CLOUD_CONCURRENCY = 10

# ══════════════════════════════════════════════════════════════════
# Compact prompt for local qwen3:30b-a3b (fits in ~2K tokens)
# ══════════════════════════════════════════════════════════════════

LOCAL_SYSTEM_PROMPT = """\
You are Alteris, an AI executive assistant. Extract commitments and action items from this email/message.

EXTRACT these types:
- inbound_request: someone asks the user to do something
- user_commitment: user promised to do something ("I'll...", "Let me...")
- deadline: explicit deadline for user action
- waiting_on: user is waiting for someone else (high-stakes only)
- payment_due: invoice/bill with amount due
- follow_up: user needs to check back on something

DO NOT extract:
- Newsletters, marketing, promotions
- Actions someone else is doing (not user's responsibility)
- Pure information with no action needed
- Meeting logistics (proposing times, confirming attendance)
- Completed and acknowledged actions (unless tracking completion)
- Credit card statements with $0 due or negative balance
- User asking someone else to do something (delegation)

DIRECTION RULES:
- Someone asks user → EXTRACT (inbound_request)
- User asks someone else → DO NOT EXTRACT (delegation)
- User commits "I'll..." → EXTRACT (user_commitment)
- User completed it in thread → EXTRACT with status=done
- iMessage marked "Direction: outbound" or "from: USER" → user sent it, treat as delegation or user_commitment, NOT inbound_request

Respond with ONLY valid JSON:
{"commitments": [
  {
    "type": "inbound_request|user_commitment|deadline|waiting_on|payment_due|follow_up",
    "who": "user|<contact name>",
    "what": "verb-first concise description",
    "to_whom": "<who expects this>|null",
    "deadline": "YYYY-MM-DD|null",
    "status": "open|done|cancelled",
    "priority": 3,
    "confidence": 0.8,
    "provenance": "user_committed|assigned_to_user|system_detected",
    "note": "1-2 sentence context"
  }
]}

If no commitments found, return: {"commitments": []}"""


LOCAL_STATE_PREFIX = """\
PREVIOUS CONTEXT from earlier messages in this thread:
{state_json}

Given the above context, analyze this next message and UPDATE the commitments.
Mark any previously-open commitments as "done" if this message completes them.
Add any new commitments. Return the FULL updated list."""


# ══════════════════════════════════════════════════════════════════
# Cloud prompt (Gemini Flash) — full thread at once
# ══════════════════════════════════════════════════════════════════

CLOUD_SYSTEM_PROMPT = """\
You are Alteris, an AI executive assistant. Extract ALL commitments, action items, and obligations from this email thread.

Read the entire thread chronologically. Later messages may complete, modify, or cancel earlier commitments.

If a "Related context from other sources" section is provided, use it ONLY to:
- Mark commitments as "done" if a calendar event or meeting confirms they were completed
- Avoid extracting duplicates of commitments already visible in other sources
- Enrich your understanding of the full situation
NEVER extract new commitments from the "Related context" section. Only extract from the primary thread messages above it.

EXTRACT these types:
- inbound_request: someone explicitly asks the user to do something
- user_commitment: user promised to do something ("I'll...", "Let me...", "I can...")
- deadline: explicit deadline requiring user action
- waiting_on: user is waiting for someone else on high-stakes item (rare)
- payment_due: invoice/bill/statement with payment required
- follow_up: user needs to check back on something

CRITICAL RULES:
1. Direction matters: someone→user = extract. user→someone = DO NOT extract (delegation).
2. iMessage direction: if "Direction: outbound" or "from: USER", the user sent it — treat as delegation or user_commitment, NOT inbound_request. Only extract as "waiting_on" if user is waiting for a reply on something high-stakes.
3. If user completed the task in the thread → extract with status="done"
4. Deduplicate: same action mentioned multiple times = one commitment
5. Final state only: if rescheduled/changed, only extract the latest version
6. No newsletters, marketing, promotions, or $0-due statements
7. User asking "Did you...?", "Can you handle...?" = delegation = NO extraction
8. Priority: almost everything = 3. Use 2 only for explicit deadlines within a week where someone external is waiting. Use 1 ONLY for same-day deadlines with real consequences (financial penalties, blocking a team). Routine personal/family tasks are ALWAYS priority 3 regardless of urgency language.

COMPLETION DETECTION:
- If someone asked user AND user did it in the thread → status="done"
- If related context shows a past calendar event matching this commitment → status="done"
- "I sent it", "Done!", past-tense confirmation → done
- Still pending → status="open"

Respond with ONLY valid JSON:
{"commitments": [
  {
    "type": "inbound_request|user_commitment|deadline|waiting_on|payment_due|follow_up",
    "who": "user|<contact name>",
    "what": "verb-first concise action (5-10 words)",
    "to_whom": "<who expects this or null>",
    "deadline": "YYYY-MM-DD or null",
    "status": "open|done|cancelled",
    "priority": 1|2|3,
    "confidence": 0.0-1.0,
    "provenance": "user_committed|assigned_to_user|group_decision|system_detected",
    "note": "2-3 sentence context: who asked, what happened, why it matters"
  }
]}

If nothing actionable, return: {"commitments": []}"""


# ══════════════════════════════════════════════════════════════════
# Thread grouping and routing
# ══════════════════════════════════════════════════════════════════

def _group_threads(store: GraphStore, tier_filter: str = "deep", days_back: int = 7) -> dict:
    """Group triaged nodes into threads for extraction.

    Args:
        tier_filter: Which tiers to include.
            "deep" — only deep tier nodes
            "lightweight" — lightweight + deep
            "all" — everything triaged
        days_back: Only include nodes from the last N days (0 = no limit)
    """
    tier_clause = {
        "deep": "COALESCE(triage_adjusted, triage_relevant) >= 7",
        "lightweight": "COALESCE(triage_adjusted, triage_relevant) >= 3",
        "all": "COALESCE(triage_adjusted, triage_relevant) IS NOT NULL",
    }.get(tier_filter, "COALESCE(triage_adjusted, triage_relevant) >= 7")

    if days_back > 0:
        cutoff = int(time.time()) - (days_back * 86400)
        time_clause = f"AND timestamp >= {cutoff}"
    else:
        time_clause = ""

    rows = store.conn.execute(
        f"""SELECT id, node_type, source, timestamp, subject, sender,
                  recipients, body_preview, thread_id, tier, pii_flags,
                  domain, topics, entities
           FROM nodes
           WHERE {tier_clause}
             AND embedding IS NOT NULL
             {time_clause}
           ORDER BY timestamp ASC"""
    ).fetchall()

    threads: dict[str, list] = defaultdict(list)
    standalone: list = []
    sensitive_count = 0

    for row in rows:
        row_dict = dict(row)
        pii = []
        if row["pii_flags"]:
            try:
                pii = json.loads(row["pii_flags"]) if isinstance(row["pii_flags"], str) else row["pii_flags"]
            except (json.JSONDecodeError, TypeError):
                pass
        row_dict["_pii"] = pii
        row_dict["_sensitive"] = bool(set(pii) & SENSITIVE_PII)
        if row_dict["_sensitive"]:
            sensitive_count += 1

        if row["thread_id"]:
            threads[row["thread_id"]].append(row_dict)
        elif row["source"] == "imessage":
            # Group 1:1 iMessages by contact as synthetic thread
            contact = None
            sender = row["sender"] or ""
            if sender.strip():
                contact = sender.strip()
            if not contact:
                recips_raw = row["recipients"] or ""
                if recips_raw:
                    try:
                        recips = json.loads(recips_raw) if isinstance(recips_raw, str) else recips_raw
                        if recips and recips[0]:
                            contact = recips[0].strip()
                    except (json.JSONDecodeError, TypeError, IndexError):
                        pass
            if contact:
                synth_thread = f"imessage:{contact}"
                threads[synth_thread].append(row_dict)
        else:
            standalone.append(row_dict)

    # Cap synthetic threads to most recent N messages (already sorted by timestamp ASC)
    MAX_MSGS_PER_THREAD = 50
    for tid in list(threads.keys()):
        if len(threads[tid]) > MAX_MSGS_PER_THREAD:
            threads[tid] = threads[tid][-MAX_MSGS_PER_THREAD:]

    return {
        "threads": dict(threads),
        "standalone": standalone,
        "stats": {
            "total": len(rows),
            "threaded": sum(len(t) for t in threads.values()),
            "standalone": len(standalone),
            "thread_count": len(threads),
            "sensitive": sensitive_count,
        },
    }


def _is_thread_sensitive(thread_nodes: list[dict]) -> bool:
    return any(n["_sensitive"] for n in thread_nodes)


def _get_full_body(node: dict) -> str:
    """Extract full body from node's msgpack data blob, falling back to body_preview."""
    import msgpack

    # Try data blob first (has full body)
    data_blob = node.get("_data_unpacked")
    if data_blob and data_blob.get("body"):
        return data_blob["body"]

    # Fallback to body_preview
    return node.get("body_preview", "") or ""


def _load_node_data(node: dict, store: GraphStore) -> dict:
    """Load and cache the msgpack data blob for a node."""
    if "_data_unpacked" not in node:
        import msgpack
        row = store.conn.execute(
            "SELECT data FROM nodes WHERE id = ?", (node["id"],)
        ).fetchone()
        if row and row["data"]:
            try:
                node["_data_unpacked"] = msgpack.unpackb(row["data"], raw=False)
            except Exception:
                node["_data_unpacked"] = {}
        else:
            node["_data_unpacked"] = {}
    return node


def _format_node_as_markdown(node: dict, index: int | None = None, user_tz: str | None = None) -> str:
    """Format a single node as markdown for LLM consumption.

    Uses markdown formatting (## headers, key: value pairs) which research
    shows is ~16% more token-efficient than JSON and significantly improves
    LLM comprehension accuracy (60.7% vs 44.3% for CSV/JSON).
    """
    from alteris_listener.graph.email_cleaner import clean_email_body

    lines = []

    # Header
    if index is not None:
        lines.append(f"## Message {index}")
        lines.append("")

    # Metadata as markdown key-value pairs
    if node["node_type"] == "email":
        lines.append(f"**From:** {node.get('sender', 'unknown')}")
        if node.get("recipients"):
            try:
                recips = json.loads(node["recipients"]) if isinstance(node["recipients"], str) else node["recipients"]
                lines.append(f"**To:** {', '.join(recips[:5])}")
            except (json.JSONDecodeError, TypeError):
                pass
        # Check for CC in data blob
        data = node.get("_data_unpacked", {})
        if data.get("cc"):
            lines.append(f"**Cc:** {data['cc']}")
        lines.append(f"**Subject:** {node.get('subject', '(no subject)')}")

    elif node["node_type"] == "meeting":
        lines.append(f"**Meeting:** {node.get('subject', '(untitled)')}")
        data = node.get("_data_unpacked", {})
        if data.get("attendees"):
            # Format attendee names if available
            attendees = data["attendees"]
            if isinstance(attendees, list):
                names = [a.get("name") or a.get("email", "?") for a in attendees[:10]]
                lines.append(f"**Attendees:** {', '.join(names)}")
            elif isinstance(attendees, str):
                lines.append(f"**Attendees:** {attendees}")

    elif node["node_type"] == "calendar_event":
        lines.append(f"**Calendar event:** {node.get('subject', '(untitled)')}")
        data = node.get("_data_unpacked", {})
        if data.get("location"):
            lines.append(f"**Location:** {data['location']}")
        if data.get("organizer_name") or data.get("organizer_email"):
            org = data.get("organizer_name") or data.get("organizer_email", "")
            lines.append(f"**Organizer:** {org}")
        if data.get("attendees"):
            attendees = data["attendees"]
            if isinstance(attendees, list):
                names = [a.get("name") or a.get("email", "?") for a in attendees[:10]]
                lines.append(f"**Attendees:** {', '.join(names)}")
        if data.get("event_url"):
            lines.append(f"**URL:** {data['event_url']}")

    elif node.get("source") == "slack":
        lines.append(f"**Slack message from:** {node.get('sender', 'unknown')}")
        if node.get("subject"):
            lines.append(f"**Channel:** {node['subject']}")

    elif node["node_type"] == "message":
        # Detect iMessage direction from is_from_me flag in data blob
        data = node.get("_data_unpacked", {})
        is_from_me = data.get("is_from_me", False)
        if is_from_me:
            recipient = node.get("recipients", "")
            if recipient:
                try:
                    recips = json.loads(recipient) if isinstance(recipient, str) else recipient
                    if recips and recips[0]:
                        lines.append(f"**iMessage from:** USER (you) → {recips[0]}")
                    else:
                        lines.append(f"**iMessage from:** USER (you, outbound)")
                except (json.JSONDecodeError, TypeError, IndexError):
                    lines.append(f"**iMessage from:** USER (you, outbound)")
            else:
                lines.append(f"**iMessage from:** USER (you, outbound)")
            lines.append(f"**Direction:** outbound (user sent this)")
        else:
            lines.append(f"**iMessage from:** {node.get('sender', 'unknown')}")
            lines.append(f"**Direction:** inbound (sent to user)")

    else:
        lines.append(f"**{node.get('source', 'unknown')}/{node['node_type']}:** {node.get('subject', '(untitled)')}")

    if node.get("timestamp"):
        from datetime import datetime, timezone
        try:
            import zoneinfo
            local_tz = zoneinfo.ZoneInfo(user_tz or "America/Los_Angeles")
            dt = datetime.fromtimestamp(node["timestamp"], tz=local_tz)
            lines.append(f"**Date:** {dt.strftime('%Y-%m-%d %H:%M %Z')}")
        except (OSError, ValueError):
            pass

    # Body — full text, cleaned of quoted replies
    lines.append("")
    lines.append("### Body")
    raw_body = _get_full_body(node)

    # Decode HTML entities (&#039; → ', &amp; → &, etc.)
    if raw_body:
        import html as html_mod
        raw_body = html_mod.unescape(raw_body)

    # Strip redundant time range line from calendar bodies (e.g. "19:30 - 20:00")
    # These are raw UTC times baked in during ingestion; the Date header has correct local time
    if node["node_type"] == "calendar_event" and raw_body:
        import re as re_mod
        # Remove lines like "19:30 - 20:00" or "14:00-15:00"
        raw_body = re_mod.sub(r"^\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\s*$", "", raw_body, flags=re_mod.MULTILINE).strip()
        # Also remove the repeated title line if it matches the subject
        subject = node.get("subject", "")
        if subject and raw_body.startswith(subject):
            raw_body = raw_body[len(subject):].strip()

    if node["node_type"] == "email" and raw_body:
        cleaned = clean_email_body(raw_body)
        lines.append(cleaned if cleaned else raw_body)
    else:
        lines.append(raw_body if raw_body else "(no content)")

    return "\n".join(lines)


def _gather_neighbor_context(
    thread_nodes: list[dict],
    store: GraphStore,
    max_neighbors: int = 8,
) -> list[dict]:
    """Walk 1 hop from thread source nodes via strong edges to find related context.

    Uses per-edge-type weight thresholds based on weight distributions:
      same_event:  no filter (always 1.0)
      same_thread: no filter (always 0.8)
      same_entity: >= 0.10 (p75 cutoff — strong entity co-occurrence)
      same_topic:  >= 0.05 (p90 cutoff — high semantic similarity)

    Returns neighbor nodes (calendar events, meetings, related emails)
    that Gemini can use to reason about completions and duplicates.
    """
    # Per-edge-type minimum weights (derived from weight distribution analysis)
    EDGE_WEIGHT_THRESHOLDS = {
        "same_event": 0.0,
        "same_thread": 0.0,
        "same_entity": 0.10,
        "same_topic": 0.05,
    }

    source_ids = {n["id"] for n in thread_nodes}

    # Collect 1-hop neighbors via relationship edges, filtered by per-type weight
    neighbor_ids: dict[str, tuple[str, float]] = {}  # node_id -> (edge_type, weight)
    for src_id in source_ids:
        for etype, min_w in EDGE_WEIGHT_THRESHOLDS.items():
            rows = store.conn.execute(
                """SELECT dst, weight FROM edges
                   WHERE src = ? AND edge_type = ? AND weight >= ?
                   UNION
                   SELECT src, weight FROM edges
                   WHERE dst = ? AND edge_type = ? AND weight >= ?""",
                (src_id, etype, min_w, src_id, etype, min_w),
            ).fetchall()
            for r in rows:
                nbr_id, weight = r[0], r[1]
                if nbr_id not in source_ids:
                    # Keep the highest-weight edge if multiple paths
                    if nbr_id not in neighbor_ids or weight > neighbor_ids[nbr_id][1]:
                        neighbor_ids[nbr_id] = (etype, weight)

    if not neighbor_ids:
        return []

    # Load neighbor node metadata and score by relevance
    neighbors = []
    for nbr_id, (etype, weight) in neighbor_ids.items():
        row = store.conn.execute(
            """SELECT id, node_type, source, timestamp, subject, sender,
                      recipients, body_preview, data
               FROM nodes WHERE id = ?""",
            (nbr_id,),
        ).fetchone()
        if not row:
            continue

        node = dict(row)

        # Prioritize by type: calendar events > meetings > emails
        type_priority = {
            "calendar_event": 0,
            "meeting": 1,
            "email": 2,
            "message": 3,
        }
        # Prioritize by edge type: same_event > same_entity > same_thread > same_topic
        edge_priority = {
            "same_event": 0,
            "same_entity": 1,
            "same_thread": 2,
            "same_topic": 3,
        }

        # Lower score = higher relevance. Weight inverted (higher weight = more relevant)
        score = type_priority.get(node["node_type"], 4) * 10 + edge_priority.get(etype, 4) - (weight * 5)
        node["_sort_score"] = score
        node["_edge_type"] = etype
        node["_edge_weight"] = weight

        # Unpack data blob for formatting
        import msgpack
        if node.get("data"):
            try:
                node["_data_unpacked"] = msgpack.unpackb(node["data"], raw=False)
            except Exception:
                node["_data_unpacked"] = {}
        else:
            node["_data_unpacked"] = {}

        neighbors.append(node)

    # Sort by relevance and cap
    neighbors.sort(key=lambda n: n["_sort_score"])
    return neighbors[:max_neighbors]


def _format_thread_for_cloud(
    thread_nodes: list[dict],
    user_email: str = "",
    user_tz: str | None = None,
    context_nodes: list[dict] | None = None,
) -> str:
    """Format a full thread as markdown for cloud processing."""
    parts = [
        f"**User email:** {user_email}",
        "",
        "The following is a thread of messages in chronological order.",
        "",
        "---",
    ]
    for i, node in enumerate(thread_nodes, 1):
        parts.append("")
        parts.append(_format_node_as_markdown(node, index=i, user_tz=user_tz))
        if i < len(thread_nodes):
            parts.append("")
            parts.append("---")

    # Append cross-source context if available
    if context_nodes:
        parts.append("")
        parts.append("---")
        parts.append("")
        parts.append("## Related context from other sources (READ-ONLY — DO NOT extract commitments from this section)")
        parts.append("This section is provided ONLY to help you determine if commitments in the thread above have been completed or are duplicates. DO NOT extract any new commitments from this section.")
        parts.append("")
        for j, ctx in enumerate(context_nodes, 1):
            # Compact format for context: just subject + date + source type
            ctx_source = ctx.get("source", "unknown")
            ctx_subject = ctx.get("subject") or "(no subject)"
            ctx_date = ""
            if ctx.get("timestamp"):
                from datetime import datetime, timezone
                try:
                    import zoneinfo
                    local_tz = zoneinfo.ZoneInfo(user_tz or "America/Los_Angeles")
                    dt = datetime.fromtimestamp(ctx["timestamp"], tz=local_tz)
                    ctx_date = dt.strftime('%Y-%m-%d %H:%M %Z')
                except (OSError, ValueError):
                    pass
            parts.append(f"- [{ctx_source}] {ctx_subject} ({ctx_date})")
        parts.append("")

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════
# Local extraction (qwen3:30b-a3b, sequential with state)
# ══════════════════════════════════════════════════════════════════

def _extract_local_sequential(
    thread_nodes: list[dict],
    user_email: str = "",
    store: GraphStore | None = None,
) -> list[dict]:
    """Process a sensitive thread locally, message by message with running state."""
    from alteris_listener.graph.local_llm import OllamaClient

    client = OllamaClient()
    model = "qwen3:30b-a3b"
    running_state = {"commitments": [], "context_summary": ""}

    for i, node in enumerate(thread_nodes):
        if store:
            _load_node_data(node, store)
        msg_text = _format_node_as_markdown(node, index=i + 1, user_tz="America/Los_Angeles")
        msg_context = f"**User email:** {user_email}\n\n{msg_text}"

        if i == 0:
            prompt = msg_context
        else:
            state_json = json.dumps(running_state, indent=2)
            prompt = LOCAL_STATE_PREFIX.format(state_json=state_json) + "\n\n" + msg_context

        result = client.generate_json(
            prompt=prompt,
            model=model,
            system=LOCAL_SYSTEM_PROMPT,
            temperature=0.1,
        )

        if result and "commitments" in result:
            running_state["commitments"] = result["commitments"]
            sender = node.get("sender", "unknown")
            subject = node.get("subject", "")
            running_state["context_summary"] = (
                f"Thread about '{subject}'. Last message ({i+1}/{len(thread_nodes)}) "
                f"from {sender}. {len(result['commitments'])} commitments tracked."
            )

    return running_state.get("commitments", [])


# ══════════════════════════════════════════════════════════════════
# Cloud extraction (Gemini Flash, async)
# ══════════════════════════════════════════════════════════════════

async def _extract_cloud_async(
    thread_nodes: list[dict],
    user_email: str = "",
    semaphore: asyncio.Semaphore | None = None,
    store: GraphStore | None = None,
) -> list[dict]:
    """Process a non-sensitive thread via Gemini Flash (async)."""
    from alteris_listener.llm.client import LLMClient

    # Load full bodies before formatting
    if store:
        for node in thread_nodes:
            _load_node_data(node, store)

    # Gather cross-source context via graph traversal
    context_nodes = _gather_neighbor_context(thread_nodes, store) if store else []

    thread_text = _format_thread_for_cloud(thread_nodes, user_email, user_tz="America/Los_Angeles", context_nodes=context_nodes)

    async def _call():
        loop = asyncio.get_event_loop()

        def _sync_call():
            client = LLMClient(provider="gemini", model="gemini-3-flash-preview", thinking_level="low")
            return client.run_json(CLOUD_SYSTEM_PROMPT, thread_text)

        return await loop.run_in_executor(None, _sync_call)

    try:
        if semaphore:
            async with semaphore:
                result = await _call()
        else:
            result = await _call()
        return result.get("commitments", [])
    except Exception as e:
        logger.error("Gemini extraction failed: %s", e)
        return []


def _extract_cloud_sync(
    thread_nodes: list[dict],
    user_email: str = "",
    store: GraphStore | None = None,
) -> list[dict]:
    """Sync wrapper for single cloud call (standalone messages)."""
    from alteris_listener.llm.client import LLMClient

    if store:
        for node in thread_nodes:
            _load_node_data(node, store)

    context_nodes = _gather_neighbor_context(thread_nodes, store) if store else []

    thread_text = _format_thread_for_cloud(thread_nodes, user_email, user_tz="America/Los_Angeles", context_nodes=context_nodes)
    try:
        client = LLMClient(provider="gemini", model="gemini-3-flash-preview", thinking_level="low")
        result = client.run_json(CLOUD_SYSTEM_PROMPT, thread_text)
        return result.get("commitments", [])
    except Exception as e:
        logger.error("Gemini extraction failed: %s", e)
        return []


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def _commitment_id(thread_id: str | None, what: str, who: str) -> str:
    key = f"{thread_id or 'standalone'}:{who}:{what}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _store_commitments(
    derived: DerivedStore,
    commitments_raw: list[dict],
    thread_id: str | None,
    node_ids: list[str],
    model_used: str,
    source_timestamp: int,
) -> int:
    count = 0
    for c in commitments_raw:
        cid = _commitment_id(thread_id, c.get("what", ""), c.get("who", "user"))
        derived.upsert_commitment({
            "id": cid,
            "thread_id": thread_id,
            "source_node_ids": node_ids,
            "commitment_type": c.get("type", "inbound_request"),
            "who": c.get("who", "user"),
            "what": c.get("what", ""),
            "to_whom": c.get("to_whom"),
            "deadline": c.get("deadline"),
            "status": c.get("status", "open"),
            "priority": c.get("priority", 3),
            "confidence": c.get("confidence", 0.5),
            "provenance": c.get("provenance"),
            "note": c.get("note"),
            "raw_extraction": json.dumps(c),
            "prompt_version": PROMPT_VERSION,
            "model_used": model_used,
            "processed_at": int(time.time()),
            "created_at": source_timestamp,
        })
        count += 1
    return count


# ══════════════════════════════════════════════════════════════════
# Async batch processing
# ══════════════════════════════════════════════════════════════════

async def _process_cloud_batch(
    cloud_items: list[tuple[str, list[dict]]],
    user_email: str,
    derived: DerivedStore,
    store: GraphStore | None = None,
) -> tuple[int, int, int]:
    """Process non-sensitive threads via async Gemini calls.

    Returns: (threads_processed, total_commitments, errors)
    """
    semaphore = asyncio.Semaphore(MAX_CLOUD_CONCURRENCY)
    threads_processed = 0
    total_commitments = 0
    errors = 0

    async def _process_one(thread_id: str, nodes: list[dict]) -> tuple[int, str | None]:
        node_ids = [n["id"] for n in nodes]
        t0 = time.time()
        try:
            commitments_raw = await _extract_cloud_async(nodes, user_email, semaphore, store)
            elapsed_ms = int((time.time() - t0) * 1000)

            count = _store_commitments(
                derived, commitments_raw, thread_id, node_ids,
                "gemini-3-flash-preview", nodes[0].get("timestamp", int(time.time())),
            )

            derived.record_run(
                thread_id=thread_id, node_ids=node_ids,
                prompt_version=PROMPT_VERSION, model_used="gemini-3-flash-preview",
                status="success", commitments_found=len(commitments_raw),
                processing_ms=elapsed_ms,
            )

            if commitments_raw:
                console.print(
                    f"    ☁️  {thread_id[:35]}... → "
                    f"{len(commitments_raw)} commitments ({elapsed_ms}ms)"
                )
            return count, None
        except Exception as e:
            derived.record_run(
                thread_id=thread_id, node_ids=node_ids,
                prompt_version=PROMPT_VERSION, model_used="error",
                status="error", error_msg=str(e),
            )
            return 0, str(e)

    tasks = [_process_one(tid, nodes) for tid, nodes in cloud_items]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, Exception):
            errors += 1
            continue
        count, err = r
        if err:
            errors += 1
        else:
            threads_processed += 1
            total_commitments += count

    return threads_processed, total_commitments, errors


# ══════════════════════════════════════════════════════════════════
# Main extraction pipeline
# ══════════════════════════════════════════════════════════════════


def _dedup_and_resolve(
    derived: DerivedStore,
    store: GraphStore,
) -> dict:
    """Post-extraction: merge duplicates and detect completions via graph traversal.

    Uses the knowledge graph edges (same_entity, same_topic, same_thread)
    to find related nodes, rather than re-embedding commitment text.

    Returns dict with 'merged' and 'completed' counts.
    """
    import time as time_mod

    stats = {"merged": 0, "completed": 0}

    all_open = derived.conn.execute(
        "SELECT * FROM commitments WHERE status = 'open'"
    ).fetchall()
    if not all_open:
        return stats

    all_open = [dict(c) for c in all_open]
    now = int(time_mod.time())

    # ── Step 1: Graph-based dedup ──
    # For each commitment, walk 1 hop from source nodes via same_entity / same_thread
    # If two commitments share a neighbor via these strong edges, they're candidates
    commitment_neighbors: dict[str, set[str]] = {}  # commitment_id -> set of neighbor node_ids

    for c in all_open:
        source_ids = json.loads(c["source_node_ids"]) if c["source_node_ids"] else []
        neighbors = set()
        for src_id in source_ids:
            # 1-hop via strong relationship edges
            rows = store.conn.execute(
                """SELECT dst FROM edges WHERE src = ? AND (
                     (edge_type = 'same_entity' AND weight >= 0.10) OR
                     (edge_type = 'same_thread' AND weight >= 0.0)
                   )
                   UNION
                   SELECT src FROM edges WHERE dst = ? AND (
                     (edge_type = 'same_entity' AND weight >= 0.10) OR
                     (edge_type = 'same_thread' AND weight >= 0.0)
                   )""",
                (src_id, src_id),
            ).fetchall()
            for r in rows:
                neighbors.add(r[0])
            neighbors.add(src_id)
        commitment_neighbors[c["id"]] = neighbors

    # Find commitment pairs that share source neighbors (same_entity/same_thread connected)
    merged_ids = set()
    for i, ci in enumerate(all_open):
        if ci["id"] in merged_ids:
            continue
        for j in range(i + 1, len(all_open)):
            cj = all_open[j]
            if cj["id"] in merged_ids:
                continue

            # Check if source nodes are graph-connected
            overlap = commitment_neighbors[ci["id"]] & commitment_neighbors[cj["id"]]
            # Exclude trivial self-overlaps; require a meaningful shared neighbor
            source_ids_i = set(json.loads(ci["source_node_ids"]) if ci["source_node_ids"] else [])
            source_ids_j = set(json.loads(cj["source_node_ids"]) if cj["source_node_ids"] else [])
            shared_external = overlap - source_ids_i - source_ids_j

            if not shared_external:
                # Also check: are the source nodes directly connected?
                direct = source_ids_i & commitment_neighbors[cj["id"]]
                if not direct:
                    continue

            # Additional check: same deadline or similar what
            same_deadline = (ci.get("deadline") and ci["deadline"] == cj.get("deadline"))
            same_person = (
                ci.get("to_whom") and cj.get("to_whom") and
                ci["to_whom"].lower().split()[0] == cj["to_whom"].lower().split()[0]
            )
            # Same what (fuzzy: first 20 chars match after lowering)
            same_what = ci["what"][:25].lower() == cj["what"][:25].lower()

            if same_deadline or same_person or same_what:
                # Additional guard: don't merge incompatible commitment types.
                # e.g. follow_up should not merge with inbound_request
                COMPATIBLE_TYPES = {
                    frozenset({"inbound_request", "user_commitment"}),
                    frozenset({"deadline", "inbound_request"}),
                    frozenset({"deadline", "user_commitment"}),
                }
                type_i = ci.get("commitment_type", "")
                type_j = cj.get("commitment_type", "")
                types_match = (
                    type_i == type_j or
                    frozenset({type_i, type_j}) in COMPATIBLE_TYPES
                )
                if not types_match:
                    continue

                # Keep higher confidence, delete the other
                keeper, dup = (ci, cj) if ci["confidence"] >= cj["confidence"] else (cj, ci)
                derived.conn.execute("DELETE FROM commitments WHERE id = ?", (dup["id"],))
                merged_ids.add(dup["id"])
                stats["merged"] += 1
                logger.info("Dedup: merged '%s' into '%s' (graph-connected)",
                            dup["what"][:50], keeper["what"][:50])

    derived.conn.commit()

    # ── Step 2: Cross-source completion via graph traversal ──
    # For each open commitment, walk 1-2 hops and check if any neighbor is
    # a past calendar event (proof the meeting happened)
    remaining = derived.conn.execute(
        "SELECT * FROM commitments WHERE status = 'open'"
    ).fetchall()

    for c in remaining:
        c = dict(c)
        what_lower = c["what"].lower()
        action_verbs = ("attend", "book", "schedule", "join", "call", "meet", "chat")
        if not any(v in what_lower for v in action_verbs):
            continue

        source_ids = json.loads(c["source_node_ids"]) if c["source_node_ids"] else []

        # Walk 1-2 hops from source nodes, looking for past calendar events
        visited = set()
        frontier = set(source_ids)
        found_completion = False

        for hop in range(2):
            next_frontier = set()
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)

                neighbors = store.conn.execute(
                    """SELECT dst, edge_type FROM edges WHERE src = ? AND (
                         (edge_type = 'same_event' AND weight >= 0.0) OR
                         (edge_type = 'same_entity' AND weight >= 0.10) OR
                         (edge_type = 'same_thread' AND weight >= 0.0) OR
                         (edge_type = 'same_topic' AND weight >= 0.05)
                       )
                       UNION
                       SELECT src, edge_type FROM edges WHERE dst = ? AND (
                         (edge_type = 'same_event' AND weight >= 0.0) OR
                         (edge_type = 'same_entity' AND weight >= 0.10) OR
                         (edge_type = 'same_thread' AND weight >= 0.0) OR
                         (edge_type = 'same_topic' AND weight >= 0.05)
                       )""",
                    (nid, nid),
                ).fetchall()

                for nbr_id, etype in neighbors:
                    if nbr_id in visited:
                        continue
                    # Check if this neighbor is a past calendar event
                    nbr = store.conn.execute(
                        "SELECT source, node_type, timestamp, subject FROM nodes WHERE id = ?",
                        (nbr_id,),
                    ).fetchone()
                    if not nbr:
                        continue

                    if nbr["source"] == "calendar" and nbr["timestamp"]:
                        # Only auto-complete if the calendar event is in the PAST
                        # (i.e., the event has already occurred)
                        event_in_past = nbr["timestamp"] < now

                        # Also check: if the commitment has a deadline, don't mark done
                        # if the deadline is still in the future
                        commitment_deadline_past = True
                        if c.get("deadline"):
                            try:
                                from datetime import datetime
                                dl = datetime.strptime(c["deadline"], "%Y-%m-%d")
                                commitment_deadline_past = dl.timestamp() < now
                            except (ValueError, TypeError):
                                pass

                        if event_in_past and commitment_deadline_past:
                            # Found a past calendar event reachable from this commitment's source
                            derived.conn.execute(
                                """UPDATE commitments
                                   SET status = 'done',
                                       note = note || ' [Auto-resolved: linked to calendar event "' || ? || '"]'
                                   WHERE id = ?""",
                                (nbr["subject"][:80], c["id"]),
                            )
                            stats["completed"] += 1
                            logger.info("Auto-completed: '%s' → calendar '%s' via %s (hop %d)",
                                        c["what"][:50], nbr["subject"][:40], etype, hop + 1)
                            found_completion = True
                            break

                    next_frontier.add(nbr_id)

                if found_completion:
                    break
            if found_completion:
                break
            frontier = next_frontier

    derived.conn.commit()
    return stats


def run_extraction(
    store: GraphStore,
    derived: DerivedStore,
    user_email: str = "",
    use_cloud: bool = True,
    tier_filter: str = "deep",
    days_back: int = 7,
    limit: int | None = None,
    force: bool = False,
) -> dict:
    """Run Pass 4 extraction on triaged threads.

    Args:
        store: Graph store (immutable source data)
        derived: Derived store (for output)
        user_email: User's email for direction detection
        use_cloud: Whether to use Gemini for non-sensitive threads
        tier_filter: "deep", "lightweight", or "all"
        days_back: Only process nodes from last N days (0 = all)
        limit: Max threads to process (None = all)
        force: Re-extract even if already processed
    """
    window = f"last {days_back} days" if days_back > 0 else "all time"
    console.print(f"\n[bold]Pass 4: Commitment Extraction[/bold] (tier={tier_filter}, {window})")

    grouped = _group_threads(store, tier_filter=tier_filter, days_back=days_back)
    stats = grouped["stats"]
    console.print(f"  {stats['total']} nodes: {stats['thread_count']} threads "
                  f"({stats['threaded']} msgs) + {stats['standalone']} standalone")
    console.print(f"  {stats['sensitive']} sensitive nodes (local processing)")

    total_commitments = 0
    threads_processed = 0
    threads_skipped = 0
    errors = 0

    # Partition threads
    local_items: list[tuple[str, list[dict]]] = []
    cloud_items: list[tuple[str, list[dict]]] = []

    thread_items = list(grouped["threads"].items())
    if limit:
        thread_items = thread_items[:limit]

    for thread_id, nodes in thread_items:
        if not force and derived.is_thread_processed(thread_id, PROMPT_VERSION):
            threads_skipped += 1
            continue
        if _is_thread_sensitive(nodes) or not use_cloud:
            local_items.append((thread_id, nodes))
        else:
            cloud_items.append((thread_id, nodes))

    console.print(f"  Routing: {len(local_items)} local, {len(cloud_items)} cloud, "
                  f"{threads_skipped} skipped")

    # ── Local (sequential) ──
    if local_items:
        console.print(f"\n  [bold]Local processing ({len(local_items)} threads):[/bold]")
        for thread_id, nodes in local_items:
            node_ids = [n["id"] for n in nodes]
            t0 = time.time()
            try:
                commitments_raw = _extract_local_sequential(nodes, user_email, store)
                elapsed_ms = int((time.time() - t0) * 1000)

                count = _store_commitments(
                    derived, commitments_raw, thread_id, node_ids,
                    "qwen3:30b-a3b", nodes[0].get("timestamp", int(time.time())),
                )
                derived.record_run(
                    thread_id=thread_id, node_ids=node_ids,
                    prompt_version=PROMPT_VERSION, model_used="qwen3:30b-a3b",
                    status="success", commitments_found=len(commitments_raw),
                    processing_ms=elapsed_ms,
                )
                total_commitments += count
                threads_processed += 1

                if commitments_raw:
                    console.print(
                        f"    🔒 {thread_id[:35]}... → "
                        f"{len(commitments_raw)} commitments ({elapsed_ms}ms)"
                    )
            except Exception as e:
                logger.error("Local extraction failed for %s: %s", thread_id[:30], e)
                derived.record_run(
                    thread_id=thread_id, node_ids=node_ids,
                    prompt_version=PROMPT_VERSION, model_used="error",
                    status="error", error_msg=str(e),
                )
                errors += 1

    # ── Cloud (async parallel) ──
    if cloud_items:
        console.print(f"\n  [bold]Cloud processing ({len(cloud_items)} threads, "
                      f"max {MAX_CLOUD_CONCURRENCY} concurrent):[/bold]")
        t0 = time.time()
        cloud_processed, cloud_commitments, cloud_errors = asyncio.run(
            _process_cloud_batch(cloud_items, user_email, derived, store)
        )
        cloud_elapsed = time.time() - t0
        threads_processed += cloud_processed
        total_commitments += cloud_commitments
        errors += cloud_errors
        console.print(f"    Done: {cloud_processed} threads in {cloud_elapsed:.1f}s "
                      f"({cloud_commitments} commitments, {cloud_errors} errors)")

    # ── Standalone (async parallel for cloud, sequential for local) ──
    standalone = grouped["standalone"]
    if limit:
        standalone = standalone[:max(10, limit // 5)]

    standalone_processed = 0
    if standalone:
        # Filter already processed
        todo = [n for n in standalone if force or not derived.is_node_processed(n["id"], PROMPT_VERSION)]
        local_standalone = [n for n in todo if n["_sensitive"] or not use_cloud]
        cloud_standalone = [n for n in todo if not n["_sensitive"] and use_cloud]

        # Local standalone (sequential)
        if local_standalone:
            console.print(f"\n  [bold]Standalone local ({len(local_standalone)}):[/bold]")
            for node in local_standalone:
                t0 = time.time()
                try:
                    commitments_raw = _extract_local_sequential([node], user_email, store)
                    count = _store_commitments(
                        derived, commitments_raw, None, [node["id"]],
                        "qwen3:30b-a3b", node.get("timestamp", int(time.time())),
                    )
                    elapsed_ms = int((time.time() - t0) * 1000)
                    derived.record_run(
                        thread_id=None, node_ids=[node["id"]],
                        prompt_version=PROMPT_VERSION, model_used="qwen3:30b-a3b",
                        status="success", commitments_found=len(commitments_raw),
                        processing_ms=elapsed_ms,
                    )
                    total_commitments += count
                    standalone_processed += 1
                except Exception as e:
                    logger.error("Standalone local extraction failed: %s", e)
                    errors += 1

        # Cloud standalone (async parallel)
        if cloud_standalone:
            console.print(f"\n  [bold]Standalone cloud ({len(cloud_standalone)}, "
                          f"max {MAX_CLOUD_CONCURRENCY} concurrent):[/bold]")
            # Wrap each standalone node as a pseudo-thread: (node_id, [node])
            pseudo_threads = [(node["id"], [node]) for node in cloud_standalone]
            t0 = time.time()
            sa_processed, sa_commitments, sa_errors = asyncio.run(
                _process_cloud_batch(pseudo_threads, user_email, derived, store)
            )
            sa_elapsed = time.time() - t0
            standalone_processed += sa_processed
            total_commitments += sa_commitments
            errors += sa_errors
            console.print(f"    Done: {sa_processed} messages in {sa_elapsed:.1f}s "
                          f"({sa_commitments} commitments, {sa_errors} errors)")

    # ── Phase 2: Dedup & Cross-source Resolution ──
    dedup_stats = _dedup_and_resolve(derived, store)
    if dedup_stats["merged"] or dedup_stats["completed"]:
        console.print(f"\n  [bold]Post-extraction resolution:[/bold]")
        if dedup_stats["merged"]:
            console.print(f"    Merged {dedup_stats['merged']} duplicate commitments")
        if dedup_stats["completed"]:
            console.print(f"    Marked {dedup_stats['completed']} commitments as done (cross-source)")
        total_commitments -= dedup_stats["merged"]

    # ── Summary ──
    result_stats = {
        "threads_processed": threads_processed,
        "threads_skipped": threads_skipped,
        "standalone_processed": standalone_processed,
        "total_commitments": total_commitments,
        "errors": errors,
    }

    console.print(f"\n  [bold]Extraction complete:[/bold]")
    console.print(f"    Threads: {threads_processed} processed, {threads_skipped} skipped")
    console.print(f"    Standalone: {standalone_processed} processed")
    console.print(f"    Commitments found: {total_commitments}")
    if errors:
        console.print(f"    [red]Errors: {errors}[/red]")

    open_commitments = derived.get_open_commitments(limit=10)
    if open_commitments:
        console.print(f"\n  [bold]Top open commitments:[/bold]")
        for c in open_commitments[:5]:
            deadline = c.get("deadline") or "no deadline"
            console.print(f"    P{c['priority']} [{c['commitment_type']}] {c['what']} ({deadline})")

    overdue = derived.get_overdue_commitments()
    if overdue:
        console.print(f"\n  [bold red]⚠ {len(overdue)} overdue commitments:[/bold red]")
        for c in overdue[:5]:
            console.print(f"    P{c['priority']} {c['what']} (due {c['deadline']})")

    return result_stats
