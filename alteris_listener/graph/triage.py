"""Pass 3: LLM triage — fast binary classification of graph nodes.

Uses qwen3:8b via Ollama to classify which embedded nodes contain
actionable content worth deep extraction in Pass 4. Runs on the
~5-6K nodes that survived the heuristic score filter and were embedded.

Each node is classified as relevant (contains task, commitment, pending
request, decision, or relationship signal) or not. Graph context is
included: sender tier, thread activity, temporal bucket.

Output is stored as a triage_result column on the nodes table.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from json_repair import repair_json

from alteris_listener.graph.local_llm import OllamaClient
from alteris_listener.graph.scoring import assign_temporal_bucket
from alteris_listener.graph.store import GraphStore

logger = logging.getLogger(__name__)

TRIAGE_MODEL = "qwen3:30b-a3b"

VALID_DOMAINS = {"work", "personal", "family", "financial", "health", "legal", "travel", "shopping", "automated"}
VALID_PII = {"financial", "medical", "legal", "credentials", "travel_docs"}


class TriageResult(BaseModel):
    """Validated triage result for a single item."""
    id: str = ""
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    domain: str = ""
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    pii: list[str] = Field(default_factory=list)

    @field_validator("score")
    @classmethod
    def round_score(cls, v):
        return round(max(0.0, min(1.0, v)), 1)

    @field_validator("reason")
    @classmethod
    def truncate_reason(cls, v):
        return str(v)[:200]

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v):
        v = str(v).lower().strip()
        return v if v in VALID_DOMAINS else ""

    @field_validator("topics")
    @classmethod
    def clean_topics(cls, v):
        if not isinstance(v, list):
            return []
        return [str(t).strip().lower()[:50] for t in v[:5] if t]

    @field_validator("entities")
    @classmethod
    def clean_entities(cls, v):
        if not isinstance(v, list):
            return []
        return [str(e).strip()[:100] for e in v[:10] if e]

    @field_validator("pii")
    @classmethod
    def validate_pii(cls, v):
        if not isinstance(v, list):
            return []
        return [str(p).lower().strip() for p in v if str(p).lower().strip() in VALID_PII]

TRIAGE_SYSTEM = """You are a strict triage classifier for a personal knowledge graph. You decide how much AI analysis a communication item deserves.

The core question: "Does the USER need to THINK, DECIDE, or ACT on this?"

Score from 0.0 to 1.0 in increments of 0.1. The score determines processing tier:

## SCORE 0.1 = IGNORE (no processing)
Automated noise. No human thought needed.
- OTPs, 2FA codes, verification emails, "confirm your email"
- Security alerts (new device, password changed)
- Delivery tracking updates, shipping notifications
- Order confirmations where user already placed the order
- Receipts for completed transactions (parking, tolls)
- Subscription renewal confirmations
- Automated service check-ins ("need help?", "how was your experience?")
- Spam, marketing with no content value
- App notifications (Slack bot, GitHub CI, build status)
- Simple acks with no context value ("ok", "thanks", "got it") from non-inner-circle

## SCORE 0.3 = LIGHTWEIGHT (quick context extraction)
Worth indexing for context, not worth deep analysis.
- Newsletters with actual content (news, analysis, industry updates)
- Financial charges and statements (not OTPs, but actual spend: "$47.99 at Amazon")
- Community/school updates with useful information
- Calendar invites with attendees and agenda (the metadata is useful)
- "Just checking in" or personal updates from known contacts
- Status updates or progress reports (FYI, no response needed)
- Acknowledgments from tier-1 contacts in active threads ("sounds good" from spouse)
- Donation receipts (useful for tax records)
- Automated summaries with useful content (daily digests with real info)

## SCORE 0.5 = LIGHTWEIGHT (higher priority context)
Coordination or soft requests.
- Scheduling back-and-forth ("when are you free?")
- Optional RSVPs, sign-ups with deadlines
- "Let me know if you have questions"
- Informational emails that may need follow-up later
- Forwarded articles or links with personal note

## SCORE 0.7 = DEEP (clear action required)
A human needs the user to do something specific.
- Explicit task requests ("please send", "please review", "can you")
- Invoices, forms, or documents requiring response
- Commitments the user made that need follow-through
- Decisions the user needs to make with consequences
- Professional opportunities requiring response (job discussions, investor outreach, partnerships)
- Legal or financial documents requiring review
- Introductions that expect a follow-up

## SCORE 0.9 = DEEP (urgent/critical)
Immediate action needed.
- Same-day or imminent deadlines
- Financial or legal risk
- Blocking issues for other people
- Escalations requiring immediate response
- Time-sensitive professional opportunities

## Key rules:
- "Personalized" does NOT mean important. A bank alert with your name is still 0.1.
- "Time-sensitive" does NOT mean important. A USPS delivery window is still 0.1.
- Automated messages from noreply/alerts/notifications addresses are 0.1 unless they contain genuinely useful financial data (then 0.3).
- A human asking the user to do something specific is 0.7+.
- Sender tier matters: acks from tier-1 in active threads = 0.3 (context). Same from tier-3 = 0.1.
- Recency matters for the 0.3 tier: recent context is more valuable than old context.

Respond with ONLY valid JSON, no other text. No markdown fences."""

TRIAGE_ITEM_TEMPLATE = """--- ITEM {idx} (id: {node_id}) ---
SENDER: {sender} (tier {sender_tier}, {sender_msgs} messages, {reply_ratio} reply ratio)
TO: {recipients}
DATE: {date} ({bucket_label}, {age_days}d ago)
SUBJECT: {subject}
THREAD: {thread_info}

BODY PREVIEW:
{body_preview}
"""

TRIAGE_BATCH_SUFFIX = """---
Classify ALL items above. Return a JSON array with one object per item, in order.
Each object must have:
- "id": the item id string
- "score": 0.0-1.0 (triage importance)
- "reason": 25 words or less
- "domain": one of: work, personal, family, financial, health, legal, travel, shopping, automated
- "topics": 1-3 tags describing WHAT was discussed, NOT the format. Bad: "email", "meeting", "agenda". Good: "fundraising", "hiring", "product roadmap", "tax documents", "school enrollment", "childcare"
- "entities": companies, products, projects, or concepts MENTIONED IN THE BODY. Do NOT include: sender/recipient names or emails already in SENDER/TO fields, generic platforms (LinkedIn, Google, Gmail, Slack, Zoom, YouTube), payment processors (Chase, Visa, Mastercard), or mailing tools (Loops, Mailchimp, Substack). E.g. ["Meta hiring team", "Series A", "Alteris onboarding"]
- "pii": array of PII types present, or empty. Types: financial (account numbers, transactions, tax docs), medical (health records, prescriptions), legal (contracts, lawsuits), credentials (passwords, API keys, tokens), travel_docs (passport, visa numbers)"""

BUCKET_LABELS = {1: "hot (≤7d)", 2: "recent (7-30d)", 3: "warm (30-90d)", 4: "aging (90d-1y)", 5: "old (1y+)"}


def build_triage_item(node: dict, store: GraphStore, now: int, idx: int = 1) -> str:
    """Build the triage text for a single item in a batch."""
    sender = node.get("sender", "unknown")
    subject = node.get("subject", "(no subject)")
    body_preview = node.get("body_preview", "")[:500]
    timestamp = node.get("timestamp") or now

    # Sender context from contact stats
    sender_tier = 3
    sender_msgs = 0
    reply_ratio = "0%"
    if sender:
        contact_id = f"contact:{sender.lower()}"
        contact = store.get_contact(contact_id)
        if contact:
            sender_tier = contact.get("importance_tier", 3)
            sender_msgs = contact.get("total_messages", 0)
            rr = contact.get("reply_ratio", 0.0)
            reply_ratio = f"{rr:.0%}"

    # Recipients
    recipients_raw = node.get("recipients")
    if isinstance(recipients_raw, str):
        recipients_raw = json.loads(recipients_raw)
    recipients = ", ".join(recipients_raw[:3]) if recipients_raw else "unknown"
    if recipients_raw and len(recipients_raw) > 3:
        recipients += f" (+{len(recipients_raw) - 3} more)"

    # Temporal context
    bucket = assign_temporal_bucket(timestamp, now)
    bucket_label = BUCKET_LABELS.get(bucket, "unknown")
    age_days = max(0, (now - timestamp) / 86400)

    # Thread context
    thread_id = node.get("thread_id")
    if thread_id:
        thread_count = store.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE thread_id = ?", (thread_id,)
        ).fetchone()[0]
        thread_info = f"{thread_count} messages in thread"
    else:
        thread_info = "standalone message"

    # Date
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    date_str = dt.strftime("%Y-%m-%d %H:%M")

    return TRIAGE_ITEM_TEMPLATE.format(
        idx=idx,
        node_id=node["id"],
        sender=sender,
        sender_tier=sender_tier,
        sender_msgs=sender_msgs,
        reply_ratio=reply_ratio,
        recipients=recipients,
        date=date_str,
        bucket_label=bucket_label,
        age_days=f"{age_days:.0f}",
        subject=subject,
        thread_info=thread_info,
        body_preview=body_preview or "(empty)",
    )


def _split_flat_response(cleaned: str, batch_ids: list[str]) -> dict[str, TriageResult | None] | None:
    """Handle LLM responses that mash multiple objects into one flat object.

    The LLM sometimes returns: {"id":"a","score":0.5,...,"id":"b","score":0.3,...}
    instead of an array. Split on the "id" boundaries and parse each chunk.
    """
    # Look for repeated "id" keys — signature of a flat mashed response
    id_positions = [m.start() for m in re.finditer(r'"id"\s*:', cleaned)]
    if len(id_positions) < 2:
        return None  # Not a mashed response

    # Split into individual object strings
    chunks = []
    for i, pos in enumerate(id_positions):
        # Find the start — back up to include the opening brace or comma
        start = pos
        end = id_positions[i + 1] if i + 1 < len(id_positions) else len(cleaned)

        chunk = cleaned[start:end].strip().rstrip(",").strip()
        # Wrap in braces if needed
        if not chunk.startswith("{"):
            chunk = "{" + chunk
        if not chunk.endswith("}"):
            chunk = chunk + "}"

        chunks.append(chunk)

    # Parse each chunk
    results: dict[str, TriageResult | None] = {nid: None for nid in batch_ids}
    parsed_items: list[tuple[str, TriageResult]] = []

    for chunk in chunks:
        try:
            repaired = repair_json(chunk, return_objects=True)
            if isinstance(repaired, dict):
                tr = _parse_one_result(repaired)
                if tr:
                    parsed_items.append((str(repaired.get("id", "")), tr))
        except Exception:
            continue

    if not parsed_items:
        return None

    # Match by id, then positional
    for fid, tr in parsed_items:
        if fid in results:
            results[fid] = tr

    # Fill remaining by position
    unmatched_results = [tr for fid, tr in parsed_items if fid not in batch_ids]
    unmatched_ids = [nid for nid in batch_ids if results[nid] is None]
    for nid, tr in zip(unmatched_ids, unmatched_results):
        results[nid] = tr

    return results


def _clean_llm_output(raw: str) -> str:
    """Strip thinking tags and markdown fences from LLM output."""
    cleaned = raw.strip()
    if "<think>" in cleaned:
        parts = cleaned.split("</think>")
        cleaned = parts[-1].strip() if len(parts) > 1 else cleaned
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _parse_one_result(obj: dict) -> TriageResult | None:
    """Parse a single dict into a validated TriageResult."""
    try:
        return TriageResult.model_validate(obj)
    except Exception:
        # Fallback: try to extract just score
        if "score" in obj:
            try:
                return TriageResult(score=float(obj["score"]), reason=str(obj.get("reason", "")))
            except Exception:
                pass
        # Handle old bool format
        if "relevant" in obj:
            try:
                return TriageResult(score=0.7 if obj["relevant"] else 0.1, reason=str(obj.get("reason", "")))
            except Exception:
                pass
        return None


def parse_triage_response(raw: str, batch_ids: list[str]) -> dict[str, TriageResult | None]:
    """Parse LLM JSON response for a batch of items.

    Uses json_repair for malformed JSON, then pydantic for validation.
    Returns {node_id: TriageResult | None} for each item.
    """
    if not raw:
        return {nid: None for nid in batch_ids}

    cleaned = _clean_llm_output(raw)

    # For batches, first try splitting flat mashed objects (most common LLM failure mode)
    if len(batch_ids) > 1:
        flat_results = _split_flat_response(cleaned, batch_ids)
        if flat_results and sum(1 for v in flat_results.values() if v is not None) > 1:
            return flat_results

    # Try json_repair first (handles truncated JSON, missing quotes, trailing commas)
    try:
        repaired = repair_json(cleaned, return_objects=True)
    except Exception:
        repaired = None

    # If repair failed, try raw json.loads
    if repaired is None:
        try:
            repaired = json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    if repaired is None:
        # Last resort: extract individual JSON objects via regex
        results: dict[str, TriageResult | None] = {}
        found = []
        for match in re.finditer(r'\{[^{}]+\}', cleaned):
            try:
                obj = json.loads(match.group())
                parsed = _parse_one_result(obj)
                if parsed:
                    found.append((obj.get("id", ""), parsed))
            except (json.JSONDecodeError, ValueError):
                continue

        for i, nid in enumerate(batch_ids):
            matched = None
            # Match by id field
            for fid, fparsed in found:
                if fid == nid:
                    matched = fparsed
                    break
            # Fallback: positional
            if matched is None and i < len(found):
                matched = found[i][1]
            results[nid] = matched

        if not any(v is not None for v in results.values()):
            logger.warning("Failed to parse triage response: %s...", raw[:300])
        return results

    # Successfully parsed — now match to batch_ids
    results = {nid: None for nid in batch_ids}

    if isinstance(repaired, list):
        # Array response — match by "id" field then positional
        id_map: dict[str, dict] = {}
        for item in repaired:
            if isinstance(item, dict) and "id" in item:
                id_map[str(item["id"])] = item

        for i, nid in enumerate(batch_ids):
            item = id_map.get(nid)
            if not item and i < len(repaired) and isinstance(repaired[i], dict):
                item = repaired[i]
            if item:
                results[nid] = _parse_one_result(item)

    elif isinstance(repaired, dict):
        # Single object — apply to first (or only) item
        parsed = _parse_one_result(repaired)
        if batch_ids:
            # Check if it has an id that matches
            obj_id = str(repaired.get("id", ""))
            if obj_id in results:
                results[obj_id] = parsed
            else:
                results[batch_ids[0]] = parsed

    return results


def run_triage(
    store: GraphStore,
    model: str = TRIAGE_MODEL,
    parallel: int = 3,
    resume: bool = True,
    batch_size: int = 5,
) -> dict:
    """Run Pass 3 LLM triage on all embedded nodes.

    Classifies each node by score using qwen3:8b with concurrent requests.
    Batches thread-related items together for context and throughput.
    Results stored in nodes table (triage_relevant as 0-10, triage_reason).

    Args:
        store: GraphStore instance.
        model: Ollama model name for triage.
        parallel: Number of concurrent Ollama requests.
        resume: If True, skip nodes that already have triage results.
        batch_size: Max items per LLM call (1-20). Default 5.

    Returns dict with triage stats.
    """
    client = OllamaClient()
    if not client.is_available():
        return {"error": "ollama_not_running"}

    now = int(time.time())

    if resume:
        rows = store.conn.execute(
            """SELECT id, node_type, source, timestamp, subject, sender,
                      recipients, body_preview, heuristic_score, tier, thread_id
               FROM nodes
               WHERE embedding IS NOT NULL
                 AND (triage_relevant IS NULL OR triage_reason = 'PARSE_FAILED')
               ORDER BY heuristic_score DESC"""
        ).fetchall()
    else:
        rows = store.conn.execute(
            """SELECT id, node_type, source, timestamp, subject, sender,
                      recipients, body_preview, heuristic_score, tier, thread_id
               FROM nodes
               WHERE embedding IS NOT NULL
               ORDER BY heuristic_score DESC"""
        ).fetchall()

    if not rows:
        return {"triaged": 0, "ignore": 0, "lightweight": 0, "deep": 0, "failed": 0}

    tier_counts = {"ignore": 0, "lightweight": 0, "deep": 0}
    score_histogram = {}
    failed_count = 0
    triaged = 0

    start_time = time.time()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

    console = Console()
    console.print(f"[bold]Pass 3: LLM Triage (model={model}, candidates={len(rows)}, parallel={parallel}, batch_size={batch_size})[/bold]")

    # Pre-build per-item text on main thread (SQLite reads aren't thread-safe)
    node_items: dict[str, str] = {}  # node_id -> item text
    node_meta: dict[str, dict] = {}  # node_id -> {node_type, sender_tier}

    for row in rows:
        node = dict(row)
        item_text = build_triage_item(node, store, now, idx=1)
        node_items[node["id"]] = item_text

        # Cache metadata for score floor rules
        sender = node.get("sender", "")
        sender_tier = 3
        if sender:
            contact_id = f"contact:{sender.lower()}"
            contact = store.get_contact(contact_id)
            if contact:
                sender_tier = contact.get("importance_tier", 3)
        node_meta[node["id"]] = {
            "node_type": node.get("node_type", ""),
            "source": node.get("source", ""),
            "sender_tier": sender_tier,
        }

    # Build batches
    MAX_BATCH = max(1, min(20, batch_size))
    all_ids = list(node_items.keys())
    batches: list[list[str]] = []
    for i in range(0, len(all_ids), MAX_BATCH):
        batches.append(all_ids[i:i + MAX_BATCH])

    total_items = len(all_ids)

    def _apply_score_floor(score: float, node_id: str) -> float:
        """Apply minimum score floors based on node type and sender tier."""
        meta = node_meta.get(node_id, {})
        ntype = meta.get("node_type", "")
        tier = meta.get("sender_tier", 3)
        if ntype == "meeting":
            score = max(score, 0.5)
        elif ntype == "calendar_event":
            score = max(score, 0.3)
        if tier == 1 and score < 0.3:
            score = 0.3
        return score

    def classify_batch(batch_ids: list[str]) -> dict[str, TriageResult | None]:
        """Classify a batch of items in one LLM call, with retry."""
        items_text = []
        for i, nid in enumerate(batch_ids, 1):
            text = node_items[nid]
            if len(batch_ids) > 1:
                text = text.replace("--- ITEM 1 (", f"--- ITEM {i} (", 1)
            items_text.append(text)

        if len(batch_ids) > 1:
            prompt = "\n".join(items_text) + "\n" + TRIAGE_BATCH_SUFFIX
        else:
            prompt = items_text[0] + "\n" + TRIAGE_BATCH_SUFFIX

        max_tokens = 200 * len(batch_ids)

        MAX_RETRIES = 2
        results: dict[str, TriageResult | None] = {nid: None for nid in batch_ids}

        for attempt in range(MAX_RETRIES + 1):
            pending_ids = [nid for nid in batch_ids if results[nid] is None]
            if not pending_ids:
                break

            if attempt > 0:
                items_text = []
                for i, nid in enumerate(pending_ids, 1):
                    text = node_items[nid]
                    if len(pending_ids) > 1:
                        text = text.replace("--- ITEM 1 (", f"--- ITEM {i} (", 1)
                    items_text.append(text)
                if len(pending_ids) > 1:
                    prompt = "\n".join(items_text) + "\n" + TRIAGE_BATCH_SUFFIX
                else:
                    prompt = items_text[0] + "\n" + TRIAGE_BATCH_SUFFIX
                max_tokens = 200 * len(pending_ids)
                logger.info("Retry %d: %d/%d items", attempt, len(pending_ids), len(batch_ids))

            raw_response = client.generate(
                prompt=prompt,
                model=model,
                system=TRIAGE_SYSTEM,
                temperature=0.1,
                max_tokens=max_tokens,
                format_json=True,
            )
            batch_results = parse_triage_response(raw_response, pending_ids)

            for nid, val in batch_results.items():
                if val is not None:
                    results[nid] = val

        return results

    pending_writes = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Triaging...", total=total_items)

        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(classify_batch, batch): batch
                for batch in batches
            }

            for future in as_completed(futures):
                batch_ids = futures[future]
                try:
                    results = future.result()
                except Exception as exc:
                    logger.error("Batch triage failed: %s", exc)
                    results = {nid: None for nid in batch_ids}

                for node_id in batch_ids:
                    result = results.get(node_id)

                    if result is not None:
                        score = _apply_score_floor(result.score, node_id)
                        result.score = score
                        pending_writes.append((
                            int(score * 10),
                            result.reason,
                            result.domain,
                            json.dumps(result.topics),
                            json.dumps(result.entities),
                            json.dumps(result.pii),
                            node_id,
                        ))
                        score_histogram[score] = score_histogram.get(score, 0) + 1
                        if score < 0.3:
                            tier_counts["ignore"] += 1
                        elif score < 0.7:
                            tier_counts["lightweight"] += 1
                        else:
                            tier_counts["deep"] += 1
                        triaged += 1
                    else:
                        failed_count += 1
                        pending_writes.append((1, "PARSE_FAILED", "", "[]", "[]", "[]", node_id))

                # Batch write to SQLite every 50 results
                if len(pending_writes) >= 50:
                    for args in pending_writes:
                        store.conn.execute(
                            """UPDATE nodes SET triage_relevant = ?, triage_reason = ?,
                               domain = ?, topics = ?, entities = ?, pii_flags = ?
                               WHERE id = ?""",
                            args,
                        )
                    store.conn.commit()
                    pending_writes.clear()

                progress.update(task, advance=len(batch_ids))

    # Flush remaining writes
    for args in pending_writes:
        store.conn.execute(
            """UPDATE nodes SET triage_relevant = ?, triage_reason = ?,
               domain = ?, topics = ?, entities = ?, pii_flags = ?
               WHERE id = ?""",
            args,
        )
    store.conn.commit()

    elapsed = time.time() - start_time
    rate = triaged / max(elapsed, 1)

    store.set_bootstrap_state("last_pass", "triage")
    store.set_bootstrap_state("triage_completed_at", str(int(time.time())))

    console.print()
    console.print(f"  [green]Triaged {triaged} nodes in {elapsed:.0f}s ({rate:.1f}/s)[/green]")

    console.print(f"  Score histogram:")
    for score_val in sorted(score_histogram.keys()):
        count = score_histogram[score_val]
        pct = count / max(triaged, 1) * 100
        bar = "█" * max(1, int(pct / 2))
        console.print(f"    {score_val:.1f}: {count:>5}  ({pct:4.1f}%)  {bar}")

    console.print()
    console.print(f"  [dim]Ignore (<0.3):[/dim]           {tier_counts['ignore']:>5}  → no processing")
    console.print(f"  [yellow]Lightweight (0.3-0.6):[/yellow]  {tier_counts['lightweight']:>5}  → qwen3:8b context extraction")
    console.print(f"  [green]Deep (0.7-1.0):[/green]         {tier_counts['deep']:>5}  → qwen3:30b-a3b full extraction")
    console.print(f"  [red]Parse failed:[/red]            {failed_count:>5}")

    return {
        "triaged": triaged,
        "ignore": tier_counts["ignore"],
        "lightweight": tier_counts["lightweight"],
        "deep": tier_counts["deep"],
        "failed": failed_count,
        "score_histogram": score_histogram,
        "elapsed_seconds": elapsed,
        "rate_per_second": rate,
    }
