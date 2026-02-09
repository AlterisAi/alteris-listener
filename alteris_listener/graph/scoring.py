"""Temporal scoring and heuristic importance estimation.

Provides type-aware temporal decay and the composite heuristic scorer
used during bootstrap Pass 1 to filter items before LLM processing.
No LLM calls — purely mathematical scoring.
"""

from __future__ import annotations

import math
import re
import time

URGENT_PATTERNS = [
    r"\burgent\b",
    r"\basap\b",
    r"\bdeadline\b",
    r"\baction required\b",
    r"\btime sensitive\b",
    r"\beod\b",
    r"\bby tomorrow\b",
    r"\bby end of day\b",
    r"\bimmediate(ly)?\b",
    r"\bcritical\b",
    r"\bfinal reminder\b",
    r"\blast chance\b",
    r"\boverdue\b",
]
_urgent_re = re.compile("|".join(URGENT_PATTERNS), re.IGNORECASE)

NOISE_SENDER_PATTERNS = [
    r"no-?reply@",
    r"notifications?@",
    r"newsletter@",
    r"marketing@",
    r"promo(tions)?@",
    r"updates?@",
    r"info@",
    r"support@",
    r"@.*\.noreply\.",
    r"calendar-notification@",
]
_noise_sender_re = re.compile("|".join(NOISE_SENDER_PATTERNS), re.IGNORECASE)

NOISE_SUBJECT_PATTERNS = [
    r"\bunsubscribe\b",
    r"\bnewsletter\b",
    r"\bpromotion(al)?\b",
    r"\bsale\b",
    r"\b\d+% off\b",
    r"\bshop now\b",
    r"\bfree shipping\b",
    r"\bnew arrivals?\b",
    r"\bview in browser\b",
]
_noise_subject_re = re.compile("|".join(NOISE_SUBJECT_PATTERNS), re.IGNORECASE)


def temporal_decay(
    timestamp: int,
    now: int | None = None,
    half_life_days: float = 14.0,
) -> float:
    """Generic exponential decay with configurable half-life.

    Returns a value in (0, 1] where 1 = now and 0.5 = half_life_days ago.
    """
    now = now or int(time.time())
    age_days = max(0, (now - timestamp)) / 86400.0
    return math.exp(-0.693 * age_days / half_life_days)


def temporal_relevance(
    timestamp: int,
    node_type: str,
    now: int | None = None,
    has_pending_action: bool = False,
    is_open_task: bool = False,
) -> float:
    """Type-aware temporal relevance scoring.

    Different node types decay at different rates:
    - Emails with pending actions get MORE relevant up to ~7 days, then fade
    - Informational emails decay fast (half-life ~14 days)
    - Meetings are relevant for follow-up window (~3 days), then drop
    - Open tasks get more urgent with age
    - Contacts don't decay (handled via activity_signals)
    """
    now = now or int(time.time())
    age_days = max(0, (now - timestamp)) / 86400.0

    if node_type in ("email", "message", "slack"):
        if has_pending_action:
            # Bell curve: peaks at ~7 days overdue, fades after ~30 days
            return math.exp(-((age_days - 7) ** 2) / 200)
        # Standard decay
        return math.exp(-age_days / 14)

    if node_type == "meeting":
        if age_days < 3:
            return 1.0
        if age_days < 14:
            return 0.5 * math.exp(-(age_days - 3) / 10)
        return 0.05

    if node_type == "calendar_event":
        # Future events are relevant, past events decay fast
        if age_days < 0:  # future
            return 1.0
        if age_days < 1:
            return 0.8
        return math.exp(-age_days / 7)

    if node_type == "contact":
        return 1.0

    if node_type == "task":
        if is_open_task:
            return min(2.0, 1.0 + age_days / 30)
        return math.exp(-age_days / 30)

    # Default decay
    return math.exp(-age_days / 14)


def heuristic_importance(
    timestamp: int,
    sender: str | None = None,
    recipients: list[str] | None = None,
    subject: str | None = None,
    body_length: int = 0,
    node_type: str = "email",
    is_direct: bool = False,
    thread_size: int = 1,
    sender_total_messages: int = 0,
    sender_reply_ratio: float = 0.0,
    sender_importance_tier: int = 3,
    user_emails: set[str] | None = None,
    now: int | None = None,
) -> float:
    """Compute heuristic importance score (0-1) without any LLM calls.

    This is the primary filter for bootstrap Pass 1. Combines:
    - Temporal decay
    - Sender importance (from contact graph)
    - Directionality (direct vs CC)
    - Thread activity
    - Urgency signals in subject
    - Noise detection (newsletters, marketing)
    """
    now = now or int(time.time())
    user_emails = user_emails or set()
    subject = subject or ""
    sender = sender or ""

    # ── Noise detection (immediate disqualification) ─────────────
    if _noise_sender_re.search(sender):
        return 0.01  # near-zero but not zero (preserves for contact graph)

    if _noise_subject_re.search(subject):
        return 0.02

    # ── Temporal component ───────────────────────────────────────
    temporal = temporal_decay(timestamp, now, half_life_days=30.0)

    # ── Sender importance ────────────────────────────────────────
    if sender_importance_tier == 1:
        sender_score = 1.0   # true inner circle (top ~25 people)
    elif sender_importance_tier == 2:
        sender_score = 0.6
    elif sender_total_messages >= 5:
        sender_score = min(0.5, sender_total_messages / 100)
        sender_score *= (0.5 + 0.5 * sender_reply_ratio)
    else:
        sender_score = 0.15

    # ── Directionality ───────────────────────────────────────────
    direct_bonus = 0.15 if is_direct else 0.0

    # ── Thread activity ──────────────────────────────────────────
    thread_score = min(0.15, thread_size * 0.03)

    # ── Urgency signals ──────────────────────────────────────────
    urgency = 0.15 if _urgent_re.search(subject) else 0.0

    # ── Body length signal (very short = likely auto, very long = likely content) ──
    if body_length < 20:
        body_signal = 0.0
    elif body_length < 200:
        body_signal = 0.05
    else:
        body_signal = 0.1

    # ── Combine ──────────────────────────────────────────────────
    raw_score = (
        sender_score * 0.30 +
        direct_bonus +
        thread_score +
        urgency +
        body_signal +
        0.15  # base relevance
    )

    # Temporal gates everything: old items need high structural signals
    return min(1.0, raw_score * temporal)


def assign_temporal_bucket(timestamp: int, now: int | None = None) -> int:
    """Assign an item to a temporal bucket for bootstrap processing.

    Returns:
        1: Last 7 days (full processing)
        2: 7-30 days (full processing, relaxed threshold)
        3: 30-90 days (heuristic + LLM triage on survivors)
        4: 90 days - 1 year (heuristic + embed, LLM only for high-scoring)
        5: 1+ years (structural only, contact graph, topic clusters)
    """
    now = now or int(time.time())
    age_days = (now - timestamp) / 86400.0

    if age_days <= 7:
        return 1
    if age_days <= 30:
        return 2
    if age_days <= 90:
        return 3
    if age_days <= 365:
        return 4
    return 5


def bucket_score_threshold(bucket: int) -> float:
    """Minimum heuristic score for an item to survive filtering in each bucket."""
    return {
        1: 0.0,    # everything in last 7 days passes
        2: 0.05,   # very permissive
        3: 0.15,   # need some signal
        4: 0.25,   # need decent signal
        5: 0.40,   # only structurally important items from deep past
    }.get(bucket, 0.15)
