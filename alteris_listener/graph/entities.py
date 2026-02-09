"""Deterministic entity extraction from messages.

No LLM calls — pure parsing of structured fields to extract contacts,
email addresses, thread references, and other entities that become
nodes and edges in the graph.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional

# Known user emails — used to determine message directionality
USER_EMAILS = {
    "aniruddha.j@gmail.com",
    "threepwood.galahad@gmail.com",
    "abhargava@me.com",
}

# Patterns to skip (automated senders, no-reply, etc.)
SKIP_SENDER_PATTERNS = [
    r"no-?reply@",
    r"noreply@",
    r"donotreply@",
    r"do-not-reply@",
    r"do_not_reply@",
    r"notifications?@",
    r"mailer-daemon@",
    r"postmaster@",
    r"calendar-notification@",
    r"calendar-server@google\.com",
    r".*@.*\.noreply\.",
    r"notify@",
    r"alerts?@",
    r"^system@",
    r"^admin@",
    # Newsletters and marketing
    r"@substack\.com",
    r"@mail\.substack\.com",
    r"newsletter@",
    r"marketing@",
    r"promo(tions)?@",
    r"updates?@",
    r"^info@",
    r"digest@",
    r"weekly@",
    r"daily@",
    # Transactional / automated services
    r"@venmo\.com",
    r"@paypal\.com",
    r"@email\.informeddelivery",
    r"uspsinformeddelivery@",
    r"@robinhood\.com",
    r"@linkedin\.com",
    r"@facebookmail\.com",
    r"@pinterest\.com",
    r"@twitter\.com",
    r"@x\.com",
    r"@amazonses\.com",
    r"@email\.amazon\.com",
    r"@gc\.email\.amazon\.com",
    r"shipment-tracking@",
    r"order-update@",
    r"receipt@",
    r"billing@",
    r"invoice@",
]

_skip_re = re.compile("|".join(SKIP_SENDER_PATTERNS), re.IGNORECASE)

# Email address extraction
_email_re = re.compile(r"[\w.+-]+@[\w.-]+\.\w{2,}", re.ASCII)

# Name from "Display Name <email>" format
_name_email_re = re.compile(r"^(.+?)\s*<(.+?)>$")


@dataclass
class ExtractedContact:
    """A contact parsed from message fields."""
    email: str
    display_name: str = ""
    source: str = ""

    @property
    def normalized_email(self) -> str:
        return self.email.lower().strip()

    @property
    def contact_id(self) -> str:
        return f"contact:{self.normalized_email}"

    @property
    def is_user(self) -> bool:
        return self.normalized_email in USER_EMAILS

    @property
    def is_automated(self) -> bool:
        return bool(_skip_re.search(self.normalized_email))


@dataclass
class ExtractedEntities:
    """All entities deterministically extracted from a single message."""
    node_id: str
    sender: Optional[ExtractedContact] = None
    recipients: list[ExtractedContact] = field(default_factory=list)
    cc_recipients: list[ExtractedContact] = field(default_factory=list)
    thread_id: Optional[str] = None
    is_from_user: bool = False
    is_to_user: bool = False
    body_hash: str = ""
    body_preview: str = ""
    subject_clean: str = ""


def parse_address(raw: str) -> Optional[ExtractedContact]:
    """Parse a sender/recipient string into an ExtractedContact.

    Handles formats:
      - "email@example.com"
      - "Display Name <email@example.com>"
      - "me" (returns None)
    """
    if not raw or raw.strip().lower() in ("me", "unknown", "meeting", ""):
        return None

    raw = raw.strip()
    match = _name_email_re.match(raw)
    if match:
        name = match.group(1).strip().strip('"').strip("'")
        email = match.group(2).strip()
        return ExtractedContact(email=email, display_name=name)

    emails = _email_re.findall(raw)
    if emails:
        return ExtractedContact(email=emails[0], display_name="")

    # If it looks like a phone number or handle (iMessage)
    if raw.startswith("+") or raw.replace("-", "").replace(" ", "").isdigit():
        return ExtractedContact(email=raw, display_name="")

    return None


def parse_recipients_field(raw: str) -> list[ExtractedContact]:
    """Parse a comma/semicolon-separated recipients field."""
    contacts = []
    if not raw or raw.strip().lower() in ("me", ""):
        return contacts

    for part in re.split(r"[;,]", raw):
        c = parse_address(part.strip())
        if c:
            contacts.append(c)

    return contacts


def clean_subject(subject: str) -> str:
    """Remove Re:/Fwd: prefixes for thread grouping."""
    if not subject:
        return ""
    cleaned = re.sub(r"^(Re|Fwd|Fw)\s*:\s*", "", subject, flags=re.IGNORECASE)
    return cleaned.strip()


def compute_body_hash(body: str) -> str:
    """SHA-256 of body text for dedup."""
    return hashlib.sha256(body.encode("utf-8", errors="replace")).hexdigest()[:16]


def make_node_id(source: str, source_id: str) -> str:
    """Create a deterministic node ID from source type and source-native ID."""
    return f"{source}:{source_id}"


def extract_entities_from_message(msg: "Message", source: str = "") -> ExtractedEntities:
    """Extract all deterministic entities from a Message object.

    This is the main entry point for Pass 0 of the bootstrap.
    No LLM calls — purely structural extraction.
    """
    from alteris_listener.sources.base import Message

    src = source or msg.source
    meta = msg.metadata or {}

    # Determine source-native ID
    if src == "mail":
        # Each email in a thread needs its own node ID.
        # Use thread_id + sender + timestamp for uniqueness.
        ts_str = str(int(msg.timestamp.timestamp())) if msg.timestamp else "0"
        sender_hash = (msg.sender or "unknown")[:50]
        raw_id = f"{msg.thread_id or 'nothd'}:{sender_hash}:{ts_str}"
        node_id = make_node_id("email", raw_id)
    elif src == "imessage":
        raw_id = meta.get("message_id", msg.thread_id or str(hash(f"{msg.sender}:{msg.body[:100]}:{msg.timestamp}")))
        node_id = make_node_id("message", raw_id)
    elif src == "slack":
        ch = meta.get("channel_id", "")
        ts = meta.get("ts", "")
        raw_id = f"{ch}_{ts}" if ch and ts else str(hash(f"{msg.sender}:{msg.body[:100]}"))
        node_id = make_node_id("slack", raw_id)
    elif src == "granola":
        raw_id = meta.get("document_id", str(hash(msg.subject or msg.body[:100])))
        node_id = make_node_id("meeting", raw_id)
    elif src == "calendar":
        raw_id = meta.get("event_id", str(hash(f"{msg.subject}:{msg.timestamp}")))
        node_id = make_node_id("calendar", raw_id)
    else:
        raw_id = str(hash(f"{src}:{msg.sender}:{msg.body[:100]}:{msg.timestamp}"))
        node_id = make_node_id(src, raw_id)

    # Parse sender
    sender = parse_address(msg.sender)
    if sender:
        sender.source = src

    # Parse recipients
    recipients = []
    if msg.recipient:
        recipients = parse_recipients_field(msg.recipient)

    # Directionality
    is_from_user = msg.is_from_me or (sender is not None and sender.is_user)
    is_to_user = any(r.is_user for r in recipients) if recipients else not is_from_user

    # Body processing
    body = msg.body or ""
    body_hash = compute_body_hash(body)
    body_preview = body[:500].replace("\n", " ").strip()

    return ExtractedEntities(
        node_id=node_id,
        sender=sender,
        recipients=recipients,
        cc_recipients=[],  # populated by mail source if CC data available
        thread_id=msg.thread_id,
        is_from_user=is_from_user,
        is_to_user=is_to_user,
        body_hash=body_hash,
        body_preview=body_preview,
        subject_clean=clean_subject(msg.subject or ""),
    )
