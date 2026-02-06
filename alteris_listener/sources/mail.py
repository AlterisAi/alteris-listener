"""Mail.app reader — reads from macOS Mail's Envelope Index SQLite database."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from alteris_listener.sources.base import Message

logger = logging.getLogger(__name__)

APPLE_EPOCH_OFFSET = 978307200  # seconds from 1970-01-01 to 2001-01-01


def _row_val(row, key, default=""):
    try:
        val = row[key]
        return val if val is not None else default
    except (IndexError, KeyError):
        return default


def _find_envelope_index() -> Optional[Path]:
    mail_dir = Path.home() / "Library" / "Mail"
    for vdir in sorted(mail_dir.glob("V*"), reverse=True):
        candidate = vdir / "MailData" / "Envelope Index"
        if candidate.exists():
            return candidate
    return None


def _parse_sender(row) -> str:
    sender_addr = _row_val(row, "sender_address", "")
    sender_name = _row_val(row, "sender_name", "")
    if sender_name and sender_addr:
        return f"{sender_name} <{sender_addr}>"
    return sender_addr or sender_name or "unknown"


def _parse_timestamp(row) -> datetime:
    date_val = _row_val(row, "date_sent", 0) or _row_val(row, "date_received", 0)
    if date_val:
        return datetime.fromtimestamp(date_val + APPLE_EPOCH_OFFSET, tz=timezone.utc)
    return datetime.now(timezone.utc)


def read_recent_emails(
    hours: int = 24,
    limit: int = 50,
    since: Optional[datetime] = None,
) -> List[Message]:
    """Read recent emails from Mail.app.

    Args:
        hours: Look back this many hours from now (ignored if since is set).
        limit: Max number of emails to return.
        since: If set, return emails after this datetime.
    """
    db_path = _find_envelope_index()
    if not db_path:
        logger.warning("Mail.app Envelope Index not found")
        return []

    if since:
        cutoff_unix = since.timestamp() - APPLE_EPOCH_OFFSET
    else:
        cutoff_unix = (datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp() - APPLE_EPOCH_OFFSET

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.OperationalError:
        logger.error("Cannot read Mail.app database — check Full Disk Access")
        return []

    query = """
        SELECT
            m.ROWID,
            a.address AS sender_address,
            a.comment AS sender_name,
            s.subject AS subject_text,
            sm.summary AS summary_text,
            m.date_sent,
            m.date_received,
            m.read,
            m.conversation_id,
            mb.url AS mailbox_url
        FROM messages m
        LEFT JOIN addresses a ON m.sender = a.ROWID
        LEFT JOIN subjects s ON m.subject = s.ROWID
        LEFT JOIN summaries sm ON m.summary = sm.ROWID
        LEFT JOIN mailboxes mb ON m.mailbox = mb.ROWID
        WHERE m.date_sent > ?
        ORDER BY m.date_sent DESC
        LIMIT ?
    """

    messages = []
    try:
        for row in conn.execute(query, (cutoff_unix, limit)):
            subject = str(_row_val(row, "subject_text", ""))
            body = str(_row_val(row, "summary_text", ""))
            if len(body) > 3000:
                body = body[:3000]

            conv_id = _row_val(row, "conversation_id", None)

            messages.append(Message(
                source="mail",
                sender=_parse_sender(row),
                recipient="me",
                subject=subject,
                body=body,
                timestamp=_parse_timestamp(row),
                thread_id=str(conv_id) if conv_id else None,
                metadata={"mailbox": _row_val(row, "mailbox_url", "")},
            ))
    except sqlite3.OperationalError as exc:
        logger.error("Mail query failed: %s", exc)
    finally:
        conn.close()

    return messages


def read_email_thread(conversation_id: str, limit: int = 20) -> List[Message]:
    """Read all emails in a Mail.app conversation/thread."""
    db_path = _find_envelope_index()
    if not db_path:
        return []

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.OperationalError:
        return []

    query = """
        SELECT
            m.ROWID,
            a.address AS sender_address,
            a.comment AS sender_name,
            s.subject AS subject_text,
            sm.summary AS summary_text,
            m.date_sent,
            m.date_received,
            m.conversation_id
        FROM messages m
        LEFT JOIN addresses a ON m.sender = a.ROWID
        LEFT JOIN subjects s ON m.subject = s.ROWID
        LEFT JOIN summaries sm ON m.summary = sm.ROWID
        WHERE m.conversation_id = ?
        ORDER BY m.date_sent ASC
        LIMIT ?
    """

    messages = []
    try:
        for row in conn.execute(query, (int(conversation_id), limit)):
            messages.append(Message(
                source="mail",
                sender=_parse_sender(row),
                recipient="me",
                subject=str(_row_val(row, "subject_text", "")),
                body=str(_row_val(row, "summary_text", "")),
                timestamp=_parse_timestamp(row),
                thread_id=str(_row_val(row, "conversation_id", "")),
            ))
    except sqlite3.OperationalError as exc:
        logger.error("Thread query failed: %s", exc)
    finally:
        conn.close()

    return messages
