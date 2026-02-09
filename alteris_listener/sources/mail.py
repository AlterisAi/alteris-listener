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

# Mail.app V10+ stores date_sent as Unix epoch (NOT Apple epoch).
# Older versions used Apple epoch (seconds since 2001-01-01).
# We auto-detect based on value range: Unix timestamps for 2000-2030
# fall in ~9.5e8 to ~1.9e9, while Apple epoch values for the same
# range would be ~0 to ~9.5e8.
_UNIX_EPOCH_THRESHOLD = 9.5e8  # ~2000-01-01 in Unix epoch


def _to_unix_timestamp(raw_date: float) -> float:
    """Convert a Mail.app date_sent value to Unix timestamp.

    Auto-detects whether the value is Unix epoch or Apple epoch.
    """
    if raw_date <= 0:
        return 0.0
    if raw_date < _UNIX_EPOCH_THRESHOLD:
        # Apple epoch (older Mail.app versions)
        return raw_date + APPLE_EPOCH_OFFSET
    # Already Unix epoch (Mail.app V10+)
    return raw_date

# All known user email addresses — used to detect outbound mail
_USER_ADDRESSES = {
    "aniruddha.j@gmail.com",
    "threepwood.galahad@gmail.com",
    "abhargava@me.com",
}


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
        unix_ts = _to_unix_timestamp(date_val)
        return datetime.fromtimestamp(unix_ts, tz=timezone.utc)
    return datetime.now(timezone.utc)


def _is_from_user(sender_addr: str) -> bool:
    """Check if sender is one of the user's email addresses."""
    if not sender_addr:
        return False
    return sender_addr.lower().strip() in _USER_ADDRESSES


def _fetch_recipients(
    conn: sqlite3.Connection,
    message_rowids: list[int],
) -> dict[int, dict[str, list[str]]]:
    """Fetch recipients for a batch of messages.

    Returns {message_rowid: {"to": [...], "cc": [...], "bcc": [...]}}.
    recipient.type: 0=To, 1=CC, 2=BCC
    """
    if not message_rowids:
        return {}

    result: dict[int, dict[str, list[str]]] = {}

    chunk_size = 500
    for i in range(0, len(message_rowids), chunk_size):
        chunk = message_rowids[i:i + chunk_size]
        placeholders = ",".join("?" * len(chunk))

        rows = conn.execute(
            f"""SELECT r.message, a.address, a.comment, r.type
                FROM recipients r
                LEFT JOIN addresses a ON r.address = a.ROWID
                WHERE r.message IN ({placeholders})
                ORDER BY r.message, r.type, r.position""",
            chunk,
        ).fetchall()

        for row in rows:
            msg_id = row["message"]
            if msg_id not in result:
                result[msg_id] = {"to": [], "cc": [], "bcc": []}

            addr = _row_val(row, "address", "")
            name = _row_val(row, "comment", "")
            if not addr:
                continue

            formatted = f"{name} <{addr}>" if name else addr
            rtype = row["type"] or 0

            if rtype == 0:
                result[msg_id]["to"].append(formatted)
            elif rtype == 1:
                result[msg_id]["cc"].append(formatted)
            elif rtype == 2:
                result[msg_id]["bcc"].append(formatted)

    return result


def read_recent_emails(
    hours: int = 24,
    limit: int = 50,
    since: Optional[datetime] = None,
) -> List[Message]:
    """Read recent emails from Mail.app.

    Deduplicates across multiple synced accounts by using
    (conversation_id, date_sent, sender_addr) as a unique key.
    """
    db_path = _find_envelope_index()
    if not db_path:
        logger.warning("Mail.app Envelope Index not found")
        return []

    if since:
        cutoff_unix = since.timestamp()
    else:
        cutoff_unix = (datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp()

    # Detect epoch format: read one row to see if values are Apple or Unix epoch
    # Then convert our cutoff to match the DB's format
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.OperationalError:
        logger.error("Cannot read Mail.app database — check Full Disk Access")
        return []

    # Check if DB uses Apple epoch by sampling one row
    sample = conn.execute(
        "SELECT date_sent FROM messages WHERE date_sent > 0 LIMIT 1"
    ).fetchone()
    if sample and sample["date_sent"] < _UNIX_EPOCH_THRESHOLD:
        # DB uses Apple epoch — convert our Unix cutoff to Apple epoch
        cutoff = cutoff_unix - APPLE_EPOCH_OFFSET
    else:
        # DB uses Unix epoch — use cutoff as-is
        cutoff = cutoff_unix

    # Fetch more than limit to account for duplicates, then dedup
    fetch_limit = limit * 3

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
    seen_keys: set[tuple] = set()

    try:
        rows = conn.execute(query, (cutoff, fetch_limit)).fetchall()

        # Batch fetch recipients
        rowids = [row["ROWID"] for row in rows]
        all_recipients = _fetch_recipients(conn, rowids)

        for row in rows:
            # Dedup across accounts: same timestamp + sender + subject = same email.
            # conversation_id is per-account so it can't be used for cross-account dedup.
            sender_addr = _row_val(row, "sender_address", "")
            conv_id = _row_val(row, "conversation_id", None)
            date_sent = _row_val(row, "date_sent", 0)
            subject = str(_row_val(row, "subject_text", ""))
            dedup_key = (date_sent, sender_addr.lower(), subject.lower().strip())

            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            subject = str(_row_val(row, "subject_text", ""))
            body = str(_row_val(row, "summary_text", ""))
            if len(body) > 3000:
                body = body[:3000]

            mailbox_url = _row_val(row, "mailbox_url", "")
            is_sent = _is_from_user(sender_addr)

            # Get recipients for this message
            msg_recips = all_recipients.get(row["ROWID"], {"to": [], "cc": [], "bcc": []})
            to_list = msg_recips["to"]
            cc_list = msg_recips["cc"]

            recip_str = ", ".join(to_list) if to_list else "me"
            if cc_list:
                recip_str += "; CC: " + ", ".join(cc_list)

            messages.append(Message(
                source="mail",
                sender=_parse_sender(row),
                recipient=recip_str,
                subject=subject,
                body=body,
                timestamp=_parse_timestamp(row),
                is_from_me=is_sent,
                thread_id=str(conv_id) if conv_id else None,
                metadata={
                    "mailbox": mailbox_url,
                    "cc": cc_list,
                    "rowid": row["ROWID"],
                },
            ))

            if len(messages) >= limit:
                break

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
            m.conversation_id,
            mb.url AS mailbox_url
        FROM messages m
        LEFT JOIN addresses a ON m.sender = a.ROWID
        LEFT JOIN subjects s ON m.subject = s.ROWID
        LEFT JOIN summaries sm ON m.summary = sm.ROWID
        LEFT JOIN mailboxes mb ON m.mailbox = mb.ROWID
        WHERE m.conversation_id = ?
        ORDER BY m.date_sent ASC
        LIMIT ?
    """

    messages = []
    seen_keys: set[tuple] = set()

    try:
        rows = conn.execute(query, (int(conversation_id), limit * 3)).fetchall()

        rowids = [row["ROWID"] for row in rows]
        all_recipients = _fetch_recipients(conn, rowids)

        for row in rows:
            sender_addr = _row_val(row, "sender_address", "")
            date_sent = _row_val(row, "date_sent", 0)
            subject_text = str(_row_val(row, "subject_text", ""))
            dedup_key = (date_sent, sender_addr.lower(), subject_text.lower().strip())

            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            is_sent = _is_from_user(sender_addr)

            msg_recips = all_recipients.get(row["ROWID"], {"to": [], "cc": [], "bcc": []})
            to_list = msg_recips["to"]
            cc_list = msg_recips["cc"]
            recip_str = ", ".join(to_list) if to_list else "me"

            messages.append(Message(
                source="mail",
                sender=_parse_sender(row),
                recipient=recip_str,
                subject=str(_row_val(row, "subject_text", "")),
                body=str(_row_val(row, "summary_text", "")),
                timestamp=_parse_timestamp(row),
                is_from_me=is_sent,
                thread_id=str(_row_val(row, "conversation_id", "")),
                metadata={
                    "mailbox": _row_val(row, "mailbox_url", ""),
                    "cc": cc_list,
                    "rowid": row["ROWID"],
                },
            ))

            if len(messages) >= limit:
                break

    except sqlite3.OperationalError as exc:
        logger.error("Thread query failed: %s", exc)
    finally:
        conn.close()

    return messages
