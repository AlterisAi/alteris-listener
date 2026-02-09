"""iMessage reader — reads from macOS Messages chat.db.

Handles:
- attributedBody decoding (typedstream format) for messages where text is NULL
- Spam filtering (is_spam column)
- Group chat participant resolution
- Service type tracking (iMessage, RCS, SMS)
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

from alteris_listener.sources.base import Message

logger = logging.getLogger(__name__)

APPLE_EPOCH_OFFSET = 978307200

# Try to import typedstream for attributedBody decoding
try:
    from typedstream.stream import TypedStreamReader
    _HAS_TYPEDSTREAM = True
except ImportError:
    _HAS_TYPEDSTREAM = False
    logger.warning(
        "pytypedstream not installed — attributedBody decoding disabled. "
        "Install with: pip install pytypedstream"
    )


def _row_val(row, key, default=""):
    try:
        val = row[key]
        return val if val is not None else default
    except (IndexError, KeyError):
        return default


def _decode_attributed_body(blob: bytes) -> str:
    """Extract plain text from attributedBody typedstream blob."""
    if not blob or not _HAS_TYPEDSTREAM:
        return ""
    try:
        for event in TypedStreamReader.from_data(blob):
            if isinstance(event, bytes):
                return event.decode("utf-8", errors="replace")
    except Exception as exc:
        logger.debug("attributedBody decode failed: %s", exc)
    return ""


def read_recent_imessages(hours: int = 24, limit: int = 50) -> List[Message]:
    """Read recent iMessages from chat.db.

    Decodes attributedBody for messages where text column is NULL,
    filters spam, and resolves group chat participants.
    """
    db_path = Path.home() / "Library" / "Messages" / "chat.db"
    if not db_path.exists():
        logger.warning("iMessage database not found")
        return []

    cutoff_ns = int(
        (datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp() - APPLE_EPOCH_OFFSET
    ) * 1_000_000_000

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.OperationalError:
        logger.error("Cannot read iMessage database — check Full Disk Access")
        return []

    query = """
        SELECT
            m.ROWID,
            m.text,
            m.attributedBody,
            m.date AS date_ns,
            m.is_from_me,
            m.is_spam,
            m.cache_roomnames,
            m.service,
            h.id AS handle_id,
            c.ROWID AS chat_rowid,
            c.display_name AS chat_display_name
        FROM message m
        LEFT JOIN handle h ON m.handle_id = h.ROWID
        LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
        LEFT JOIN chat c ON c.ROWID = cmj.chat_id
        WHERE m.date > ?
          AND m.is_spam = 0
        ORDER BY m.date DESC
        LIMIT ?
    """

    # Pre-cache chat participants
    participant_cache: dict[int, list[str]] = {}

    def _get_participants(chat_rowid: int) -> list[str]:
        if chat_rowid in participant_cache:
            return participant_cache[chat_rowid]
        parts = [
            r["id"] for r in conn.execute(
                "SELECT h.id FROM chat_handle_join chj "
                "JOIN handle h ON h.ROWID = chj.handle_id "
                "WHERE chj.chat_id = ?",
                (chat_rowid,),
            )
        ]
        participant_cache[chat_rowid] = parts
        return parts

    messages = []
    try:
        for row in conn.execute(query, (cutoff_ns, limit)):
            date_ns = _row_val(row, "date_ns", 0)
            if date_ns:
                ts = datetime.fromtimestamp(date_ns / 1e9 + APPLE_EPOCH_OFFSET, tz=timezone.utc)
            else:
                ts = datetime.now(timezone.utc)

            handle = _row_val(row, "handle_id", "unknown")
            is_from_me = bool(_row_val(row, "is_from_me", 0))
            service = _row_val(row, "service", "")
            chat_rowid = _row_val(row, "chat_rowid", None)
            chat_name = _row_val(row, "chat_display_name", "")

            # Get message body: try text column first, then attributedBody
            text = _row_val(row, "text", "")
            if not text:
                text = _decode_attributed_body(row["attributedBody"])
            if not text:
                # Skip messages with no text at all (attachments, reactions, etc.)
                continue

            # Resolve sender and recipients
            if is_from_me:
                sender = "me"
                if chat_rowid:
                    participants = _get_participants(chat_rowid)
                    recipient = ", ".join(participants) if participants else handle
                else:
                    recipient = handle
            else:
                sender = handle
                if chat_rowid:
                    participants = _get_participants(chat_rowid)
                    # Recipients are ME plus other participants (excluding sender)
                    others = [p for p in participants if p != handle]
                    recipient = ", ".join(["me"] + others) if others else "me"
                else:
                    recipient = "me"

            messages.append(Message(
                source="imessage",
                sender=sender,
                recipient=recipient,
                subject="",
                body=text,
                timestamp=ts,
                is_from_me=is_from_me,
                thread_id=_row_val(row, "cache_roomnames", None),
                metadata={
                    "service": service,
                    "chat_display_name": chat_name,
                },
            ))
    except sqlite3.OperationalError as exc:
        logger.error("iMessage query failed: %s", exc)
    finally:
        conn.close()

    logger.info("Read %d iMessages (last %d hours)", len(messages), hours)
    return messages
