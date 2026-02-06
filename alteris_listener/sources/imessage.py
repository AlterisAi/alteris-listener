"""iMessage reader — reads from macOS Messages chat.db."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

from alteris_listener.sources.base import Message

logger = logging.getLogger(__name__)

APPLE_EPOCH_OFFSET = 978307200


def _row_val(row, key, default=""):
    try:
        val = row[key]
        return val if val is not None else default
    except (IndexError, KeyError):
        return default


def read_recent_imessages(hours: int = 24, limit: int = 50) -> List[Message]:
    """Read recent iMessages from chat.db."""
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
            m.date AS date_ns,
            m.is_from_me,
            m.cache_roomnames,
            h.id AS handle_id
        FROM message m
        LEFT JOIN handle h ON m.handle_id = h.ROWID
        WHERE m.date > ?
        ORDER BY m.date DESC
        LIMIT ?
    """

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
            text = str(_row_val(row, "text", ""))

            messages.append(Message(
                source="imessage",
                sender="me" if is_from_me else handle,
                recipient=handle if is_from_me else "me",
                subject="",
                body=text,
                timestamp=ts,
                is_from_me=is_from_me,
                thread_id=_row_val(row, "cache_roomnames", None),
            ))
    except sqlite3.OperationalError as exc:
        logger.error("iMessage query failed: %s", exc)
    finally:
        conn.close()

    return messages
