"""Core message dataclass used across all sources."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Message:
    """Lightweight message struct for query context building.

    Used by all source readers (mail, iMessage, Slack, Granola, calendar)
    and consumed by LLM context formatters and the CLI.
    """

    source: str
    sender: str
    recipient: str
    subject: str
    body: str
    timestamp: datetime
    is_from_me: bool = False
    thread_id: Optional[str] = None
    metadata: Optional[dict] = None
