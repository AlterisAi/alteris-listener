"""Build LLM context strings from messages.

Formats messages into the structure expected by query prompts,
matching the same format the Alteris cloud processors use.
"""

from __future__ import annotations

from typing import List

from alteris_listener.sources.base import Message


def format_email_context(msg: Message, user_email: str = "") -> str:
    """Format a single email into the snapshot format expected by queries."""
    lines = [
        "## Email Snapshot",
        f"To: {msg.recipient}",
        f"From: {msg.sender}",
        f"Subject: {msg.subject}",
        f"Date: {msg.timestamp.isoformat()}",
        f"User email: {user_email}",
        "",
        "### Body",
        msg.body,
    ]
    return "\n".join(lines)


def format_thread_context(
    messages: List[Message],
    user_email: str = "",
    user_name: str = "",
    user_timezone: str = "America/Los_Angeles",
) -> str:
    """Format an email thread into the chronological format expected by thread queries.

    This matches the _format_thread_context_async output from
    EmailQueryProcessor so thread-based prompts (like todo_extractor_thread)
    work without modification.
    """
    header = [
        f"User name: {user_name}" if user_name else "",
        f"User email: {user_email}",
        f"User timezone: {user_timezone}",
        "",
        "The following is a thread of emails in chronological order",
        "",
        "---",
    ]
    header = [line for line in header if line or line == ""]

    context_lines = list(header)
    sorted_msgs = sorted(messages, key=lambda m: m.timestamp)

    for i, msg in enumerate(sorted_msgs, start=1):
        snapshot = [
            "",
            f"## Message {i}:",
            "",
            f"from {msg.sender} to {msg.recipient}",
            "",
            "### Email Snapshot",
            f"To: {msg.recipient}",
            f"From: {msg.sender}",
            f"Subject: {msg.subject}",
            f"Date: {msg.timestamp.isoformat()}",
            "",
            "### Body",
            msg.body,
        ]
        context_lines.extend(snapshot)

        if i < len(sorted_msgs):
            context_lines.append("")
            context_lines.append("---")

    return "\n".join(context_lines)


def format_imessage_context(msg: Message) -> str:
    """Format an iMessage for query processing."""
    direction = "Sent" if msg.is_from_me else "Received"
    lines = [
        "## iMessage",
        f"Direction: {direction}",
        f"From: {msg.sender}",
        f"To: {msg.recipient}",
        f"Date: {msg.timestamp.isoformat()}",
        "",
        "### Body",
        msg.body,
    ]
    return "\n".join(lines)


def format_calendar_context(msg: Message) -> str:
    """Format a calendar event for query processing."""
    meta = msg.metadata or {}
    lines = [
        "## Calendar Event",
        f"Title: {msg.subject}",
        f"Calendar: {meta.get('calendar', '')}",
        f"Start: {msg.timestamp.isoformat()}",
        f"End: {meta.get('end', '')}",
        f"Location: {meta.get('location', '')}",
        f"All Day: {meta.get('is_all_day', False)}",
    ]
    if msg.body:
        lines.extend(["", "### Details", msg.body])
    return "\n".join(lines)


def format_slack_context(msg: Message) -> str:
    """Format a single Slack message for query processing."""
    meta = msg.metadata or {}
    lines = [
        "## Slack Message",
        f"Channel: #{meta.get('channel_name', '')}",
        f"From: {msg.sender}",
        f"Date: {msg.timestamp.isoformat()}",
        "",
        "### Body",
        msg.body,
    ]
    return "\n".join(lines)


def format_slack_thread_context(
    messages: List[Message],
    user_name: str = "",
) -> str:
    """Format a Slack thread as chronological context."""
    if not messages:
        return ""

    meta = messages[0].metadata or {}
    channel = meta.get("channel_name", "unknown")

    header = [
        f"User name: {user_name}" if user_name else "",
        f"Channel: #{channel}",
        "",
        "The following is a Slack thread in chronological order",
        "",
        "---",
    ]
    header = [line for line in header if line or line == ""]

    context_lines = list(header)
    sorted_msgs = sorted(messages, key=lambda m: m.timestamp)

    for i, msg in enumerate(sorted_msgs, start=1):
        context_lines.extend([
            "",
            f"## Message {i}:",
            f"From: {msg.sender}",
            f"Time: {msg.timestamp.isoformat()}",
            "",
            msg.body,
        ])

        if i < len(sorted_msgs):
            context_lines.append("")
            context_lines.append("---")

    return "\n".join(context_lines)


def format_granola_context(msg: Message, user_email: str = "") -> str:
    """Format a Granola meeting transcript for query processing."""
    meta = msg.metadata or {}
    lines = [
        "## Meeting Transcript",
        f"User email: {user_email or meta.get('user_email', 'unknown')}",
        f"User timezone: {meta.get('timezone', 'America/Los_Angeles')}",
        f"Title: {msg.subject}",
        f"Meeting date: {msg.timestamp.strftime('%Y-%m-%d')}",
        f"Document ID: {meta.get('document_id', '')}",
        f"Has transcript: {meta.get('has_transcript', False)}",
        "",
        msg.body,
    ]
    return "\n".join(lines)
