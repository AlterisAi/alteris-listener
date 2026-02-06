"""CLI command for asking natural language questions about local data."""

from __future__ import annotations

from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel

from alteris_listener.llm.client import LLMClient
from alteris_listener.sources.calendar import read_upcoming_events
from alteris_listener.sources.granola import check_granola_available, read_recent_meetings
from alteris_listener.sources.imessage import read_recent_imessages
from alteris_listener.sources.mail import read_recent_emails
from alteris_listener.sources.slack import check_slack_available, read_recent_slack_messages

console = Console()

ALL_SOURCES = ["mail", "imessage", "calendar", "slack", "granola"]


@click.command("ask")
@click.argument("question")
@click.option("--provider", type=click.Choice(["gemini", "claude"]), default="gemini")
@click.option("--model", default=None)
@click.option("--hours", default=24, help="Hours of history to include")
@click.option("--source", type=click.Choice(ALL_SOURCES + ["all"]), default="all")
def ask(question: str, provider: str, model: Optional[str], hours: int, source: str):
    """Ask a natural language question about your messages.

    \b
    Examples:
        alteris-listener ask "What meetings do I have tomorrow?"
        alteris-listener ask "Summarize my unread emails from today"
        alteris-listener ask "What happened in Slack today?" --source slack
        alteris-listener ask "What was discussed in my last meeting?" --source granola
    """
    llm = LLMClient(provider=provider, model=model, thinking_level="low")
    console.print(f"[dim]Using {llm.provider} / {llm.model}[/dim]")
    console.print()

    context_parts = []

    if source in ("mail", "all"):
        emails = read_recent_emails(hours=hours, limit=30)
        if emails:
            context_parts.append(f"## Recent Emails ({len(emails)} messages)\n")
            for msg in emails[:20]:
                context_parts.append(
                    f"- **From:** {msg.sender} | **Subject:** {msg.subject} | "
                    f"**Date:** {msg.timestamp.strftime('%Y-%m-%d %H:%M')} | "
                    f"**Preview:** {msg.body[:200]}\n"
                )

    if source in ("imessage", "all"):
        msgs = read_recent_imessages(hours=hours, limit=30)
        if msgs:
            context_parts.append(f"\n## Recent iMessages ({len(msgs)} messages)\n")
            for msg in msgs[:20]:
                direction = "→" if msg.is_from_me else "←"
                contact = msg.recipient if msg.is_from_me else msg.sender
                context_parts.append(
                    f"- {direction} **{contact}** ({msg.timestamp.strftime('%H:%M')}): {msg.body[:200]}\n"
                )

    if source in ("calendar", "all"):
        events = read_upcoming_events(days_ahead=7, days_behind=1)
        if events:
            context_parts.append(f"\n## Calendar Events ({len(events)} events)\n")
            for ev in events:
                meta = ev.metadata or {}
                context_parts.append(
                    f"- **{ev.subject}** | {ev.timestamp.strftime('%Y-%m-%d %H:%M')} - "
                    f"{meta.get('end', '')} | Calendar: {meta.get('calendar', '')} | "
                    f"Location: {meta.get('location', '')}\n"
                )

    if source in ("slack", "all"):
        available, detail = check_slack_available()
        if available:
            slack_msgs = read_recent_slack_messages(hours=hours, limit=30)
            if slack_msgs:
                context_parts.append(f"\n## Recent Slack Messages ({len(slack_msgs)} messages)\n")
                for msg in slack_msgs[:20]:
                    meta = msg.metadata or {}
                    context_parts.append(
                        f"- **#{meta.get('channel_name', '')}** {msg.sender} "
                        f"({msg.timestamp.strftime('%H:%M')}): {msg.body[:200]}\n"
                    )
        elif source == "slack":
            console.print(f"[yellow]{detail}[/yellow]")

    if source in ("granola", "all"):
        available, detail = check_granola_available()
        if available:
            meetings = read_recent_meetings(limit=5, hours=hours, include_transcript=False)
            if meetings:
                context_parts.append(f"\n## Recent Meetings ({len(meetings)} meetings)\n")
                for m in meetings:
                    body_preview = m.body[:500] if m.body else "(no notes)"
                    context_parts.append(
                        f"- **{m.subject}** | {m.timestamp.strftime('%Y-%m-%d %H:%M')}\n"
                        f"  {body_preview}\n"
                    )
        elif source == "granola":
            console.print(f"[yellow]{detail}[/yellow]")

    if not context_parts:
        console.print("[yellow]No data found for the requested sources and time range.[/yellow]")
        return

    context = "".join(context_parts)

    system_prompt = (
        "You are Alteris, a helpful AI assistant with access to the user's emails, "
        "messages, calendar, Slack messages, and meeting transcripts. Answer the user's "
        "question based on the context provided. Be concise and specific. "
        "If the answer isn't in the provided data, say so."
    )

    user_message = f"## Context\n\n{context}\n\n## Question\n\n{question}"

    with console.status("[bold green]Thinking..."):
        response = llm.run(system_prompt, user_message)

    console.print(Panel(response, title="[bold green]Alteris[/bold green]", border_style="green"))
