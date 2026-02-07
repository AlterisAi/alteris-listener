"""CLI command for running queries against local data sources."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from rich.console import Console
from rich.syntax import Syntax

from alteris_listener.api.context import fetch_user_context, format_user_context
from alteris_listener.api.session import AlterisSession
from alteris_listener.api.upload import upload_query_results
from alteris_listener.llm.client import LLMClient
from alteris_listener.llm.context import (
    format_calendar_context,
    format_email_context,
    format_granola_context,
    format_imessage_context,
    format_slack_context,
    format_thread_context,
)
from alteris_listener.llm.loader import load_queries_from_dir
from alteris_listener.sources.base import Message
from alteris_listener.sources.calendar import read_upcoming_events
from alteris_listener.sources.granola import check_granola_available, read_recent_meetings
from alteris_listener.sources.imessage import read_recent_imessages
from alteris_listener.sources.mail import read_email_thread, read_recent_emails
from alteris_listener.sources.slack import check_slack_available, read_recent_slack_messages

console = Console()
logger = logging.getLogger(__name__)

ALL_SOURCES = ["mail", "imessage", "calendar", "slack", "granola"]


def _resolve_queries_dir(cli_arg: Optional[str]) -> Path:
    """Resolve the queries directory from CLI arg, env var, or default.

    Priority:
    1. --queries-dir CLI argument
    2. ALTERIS_QUERIES_DIR environment variable
    3. ./queries/ relative to current directory
    """
    if cli_arg:
        return Path(cli_arg)

    env_dir = os.environ.get("ALTERIS_QUERIES_DIR")
    if env_dir:
        return Path(env_dir)

    return Path.cwd() / "queries"


# ══════════════════════════════════════════════════════════════════
# Source → item helpers
# ══════════════════════════════════════════════════════════════════


def _item_id_for(msg: Message) -> str:
    """Derive a stable document ID for a message."""
    meta = msg.metadata or {}

    if msg.source == "granola":
        return meta.get("document_id", "")
    if msg.source == "mail":
        return msg.thread_id or ""
    if msg.source == "slack":
        ch = meta.get("channel_id", meta.get("channel_name", ""))
        ts = meta.get("ts", "")
        if ch and ts:
            return f"{ch}_{ts}"
    if msg.source == "imessage":
        return meta.get("message_id", msg.thread_id or "")
    if msg.source == "calendar":
        return meta.get("event_id", "")

    content = json.dumps(
        {"s": msg.subject, "b": msg.body[:500], "t": str(msg.timestamp)},
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _provider_id_for(msg: Message, user_email: str = "") -> str:
    """Derive the provider_id for multi-account routing."""
    meta = msg.metadata or {}

    if msg.source == "mail":
        return user_email or msg.recipient or msg.sender or "unknown"
    if msg.source == "slack":
        return meta.get("workspace", "default")
    if msg.source == "granola":
        return meta.get("user_email", user_email or "default")
    if msg.source in ("imessage", "calendar"):
        return "local"

    return "default"


def _format_context(msg: Message, user_email: str = "") -> str:
    """Format a message into LLM context based on source type."""
    if msg.source == "mail":
        return format_email_context(msg, user_email=user_email)
    if msg.source == "imessage":
        return format_imessage_context(msg)
    if msg.source == "slack":
        return format_slack_context(msg)
    if msg.source == "granola":
        return format_granola_context(msg, user_email=user_email)
    if msg.source == "calendar":
        return format_calendar_context(msg)
    return msg.body


# ══════════════════════════════════════════════════════════════════
# Fetch helpers
# ══════════════════════════════════════════════════════════════════


def _fetch_items(
    source: str,
    hours: int,
    limit: int,
    user_email: str = "",
    thread_id: Optional[str] = None,
) -> Tuple[List[Message], Optional[List[List[Message]]]]:
    """Fetch items from a source.

    Returns:
        (flat_items, grouped_threads)
        grouped_threads is non-None only for mail source (email threads).
    """
    if source == "mail":
        if thread_id:
            msgs = read_email_thread(thread_id, limit=20)
            return msgs, None

        emails = read_recent_emails(hours=hours, limit=limit * 5)
        if not emails:
            return [], None

        thread_map: Dict[str, List[Message]] = {}
        for msg in emails:
            tid = msg.thread_id or f"single-{id(msg)}"
            thread_map.setdefault(tid, []).append(msg)

        sorted_threads = sorted(
            thread_map.values(),
            key=lambda t: max(m.timestamp for m in t),
            reverse=True,
        )[:limit]

        return emails, sorted_threads

    if source == "imessage":
        return read_recent_imessages(hours=hours, limit=limit), None

    if source == "calendar":
        days = max(1, hours // 24)
        return read_upcoming_events(days_ahead=days, days_behind=1), None

    if source == "slack":
        available, detail = check_slack_available()
        if not available:
            console.print(f"[yellow]{detail}[/yellow]")
            return [], None
        return read_recent_slack_messages(hours=hours, limit=limit), None

    if source == "granola":
        available, detail = check_granola_available()
        if not available:
            console.print(f"[yellow]{detail}[/yellow]")
            return [], None
        return read_recent_meetings(limit=limit, hours=hours, include_transcript=True), None

    return [], None


# ══════════════════════════════════════════════════════════════════
# Display helpers
# ══════════════════════════════════════════════════════════════════


def _display_result(result: dict, indent: str = "  ") -> None:
    """Pretty-print a query result."""
    if "tasks" in result:
        tasks = result["tasks"]
        if not tasks:
            console.print(f"{indent}[dim]No tasks extracted[/dim]")
        else:
            for task in tasks:
                icon = "✅" if task.get("done") else "⬜"
                p = task.get("priority", 3)
                color = {1: "red", 2: "yellow", 3: "white"}.get(p, "white")
                console.print(f"{indent}{icon} [{color}]P{p}[/{color}] {task.get('title', '?')}")
                if task.get("due_date"):
                    console.print(f"{indent}    📅 Due: {task['due_date']}")
                if task.get("assignee"):
                    console.print(f"{indent}    👤 {task['assignee']}")
                if task.get("note"):
                    first_line = task["note"].split("\n")[0]
                    console.print(f"{indent}    [dim]{first_line}[/dim]")

    elif "summary" in result:
        s = result["summary"]
        console.print(f"{indent}[bold]{s.get('title', '?')}[/bold]")
        console.print(
            f"{indent}[dim]{s.get('meeting_type', '?')} · "
            f"{s.get('sentiment', '?')} · {s.get('duration_estimate', '?')}[/dim]"
        )
        if s.get("tldr"):
            console.print(f"{indent}[italic]{s['tldr']}[/italic]")
        if s.get("decisions_made"):
            for d in s["decisions_made"]:
                console.print(f"{indent}  ✓ {d.get('decision', '?')}")
        if s.get("open_questions"):
            for q in s["open_questions"]:
                console.print(f"{indent}  ? {q}")

    elif "raw_text" in result:
        console.print(result["raw_text"])

    else:
        console.print(Syntax(json.dumps(result, indent=2, ensure_ascii=False), "json"))


# ══════════════════════════════════════════════════════════════════
# Query execution + upload helpers
# ══════════════════════════════════════════════════════════════════


def _run_queries(
    llm: LLMClient,
    query_defs: Dict[str, Any],
    context: str,
    raw: bool,
    multi: bool,
    user_context: str = "",
    json_mode: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Run all query definitions against context, accumulate results."""
    query_results: Dict[str, Any] = {}
    query_versions: Dict[str, str] = {}

    full_context = f"{user_context}\n\n{context}" if user_context else context

    for qname, qdef in query_defs.items():
        if multi:
            console.print(f"  [bold]▶ {qname}[/bold]")

        with console.status(f"[green]Running {qname}..."):
            result = llm.run_json(qdef.prompt_body, full_context)

        if "raw_text" not in result:
            query_results[qname] = result
            query_versions[qname] = datetime.now(timezone.utc).isoformat()

        if json_mode:
            pass  # Don't print per-item; collected and printed at end
        elif raw:
            click.echo(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            _display_result(result, indent="    " if multi else "  ")
        console.print()

    return query_results, query_versions


def _upload_enriched(
    item_id: str,
    source: str,
    provider_id: str,
    subject: str,
    metadata: Optional[dict],
    timestamp: datetime,
    query_results: Dict[str, Any],
    query_versions: Dict[str, str],
) -> bool:
    """Upload an enriched document with accumulated query_results."""
    meta = metadata or {}

    enriched_doc: Dict[str, Any] = {
        "id": item_id,
        "source_item_id": item_id,
        "subject": subject,
        "query_results": query_results,
        "query_versions": query_versions,
    }

    if source == "granola":
        enriched_doc["document_id"] = meta.get("document_id", item_id)
        enriched_doc["created_at"] = meta.get("created_at", "")
        enriched_doc["updated_at"] = meta.get("updated_at", "")
        enriched_doc["has_transcript"] = meta.get("has_transcript", False)
    elif source == "mail":
        enriched_doc["thread_id"] = item_id
    elif source == "slack":
        enriched_doc["channel"] = meta.get("channel_name", "")
        enriched_doc["ts"] = meta.get("ts", "")
    elif source == "calendar":
        enriched_doc["event_id"] = meta.get("event_id", item_id)

    return upload_query_results(
        query_name="_enriched",
        results=[enriched_doc],
        source=source,
        provider_id=provider_id,
        thread_subject=subject,
    )


def _print_summary(item_count: int, query_count: int, uploaded: bool) -> None:
    console.print(f"[bold green]Done.[/bold green] Processed {item_count} items × {query_count} queries")
    if uploaded:
        console.print("[green]Results synced to Alteris[/green]")


# ══════════════════════════════════════════════════════════════════
# run-query command
# ══════════════════════════════════════════════════════════════════


@click.command("run-query")
@click.argument("query_names", nargs=-1, required=True)
@click.option("--source", "-s", type=click.Choice(ALL_SOURCES), required=True,
              help="Data source to query")
@click.option("--queries-dir", type=click.Path(exists=True), default=None,
              help="Path to queries directory (default: $ALTERIS_QUERIES_DIR or ./queries/)")
@click.option("--provider", type=click.Choice(["gemini", "claude"]), default="gemini")
@click.option("--model", default=None, help="Override model name")
@click.option("--thinking", type=click.Choice(["off", "minimal", "low", "medium", "high"]),
              default="low")
@click.option("--hours", default=24, help="Hours of history to look back")
@click.option("--max", "limit", default=10, help="Max items to process")
@click.option("--thread-id", default=None, help="Process a specific email thread (mail only)")
@click.option("--user-email", default=None, help="Your email address")
@click.option("--raw", is_flag=True, help="Print raw JSON per item (verbose)")
@click.option("--json", "json_mode", is_flag=True,
              help="Output only clean JSON to stdout. All progress goes to stderr.")
@click.option("--upload", is_flag=True, help="Upload enriched results to Alteris")
@click.option("--context", "-c", "context_names", multiple=True, default=None,
              help="Context docs to load from cloud (default: clarity_queue). "
                   "E.g. -c clarity_queue -c goals_and_values")
@click.option("--no-context", is_flag=True, help="Skip fetching user context from cloud")
def run_query(
    query_names: tuple,
    source: str,
    queries_dir: Optional[str],
    provider: str,
    model: Optional[str],
    thinking: str,
    hours: int,
    limit: int,
    thread_id: Optional[str],
    user_email: Optional[str],
    raw: bool,
    json_mode: bool,
    upload: bool,
    context_names: tuple,
    no_context: bool,
):
    """Run one or more queries against a data source.

    Each item is processed through ALL specified queries. Results are
    accumulated under query_results.{query_name} per item — matching
    the Firestore schema — then uploaded as one enriched document.

    \b
    Examples:
        alteris-listener run-query meeting_summary meeting_todo_extractor -s granola --hours 168
        alteris-listener run-query todo_extractor_thread -s mail --hours 48 --upload
        alteris-listener run-query meeting_summary -s granola --max 5 -c clarity_queue
        alteris-listener run-query todo_extractor -s mail --json | jq .
    """
    global console

    # In JSON mode, redirect all Rich output to stderr so stdout is clean JSON
    if json_mode:
        console = Console(stderr=True)
        raw = True  # json mode implies raw (we need the parsed dicts)

    # Collector for --json mode
    json_results: List[Dict[str, Any]] = []
    # ── Load query definitions ────────────────────────────────────
    qdir = _resolve_queries_dir(queries_dir)
    all_queries = load_queries_from_dir(qdir)

    if not all_queries:
        console.print(f"[red]No queries found in {qdir}[/red]")
        console.print("Set ALTERIS_QUERIES_DIR or use --queries-dir to point to your queries folder.")
        sys.exit(1)

    query_defs: Dict[str, Any] = {}
    for qn in query_names:
        if qn not in all_queries:
            console.print(f"[red]Query '{qn}' not found.[/red]")
            console.print(f"Available: {', '.join(sorted(all_queries.keys()))}")
            sys.exit(1)
        query_defs[qn] = all_queries[qn]

    multi = len(query_defs) > 1

    console.print(f"[bold]Source:[/bold] {source}")
    console.print(f"[bold]Queries:[/bold] {', '.join(query_defs.keys())}")
    console.print()

    # ── Initialize LLM ────────────────────────────────────────────
    llm = LLMClient(provider=provider, model=model, thinking_level=thinking)
    console.print(f"[dim]Using {llm.provider} / {llm.model} (thinking={thinking})[/dim]")
    console.print()

    user_email = user_email or ""

    # ── Fetch user context from Alteris cloud ─────────────────────
    user_context_text = ""
    if not no_context:
        try:
            session = AlterisSession()
            if session.load_from_keychain():
                names = list(context_names) if context_names else None
                with console.status("[dim]Loading user context..."):
                    ctx_data = fetch_user_context(session, context_names=names)
                if ctx_data and ctx_data.get("contexts"):
                    user_context_text = format_user_context(ctx_data)
                    if not user_email and ctx_data.get("user_email"):
                        user_email = ctx_data["user_email"]
                    loaded = list(ctx_data["contexts"].keys())
                    console.print(f"[dim]Context loaded: {', '.join(loaded)}[/dim]")
                else:
                    console.print("[dim]No user context found in context_store[/dim]")
            else:
                console.print("[dim]Not logged in — skipping context. Run: alteris-listener login[/dim]")
        except Exception as exc:
            logger.debug("Context fetch failed: %s", exc)
            console.print("[dim]Could not fetch user context — continuing without it[/dim]")

    # ── Fetch items ───────────────────────────────────────────────
    console.print(f"[bold]Loading {source} ({hours}h, max {limit})...[/bold]")
    items, threads = _fetch_items(source, hours, limit, user_email, thread_id)

    # ── Handle mail threads ───────────────────────────────────────
    if source == "mail" and threads is not None:
        console.print(f"  Found {len(items)} emails in {len(threads)} threads")
        console.print(f"  Processing top {len(threads)} threads")
        console.print()

        for thread_msgs in threads:
            subject = thread_msgs[0].subject or "(no subject)"
            tid = thread_msgs[0].thread_id or ""
            pid = _provider_id_for(thread_msgs[0], user_email)

            console.print(f"[bold cyan]━━━ Thread: {subject} ({len(thread_msgs)} msgs) ━━━[/bold cyan]")

            context = format_thread_context(thread_msgs, user_email=user_email)

            query_results, query_versions = _run_queries(
                llm, query_defs, context, raw, multi,
                user_context=user_context_text,
                json_mode=json_mode,
            )

            if json_mode and query_results:
                json_results.append({
                    "source": source,
                    "item_id": tid,
                    "subject": subject,
                    "timestamp": thread_msgs[0].timestamp.isoformat(),
                    "query_results": query_results,
                })

            if upload and query_results:
                _upload_enriched(
                    item_id=tid, source=source, provider_id=pid,
                    subject=subject, metadata=thread_msgs[0].metadata,
                    timestamp=thread_msgs[0].timestamp,
                    query_results=query_results,
                    query_versions=query_versions,
                )
            console.print()

        _print_summary(len(threads), len(query_defs), upload)
        if json_mode:
            click.echo(json.dumps(json_results, indent=2, ensure_ascii=False))
        return

    # ── Handle individual items (all other sources) ───────────────
    if not items:
        console.print("[yellow]No items found.[/yellow]")
        return

    items.sort(key=lambda m: m.timestamp, reverse=True)
    items = items[:limit]

    console.print(f"  Found {len(items)} items")
    console.print()

    for msg in items:
        item_id = _item_id_for(msg)
        pid = _provider_id_for(msg, user_email)
        label = f"{msg.source}: {msg.subject or msg.body[:60]}"

        console.print(f"[bold cyan]━━━ {label} ━━━[/bold cyan]")
        if item_id:
            console.print(f"[dim]   ID: {item_id}[/dim]")

        context = _format_context(msg, user_email)

        query_results, query_versions = _run_queries(
            llm, query_defs, context, raw, multi,
            user_context=user_context_text,
            json_mode=json_mode,
        )

        if json_mode and query_results:
            json_results.append({
                "source": source,
                "item_id": item_id,
                "subject": msg.subject or "",
                "timestamp": msg.timestamp.isoformat(),
                "query_results": query_results,
            })

        if upload and query_results:
            _upload_enriched(
                item_id=item_id, source=source, provider_id=pid,
                subject=msg.subject or "",
                metadata=msg.metadata, timestamp=msg.timestamp,
                query_results=query_results,
                query_versions=query_versions,
            )
        console.print()

    _print_summary(len(items), len(query_defs), upload)
    if json_mode:
        click.echo(json.dumps(json_results, indent=2, ensure_ascii=False))
