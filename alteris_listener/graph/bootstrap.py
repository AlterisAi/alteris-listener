"""Bootstrap pipeline — multi-pass initialization of the knowledge graph.

Orchestrates the progression from raw data → populated graph:
  Pass 0: Structural extraction (no LLM, minutes)
  Pass 1: Heuristic scoring (no LLM, seconds)
  Pass 2: Embedding + clustering (embedding model, ~30 min for 40K items)
  Pass 3: LLM triage (8B model, selective)
  Pass 4: Deep extraction (30B model, high-value only)

Passes 0-1 are fully deterministic and run here.
Passes 2-4 require local models and are defined but need Ollama running.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from alteris_listener.graph.ingest import ingest_messages
from alteris_listener.graph.scoring import assign_temporal_bucket, bucket_score_threshold
from alteris_listener.graph.store import GraphStore
from alteris_listener.sources.base import Message

logger = logging.getLogger(__name__)
console = Console()


class BootstrapPipeline:
    """Multi-pass bootstrap for initializing the knowledge graph."""

    def __init__(self, store: GraphStore):
        self.store = store

    # ══════════════════════════════════════════════════════════════
    # Pass 0 + 1: Structural extraction + heuristic scoring
    # ══════════════════════════════════════════════════════════════

    def run_structural_pass(
        self,
        hours: int | None = None,
        max_items: int | None = None,
        sources: list[str] | None = None,
    ) -> dict:
        """Run Pass 0 (parse) and Pass 1 (score) on local data sources.

        This is fully deterministic — no LLM calls.
        Reads from Mail.app, iMessage, Calendar, Slack, Granola
        and populates the graph with nodes, edges, and contact stats.

        Args:
            hours: Lookback hours (None = all available data).
            max_items: Max items per source (None = unlimited).
            sources: Which sources to process (None = all available).

        Returns dict with ingest stats.
        """
        all_sources = sources or ["mail", "imessage", "calendar", "slack", "granola"]
        total_stats = {"ingested": 0, "skipped": 0, "total": 0, "sources": {}}

        console.print("[bold]Pass 0+1: Structural extraction + heuristic scoring[/bold]")
        console.print()

        # Load macOS Contacts for phone→name resolution
        contact_lookup = None
        try:
            from alteris_listener.graph.contacts_resolver import ContactLookup
            contact_lookup = ContactLookup()
            count = contact_lookup.load()
            if count > 0:
                console.print(f"  [green]Loaded {count} contacts ({contact_lookup.email_count} emails, {contact_lookup.phone_count} phones)[/green]")
            else:
                console.print("  [dim]No contacts loaded from Contacts.app[/dim]")
                contact_lookup = None
        except Exception as exc:
            console.print(f"  [dim]Contacts.app not available: {exc}[/dim]")
        console.print()

        for source_name in all_sources:
            messages = self._fetch_source(source_name, hours, max_items)
            if not messages:
                console.print(f"  [dim]{source_name}: no items found[/dim]")
                continue

            console.print(f"  [cyan]{source_name}[/cyan]: {len(messages)} items")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Ingesting {source_name}...", total=len(messages))

                # Batch ingest
                batch_size = 500
                source_stats = {"ingested": 0, "skipped": 0, "total": len(messages)}

                for i in range(0, len(messages), batch_size):
                    batch = messages[i:i + batch_size]
                    stats = ingest_messages(self.store, batch, contact_lookup=contact_lookup)
                    source_stats["ingested"] += stats["ingested"]
                    source_stats["skipped"] += stats["skipped"]
                    progress.update(task, advance=len(batch))

                total_stats["sources"][source_name] = source_stats
                total_stats["ingested"] += source_stats["ingested"]
                total_stats["skipped"] += source_stats["skipped"]
                total_stats["total"] += source_stats["total"]

        # Record bootstrap state
        self.store.set_bootstrap_state("last_pass", "structural")
        self.store.set_bootstrap_state("structural_completed_at", str(int(time.time())))

        # Link calendar events to meeting notes
        from alteris_listener.graph.ingest import link_calendar_to_meetings
        link_stats = link_calendar_to_meetings(self.store)
        if link_stats["linked"] or link_stats["attendee_edges"]:
            console.print(
                f"  Calendar→Meeting links: {link_stats['linked']}, "
                f"attendee edges: {link_stats['attendee_edges']}"
            )

        console.print()
        self._print_stats(total_stats)
        self._print_graph_summary()

        return total_stats

    def run_embedding_pass(
        self,
        min_score: float = 0.05,
        embedding_model: str = "nomic-embed-text",
        batch_size: int = 32,
    ) -> dict:
        """Pass 2: Generate embeddings for nodes above score threshold.

        Requires Ollama running with the embedding model pulled.
        Embeds subject + body_preview for each qualifying node.

        Returns dict with embedding stats.
        """
        from alteris_listener.graph.local_llm import OllamaClient
        from alteris_listener.graph.embeddings import EmbeddingIndex, vector_to_blob

        client = OllamaClient()
        if not client.is_available():
            console.print("[red]Ollama is not running. Start it with: ollama serve[/red]")
            return {"embedded": 0, "error": "ollama_not_running"}

        console.print(f"[bold]Pass 2: Embedding (model={embedding_model}, min_score={min_score})[/bold]")

        # Get nodes that need embedding
        rows = self.store.conn.execute(
            """SELECT id, subject, body_preview FROM nodes
               WHERE embedding IS NULL AND heuristic_score >= ?
               ORDER BY heuristic_score DESC""",
            (min_score,),
        ).fetchall()

        if not rows:
            console.print("  [dim]No nodes need embedding[/dim]")
            return {"embedded": 0}

        console.print(f"  Embedding {len(rows)} nodes...")

        embedded = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Embedding...", total=len(rows))

            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                texts = []
                for row in batch:
                    text = f"{row['subject'] or ''} {row['body_preview'] or ''}".strip()
                    texts.append(text[:512])  # Truncate for embedding model

                vectors = client.embed_batch(texts, model=embedding_model)

                for row, vec in zip(batch, vectors):
                    if vec is not None:
                        blob = vector_to_blob(vec.astype("float16"))
                        self.store.update_node_embedding(row["id"], blob)
                        embedded += 1

                progress.update(task, advance=len(batch))

        self.store.set_bootstrap_state("last_pass", "embedding")
        self.store.set_bootstrap_state("embedding_completed_at", str(int(time.time())))

        console.print(f"  [green]Embedded {embedded}/{len(rows)} nodes[/green]")
        return {"embedded": embedded, "total_candidates": len(rows)}

    # ══════════════════════════════════════════════════════════════
    # Source fetchers
    # ══════════════════════════════════════════════════════════════

    def _fetch_source(
        self,
        source: str,
        hours: int | None,
        max_items: int | None,
    ) -> list[Message]:
        """Fetch items from a data source."""
        limit = max_items or 100_000  # effectively unlimited
        hrs = hours or 87600  # ~10 years

        if source == "mail":
            try:
                from alteris_listener.sources.mail import read_recent_emails
                return read_recent_emails(hours=hrs, limit=limit)
            except Exception as exc:
                logger.warning("Failed to read mail: %s", exc)
                return []

        if source == "imessage":
            try:
                from alteris_listener.sources.imessage import read_recent_imessages
                return read_recent_imessages(hours=hrs, limit=limit)
            except Exception as exc:
                logger.warning("Failed to read imessage: %s", exc)
                return []

        if source == "calendar":
            try:
                from alteris_listener.sources.calendar import read_upcoming_events
                days = max(1, hrs // 24)
                return read_upcoming_events(days_ahead=days, days_behind=days)
            except Exception as exc:
                logger.warning("Failed to read calendar: %s", exc)
                return []

        if source == "slack":
            try:
                from alteris_listener.sources.slack import check_slack_available, read_recent_slack_messages
                available, _ = check_slack_available()
                if not available:
                    return []
                return read_recent_slack_messages(hours=hrs, limit=limit)
            except Exception as exc:
                logger.warning("Failed to read slack: %s", exc)
                return []

        if source == "granola":
            try:
                from alteris_listener.sources.granola import check_granola_available, read_recent_meetings
                available, _ = check_granola_available()
                if not available:
                    return []
                return read_recent_meetings(limit=limit, hours=hrs, include_transcript=False)
            except Exception as exc:
                logger.warning("Failed to read granola: %s", exc)
                return []

        return []

    # ══════════════════════════════════════════════════════════════
    # Display helpers
    # ══════════════════════════════════════════════════════════════

    def _print_stats(self, stats: dict):
        """Print ingest statistics."""
        table = Table(title="Bootstrap Results")
        table.add_column("Source", style="cyan")
        table.add_column("Ingested", justify="right", style="green")
        table.add_column("Skipped", justify="right", style="yellow")
        table.add_column("Total", justify="right")

        for src, s in stats.get("sources", {}).items():
            table.add_row(src, str(s["ingested"]), str(s["skipped"]), str(s["total"]))

        table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{stats['ingested']}[/bold]",
            f"[bold]{stats['skipped']}[/bold]",
            f"[bold]{stats['total']}[/bold]",
        )
        console.print(table)

    def _print_graph_summary(self):
        """Print current graph state."""
        stats = self.store.graph_stats()

        console.print()
        console.print("[bold]Graph State:[/bold]")
        console.print(f"  Nodes: {stats['total_nodes']}")
        console.print(f"  Edges: {stats['total_edges']}")
        console.print(f"  Contacts: {stats['contacts']}")

        for tier in (1, 2, 3):
            key = f"contacts_tier_{tier}"
            label = {1: "inner circle", 2: "regular", 3: "peripheral"}[tier]
            console.print(f"    Tier {tier} ({label}): {stats.get(key, 0)}")

        # Node type breakdown
        console.print()
        for ntype in ("email", "message", "meeting", "contact", "calendar_event"):
            key = f"nodes_{ntype}"
            if stats.get(key, 0) > 0:
                console.print(f"  {ntype}: {stats[key]}")

        # Temporal bucket distribution
        buckets = defaultdict(int)
        now = int(time.time())
        rows = self.store.conn.execute(
            "SELECT timestamp FROM nodes WHERE timestamp IS NOT NULL"
        ).fetchall()
        for row in rows:
            b = assign_temporal_bucket(row["timestamp"], now)
            buckets[b] += 1

        if buckets:
            console.print()
            console.print("[bold]Temporal distribution:[/bold]")
            labels = {1: "≤7d", 2: "7-30d", 3: "30-90d", 4: "90d-1y", 5: "1y+"}
            for b in sorted(buckets.keys()):
                console.print(f"  Bucket {b} ({labels.get(b, '?')}): {buckets[b]}")

        # Top contacts
        top = self.store.get_top_contacts(limit=10)
        if top:
            console.print()
            t2 = Table(title="Top Contacts")
            t2.add_column("Contact", style="cyan")
            t2.add_column("Messages", justify="right")
            t2.add_column("Sent/Recv", justify="right")
            t2.add_column("Reply Ratio", justify="right")
            t2.add_column("Tier", justify="center")

            for c in top:
                cid = c["contact_id"].replace("contact:", "")
                t2.add_row(
                    cid[:40],
                    str(c["total_messages"]),
                    f"{c['sent_to_count']}/{c['recv_from_count']}",
                    f"{c['reply_ratio']:.2f}",
                    str(c["importance_tier"]),
                )
            console.print(t2)

    # ══════════════════════════════════════════════════════════════
    # Score distribution analysis
    # ══════════════════════════════════════════════════════════════

    def analyze_scores(self) -> dict:
        """Analyze the distribution of heuristic scores for tuning thresholds."""
        rows = self.store.conn.execute(
            "SELECT heuristic_score, tier, node_type FROM nodes"
        ).fetchall()

        if not rows:
            return {}

        scores = [r["heuristic_score"] for r in rows]
        import statistics

        analysis = {
            "count": len(scores),
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
            "min": min(scores),
            "max": max(scores),
        }

        # Percentiles
        sorted_scores = sorted(scores)
        for pct in (10, 25, 50, 75, 90, 95, 99):
            idx = int(len(sorted_scores) * pct / 100)
            analysis[f"p{pct}"] = sorted_scores[min(idx, len(sorted_scores) - 1)]

        # By tier
        tier_counts = defaultdict(int)
        for r in rows:
            tier_counts[r["tier"]] += 1
        analysis["tier_distribution"] = dict(tier_counts)

        # By type
        type_counts = defaultdict(int)
        for r in rows:
            type_counts[r["node_type"]] += 1
        analysis["type_distribution"] = dict(type_counts)

        return analysis
