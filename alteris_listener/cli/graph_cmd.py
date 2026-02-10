"""CLI commands for the Alteris knowledge graph.

Commands:
  graph bootstrap    — Run structural pass (Pass 0+1) to initialize the graph
  graph embed        — Run embedding pass (Pass 2) on scored nodes
  graph status       — Show graph statistics and health
  graph neighbors    — Show neighborhood for a node
  graph contacts     — Show contact rankings
  graph scores       — Analyze heuristic score distribution
  graph summarize    — Show summarization candidates
  graph llm-check    — Check local LLM setup
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group("graph")
def graph_cli():
    """Knowledge graph operations — build, query, and manage the local graph."""
    pass


@graph_cli.command("bootstrap")
@click.option("--hours", default=None, type=int,
              help="Lookback hours (default: all available data)")
@click.option("--max", "max_items", default=None, type=int,
              help="Max items per source (default: unlimited)")
@click.option("--source", "-s", "sources", multiple=True,
              help="Specific sources to process (default: all). Can repeat: -s mail -s slack")
def bootstrap(hours: Optional[int], max_items: Optional[int], sources: tuple):
    """Initialize the knowledge graph from local data sources.

    Runs Pass 0 (structural extraction) and Pass 1 (heuristic scoring).
    This is fully deterministic — no LLM calls required.

    \b
    Examples:
        alteris-listener graph bootstrap                     # All sources, all data
        alteris-listener graph bootstrap --hours 720         # Last 30 days
        alteris-listener graph bootstrap -s mail -s granola  # Only mail + meetings
        alteris-listener graph bootstrap --max 1000          # Limit per source
    """
    from alteris_listener.graph.bootstrap import BootstrapPipeline
    from alteris_listener.graph.store import GraphStore

    store = GraphStore()
    pipeline = BootstrapPipeline(store)

    src_list = list(sources) if sources else None
    pipeline.run_structural_pass(hours=hours, max_items=max_items, sources=src_list)
    store.close()


@graph_cli.command("discover")
@click.option("--dir", "scan_dirs", multiple=True,
              help="Directories to scan (default: ~/Documents, ~/Desktop, ~/Downloads)")
@click.option("--obsidian", "obsidian_vaults", multiple=True,
              help="Obsidian vault paths to index")
@click.option("--no-notes", is_flag=True, default=False,
              help="Skip Apple Notes")
@click.option("--max-files", default=None, type=int,
              help="Max files to discover (default: unlimited)")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show what would be indexed without ingesting")
def discover(scan_dirs: tuple, obsidian_vaults: tuple, no_notes: bool,
             max_files: Optional[int], dry_run: bool):
    """Discover and index local documents into the graph.

    Crawls filesystem, Apple Notes, and Obsidian vaults to find documents
    that can be linked to your communication graph.

    \b
    Examples:
        alteris-listener graph discover                         # Default dirs
        alteris-listener graph discover --dir ~/Projects        # Custom dir
        alteris-listener graph discover --obsidian ~/vault      # Obsidian vault
        alteris-listener graph discover --dry-run               # Preview only
    """
    from alteris_listener.graph.discover import (
        run_discovery, DiscoveryConfig, discover_filesystem,
        discover_apple_notes, discover_obsidian_vault,
    )

    config = DiscoveryConfig(
        scan_dirs=list(scan_dirs) if scan_dirs else [
            str(__import__("pathlib").Path.home() / "Documents"),
            str(__import__("pathlib").Path.home() / "Desktop"),
            str(__import__("pathlib").Path.home() / "Downloads"),
        ],
        apple_notes=not no_notes,
        obsidian_vaults=list(obsidian_vaults),
        max_files=max_files,
    )

    console.print("[bold]Phase 0: Document Discovery[/bold]\n")

    # Discover files
    messages = run_discovery(config)

    # Summarize findings
    from collections import Counter
    by_source = Counter(m.metadata.get("source_app", "unknown") for m in messages if m.metadata)
    by_ext = Counter(m.metadata.get("file_ext", "unknown") for m in messages if m.metadata)
    chunks = sum(1 for m in messages if m.metadata and m.metadata.get("doc_type") == "chunk")
    files = len(messages) - chunks

    console.print(f"  Discovered {files} files ({chunks} chunks from large docs)\n")
    console.print("  By source:")
    for src, cnt in by_source.most_common():
        console.print(f"    {src}: {cnt}")
    console.print("\n  By type:")
    for ext, cnt in by_ext.most_common(10):
        console.print(f"    {ext}: {cnt}")

    if dry_run:
        console.print("\n  [yellow]Dry run — nothing ingested[/yellow]")
        # Show sample
        console.print("\n  Sample files:")
        for m in messages[:20]:
            if m.metadata and m.metadata.get("doc_type") != "chunk":
                path = m.metadata.get("file_path", "")
                console.print(f"    {m.subject} [{m.metadata.get('file_ext', '')}] {path}")
        return

    # Ingest into graph
    from alteris_listener.graph.store import GraphStore
    from alteris_listener.graph.ingest import ingest_messages

    store = GraphStore()
    stats = ingest_messages(store, messages)
    store.close()

    console.print(f"\n  Ingested {stats['ingested']} nodes ({stats['skipped']} skipped)")
    console.print("  Run [bold]graph triage[/bold] to classify discovered documents")


@graph_cli.command("embed")
@click.option("--min-score", default=0.05, type=float,
              help="Minimum heuristic score to embed (default: 0.05)")
@click.option("--model", default="nomic-embed-text",
              help="Ollama embedding model (default: nomic-embed-text)")
@click.option("--batch-size", default=32, type=int,
              help="Batch size for embedding requests")
def embed(min_score: float, model: str, batch_size: int):
    """Generate embeddings for graph nodes (requires Ollama).

    Embeds all nodes with heuristic_score >= min_score that don't
    already have embeddings. Uses a local embedding model via Ollama.

    \b
    Prerequisites:
        ollama serve &
        ollama pull nomic-embed-text

    \b
    Examples:
        alteris-listener graph embed                    # Default settings
        alteris-listener graph embed --min-score 0.1    # Only higher-scoring nodes
        alteris-listener graph embed --model all-minilm # Use smaller model
    """
    from alteris_listener.graph.bootstrap import BootstrapPipeline
    from alteris_listener.graph.store import GraphStore

    store = GraphStore()
    pipeline = BootstrapPipeline(store)
    pipeline.run_embedding_pass(min_score=min_score, embedding_model=model, batch_size=batch_size)
    store.close()


@graph_cli.command("status")
def status():
    """Show graph statistics and health."""
    from alteris_listener.graph.store import GraphStore
    from alteris_listener.graph.scoring import assign_temporal_bucket
    import time
    from collections import defaultdict

    store = GraphStore()
    stats = store.graph_stats()

    if stats["total_nodes"] == 0:
        console.print("[yellow]Graph is empty. Run: alteris-listener graph bootstrap[/yellow]")
        store.close()
        return

    console.print("[bold]Graph Status[/bold]")
    console.print(f"  Total nodes: {stats['total_nodes']}")
    console.print(f"  Total edges: {stats['total_edges']}")
    console.print(f"  Contacts:    {stats['contacts']}")
    console.print()

    # Node types
    console.print("[bold]Node Types:[/bold]")
    for ntype in ("email", "message", "meeting", "contact", "calendar_event"):
        key = f"nodes_{ntype}"
        if stats.get(key, 0) > 0:
            console.print(f"  {ntype}: {stats[key]}")

    # Edge types
    console.print()
    console.print("[bold]Edge Types:[/bold]")
    for etype in ("sent", "sent_to", "same_thread", "cc_to", "attendee", "reply_to", "same_entity", "involved"):
        count = store.count_edges(etype)
        if count > 0:
            console.print(f"  {etype}: {count}")

    # Contact tiers
    console.print()
    console.print("[bold]Contact Tiers:[/bold]")
    for tier in (1, 2, 3):
        label = {1: "inner circle", 2: "regular", 3: "peripheral"}[tier]
        console.print(f"  Tier {tier} ({label}): {stats.get(f'contacts_tier_{tier}', 0)}")

    # Embedding coverage
    emb_count = store.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    if stats["total_nodes"] > 0:
        pct = (emb_count / stats["total_nodes"]) * 100
        console.print()
        console.print(f"[bold]Embeddings:[/bold] {emb_count}/{stats['total_nodes']} ({pct:.1f}%)")

    # Temporal buckets
    now = int(time.time())
    rows = store.conn.execute(
        "SELECT timestamp FROM nodes WHERE timestamp IS NOT NULL"
    ).fetchall()
    if rows:
        buckets = defaultdict(int)
        for row in rows:
            b = assign_temporal_bucket(row["timestamp"], now)
            buckets[b] += 1

        console.print()
        console.print("[bold]Temporal Distribution:[/bold]")
        labels = {1: "≤7d", 2: "7-30d", 3: "30-90d", 4: "90d-1y", 5: "1y+"}
        for b in sorted(buckets.keys()):
            console.print(f"  Bucket {b} ({labels.get(b, '?')}): {buckets[b]}")

    # Tier distribution
    tier_rows = store.conn.execute(
        "SELECT tier, COUNT(*) as cnt FROM nodes GROUP BY tier ORDER BY cnt DESC"
    ).fetchall()
    if tier_rows:
        console.print()
        console.print("[bold]Node Tiers:[/bold]")
        for row in tier_rows:
            console.print(f"  {row['tier']}: {row['cnt']}")

    # Bootstrap state
    last_pass = stats.get("bootstrap_last_pass")
    if last_pass:
        console.print()
        console.print(f"[bold]Bootstrap:[/bold] last pass = {last_pass}")

    # Triage stats
    if stats.get("triaged_total", 0) > 0:
        console.print()
        console.print("[bold]Triage (Pass 3):[/bold]")
        console.print(f"  Triaged:      {stats['triaged_total']}")
        if "triaged_relevant" in stats:
            console.print(f"  Relevant:     {stats['triaged_relevant']}")
        if "triaged_not_relevant" in stats:
            console.print(f"  Not relevant: {stats['triaged_not_relevant']}")

    store.close()


@graph_cli.command("contacts")
@click.option("--limit", default=20, type=int, help="Number of contacts to show")
@click.option("--tier", default=None, type=int, help="Filter by importance tier (1-3)")
def contacts(limit: int, tier: Optional[int]):
    """Show contact rankings from the knowledge graph.

    \b
    Tiers:
        1 = Inner circle (high volume, bidirectional, recent)
        2 = Regular (moderate volume or recent)
        3 = Peripheral (low volume, infrequent)
    """
    from alteris_listener.graph.store import GraphStore

    store = GraphStore()

    if tier:
        rows = store.conn.execute(
            """SELECT * FROM contact_stats
               WHERE importance_tier = ?
               ORDER BY total_messages DESC LIMIT ?""",
            (tier, limit),
        ).fetchall()
    else:
        rows = store.conn.execute(
            """SELECT * FROM contact_stats
               ORDER BY importance_tier ASC, total_messages DESC LIMIT ?""",
            (limit,),
        ).fetchall()

    if not rows:
        console.print("[yellow]No contacts found. Run: alteris-listener graph bootstrap[/yellow]")
        store.close()
        return

    table = Table(title="Contact Rankings")
    table.add_column("Contact", style="cyan", max_width=40)
    table.add_column("Name", style="dim", max_width=25)
    table.add_column("Msgs", justify="right")
    table.add_column("Sent", justify="right", style="green")
    table.add_column("Recv", justify="right", style="blue")
    table.add_column("Reply%", justify="right")
    table.add_column("Tier", justify="center")
    table.add_column("Last Seen", style="dim")
    table.add_column("Sources", style="dim")

    for row in rows:
        cid = row["contact_id"].replace("contact:", "")
        name = row["display_name"] or ""
        last = ""
        if row["last_seen"]:
            last = datetime.fromtimestamp(row["last_seen"], tz=timezone.utc).strftime("%Y-%m-%d")

        sources_raw = row["sources"]
        if isinstance(sources_raw, str):
            sources_list = json.loads(sources_raw)
        else:
            sources_list = sources_raw or []
        sources_str = ",".join(sources_list) if sources_list else ""

        tier_colors = {1: "[bold green]1[/bold green]", 2: "[yellow]2[/yellow]", 3: "[dim]3[/dim]"}
        tier_display = tier_colors.get(row["importance_tier"], str(row["importance_tier"]))

        table.add_row(
            cid[:40],
            name[:25],
            str(row["total_messages"]),
            str(row["sent_to_count"]),
            str(row["recv_from_count"]),
            f"{row['reply_ratio']:.0%}",
            tier_display,
            last,
            sources_str,
        )

    console.print(table)
    store.close()


@graph_cli.command("scores")
def scores():
    """Analyze heuristic score distribution for threshold tuning."""
    from alteris_listener.graph.bootstrap import BootstrapPipeline
    from alteris_listener.graph.store import GraphStore

    store = GraphStore()
    pipeline = BootstrapPipeline(store)
    analysis = pipeline.analyze_scores()

    if not analysis:
        console.print("[yellow]No nodes to analyze. Run: alteris-listener graph bootstrap[/yellow]")
        store.close()
        return

    console.print("[bold]Score Distribution[/bold]")
    console.print(f"  Count:  {analysis['count']}")
    console.print(f"  Mean:   {analysis['mean']:.4f}")
    console.print(f"  Median: {analysis['median']:.4f}")
    console.print(f"  Stdev:  {analysis['stdev']:.4f}")
    console.print(f"  Min:    {analysis['min']:.4f}")
    console.print(f"  Max:    {analysis['max']:.4f}")

    console.print()
    console.print("[bold]Percentiles:[/bold]")
    for pct in (10, 25, 50, 75, 90, 95, 99):
        console.print(f"  P{pct:2d}: {analysis.get(f'p{pct}', 0):.4f}")

    console.print()
    console.print("[bold]By Tier:[/bold]")
    for tier_name, count in sorted(analysis.get("tier_distribution", {}).items()):
        console.print(f"  {tier_name}: {count}")

    console.print()
    console.print("[bold]By Type:[/bold]")
    for type_name, count in sorted(analysis.get("type_distribution", {}).items(), key=lambda x: -x[1]):
        console.print(f"  {type_name}: {count}")

    store.close()


@graph_cli.command("neighbors")
@click.argument("node_id")
@click.option("--depth", default=2, type=int, help="Max traversal depth (1-3)")
@click.option("--budget", default=4000, type=int, help="Token budget for results")
@click.option("--edge-type", "-e", "edge_types", multiple=True,
              help="Filter by edge type. Can repeat.")
def neighbors(node_id: str, depth: int, budget: int, edge_types: tuple):
    """Show neighborhood around a node.

    \b
    Examples:
        alteris-listener graph neighbors email:12345
        alteris-listener graph neighbors contact:jane@example.com --depth 1
        alteris-listener graph neighbors email:12345 -e sent_to -e same_thread
    """
    from alteris_listener.graph.embeddings import EmbeddingIndex
    from alteris_listener.graph.neighbors import NeighborhoodQuery
    from alteris_listener.graph.store import GraphStore

    store = GraphStore()

    # Check node exists
    node = store.get_node(node_id)
    if not node:
        console.print(f"[red]Node not found: {node_id}[/red]")
        store.close()
        return

    console.print(f"[bold]Center node:[/bold] {node_id}")
    console.print(f"  Type: {node.get('node_type')}")
    console.print(f"  Subject: {node.get('subject', '(none)')}")
    console.print(f"  Score: {node.get('heuristic_score', 0):.4f}")
    console.print()

    # Try to load embedding index for semantic search
    emb_index = None
    emb_count = store.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    if emb_count > 0:
        from alteris_listener.graph.local_llm import EMBEDDING_DIMS
        dim = 768  # nomic-embed-text default
        emb_index = EmbeddingIndex(store, dim=dim)
        loaded = emb_index.load()
        if loaded > 0:
            console.print(f"[dim]Loaded {loaded} embeddings for semantic search[/dim]")

    query = NeighborhoodQuery(store, embedding_index=emb_index)
    et = list(edge_types) if edge_types else None

    results = query.get_neighborhood(
        node_id=node_id,
        max_depth=depth,
        budget=budget,
        edge_types=et,
    )

    if not results:
        console.print("[yellow]No neighbors found.[/yellow]")
        store.close()
        return

    table = Table(title=f"Neighborhood ({len(results)} nodes)")
    table.add_column("Node ID", style="cyan", max_width=45)
    table.add_column("Type", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("Depth", justify="center")
    table.add_column("Source", style="dim")
    table.add_column("Edges", style="dim", max_width=30)

    for n in results:
        table.add_row(
            n.node_id[:45],
            n.node_type,
            f"{n.score:.3f}",
            str(n.depth),
            n.source,
            ", ".join(set(n.edge_types))[:30],
        )

    console.print(table)
    store.close()


@graph_cli.command("summarize-candidates")
@click.option("--max-age", default=90, type=int,
              help="Minimum age in days for summarization eligibility")
def summarize_candidates(max_age: int):
    """Show nodes eligible for progressive summarization.

    Lists thread groups, contact period groups, and recurring patterns
    that can be compressed into summary nodes.
    """
    from alteris_listener.graph.store import GraphStore
    from alteris_listener.graph.summarize import get_summarization_candidates

    store = GraphStore()
    candidates = get_summarization_candidates(store, max_age_days=max_age)

    # Thread groups
    threads = candidates.get("thread_groups", [])
    if threads:
        console.print(f"[bold]Thread Groups ({len(threads)}):[/bold]")
        table = Table()
        table.add_column("Thread ID", style="cyan", max_width=30)
        table.add_column("Messages", justify="right")
        table.add_column("Participants", style="dim", max_width=40)
        table.add_column("Age (days)", justify="right")

        now = int(__import__("time").time())
        for t in threads[:20]:
            age = (now - t["last_timestamp"]) / 86400
            table.add_row(
                str(t["thread_id"])[:30],
                str(t["message_count"]),
                ", ".join(t["participants"][:3]),
                f"{age:.0f}",
            )
        console.print(table)
    else:
        console.print("[dim]No thread groups eligible for summarization[/dim]")

    # Contact period groups
    contacts_g = candidates.get("contact_period_groups", [])
    if contacts_g:
        console.print()
        console.print(f"[bold]Contact Period Groups ({len(contacts_g)}):[/bold]")
        table = Table()
        table.add_column("Contact", style="cyan", max_width=35)
        table.add_column("Messages", justify="right")
        table.add_column("Period", style="dim")

        for c in contacts_g[:20]:
            start = datetime.fromtimestamp(c["first_timestamp"], tz=timezone.utc).strftime("%Y-%m")
            end = datetime.fromtimestamp(c["last_timestamp"], tz=timezone.utc).strftime("%Y-%m")
            table.add_row(
                c["contact"][:35],
                str(c["message_count"]),
                f"{start} to {end}",
            )
        console.print(table)
    else:
        console.print("[dim]No contact period groups eligible[/dim]")

    # Recurring patterns
    patterns = candidates.get("recurring_patterns", [])
    if patterns:
        console.print()
        console.print(f"[bold]Recurring Patterns ({len(patterns)}):[/bold]")
        table = Table()
        table.add_column("Subject", style="cyan", max_width=45)
        table.add_column("Occurrences", justify="right")
        table.add_column("Avg Gap (days)", justify="right")

        for p in patterns[:20]:
            table.add_row(
                p["subject"][:45],
                str(p["occurrences"]),
                f"{p['avg_gap_days']:.0f}",
            )
        console.print(table)
    else:
        console.print("[dim]No recurring patterns found[/dim]")

    store.close()


@graph_cli.command("llm-check")
def llm_check():
    """Check local LLM setup (Ollama installation and models).

    Verifies that Ollama is running and required models are available
    for embedding and inference passes.
    """
    from alteris_listener.graph.local_llm import check_local_llm_setup

    result = check_local_llm_setup()

    if result["ready"]:
        console.print("[bold green]Local LLM setup is ready.[/bold green]")
        console.print()
        console.print("[bold]Available models:[/bold]")
        for m in result["models_available"]:
            console.print(f"  {m}")
    else:
        if not result["ollama_running"]:
            console.print("[bold red]Ollama is not running.[/bold red]")
        else:
            console.print("[bold yellow]Ollama is running but missing models.[/bold yellow]")
            console.print()
            console.print("[bold]Available:[/bold]")
            for m in result["models_available"]:
                console.print(f"  [green]✓[/green] {m}")
            console.print()
            console.print("[bold]Missing:[/bold]")
            for m in result["models_missing"]:
                console.print(f"  [red]✗[/red] {m}")

        console.print()
        console.print("[bold]Setup instructions:[/bold]")
        for line in result["instructions"]:
            console.print(f"  {line}")


@graph_cli.command("node")
@click.argument("node_id")
def show_node(node_id: str):
    """Show details for a specific node.

    \b
    Examples:
        alteris-listener graph node email:12345
        alteris-listener graph node contact:jane@example.com
    """
    from alteris_listener.graph.store import GraphStore

    store = GraphStore()
    node = store.get_node(node_id)

    if not node:
        console.print(f"[red]Node not found: {node_id}[/red]")
        store.close()
        return

    console.print(f"[bold]Node: {node_id}[/bold]")
    console.print(f"  Type:    {node.get('node_type')}")
    console.print(f"  Source:  {node.get('source')}")
    console.print(f"  Subject: {node.get('subject', '(none)')}")
    console.print(f"  Sender:  {node.get('sender', '(none)')}")
    console.print(f"  Score:   {node.get('heuristic_score', 0):.4f}")
    console.print(f"  Tier:    {node.get('tier')}")

    if node.get("timestamp"):
        ts = datetime.fromtimestamp(node["timestamp"], tz=timezone.utc)
        console.print(f"  Time:    {ts.isoformat()}")

    if node.get("thread_id"):
        console.print(f"  Thread:  {node['thread_id']}")

    has_emb = node.get("embedding") is not None
    console.print(f"  Embedding: {'yes' if has_emb else 'no'}")

    if node.get("recipients"):
        console.print(f"  Recipients: {', '.join(node['recipients'][:5])}")

    if node.get("body_preview"):
        console.print()
        console.print("[bold]Preview:[/bold]")
        console.print(f"  {node['body_preview'][:300]}")

    # Show edges
    outgoing = store.get_edges_from(node_id)
    incoming = store.get_edges_to(node_id)

    if outgoing or incoming:
        console.print()
        console.print(f"[bold]Edges ({len(outgoing)} out, {len(incoming)} in):[/bold]")
        for e in outgoing[:10]:
            console.print(f"  → {e['dst'][:45]} [{e['edge_type']}] w={e.get('weight', 1):.1f}")
        for e in incoming[:10]:
            console.print(f"  ← {e['src'][:45]} [{e['edge_type']}] w={e.get('weight', 1):.1f}")
        total_edges = len(outgoing) + len(incoming)
        if total_edges > 20:
            console.print(f"  [dim]... and {total_edges - 20} more[/dim]")

    store.close()


@graph_cli.command("triage")
@click.option("--model", default="qwen3:30b-a3b",
              help="Model for triage (default: qwen3:30b-a3b for Ollama, gemini-2.5-flash for Gemini)")
@click.option("--parallel", default=3, type=int,
              help="Number of concurrent requests (default: 3)")
@click.option("--batch-size", default=5, type=int,
              help="Max items per LLM call (default: 5)")
@click.option("--no-resume", is_flag=True,
              help="Re-triage all embedded nodes, not just untriaged ones")
@click.option("--fix-failed", is_flag=True,
              help="Retry only PARSE_FAILED nodes (e.g. with gemini after ollama failures)")
@click.option("--provider", default="ollama", type=click.Choice(["ollama", "gemini"]),
              help="LLM provider: ollama (local) or gemini (API, faster)")
def triage(model: str, parallel: int, batch_size: int, no_resume: bool, fix_failed: bool, provider: str):
    """Run Pass 3: LLM triage on embedded nodes.

    Uses LLM to score embedded nodes on a 0-1 scale.
    Batches thread-related items together for context and throughput.
    Scores route to processing tiers:
      <0.3 = ignore, 0.3-0.6 = lightweight (8B), 0.7+ = deep (30B)

    Resumable by default — re-run to continue after interruption.

    \b
    Examples:
        alteris-listener graph triage                         # Local Ollama
        alteris-listener graph triage --provider gemini       # Fast via Gemini API
        alteris-listener graph triage --provider gemini --parallel 10 --batch-size 10
        alteris-listener graph triage --no-resume             # Re-triage everything
        alteris-listener graph triage --fix-failed --provider gemini  # Fix parse failures
    """
    from alteris_listener.graph.triage import run_triage
    from alteris_listener.graph.store import GraphStore

    store = GraphStore()
    result = run_triage(store, model=model, parallel=parallel,
                        resume=not no_resume, batch_size=batch_size,
                        provider=provider, fix_failed=fix_failed)

    if result.get("error") == "ollama_not_running":
        console.print("[red]Ollama is not running. Start it with: ollama serve[/red]")
    if result.get("error") == "gemini_api_key_missing":
        console.print("[red]GEMINI_API_KEY not set. Run: alteris-listener set-key gemini[/red]")
    store.close()


@graph_cli.command("refresh")
@click.option("--hours", default=48, type=int,
              help="Lookback hours for new data (default: 48)")
@click.option("--provider", default="ollama", type=click.Choice(["ollama", "gemini"]),
              help="LLM provider for triage (default: ollama)")
@click.option("--parallel", default=3, type=int,
              help="Concurrent triage requests (default: 3, try 10 for gemini)")
def refresh(hours: int, provider: str, parallel: int):
    """Quick incremental refresh: pull new data + triage.

    Chains bootstrap (incremental) → triage (new nodes only).
    Much faster than a full bootstrap since it only reads recent data.

    \b
    Examples:
        alteris-listener graph refresh                        # Last 48h, local Ollama
        alteris-listener graph refresh --provider gemini      # Last 48h, fast Gemini
        alteris-listener graph refresh --hours 168            # Last week
    """
    import time
    from alteris_listener.graph.bootstrap import BootstrapPipeline
    from alteris_listener.graph.triage import run_triage
    from alteris_listener.graph.store import GraphStore

    t0 = time.time()
    store = GraphStore()

    # Step 1: Incremental bootstrap
    console.print(f"[bold]Step 1: Pull new data (last {hours}h)[/bold]\n")
    pipeline = BootstrapPipeline(store)
    stats = pipeline.run_structural_pass(hours=hours)

    new_count = stats.get("ingested", 0)
    if new_count == 0:
        console.print("  [dim]No new items found. Graph is up to date.[/dim]")
        store.close()
        elapsed = time.time() - t0
        console.print(f"\n[green]Refresh complete in {elapsed:.0f}s (nothing new)[/green]")
        return

    console.print(f"  [green]{new_count} new items ingested[/green]\n")

    # Step 2: Embed new nodes (required before triage)
    console.print("[bold]Step 2: Embed new nodes[/bold]\n")
    embed_stats = pipeline.run_embedding_pass(min_score=0.05)
    console.print()

    # Step 3: Triage new nodes
    console.print(f"[bold]Step 3: Triage new nodes (provider={provider})[/bold]\n")
    triage_result = run_triage(
        store, parallel=parallel, resume=True,
        batch_size=10 if provider == "gemini" else 5,
        provider=provider,
    )

    if triage_result.get("error") == "ollama_not_running":
        console.print("[red]Ollama is not running. Start it with: ollama serve[/red]")
    elif triage_result.get("error") == "gemini_api_key_missing":
        console.print("[red]GEMINI_API_KEY not set. Run: alteris-listener set-key gemini[/red]")

    store.close()
    elapsed = time.time() - t0
    console.print(f"\n[green]Refresh complete in {elapsed:.0f}s[/green]")


@graph_cli.command("triage-bench")
@click.option("--model", default="qwen3:30b-a3b", help="Ollama model")
@click.option("--sample", default=50, type=int, help="Number of items to benchmark")
def triage_bench(model: str, sample: int):
    """Benchmark triage batch sizes (1, 5, 10, 20).

    Measures throughput and parse success rate for each batch size
    on a sample of items. Does NOT write results to the DB.

    \\b
    Examples:
        alteris-listener graph triage-bench
        alteris-listener graph triage-bench --sample 100
    """
    import time as _time
    from alteris_listener.graph.triage import (
        build_triage_item, parse_triage_response,
        TRIAGE_SYSTEM, TRIAGE_BATCH_SUFFIX,
    )
    from alteris_listener.graph.local_llm import OllamaClient
    from alteris_listener.graph.store import GraphStore

    store = GraphStore()
    client = OllamaClient()
    if not client.is_available():
        console.print("[red]Ollama not running[/red]")
        store.close()
        return

    now = int(_time.time())
    rows = store.conn.execute(
        """SELECT id, node_type, source, timestamp, subject, sender,
                  recipients, body_preview, heuristic_score, tier, thread_id
           FROM nodes WHERE embedding IS NOT NULL
           ORDER BY heuristic_score DESC LIMIT ?""",
        (sample,),
    ).fetchall()

    if not rows:
        console.print("[yellow]No embedded nodes[/yellow]")
        store.close()
        return

    # Pre-build items
    items: list[tuple[str, str]] = []
    for row in rows:
        node = dict(row)
        text = build_triage_item(node, store, now, idx=0)
        items.append((node["id"], text))

    console.print(f"[bold]Triage Benchmark: {len(items)} items, model={model}[/bold]\n")

    for batch_sz in [1, 2, 3, 5]:
        batches = []
        for i in range(0, len(items), batch_sz):
            batch = items[i:i + batch_sz]
            batches.append(batch)

        total_parsed = 0
        total_failed = 0
        start = _time.time()

        for batch in batches:
            batch_ids = [nid for nid, _ in batch]
            texts = []
            for i, (nid, text) in enumerate(batch, 1):
                texts.append(text.replace("--- ITEM 0 (", f"--- ITEM {i} (", 1))

            prompt = "\n".join(texts) + "\n" + TRIAGE_BATCH_SUFFIX
            max_tokens = 200 * len(batch)

            raw = client.generate(
                prompt=prompt, model=model, system=TRIAGE_SYSTEM,
                temperature=0.1, max_tokens=max_tokens, format_json=True,
            )
            results = parse_triage_response(raw, batch_ids)

            for nid, val in results.items():
                if val is not None:
                    total_parsed += 1
                else:
                    total_failed += 1

        elapsed = _time.time() - start
        rate = len(items) / max(elapsed, 0.1)
        parse_pct = total_parsed / max(total_parsed + total_failed, 1) * 100

        console.print(
            f"  batch_size={batch_sz:>2}: "
            f"{elapsed:>6.1f}s | "
            f"{rate:>5.1f} items/s | "
            f"{len(batches):>4} calls | "
            f"parsed {total_parsed}/{total_parsed + total_failed} ({parse_pct:.0f}%)"
        )

    console.print()
    store.close()


@graph_cli.command("retier")
def retier():
    """Recompute contact importance tiers without re-ingesting.

    Useful after tuning tier thresholds. Runs in <1 second.
    """
    from alteris_listener.graph.store import GraphStore

    store = GraphStore()

    before = {}
    for tier in (1, 2, 3):
        ct = store.conn.execute(
            "SELECT COUNT(*) FROM contact_stats WHERE importance_tier = ?", (tier,)
        ).fetchone()[0]
        before[tier] = ct

    store.recompute_contact_tiers()

    after = {}
    for tier in (1, 2, 3):
        ct = store.conn.execute(
            "SELECT COUNT(*) FROM contact_stats WHERE importance_tier = ?", (tier,)
        ).fetchone()[0]
        after[tier] = ct

    console.print("[bold]Contact tier recomputation:[/bold]")
    for tier in (1, 2, 3):
        label = {1: "inner circle", 2: "regular", 3: "peripheral"}[tier]
        console.print(f"  Tier {tier} ({label}): {before[tier]} → {after[tier]}")

    # Show tier 1 contacts
    t1 = store.conn.execute(
        """SELECT contact_id, display_name, total_messages, reply_ratio, sources
           FROM contact_stats WHERE importance_tier = 1
           ORDER BY total_messages DESC"""
    ).fetchall()
    if t1:
        console.print()
        table = Table(title="Tier 1 — Inner Circle")
        table.add_column("Contact", style="cyan", max_width=35)
        table.add_column("Name", style="dim", max_width=20)
        table.add_column("Msgs", justify="right")
        table.add_column("Reply%", justify="right")
        table.add_column("Sources", style="dim")

        for row in t1:
            cid = row["contact_id"].replace("contact:", "")
            sources_raw = row["sources"]
            if isinstance(sources_raw, str):
                import json as _json
                sources_list = _json.loads(sources_raw)
            else:
                sources_list = sources_raw or []

            table.add_row(
                cid[:35],
                (row["display_name"] or "")[:20],
                str(row["total_messages"]),
                f"{row['reply_ratio']:.0%}",
                ",".join(sources_list),
            )
        console.print(table)

    store.close()


@graph_cli.command("entity-edges")
def entity_edges_cmd():
    """Build weak edges from shared entities and topics.

    Connects nodes that mention the same entities (same_entity edges,
    weight 0.5) or share topic tags (same_topic edges, weight 0.2).
    Enables graph-based discovery across sources.

    Run after triage, before propagate.

    \\b
    Examples:
        alteris-listener graph entity-edges
    """
    from alteris_listener.graph.entity_edges import build_entity_edges
    from alteris_listener.graph.store import GraphStore

    store = GraphStore()
    stats = build_entity_edges(store)
    store.close()


@graph_cli.command("propagate")
@click.option("--iterations", default=3, type=int,
              help="Maximum propagation iterations (default: 3)")
@click.option("--dry-run", is_flag=True,
              help="Show what would change without saving")
def propagate(iterations: int, dry_run: bool):
    """Run Pass 3.5: Message passing to adjust triage scores.

    Uses graph structure to correct triage misclassifications:
    - Newsletter urgency-bait dampened via sender reputation
    - Thread coherence promotes/demotes outliers
    - Inner circle messages get slight boost
    - CC'd and mass emails get dampened

    Pure SQL, no LLM calls. Runs in seconds.

    \\b
    Examples:
        alteris-listener graph propagate            # Run propagation
        alteris-listener graph propagate --dry-run  # Preview changes
    """
    from alteris_listener.graph.propagate import run_propagation
    from alteris_listener.graph.store import GraphStore

    store = GraphStore()
    run_propagation(store, max_iterations=iterations, dry_run=dry_run)
    store.close()


@graph_cli.command("dedup")
@click.option("--dry-run", is_flag=True, help="Preview changes without applying")
def dedup_contacts(dry_run: bool):
    """Deduplicate contacts — merge aliases for the same person.

    Merges contacts sharing display_name, Gmail dot-equivalence,
    and automated sender patterns (Slack, AT&T, Yelp, etc.).
    Updates edges, tiers, and message counts.

    \\b
    Examples:
        alteris-listener graph dedup            # Run deduplication
        alteris-listener graph dedup --dry-run  # Preview changes
    """
    from alteris_listener.graph.dedup import run_dedup
    from alteris_listener.graph.store import GraphStore

    store = GraphStore()
    run_dedup(store, dry_run=dry_run)
    store.close()


@graph_cli.command("merge-contacts")
@click.option("--dry-run", is_flag=True, help="Preview merges without applying")
def merge_contacts(dry_run: bool):
    """Merge phone-number contacts into their email contacts.

    Uses macOS Contacts.app to find people who have both a phone and
    an email. If both contact:+phone and contact:email exist in the
    graph, merges the phone contact into the email one (sums stats,
    re-points edges, removes phone contact row).

    \\b
    Examples:
        alteris-listener graph merge-contacts --dry-run   # Preview
        alteris-listener graph merge-contacts              # Apply
    """
    import json
    from rich.table import Table
    from alteris_listener.graph.contacts_resolver import ContactLookup
    from alteris_listener.graph.store import GraphStore

    store = GraphStore()
    cl = ContactLookup()
    cl.load()

    bridges = cl.phone_email_bridges()
    if not bridges:
        console.print("  [dim]No phone→email bridges found in Contacts.app[/dim]")
        store.close()
        return

    # Find mergeable pairs: phone contact AND email contact both exist
    merge_plan = []
    for phone_norm, emails in bridges.items():
        phone_cid = f"contact:{phone_norm}"
        phone_row = store.conn.execute(
            """SELECT contact_id, display_name, total_messages, sent_to_count,
                      recv_from_count, first_seen, last_seen, sources
               FROM contact_stats WHERE contact_id = ?""",
            (phone_cid,),
        ).fetchone()
        if not phone_row or phone_row["total_messages"] == 0:
            continue

        # Find first email that exists in graph
        email_cid = None
        for email in emails:
            candidate = f"contact:{email}"
            email_row = store.conn.execute(
                "SELECT contact_id FROM contact_stats WHERE contact_id = ?",
                (candidate,),
            ).fetchone()
            if email_row:
                email_cid = candidate
                break

        if not email_cid:
            continue

        merge_plan.append({
            "phone_cid": phone_cid,
            "email_cid": email_cid,
            "name": phone_row["display_name"] or cl.resolve_name(phone_norm) or phone_norm,
            "phone_msgs": phone_row["total_messages"],
            "phone_sent": phone_row["sent_to_count"],
            "phone_recv": phone_row["recv_from_count"],
        })

    if not merge_plan:
        console.print("  [dim]No phone→email merges found (no overlapping contacts)[/dim]")
        store.close()
        return

    # Display plan
    table = Table(title=f"{'DRY RUN — ' if dry_run else ''}Merge Plan ({len(merge_plan)} contacts)")
    table.add_column("Name")
    table.add_column("Phone Contact")
    table.add_column("→ Email Contact")
    table.add_column("Msgs")
    table.add_column("Sent/Recv")

    for m in sorted(merge_plan, key=lambda x: x["phone_msgs"], reverse=True):
        table.add_row(
            m["name"],
            m["phone_cid"].replace("contact:", ""),
            m["email_cid"].replace("contact:", ""),
            str(m["phone_msgs"]),
            f"{m['phone_sent']}/{m['phone_recv']}",
        )
    console.print(table)

    if dry_run:
        console.print("\n  [yellow]Dry run — no changes applied[/yellow]")
        store.close()
        return

    # Execute merges
    merged = 0
    for m in merge_plan:
        phone_cid = m["phone_cid"]
        email_cid = m["email_cid"]

        # 1. Sum contact stats into email contact
        store.conn.execute(
            """UPDATE contact_stats SET
                 total_messages = total_messages + ?,
                 sent_to_count = sent_to_count + ?,
                 recv_from_count = recv_from_count + ?,
                 first_seen = MIN(first_seen, ?),
                 last_seen = MAX(last_seen, ?),
                 display_name = COALESCE(NULLIF(display_name, ''), ?)
               WHERE contact_id = ?""",
            (
                m["phone_msgs"], m["phone_sent"], m["phone_recv"],
                store.conn.execute(
                    "SELECT first_seen FROM contact_stats WHERE contact_id = ?",
                    (phone_cid,),
                ).fetchone()["first_seen"],
                store.conn.execute(
                    "SELECT last_seen FROM contact_stats WHERE contact_id = ?",
                    (phone_cid,),
                ).fetchone()["last_seen"],
                m["name"],
                email_cid,
            ),
        )

        # 2. Recompute reply_ratio on merged contact
        row = store.conn.execute(
            "SELECT sent_to_count, recv_from_count, total_messages FROM contact_stats WHERE contact_id = ?",
            (email_cid,),
        ).fetchone()
        if row and row["total_messages"] > 0:
            rr = min(row["sent_to_count"], row["recv_from_count"]) / row["total_messages"]
            store.conn.execute(
                "UPDATE contact_stats SET reply_ratio = ? WHERE contact_id = ?",
                (rr, email_cid),
            )

        # 3. Merge sources
        phone_sources = store.conn.execute(
            "SELECT sources FROM contact_stats WHERE contact_id = ?", (phone_cid,)
        ).fetchone()
        email_sources = store.conn.execute(
            "SELECT sources FROM contact_stats WHERE contact_id = ?", (email_cid,)
        ).fetchone()
        if phone_sources and email_sources:
            ps = set(json.loads(phone_sources["sources"]))
            es = set(json.loads(email_sources["sources"]))
            merged_sources = json.dumps(sorted(ps | es))
            store.conn.execute(
                "UPDATE contact_stats SET sources = ? WHERE contact_id = ?",
                (merged_sources, email_cid),
            )

        # 4. Re-point edges from phone contact to email contact
        store.conn.execute(
            "UPDATE OR IGNORE edges SET src = ? WHERE src = ?",
            (email_cid, phone_cid),
        )
        store.conn.execute(
            "UPDATE OR IGNORE edges SET dst = ? WHERE dst = ?",
            (email_cid, phone_cid),
        )
        # Clean up any duplicate edges that couldn't be updated
        store.conn.execute("DELETE FROM edges WHERE src = ? OR dst = ?", (phone_cid, phone_cid))

        # 5. Re-point nodes referencing this contact
        store.conn.execute(
            "UPDATE nodes SET sender = ? WHERE sender = ?",
            (email_cid.replace("contact:", ""), phone_cid.replace("contact:", "")),
        )

        # 6. Delete phone contact stat
        store.conn.execute("DELETE FROM contact_stats WHERE contact_id = ?", (phone_cid,))

        # 7. Delete phone contact node if it exists
        store.conn.execute("DELETE FROM nodes WHERE id = ?", (phone_cid,))

        merged += 1

    store.conn.commit()

    # Recompute tiers
    store.recompute_contact_tiers()
    store.close()

    console.print(f"\n  [green]Merged {merged} phone contacts into email contacts[/green]")


@graph_cli.command("reset")
@click.confirmation_option(prompt="This will delete the entire graph. Are you sure?")
def reset():
    """Delete the graph database and start fresh."""
    from alteris_listener.graph.schema import GRAPH_DB_PATH

    if GRAPH_DB_PATH.exists():
        GRAPH_DB_PATH.unlink()
        console.print(f"[green]✓[/green] Deleted {GRAPH_DB_PATH}")
    else:
        console.print("[dim]No graph database found[/dim]")


@graph_cli.command("backfill-names")
def backfill_names():
    """Fill in missing display names on contacts from Contacts.app.

    Looks up each contact's email or phone in macOS Contacts.app
    and updates the display_name if it's currently empty.

    \\b
    Examples:
        alteris-listener graph backfill-names
    """
    from alteris_listener.graph.contacts_resolver import ContactLookup
    from alteris_listener.graph.store import GraphStore

    store = GraphStore()
    cl = ContactLookup()
    cl.load()

    # Find contacts with empty display names
    rows = store.conn.execute(
        "SELECT contact_id, display_name FROM contact_stats WHERE display_name IS NULL OR display_name = ''"
    ).fetchall()

    if not rows:
        console.print("  [dim]All contacts already have display names[/dim]")
        store.close()
        return

    updated = 0
    for row in rows:
        cid = row["contact_id"]
        identifier = cid.replace("contact:", "")
        name = cl.resolve_name(identifier)
        if name:
            store.conn.execute(
                "UPDATE contact_stats SET display_name = ? WHERE contact_id = ?",
                (name, cid),
            )
            updated += 1

    store.conn.commit()
    store.close()
    console.print(f"  [green]Updated {updated}/{len(rows)} contacts with display names[/green]")


@graph_cli.command("extract")
@click.option("--limit", "-n", type=int, default=None, help="Max threads to process")
@click.option("--local-only", is_flag=True, help="Use only local LLM (no cloud)")
@click.option("--force", is_flag=True, help="Re-extract already processed threads")
@click.option("--user-email", default="", help="User email for direction detection")
@click.option("--tier", default="deep", type=click.Choice(["deep", "lightweight", "all"]),
              help="Which tier to extract (default: deep)")
@click.option("--days", default=7, type=int, help="Only process last N days (0=all)")
def extract_cmd(limit, local_only, force, user_email, tier, days):
    """Pass 4: Extract commitments and action items from triaged threads."""
    from alteris_listener.graph.derived_store import DerivedStore
    from alteris_listener.graph.extract import run_extraction
    from alteris_listener.graph.store import GraphStore

    store = GraphStore()
    derived = DerivedStore()

    try:
        stats = run_extraction(
            store=store,
            derived=derived,
            user_email=user_email,
            use_cloud=not local_only,
            tier_filter=tier,
            days_back=days,
            limit=limit,
            force=force,
        )
    finally:
        store.close()
        derived.close()


@graph_cli.command("extract-status")
@click.option("--detail", "-d", is_flag=True, help="Show full commitment details")
@click.option("--limit", "-n", type=int, default=20, help="Max commitments to show")
def extract_status_cmd(detail, limit):
    """Show extraction stats and top commitments."""
    from alteris_listener.graph.derived_store import DerivedStore
    from alteris_listener.graph.store import GraphStore

    derived = DerivedStore()
    store = GraphStore()

    try:
        # Run stats
        runs = derived.conn.execute(
            "SELECT status, COUNT(*) as cnt FROM extraction_runs GROUP BY status"
        ).fetchall()
        console.print("\n[bold]Extraction runs:[/bold]")
        for r in runs:
            console.print(f"  {r['status']}: {r['cnt']}")

        # Commitment stats
        types = derived.conn.execute(
            """SELECT commitment_type, status, COUNT(*) as cnt
               FROM commitments GROUP BY commitment_type, status
               ORDER BY commitment_type, status"""
        ).fetchall()
        console.print("\n[bold]Commitments by type:[/bold]")
        for t in types:
            console.print(f"  {t['commitment_type']} ({t['status']}): {t['cnt']}")

        # All commitments with detail
        all_c = derived.conn.execute(
            """SELECT * FROM commitments
               ORDER BY priority ASC,
                        CASE WHEN deadline IS NOT NULL THEN 0 ELSE 1 END,
                        deadline ASC
               LIMIT ?""",
            (limit,),
        ).fetchall()

        if all_c:
            console.print(f"\n[bold]All commitments ({len(all_c)}):[/bold]")
            for c in all_c:
                c = dict(c)
                deadline = c.get("deadline") or "no deadline"
                status_icon = {"open": "⬜", "done": "✅", "overdue": "🔴", "cancelled": "❌"}.get(c["status"], "❓")
                console.print(
                    f"\n  {status_icon} P{c['priority']} [{c['commitment_type']}] "
                    f"[bold]{c['what']}[/bold] ({deadline})"
                )
                console.print(f"    Model: {c['model_used']} | Confidence: {c.get('confidence', '?')}")
                console.print(f"    Thread: {(c.get('thread_id') or 'standalone')[:50]}")

                if detail:
                    # Show source node info
                    node_ids = json.loads(c["source_node_ids"]) if c["source_node_ids"] else []
                    if node_ids:
                        for nid in node_ids[:3]:
                            node = store.conn.execute(
                                "SELECT sender, subject, body_preview, source, timestamp FROM nodes WHERE id = ?",
                                (nid,),
                            ).fetchone()
                            if node:
                                from datetime import datetime, timezone
                                try:
                                    dt = datetime.fromtimestamp(node["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d")
                                except (OSError, ValueError, TypeError):
                                    dt = "?"
                                console.print(
                                    f"    Source: [{node['source']}] {dt} "
                                    f"from={node['sender'] or '(you)'} "
                                    f"subj={node['subject'] or '(none)'}"
                                )
                                preview = (node["body_preview"] or "")[:120].replace("\n", " ")
                                console.print(f"      {preview}")

                    if c.get("note"):
                        console.print(f"    Note: {c['note'][:200]}")

                    if c.get("raw_extraction"):
                        try:
                            raw = json.loads(c["raw_extraction"])
                            console.print(f"    Raw: {json.dumps(raw, indent=2)[:300]}")
                        except (json.JSONDecodeError, TypeError):
                            pass

    finally:
        derived.close()
        store.close()


@graph_cli.command("extract-reset")
@click.confirmation_option(prompt="This will delete all extracted data. Are you sure?")
def extract_reset_cmd():
    """Delete all extracted data (derived.db). Source data is untouched."""
    from alteris_listener.graph.derived_store import DERIVED_DB_PATH

    if DERIVED_DB_PATH.exists():
        DERIVED_DB_PATH.unlink()
        console.print(f"[green]✓[/green] Deleted {DERIVED_DB_PATH}")
    else:
        console.print("[dim]No derived database found[/dim]")


# ══════════════════════════════════════════════════════════════════
# Pass 5: Meeting Briefing
# ══════════════════════════════════════════════════════════════════

@graph_cli.command("brief")
@click.option("--days", default=7, type=int, help="Days ahead to look for meetings")
@click.option("--web/--no-web", "web_search", default=False, help="Enable web search for context")
@click.option("--ask/--no-ask", "interactive", default=False, help="Ask clarifying questions before generating briefing")
@click.option("--model", default="gemini-3-flash-preview", help="LLM model for synthesis")
@click.option("--thinking", default="medium", help="Thinking level: off|minimal|low|medium|high")
@click.option("--save", "save_path", default=None, type=click.Path(), help="Save briefing to file")
def brief_cmd(days, web_search, interactive, model, thinking, save_path):
    """Generate meeting briefings for upcoming calendar events.

    \b
    Walks the knowledge graph to pull related emails, past meetings,
    messages, and open commitments for each upcoming meeting, then
    synthesizes a concise briefing via LLM.

    \b
    Use --ask to enable interactive mode where the agent asks you
    questions it can't answer from the graph (logistics, preferences,
    decisions). You can provide text answers or file paths.

    \b
    Examples:
        alteris-listener graph brief
        alteris-listener graph brief --days 3
        alteris-listener graph brief --web --ask
        alteris-listener graph brief --web --save ~/Desktop/brief.md
        alteris-listener graph brief --days 1 --thinking high
    """
    from alteris_listener.graph.briefing import run_briefing, _load_user_profile
    from alteris_listener.graph.derived_store import DerivedStore
    from alteris_listener.graph.store import GraphStore

    store = GraphStore()
    derived = DerivedStore()

    # Load timezone from profile, with sensible fallback
    profile = _load_user_profile()
    user_tz = profile.get("timezone", "America/Los_Angeles")

    result = run_briefing(
        store=store,
        derived=derived,
        days_ahead=days,
        web_search=web_search,
        interactive=interactive,
        user_tz=user_tz,
        model=model,
        thinking=thinking,
    )

    if result["events_count"] == 0:
        store.close()
        derived.close()
        return

    # Print the briefing
    console.print()
    console.rule("[bold]Meeting Briefing[/bold]")
    console.print()
    from rich.markdown import Markdown
    console.print(Markdown(result["briefing"]))
    console.print()
    console.rule()

    # Save if requested
    if save_path:
        from pathlib import Path
        path = Path(save_path).expanduser()
        if path.suffix.lower() == ".pdf":
            from alteris_listener.graph.briefing_pdf import render_briefing_pdf
            render_briefing_pdf(
                result["briefing"],
                path,
                title=f"Meeting Briefing — next {days} days",
            )
            console.print(f"[green]✓[/green] Saved PDF to {path}")
        else:
            path.write_text(result["briefing"])
            console.print(f"[green]✓[/green] Saved to {path}")

    store.close()
    derived.close()


@graph_cli.command("voice")
@click.option("--days", default=7, type=int, help="Days ahead to brief on")
@click.option("--thinking", default="medium", help="Thinking level: off|minimal|low|medium|high")
@click.option("--model", default=None, help="Override voice model")
@click.option("--no-tools", is_flag=True, default=False, help="Disable tools for testing voice-only")
@click.option("--vertex", is_flag=True, default=False, help="Use Vertex AI (GA model, function calling supported)")
@click.option("--project", default="ordinal-virtue-462602-p5", help="GCP project ID for Vertex AI")
@click.option("--location", default="us-central1", help="GCP region for Vertex AI")
def voice_cmd(days, thinking, model, no_tools, vertex, project, location):
    """Start a real-time voice briefing session.

    \b
    Opens a live conversation with Alteris using Gemini's voice API.
    Speak naturally — the agent will walk you through your week,
    pull context from your knowledge graph, ask questions, and
    generate a written briefing at the end.

    \b
    Requires: pip install pyaudio google-genai
    Developer API: GEMINI_API_KEY set in Keychain or environment
    Vertex AI:     gcloud auth application-default login

    \b
    Examples:
        alteris-listener graph voice                  # Developer API
        alteris-listener graph voice --vertex         # Vertex AI (production)
        alteris-listener graph voice --no-tools       # test voice only
        alteris-listener graph voice --model gemini-live-2.5-flash-preview
    """
    try:
        import pyaudio  # noqa: F401
    except ImportError:
        console.print("[red]Error:[/red] pyaudio not installed.")
        console.print("  Install with: pip install pyaudio")
        console.print("  On macOS you may need: brew install portaudio")
        return

    from alteris_listener.graph.voice_agent import run_voice_session
    run_voice_session(days=days, thinking=thinking, model=model,
                      no_tools=no_tools, vertex=vertex,
                      project=project, location=location)
