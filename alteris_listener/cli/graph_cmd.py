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
        console.print(f"  Relevant:     {stats['triaged_relevant']}")
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
              help="Ollama model for triage (default: qwen3:8b)")
@click.option("--parallel", default=3, type=int,
              help="Number of concurrent Ollama requests (default: 3)")
@click.option("--batch-size", default=5, type=int,
              help="Max items per LLM call (default: 5)")
@click.option("--no-resume", is_flag=True,
              help="Re-triage all embedded nodes, not just untriaged ones")
def triage(model: str, parallel: int, batch_size: int, no_resume: bool):
    """Run Pass 3: LLM triage on embedded nodes (requires Ollama).

    Uses qwen3:8b to score embedded nodes on a 0-1 scale.
    Batches thread-related items together for context and throughput.
    Scores route to processing tiers:
      <0.3 = ignore, 0.3-0.6 = lightweight (8B), 0.7+ = deep (30B)

    Resumable by default — re-run to continue after interruption.
    Set OLLAMA_NUM_PARALLEL=3 before starting ollama serve for best speed.

    \\b
    Examples:
        alteris-listener graph triage                     # Default
        alteris-listener graph triage --batch-size 10     # Larger batches
        alteris-listener graph triage --no-resume         # Re-triage everything
    """
    from alteris_listener.graph.triage import run_triage
    from alteris_listener.graph.store import GraphStore

    store = GraphStore()
    result = run_triage(store, model=model, parallel=parallel,
                        resume=not no_resume, batch_size=batch_size)

    if result.get("error") == "ollama_not_running":
        console.print("[red]Ollama is not running. Start it with: ollama serve[/red]")
    store.close()


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
        path.write_text(result["briefing"])
        console.print(f"[green]✓[/green] Saved to {path}")

    store.close()
    derived.close()
