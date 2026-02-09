"""Build weak edges from triage-extracted entities and topics.

Runs after Pass 3 (LLM triage). Creates edges between nodes that share
the same extracted entities, enabling graph-based discovery of related
content across sources (e.g., an email about "Meta hiring" links to a
meeting about "Meta interview prep").

Edge types created:
  - same_entity: two nodes mention the same entity (weight by co-occurrence)
  - same_topic: two nodes share a topic tag (lower weight, broader signal)
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict

import numpy as np
from rich.console import Console

from alteris_listener.graph.store import GraphStore

import re

logger = logging.getLogger(__name__)

# Regex to strip noise characters from entity names
_NOISE_CHARS = re.compile(r'[®™©*#†‡§¶•→←↑↓|~`]+')
_MULTI_SPACE = re.compile(r'\s+')
_LEADING_JUNK = re.compile(r'^(sq |rf |lee)\*', re.IGNORECASE)


def _clean_entity(name: str) -> str:
    """Normalize an entity name: lowercase, strip noise chars, collapse whitespace."""
    name = name.strip().lower()
    name = _LEADING_JUNK.sub('', name)
    name = _NOISE_CHARS.sub('', name)
    # Strip common URL/domain suffixes
    for suffix in ('.com', '.org', '.net', '.io', '.ai', '.co'):
        if name.endswith(suffix) and len(name) > len(suffix) + 2:
            name = name[:-len(suffix)]
    name = _MULTI_SPACE.sub(' ', name).strip()
    return name

# Generic stop-lists: platform/tool names that are never meaningful as edges.
# These apply to any user, not specific to anyone's data.
ENTITY_STOPLIST = {
    # Communication platforms
    "linkedin", "google", "gmail", "slack", "zoom", "youtube",
    "google meet", "google account", "microsoft", "apple", "icloud",
    "outlook", "teams", "calendly", "whatsapp", "telegram", "discord",
    # Payment networks (the network, not the bank)
    "visa", "mastercard", "amex", "american express",
    # Mailing infrastructure
    "loops", "mailchimp", "substack", "sendgrid", "constant contact",
    # Auth/security generic terms
    "2-factor authentication", "authenticator app", "two-factor authentication",
}

TOPIC_STOPLIST = {
    "email", "message", "meeting", "calendar", "notification",
    "automated", "personal", "update", "reminder",
}


def _names_share_root(a: str, b: str) -> bool:
    """Check if two entity names likely refer to the same thing via string overlap.
    
    Returns True if one contains the other (min 4 chars, and contained part is
    at least 50% of the longer name), or they share a 4+ char word.
    """
    # One contains the other, but:
    # - contained part must be 4+ chars
    # - contained part must be ≥50% of the longer string's length
    #   (prevents "interbay" matching "bright horizons at interbay")
    longer = max(len(a), len(b))
    if len(a) >= 4 and a in b and len(a) / longer >= 0.5:
        return True
    if len(b) >= 4 and b in a and len(b) / longer >= 0.5:
        return True
    # Share a word that's 4+ chars
    words_a = set(w for w in a.split() if len(w) >= 4)
    words_b = set(w for w in b.split() if len(w) >= 4)
    if words_a & words_b:
        return True
    return False


def _pick_canonical(names: list[str], entity_counts: Counter | None, contact_names: set[str] | None = None) -> str:
    """Pick the best canonical name from a cluster.
    
    Picks the name that is the best "root" — appears as substring in the
    most other names. If a name matches a known contact, prefer the full
    contact name. Tiebreakers: no special chars, has uppercase, shorter.
    """
    LEGAL_SUFFIXES = {"inc.", "inc", "llc", "pllc", "corp", "corp.", "ltd", "ltd."}

    def score(name: str) -> tuple:
        clean = name
        for suffix in LEGAL_SUFFIXES:
            if clean.endswith(suffix):
                clean = clean[: -len(suffix)].rstrip(", ")

        # How many other names contain this one as a substring?
        coverage = sum(1 for other in names if other != name and clean in other)
        # Bonus: matches a known contact full name, or shares a last name
        # (catches kids like "kiran bhargava" via parent "ananya bhargava")
        is_contact = 0
        if contact_names:
            if clean in contact_names:
                is_contact = 2  # exact match
            elif " " in clean:
                last_name = clean.split()[-1]
                if len(last_name) >= 3 and any(last_name in cn for cn in contact_names):
                    is_contact = 1  # last name match
        no_special = not any(c in name for c in "*_/.®")
        has_upper = any(c.isupper() for c in name)
        no_suffix = not any(name.endswith(s) for s in LEGAL_SUFFIXES)
        looks_natural = " " in name or len(name) <= 15

        # Primary: contact match, then coverage, then quality signals, then shorter
        return (is_contact, coverage, looks_natural, no_special, has_upper, no_suffix, -len(clean))

    return max(names, key=score)


def _run_entity_dedup(entities: list[str], store: GraphStore, entity_counts: Counter | None = None, contact_names: set[str] | None = None) -> dict[str, str]:
    """Use embeddings + agglomerative clustering to find entity aliases.

    Embeds entity names, clusters hierarchically by cosine distance,
    validates merges via string overlap, and returns a normalization map.
    """
    from alteris_listener.graph.local_llm import OllamaClient
    from sklearn.cluster import AgglomerativeClustering

    if len(entities) < 5:
        return {}

    entities = entities[:200]

    client = OllamaClient()
    embeddings = client.embed_batch(entities)

    valid = [(e, emb) for e, emb in zip(entities, embeddings) if emb is not None]
    if len(valid) < 5:
        return {}

    names = [v[0] for v in valid]
    vecs = np.stack([v[1] for v in valid])

    # Normalize for cosine distance
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs_normed = vecs / norms

    # Hierarchical clustering with complete linkage:
    # Every pair within a cluster must be within distance_threshold.
    # This prevents chain drift (A~B~C merging when A and C are dissimilar).
    # Tested: dist<0.35 is the sweet spot — catches abbreviations and spelling
    # variants without merging competitors or related-but-distinct entities.
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.35,
        metric="cosine",
        linkage="complete",
    )
    labels = clustering.fit_predict(vecs_normed)

    # Group by cluster label
    clusters: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    # Build alias map from clusters with 2+ members
    alias_map: dict[str, str] = {}
    for label, indices in clusters.items():
        if len(indices) < 2:
            continue

        cluster_names = [names[i] for i in indices]

        # Validate: at least one pair must share a string root.
        # This prevents semantically similar but distinct entities
        # (e.g. "PayPal" / "Venmo") from merging.
        has_valid_overlap = False
        for i in range(len(cluster_names)):
            for j in range(i + 1, len(cluster_names)):
                if _names_share_root(cluster_names[i], cluster_names[j]):
                    has_valid_overlap = True
                    break
            if has_valid_overlap:
                break

        if not has_valid_overlap:
            continue

        # Within a validated cluster, only include members connected by
        # string overlap (don't drag in unrelated hitchhikers).
        connected = set()
        for i in range(len(cluster_names)):
            for j in range(i + 1, len(cluster_names)):
                if _names_share_root(cluster_names[i], cluster_names[j]):
                    connected.add(i)
                    connected.add(j)

        if len(connected) < 2:
            continue

        connected_names = [cluster_names[i] for i in sorted(connected)]
        canonical = _pick_canonical(connected_names, entity_counts, contact_names)
        for name in connected_names:
            if name != canonical:
                alias_map[name] = canonical

    return alias_map


def build_entity_edges(store: GraphStore, min_score: int = 1) -> dict:
    """Create same_entity and same_topic edges from triage classification.

    Args:
        store: Graph store instance.
        min_score: Minimum triage_relevant to include (default 1 = all triaged).

    Returns:
        Dict with counts: {"entity_edges", "topic_edges", "entities_found", "topics_found"}
    """
    console = Console()
    now = int(time.time())

    # Load all triaged nodes with entities/topics
    rows = store.conn.execute(
        """SELECT id, entities, topics, triage_relevant
           FROM nodes
           WHERE triage_relevant IS NOT NULL
             AND triage_relevant >= ?
             AND (entities IS NOT NULL OR topics IS NOT NULL)""",
        (min_score,),
    ).fetchall()

    console.print(f"[bold]Entity Edge Builder: {len(rows)} triaged nodes[/bold]")

    # First pass: collect all entities to find duplicates
    raw_entity_counts: Counter = Counter()
    for row in rows:
        entities_raw = row["entities"]
        if entities_raw:
            try:
                entities = json.loads(entities_raw) if isinstance(entities_raw, str) else entities_raw
                for entity in entities:
                    key = _clean_entity(entity)
                    if key and len(key) > 1 and key not in ENTITY_STOPLIST:
                        raw_entity_counts[key] += 1
            except (json.JSONDecodeError, TypeError):
                pass

    # Load contact names for canonical selection (prefer full names over first-only)
    contact_rows = store.conn.execute(
        "SELECT display_name FROM contact_stats WHERE display_name IS NOT NULL"
    ).fetchall()
    # Build a set of all name forms: full name, reversed, first, last
    contact_names: set[str] = set()
    for row in contact_rows:
        name = row["display_name"].strip().lower()
        if not name:
            continue
        contact_names.add(name)
        parts = name.split()
        if len(parts) >= 2:
            # "emily voigt" also matches "voigt emily"
            contact_names.add(f"{parts[-1]} {' '.join(parts[:-1])}")
    console.print(f"  Loaded {len(contact_names)} contact names for canonical matching")

    # Run embedding-based dedup on entities that appear 2+ times
    frequent_entities = [name for name, cnt in raw_entity_counts.most_common() if cnt >= 2]
    console.print(f"  Running entity dedup on {len(frequent_entities)} entities...")
    alias_map = _run_entity_dedup(frequent_entities, store, entity_counts=raw_entity_counts, contact_names=contact_names)
    if alias_map:
        console.print(f"  Found {len(alias_map)} aliases:")
        for alias, canonical in sorted(alias_map.items()):
            console.print(f"    {alias} → {canonical}")

    # Second pass: build inverted indexes with normalization
    entity_index: dict[str, list[str]] = defaultdict(list)
    topic_index: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        node_id = row["id"]

        # Parse entities
        entities_raw = row["entities"]
        if entities_raw:
            try:
                entities = json.loads(entities_raw) if isinstance(entities_raw, str) else entities_raw
                for entity in entities:
                    key = _clean_entity(entity)
                    if key and len(key) > 1 and key not in ENTITY_STOPLIST:
                        key = alias_map.get(key, key)
                        entity_index[key].append(node_id)
            except (json.JSONDecodeError, TypeError):
                pass

        # Parse topics
        topics_raw = row["topics"]
        if topics_raw:
            try:
                topics = json.loads(topics_raw) if isinstance(topics_raw, str) else topics_raw
                for topic in topics:
                    key = topic.strip().lower()
                    if key and len(key) > 1 and key not in TOPIC_STOPLIST:
                        topic_index[key].append(node_id)
            except (json.JSONDecodeError, TypeError):
                pass

    entity_groups = {k: v for k, v in entity_index.items() if len(v) >= 2}
    topic_groups = {k: v for k, v in topic_index.items() if len(v) >= 2}

    console.print(f"  Entities with 2+ nodes: {len(entity_groups)} (of {len(entity_index)} unique)")
    console.print(f"  Topics with 2+ nodes: {len(topic_groups)} (of {len(topic_index)} unique)")

    # Show top entities
    top_entities = sorted(entity_groups.items(), key=lambda x: -len(x[1]))[:10]
    if top_entities:
        console.print("  Top entities:")
        for name, nodes in top_entities:
            console.print(f"    {name}: {len(nodes)} nodes")

    # Create edges
    entity_edge_count = 0
    topic_edge_count = 0

    import math
    import random

    # IDF-style weight: smaller clusters = stronger signal per edge.
    def idf_weight(base: float, cluster_size: int) -> float:
        return round(base / max(1.0, math.log2(cluster_size)), 3)

    # For large clusters, use k-nearest neighbors instead of all-pairs.
    # Each node connects to at most MAX_EDGES_PER_NODE random peers in the cluster.
    # This keeps Alteris (763 nodes) at ~7600 edges instead of ~290K.
    MAX_EDGES_PER_NODE = 10

    def _create_cluster_edges(
        node_ids: list[str], edge_type: str, base_weight: float,
    ) -> int:
        n = len(node_ids)
        w = idf_weight(base_weight, n)
        count = 0

        if n * (n - 1) // 2 <= MAX_EDGES_PER_NODE * n:
            # Small cluster: all-pairs
            for i in range(n):
                for j in range(i + 1, n):
                    src, dst = node_ids[i], node_ids[j]
                    if src > dst:
                        src, dst = dst, src
                    try:
                        store.conn.execute(
                            """INSERT OR IGNORE INTO edges (src, dst, edge_type, weight, created_at)
                               VALUES (?, ?, ?, ?, ?)""",
                            (src, dst, edge_type, w, now),
                        )
                        count += 1
                    except Exception:
                        pass
        else:
            # Large cluster: each node connects to K random peers
            for i in range(n):
                peers = random.sample(range(n), min(MAX_EDGES_PER_NODE, n - 1))
                for j in peers:
                    if i == j:
                        continue
                    src, dst = node_ids[i], node_ids[j]
                    if src > dst:
                        src, dst = dst, src
                    try:
                        store.conn.execute(
                            """INSERT OR IGNORE INTO edges (src, dst, edge_type, weight, created_at)
                               VALUES (?, ?, ?, ?, ?)""",
                            (src, dst, edge_type, w, now),
                        )
                        count += 1
                    except Exception:
                        pass
        return count

    # same_entity edges (base weight 0.5)
    for entity, node_ids in entity_groups.items():
        entity_edge_count += _create_cluster_edges(node_ids, "same_entity", 0.5)

    # same_topic edges (base weight 0.2)
    for topic, node_ids in topic_groups.items():
        topic_edge_count += _create_cluster_edges(node_ids, "same_topic", 0.2)

    store.conn.commit()

    console.print(f"\n  Created {entity_edge_count} same_entity edges")
    console.print(f"  Created {topic_edge_count} same_topic edges")

    return {
        "entity_edges": entity_edge_count,
        "topic_edges": topic_edge_count,
        "entities_found": len(entity_groups),
        "topics_found": len(topic_groups),
    }
