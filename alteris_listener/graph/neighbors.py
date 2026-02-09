"""Neighborhood retrieval for message passing.

Combines structural (edge-based) and semantic (embedding similarity)
retrieval to find the relevant subgraph around a node. Uses budget-aware
traversal to control LLM token spend.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

from alteris_listener.graph.scoring import temporal_decay
from alteris_listener.graph.store import GraphStore

logger = logging.getLogger(__name__)


@dataclass
class ScoredNeighbor:
    """A neighbor node with combined relevance score."""
    node_id: str
    node_type: str
    score: float
    depth: int
    edge_types: list[str] = field(default_factory=list)
    source: str = ""  # "structural", "semantic", or "both"

    @property
    def cost(self) -> float:
        """Estimated token cost to include this node in LLM context."""
        base_costs = {
            "email": 200,
            "message": 80,
            "meeting": 500,
            "contact": 30,
            "calendar_event": 50,
            "task": 60,
            "summary": 150,
        }
        base = base_costs.get(self.node_type, 100)
        # Deeper nodes are cheaper (already summarized in our mental model)
        return base * (0.7 ** self.depth)


class NeighborhoodQuery:
    """Budget-aware neighborhood retrieval from the graph.

    Combines two retrieval paths:
    1. Structural: follow edges in SQLite (microseconds)
    2. Semantic: embedding similarity in the EmbeddingIndex (milliseconds)

    Results are merged, deduped, and ranked by combined score.
    """

    def __init__(
        self,
        store: GraphStore,
        embedding_index: Optional["EmbeddingIndex"] = None,
    ):
        self.store = store
        self.embedding_index = embedding_index

    def get_neighborhood(
        self,
        node_id: str,
        max_depth: int = 2,
        budget: int = 4000,
        edge_types: list[str] | None = None,
        time_decay_half_life: float = 14.0,
        semantic_k: int = 20,
        min_score: float = 0.05,
        now: int | None = None,
    ) -> list[ScoredNeighbor]:
        """Retrieve the relevant neighborhood around a node.

        Args:
            node_id: Center node to explore from.
            max_depth: Maximum hops to traverse (1-3).
            budget: Token budget for included nodes.
            edge_types: Filter to specific edge types, or None for all.
            time_decay_half_life: How fast old nodes lose relevance.
            semantic_k: How many semantic neighbors to retrieve.
            min_score: Minimum combined score to include.
            now: Current timestamp (for temporal scoring).

        Returns:
            List of ScoredNeighbor sorted by score descending,
            with total cost within budget.
        """
        now = now or int(time.time())
        candidates: dict[str, ScoredNeighbor] = {}

        # ── Structural neighbors ─────────────────────────────────
        self._traverse_structural(
            node_id=node_id,
            candidates=candidates,
            edge_types=edge_types,
            max_depth=max_depth,
            depth=0,
            now=now,
            time_decay_half_life=time_decay_half_life,
            visited={node_id},
        )

        # ── Semantic neighbors ───────────────────────────────────
        if self.embedding_index:
            self._find_semantic(
                node_id=node_id,
                candidates=candidates,
                k=semantic_k,
                now=now,
                time_decay_half_life=time_decay_half_life,
            )

        # ── Filter and budget ────────────────────────────────────
        scored = [n for n in candidates.values() if n.score >= min_score]
        scored.sort(key=lambda n: n.score, reverse=True)

        # Apply budget
        result = []
        remaining = budget
        for neighbor in scored:
            cost = neighbor.cost
            if cost > remaining:
                continue
            remaining -= cost
            result.append(neighbor)

        logger.debug(
            "Neighborhood for %s: %d structural + semantic candidates → %d within budget (%d tokens remaining)",
            node_id, len(candidates), len(result), remaining,
        )
        return result

    def _traverse_structural(
        self,
        node_id: str,
        candidates: dict[str, ScoredNeighbor],
        edge_types: list[str] | None,
        max_depth: int,
        depth: int,
        now: int,
        time_decay_half_life: float,
        visited: set[str],
    ):
        """BFS traversal through graph edges."""
        if depth >= max_depth:
            return

        edges = self.store.get_neighbors(node_id, edge_types)

        for edge in edges:
            neighbor_id = edge["dst"] if edge["src"] == node_id else edge["src"]

            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)

            # Fetch the neighbor node for scoring
            neighbor_node = self.store.get_node(neighbor_id)
            if not neighbor_node:
                continue

            # Score = edge_weight × time_decay × depth_discount
            edge_weight = edge.get("weight", 1.0)
            ts = neighbor_node.get("timestamp")
            time_score = temporal_decay(ts, now, time_decay_half_life) if ts else 0.5
            depth_discount = 0.7 ** depth

            combined_score = edge_weight * time_score * depth_discount

            if neighbor_id in candidates:
                # Boost if found via multiple paths
                existing = candidates[neighbor_id]
                existing.score = max(existing.score, combined_score * 1.2)
                existing.edge_types.append(edge["edge_type"])
                if existing.source == "semantic":
                    existing.source = "both"
            else:
                candidates[neighbor_id] = ScoredNeighbor(
                    node_id=neighbor_id,
                    node_type=neighbor_node.get("node_type", "unknown"),
                    score=combined_score,
                    depth=depth + 1,
                    edge_types=[edge["edge_type"]],
                    source="structural",
                )

            # Recurse if high enough score
            if combined_score > 0.3:
                self._traverse_structural(
                    node_id=neighbor_id,
                    candidates=candidates,
                    edge_types=edge_types,
                    max_depth=max_depth,
                    depth=depth + 1,
                    now=now,
                    time_decay_half_life=time_decay_half_life,
                    visited=visited,
                )

    def _find_semantic(
        self,
        node_id: str,
        candidates: dict[str, ScoredNeighbor],
        k: int,
        now: int,
        time_decay_half_life: float,
    ):
        """Find semantically similar nodes via embedding index."""
        if not self.embedding_index:
            return

        center_node = self.store.get_node(node_id)
        if not center_node or not center_node.get("embedding"):
            return

        # The embedding index search returns (node_id, similarity) pairs
        similar = self.embedding_index.search_by_node_id(node_id, k=k)

        for sim_id, similarity in similar:
            if sim_id == node_id:
                continue

            sim_node = self.store.get_node(sim_id)
            if not sim_node:
                continue

            ts = sim_node.get("timestamp")
            time_score = temporal_decay(ts, now, time_decay_half_life) if ts else 0.5

            # Semantic score = similarity × time_decay
            semantic_score = similarity * time_score

            if sim_id in candidates:
                # Convergence bonus: found via both structural and semantic
                existing = candidates[sim_id]
                existing.score = max(existing.score, semantic_score) * 1.3
                existing.source = "both"
            else:
                candidates[sim_id] = ScoredNeighbor(
                    node_id=sim_id,
                    node_type=sim_node.get("node_type", "unknown"),
                    score=semantic_score,
                    depth=0,
                    source="semantic",
                )

    def get_scored_neighborhood_sql(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        max_neighbors: int = 50,
        time_cutoff_days: int = 90,
        now: int | None = None,
    ) -> list[dict]:
        """Fast SQL-only neighborhood retrieval with inline scoring.

        For when you need speed over depth. Single SQL query,
        no Python traversal. Returns scored neighbors with
        weight × time_decay ranking.
        """
        now = now or int(time.time())
        cutoff = now - (time_cutoff_days * 86400)

        if edge_types:
            type_filter = "AND e.edge_type IN ({})".format(
                ",".join("?" * len(edge_types))
            )
            params = [node_id, node_id] + edge_types + edge_types + [cutoff, max_neighbors]
        else:
            type_filter = ""
            params = [node_id, node_id, cutoff, max_neighbors]

        sql = f"""
            WITH neighborhood AS (
                SELECT dst AS neighbor_id, edge_type, weight
                FROM edges e
                WHERE src = ? {type_filter}
                UNION ALL
                SELECT src AS neighbor_id, edge_type, weight
                FROM edges e
                WHERE dst = ? {type_filter}
            )
            SELECT
                n.id, n.node_type, n.source, n.timestamp, n.subject,
                n.sender, n.body_preview, n.heuristic_score,
                nb.edge_type, nb.weight,
                nb.weight * (1.0 / (1 + (? - COALESCE(n.timestamp, ?)) / 86400.0)) AS combined_score
            FROM neighborhood nb
            JOIN nodes n ON n.id = nb.neighbor_id
            WHERE n.timestamp > ? OR n.timestamp IS NULL
            ORDER BY combined_score DESC
            LIMIT ?
        """

        # Adjust params for the inline now reference
        if edge_types:
            params = [node_id] + edge_types + [node_id] + edge_types + [now, now, cutoff, max_neighbors]
        else:
            params = [node_id, node_id, now, now, cutoff, max_neighbors]

        rows = self.store.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
