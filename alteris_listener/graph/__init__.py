"""Alteris Knowledge Graph — local-first knowledge graph over user data."""

from alteris_listener.graph.store import GraphStore
from alteris_listener.graph.schema import init_db
from alteris_listener.graph.ingest import ingest_message, ingest_messages
from alteris_listener.graph.neighbors import NeighborhoodQuery
from alteris_listener.graph.embeddings import EmbeddingIndex

__all__ = [
    "GraphStore",
    "init_db",
    "ingest_message",
    "ingest_messages",
    "NeighborhoodQuery",
    "EmbeddingIndex",
]
