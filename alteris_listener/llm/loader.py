"""Load query definitions from Markdown files with YAML frontmatter."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class QueryDefinition:
    """A parsed query from a markdown file."""

    name: str
    sources: List[str]
    description: str
    prompt_body: str
    model_params: Dict[str, Any] = field(default_factory=dict)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[str] = None

    @property
    def applies_to(self) -> List[str]:
        """Sources this query applies to (e.g. ['emails', 'any'])."""
        return self.sources


def load_query_file(path: Path) -> Optional[QueryDefinition]:
    """Parse a single query markdown file.

    Expected format:
        ---
        query_name: ...
        sources: [...]
        ...
        ---
        # Prompt body in markdown
    """
    text = path.read_text(encoding="utf-8")

    if not text.startswith("---"):
        logger.warning("Query file %s missing YAML frontmatter, skipping", path)
        return None

    parts = text.split("---", 2)
    if len(parts) < 3:
        logger.warning("Query file %s has malformed frontmatter, skipping", path)
        return None

    frontmatter_raw = parts[1].strip()
    prompt_body = parts[2].strip()

    try:
        meta = yaml.safe_load(frontmatter_raw)
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse YAML in %s: %s", path, exc)
        return None

    if not isinstance(meta, dict):
        logger.warning("Frontmatter in %s is not a dict, skipping", path)
        return None

    return QueryDefinition(
        name=meta.get("query_name", path.stem),
        sources=meta.get("sources", ["any"]),
        description=meta.get("description", ""),
        prompt_body=prompt_body,
        model_params=meta.get("model_params", {}),
        input_schema=meta.get("input_schema", {}),
        output_schema=meta.get("output_schema", {}),
        file_path=str(path),
    )


def load_queries_from_dir(
    queries_dir: Path,
    source_filter: Optional[str] = None,
) -> Dict[str, QueryDefinition]:
    """Load all query .md files from a directory.

    Args:
        queries_dir: Path to directory containing query .md files.
        source_filter: If provided, only return queries whose sources
                       include this value or 'any'.

    Returns:
        Dict mapping query_name -> QueryDefinition.
    """
    if not queries_dir.is_dir():
        logger.error("Queries directory not found: %s", queries_dir)
        return {}

    queries: Dict[str, QueryDefinition] = {}
    for md_file in sorted(queries_dir.glob("*.md")):
        defn = load_query_file(md_file)
        if defn is None:
            continue

        if source_filter and source_filter not in defn.sources and "any" not in defn.sources:
            continue

        queries[defn.name] = defn
        logger.debug("Loaded query: %s (sources=%s)", defn.name, defn.sources)

    logger.info("Loaded %d queries from %s", len(queries), queries_dir)
    return queries
