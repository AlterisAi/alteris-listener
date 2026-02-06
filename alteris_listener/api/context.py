"""Fetch and format user context from the Alteris backend."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from alteris_listener.api.session import AlterisSession

logger = logging.getLogger(__name__)


def fetch_user_context(
    session: AlterisSession,
    context_names: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch user context from context_store via the Alteris backend.

    Args:
        session: Authenticated session.
        context_names: Which context docs to load, e.g. ["clarity_queue", "goals_and_values"].
                       Default: ["clarity_queue"].

    Returns dict with:
        contexts: { doc_name: context_text, ... }
        user_email: str
        display_name: str

    Returns None if fetch fails.
    """
    id_token = session.get_id_token()
    if not id_token:
        return None

    api_url = session.api_url
    context_endpoint = session.config.get("context_endpoint", "/cli_get_context")
    names = context_names or ["clarity_queue"]

    try:
        resp = requests.post(
            f"{api_url}{context_endpoint}",
            headers={
                "Authorization": f"Bearer {id_token}",
                "Content-Type": "application/json",
            },
            json={"contexts": names},
            timeout=10,
        )

        if resp.status_code != 200:
            logger.warning("Failed to fetch user context: %s", resp.status_code)
            return None

        return resp.json()

    except requests.RequestException as exc:
        logger.warning("Error fetching user context: %s", exc)
        return None


def format_user_context(data: Dict[str, Any]) -> str:
    """Format context_store response into a text block for LLM injection.

    The context docs already contain well-formatted text (markdown profile,
    goals, etc.), so we just concatenate them with headers.
    """
    if not data:
        return ""

    parts = []

    email = data.get("user_email", "")
    name = data.get("display_name", "")
    if email or name:
        parts.append(f"## User: {name} ({email})" if name else f"## User: {email}")

    contexts = data.get("contexts", {})
    for doc_name, ctx_text in contexts.items():
        if ctx_text:
            parts.append(ctx_text)

    return "\n\n".join(parts)
