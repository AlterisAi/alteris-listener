"""Upload query results to the Alteris backend."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests
from rich.console import Console

from alteris_listener.api.session import AlterisSession

logger = logging.getLogger(__name__)
console = Console()


def upload_results(
    session: AlterisSession,
    query_name: str,
    results: Any,
    source: str = "",
    provider_id: str = "",
    thread_subject: str = "",
    cursor: Optional[Dict[str, str]] = None,
) -> bool:
    """Upload query results to the Alteris backend.

    The backend handles all database writes — the CLI just POSTs JSON.
    """
    id_token = session.get_id_token()
    if not id_token:
        return False

    api_url = session.api_url
    upload_endpoint = session.config.get("upload_endpoint", "/cli_upload_results")

    if isinstance(results, dict):
        payload_results = results.get("tasks", results)
    else:
        payload_results = results

    payload: Dict[str, Any] = {
        "query_name": query_name,
        "results": payload_results,
        "source": source,
        "provider_id": provider_id,
        "thread_subject": thread_subject,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "client": "alteris-listener",
    }

    if cursor:
        payload["cursor"] = cursor

    resp = requests.post(
        f"{api_url}{upload_endpoint}",
        headers={
            "Authorization": f"Bearer {id_token}",
            "Content-Type": "application/json",
        },
        json=payload,
    )

    if resp.status_code != 200:
        logger.error("Upload failed: %s %s", resp.status_code, resp.text[:500])
        return False

    data = resp.json()
    if data.get("success"):
        count = data.get("stored_count", 0)
        account_id = data.get("account_id", "")
        logger.info("Uploaded %d items to Alteris (account: %s)", count, account_id)
        return True

    error = data.get("error", "Unknown error")
    logger.error("Upload rejected: %s", error)
    return False


def upload_query_results(
    query_name: str,
    results: dict,
    source: str = "",
    provider_id: str = "",
    thread_subject: str = "",
    cursor: Optional[dict] = None,
) -> bool:
    """Convenience wrapper: load session from Keychain and upload.

    Called from CLI commands when --upload flag is set.
    """
    session = AlterisSession()
    if not session.load_from_keychain():
        console.print("[yellow]Not logged in — skipping upload. Run: alteris-listener login[/yellow]")
        return False

    success = upload_results(
        session, query_name, results,
        source=source,
        provider_id=provider_id,
        thread_subject=thread_subject,
        cursor=cursor,
    )

    if success:
        console.print("  [green]↑ Synced to Alteris[/green]")
    else:
        console.print("  [red]↑ Sync failed[/red]")

    return success
