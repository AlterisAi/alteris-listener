"""Granola meeting transcript reader.

Granola stores auth tokens locally at ~/Library/Application Support/Granola/
but documents and transcripts are fetched via the Granola API.

Auth flow: reads WorkOS refresh token from local storage, exchanges for
access token, then fetches documents and transcripts.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from alteris_listener.sources.base import Message

logger = logging.getLogger(__name__)

GRANOLA_DATA_DIR = Path.home() / "Library" / "Application Support" / "Granola"
GRANOLA_API = "https://api.granola.ai"
WORKOS_AUTH_URL = "https://api.workos.com/user_management/authenticate"
CLIENT_VERSION = "5.354.0"


def _find_auth_token() -> Optional[Dict[str, Any]]:
    """Find Granola auth credentials from local storage.

    Granola stores WorkOS tokens locally. We look for either:
    - workos_tokens.json (newer versions)
    - supabase.json (older versions, has access_token directly)
    """
    for candidate in [
        GRANOLA_DATA_DIR / "workos_tokens.json",
        GRANOLA_DATA_DIR / "auth.json",
        GRANOLA_DATA_DIR / "tokens.json",
    ]:
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text())
                if "refresh_token" in data:
                    return {"type": "workos", "data": data, "path": str(candidate)}
                if "access_token" in data:
                    return {"type": "direct", "data": data, "path": str(candidate)}
            except (json.JSONDecodeError, KeyError):
                continue

    # Try supabase.json (Granola stores WorkOS tokens here as a JSON string)
    supabase_path = GRANOLA_DATA_DIR / "supabase.json"
    if supabase_path.exists():
        try:
            data = json.loads(supabase_path.read_text())
            if isinstance(data, dict):
                workos_str = data.get("workos_tokens", "")
                if workos_str and isinstance(workos_str, str):
                    try:
                        workos_data = json.loads(workos_str)
                        if "refresh_token" in workos_data:
                            return {"type": "workos", "data": workos_data, "path": str(supabase_path)}
                        if "access_token" in workos_data:
                            return {"type": "direct", "data": workos_data, "path": str(supabase_path)}
                    except json.JSONDecodeError:
                        pass
                for key, val in data.items():
                    if isinstance(val, dict) and "access_token" in val:
                        return {"type": "direct", "data": val, "path": str(supabase_path)}
        except (json.JSONDecodeError, KeyError):
            pass

    # Scan all JSON files in the directory
    if GRANOLA_DATA_DIR.exists():
        for json_file in GRANOLA_DATA_DIR.glob("*.json"):
            try:
                data = json.loads(json_file.read_text())
                if isinstance(data, dict):
                    if "refresh_token" in data:
                        return {"type": "workos", "data": data, "path": str(json_file)}
                    if "access_token" in data:
                        return {"type": "direct", "data": data, "path": str(json_file)}
            except (json.JSONDecodeError, KeyError):
                continue

    return None


def _refresh_access_token(refresh_token: str, client_id: str, token_path: str) -> Optional[str]:
    """Exchange refresh token for a new access token via WorkOS.

    WorkOS uses refresh token rotation — each refresh token is single-use,
    so we must save the new one.
    """
    resp = requests.post(WORKOS_AUTH_URL, json={
        "client_id": client_id,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    })

    if resp.status_code != 200:
        logger.error("WorkOS token refresh failed: %s %s", resp.status_code, resp.text)
        return None

    token_data = resp.json()
    new_access = token_data.get("access_token")
    new_refresh = token_data.get("refresh_token")

    # Save rotated refresh token back to disk
    if new_refresh and token_path:
        try:
            path = Path(token_path)
            existing = json.loads(path.read_text())

            if path.name == "supabase.json" and "workos_tokens" in existing:
                workos_data = json.loads(existing["workos_tokens"])
                workos_data["refresh_token"] = new_refresh
                if new_access:
                    workos_data["access_token"] = new_access
                existing["workos_tokens"] = json.dumps(workos_data)
            else:
                existing["refresh_token"] = new_refresh
                if new_access:
                    existing["access_token"] = new_access

            path.write_text(json.dumps(existing, indent=2))
            logger.debug("Saved rotated refresh token to %s", token_path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not save rotated token: %s", exc)

    return new_access


def _get_access_token() -> Optional[str]:
    """Get a valid Granola API access token."""
    auth = _find_auth_token()
    if not auth:
        logger.error(
            "Granola auth not found. Make sure Granola is installed and you're logged in. "
            "Looked in: %s", GRANOLA_DATA_DIR
        )
        return None

    if auth["type"] == "direct":
        return auth["data"]["access_token"]

    # WorkOS flow — check if access token is still valid before refreshing
    data = auth["data"]
    access_token = data.get("access_token", "")
    obtained_at = data.get("obtained_at", 0)  # milliseconds
    expires_in = data.get("expires_in", 0)  # seconds

    if access_token and obtained_at and expires_in:
        expiry_ms = obtained_at + (expires_in * 1000)
        now_ms = time.time() * 1000
        if now_ms < expiry_ms - 60000:  # 1 minute buffer
            logger.debug("Using existing Granola access token (still valid)")
            return access_token

    refresh_token = data.get("refresh_token")
    client_id = data.get("client_id", "")

    if not client_id:
        # Extract client_id from JWT issuer claim
        access_token = data.get("access_token", "")
        if access_token:
            try:
                payload = access_token.split(".")[1]
                payload += "=" * (4 - len(payload) % 4)
                jwt_data = json.loads(base64.urlsafe_b64decode(payload))
                iss = jwt_data.get("iss", "")
                if "/client_" in iss:
                    client_id = iss.rsplit("/", 1)[-1]
            except Exception:
                pass

    if not client_id:
        for candidate in GRANOLA_DATA_DIR.glob("*.json"):
            try:
                d = json.loads(candidate.read_text())
                if isinstance(d, dict) and "client_id" in d:
                    client_id = d["client_id"]
                    break
            except (json.JSONDecodeError, KeyError):
                continue

    if not client_id:
        logger.error("WorkOS client_id not found — cannot refresh token")
        return None

    return _refresh_access_token(refresh_token, client_id, auth["path"])


def _api_headers(access_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": f"Granola/{CLIENT_VERSION}",
        "X-Client-Version": CLIENT_VERSION,
    }


def _fetch_documents(access_token: str, limit: int = 50, offset: int = 0) -> List[Dict]:
    """Fetch recent documents from Granola API."""
    resp = requests.post(
        f"{GRANOLA_API}/v2/get-documents",
        headers=_api_headers(access_token),
        json={
            "limit": limit,
            "offset": offset,
            "include_last_viewed_panel": True,
        },
    )

    if resp.status_code != 200:
        logger.error("Failed to fetch Granola documents: %s", resp.status_code)
        return []

    data = resp.json()
    return data.get("docs", [])


def _fetch_transcript(access_token: str, document_id: str) -> List[Dict]:
    """Fetch transcript for a specific document."""
    resp = requests.post(
        f"{GRANOLA_API}/v1/get-document-transcript",
        headers=_api_headers(access_token),
        json={"document_id": document_id},
    )

    if resp.status_code == 404:
        logger.debug("No transcript for document %s", document_id)
        return []

    if resp.status_code != 200:
        logger.error("Failed to fetch transcript for %s: %s", document_id, resp.status_code)
        return []

    data = resp.json()
    return data if isinstance(data, list) else []


def _format_transcript(utterances: List[Dict]) -> str:
    """Format transcript utterances into readable text."""
    lines = []
    for u in utterances:
        source = u.get("source", "unknown")
        text = u.get("text", "").strip()
        start = u.get("start_timestamp", "")

        if not text:
            continue

        ts_display = ""
        if start:
            try:
                dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                ts_display = dt.strftime("%H:%M:%S")
            except ValueError:
                ts_display = start[:8]

        speaker = "🎤 You" if source == "microphone" else "🔊 Other"
        lines.append(f"[{ts_display}] {speaker}: {text}")

    return "\n".join(lines)


def check_granola_available() -> tuple[bool, str]:
    """Check if Granola is installed and auth is available."""
    if not GRANOLA_DATA_DIR.exists():
        return False, "Granola not installed (directory not found)"

    auth = _find_auth_token()
    if not auth:
        return False, "Granola installed but no auth tokens found — are you logged in?"

    return True, f"Granola auth found at {auth['path']}"


def read_recent_meetings(
    limit: int = 10,
    include_transcript: bool = True,
    hours: Optional[int] = None,
) -> List[Message]:
    """Read recent meeting notes and transcripts from Granola.

    Args:
        limit: Max number of meetings to fetch.
        include_transcript: Whether to fetch full transcripts (slower, one API call per meeting).
        hours: If set, only return meetings from the last N hours.
    """
    access_token = _get_access_token()
    if not access_token:
        return []

    documents = _fetch_documents(access_token, limit=limit)
    if not documents:
        logger.info("No Granola documents found")
        return []

    cutoff = None
    if hours:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    messages = []
    for doc in documents:
        doc_id = doc.get("id", "")
        title = doc.get("title", "Untitled Meeting")
        created = doc.get("created_at", "")
        updated = doc.get("updated_at", "")

        ts_str = created or updated
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            ts = datetime.now(timezone.utc)

        if cutoff and ts < cutoff:
            continue

        # Extract notes from ProseMirror content
        notes = ""
        panel = doc.get("last_viewed_panel")
        if panel and isinstance(panel, dict):
            content = panel.get("content")
            if content and isinstance(content, dict):
                notes = _prosemirror_to_text(content)

        transcript_text = ""
        if include_transcript:
            utterances = _fetch_transcript(access_token, doc_id)
            if utterances:
                transcript_text = _format_transcript(utterances)

        body_parts = []
        if notes:
            body_parts.append("## Meeting Notes\n" + notes)
        if transcript_text:
            body_parts.append("## Transcript\n" + transcript_text)

        body = "\n\n".join(body_parts) if body_parts else "(no content)"

        messages.append(Message(
            source="granola",
            sender="meeting",
            recipient="me",
            subject=title,
            body=body,
            timestamp=ts,
            thread_id=doc_id,
            metadata={
                "document_id": doc_id,
                "created_at": created,
                "updated_at": updated,
                "has_transcript": bool(transcript_text),
            },
        ))

    return messages


def _prosemirror_to_text(node: Dict) -> str:
    """Convert ProseMirror JSON to plain text."""
    if not isinstance(node, dict):
        return ""

    node_type = node.get("type", "")
    text_parts = []

    if node_type == "text":
        return node.get("text", "")

    if node_type == "heading":
        level = node.get("attrs", {}).get("level", 1)
        prefix = "#" * level + " "
        children_text = _prosemirror_children_text(node)
        if children_text:
            text_parts.append(prefix + children_text)

    elif node_type == "paragraph":
        children_text = _prosemirror_children_text(node)
        if children_text:
            text_parts.append(children_text)

    elif node_type in ("bulletList", "orderedList"):
        for i, item in enumerate(node.get("content", []), 1):
            item_text = _prosemirror_children_text(item)
            if item_text:
                prefix = f"{i}. " if node_type == "orderedList" else "- "
                text_parts.append(prefix + item_text)

    else:
        for child in node.get("content", []):
            child_text = _prosemirror_to_text(child)
            if child_text:
                text_parts.append(child_text)

    return "\n".join(text_parts)


def _prosemirror_children_text(node: Dict) -> str:
    """Get concatenated text from a node's children."""
    parts = []
    for child in node.get("content", []):
        parts.append(_prosemirror_to_text(child))
    return "".join(parts)
