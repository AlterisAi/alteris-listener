"""Alteris API configuration and Keychain helpers.

Shared by session, upload, and CLI auth modules.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".alteris" / "config.json"
KEYCHAIN_SERVICE = "alteris-listener"
KEYCHAIN_ACCOUNT = "alteris-auth"

# Default API URL — override via config file or `alteris-listener login --api-url`
DEFAULT_API_URL = "https://us-central1-ordinal-virtue-462602-p5.cloudfunctions.net"


def load_api_config() -> Dict[str, str]:
    """Load Alteris API config (URL, endpoints)."""
    if DEFAULT_CONFIG_PATH.exists():
        try:
            return json.loads(DEFAULT_CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "api_url": DEFAULT_API_URL,
        "auth_endpoint": "/cli_auth",
        "upload_endpoint": "/cli_upload_results",
    }


def save_api_config(config: Dict[str, str]):
    """Save config to disk."""
    DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_CONFIG_PATH.write_text(json.dumps(config, indent=2))


def store_auth(data: Dict[str, str]):
    """Store auth data in macOS Keychain."""
    payload = json.dumps(data)
    subprocess.run(
        ["security", "delete-generic-password", "-a", KEYCHAIN_ACCOUNT, "-s", KEYCHAIN_SERVICE],
        capture_output=True,
    )
    subprocess.run(
        ["security", "add-generic-password", "-a", KEYCHAIN_ACCOUNT, "-s", KEYCHAIN_SERVICE, "-w", payload],
        capture_output=True, text=True,
    )


def load_auth() -> Optional[Dict[str, str]]:
    """Load auth data from macOS Keychain."""
    result = subprocess.run(
        ["security", "find-generic-password", "-a", KEYCHAIN_ACCOUNT, "-s", KEYCHAIN_SERVICE, "-w"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return None


def clear_auth():
    """Remove auth data from Keychain."""
    subprocess.run(
        ["security", "delete-generic-password", "-a", KEYCHAIN_ACCOUNT, "-s", KEYCHAIN_SERVICE],
        capture_output=True,
    )
