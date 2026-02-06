"""Alteris API session management and browser-based login.

Handles authentication with the Alteris backend. The CLI never touches
Firebase directly — it talks to the Alteris API, which handles all
database operations server-side.
"""

from __future__ import annotations

import logging
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse

import requests

from alteris_listener.api.config import (
    load_api_config,
    load_auth,
    store_auth,
    clear_auth,
)

logger = logging.getLogger(__name__)


class AlterisSession:
    """Manages auth session with the Alteris API.

    Stores an opaque ID token + refresh token. The CLI doesn't know
    or care what these tokens are internally (Firebase, JWT, etc.) —
    it just sends them as Bearer tokens to the API.
    """

    def __init__(self):
        self._id_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._uid: Optional[str] = None
        self._email: Optional[str] = None
        self._token_expiry: float = 0
        self._config = load_api_config()

    @property
    def is_authenticated(self) -> bool:
        return self._refresh_token is not None

    @property
    def uid(self) -> Optional[str]:
        return self._uid

    @property
    def email(self) -> Optional[str]:
        return self._email

    @property
    def api_url(self) -> str:
        return self._config.get("api_url", "")

    @property
    def config(self) -> Dict[str, str]:
        return self._config

    def load_from_keychain(self) -> bool:
        """Restore session from Keychain."""
        auth = load_auth()
        if not auth or "refresh_token" not in auth:
            return False
        self._refresh_token = auth["refresh_token"]
        self._uid = auth.get("uid")
        self._email = auth.get("email")
        self._config = load_api_config()
        return True

    def get_id_token(self) -> Optional[str]:
        """Get a valid ID token, refreshing if expired."""
        if not self._refresh_token:
            logger.error("Not authenticated — run 'alteris-listener login' first")
            return None

        if self._id_token and time.time() < self._token_expiry - 60:
            return self._id_token

        api_url = self._config.get("api_url", "")
        refresh_endpoint = self._config.get("refresh_endpoint", "/cli_refresh_token")

        if not api_url:
            logger.error("Missing api_url in config — run 'alteris-listener login' again")
            return None

        resp = requests.post(
            f"{api_url}{refresh_endpoint}",
            json={"refresh_token": self._refresh_token},
        )

        if resp.status_code != 200:
            logger.error("Token refresh failed — run 'alteris-listener login' again")
            return None

        data = resp.json()
        self._id_token = data.get("id_token")
        self._refresh_token = data.get("refresh_token", self._refresh_token)
        self._token_expiry = time.time() + int(data.get("expires_in", 3600))

        store_auth({
            "refresh_token": self._refresh_token,
            "uid": self._uid or "",
            "email": self._email or "",
        })

        return self._id_token

    def login_with_tokens(self, id_token: str, refresh_token: str, uid: str, email: str):
        """Complete login with tokens received from the auth callback."""
        self._id_token = id_token
        self._refresh_token = refresh_token
        self._uid = uid
        self._email = email
        self._token_expiry = time.time() + 3600

        store_auth({
            "refresh_token": refresh_token,
            "uid": uid,
            "email": email,
        })

    def logout(self):
        """Clear stored credentials."""
        self._id_token = None
        self._refresh_token = None
        self._uid = None
        self._email = None
        self._token_expiry = 0
        clear_auth()


def login_via_browser(auth_url: str, port: int = 9876) -> Optional[Dict[str, str]]:
    """Open browser for Alteris login, capture tokens from callback.

    Flow:
    1. Opens browser to auth_url (Alteris CLI auth page)
    2. User signs in with Google
    3. Alteris backend mints tokens
    4. Browser redirects to localhost:PORT/callback with tokens
    5. We capture and return them
    """
    captured: Dict[str, Optional[Dict[str, str]]] = {"data": None}

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)

            token = params.get("token", [None])[0]
            error = params.get("error", [None])[0]

            if token:
                captured["data"] = {
                    "id_token": token,
                    "refresh_token": params.get("refresh_token", [""])[0],
                    "uid": params.get("uid", [""])[0],
                    "email": params.get("email", [""])[0],
                }
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"""
                    <html><body style="font-family:sans-serif;text-align:center;padding:50px">
                    <h1>&#10004; Logged in to Alteris</h1>
                    <p>You can close this tab and return to your terminal.</p>
                    </body></html>
                """)
            else:
                self.send_response(400)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                msg = error or "Authentication failed"
                self.wfile.write(f"""
                    <html><body style="font-family:sans-serif;text-align:center;padding:50px">
                    <h1>&#10008; Login failed</h1><p>{msg}</p>
                    </body></html>
                """.encode())

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), CallbackHandler)
    server.timeout = 120

    full_url = f"{auth_url}?redirect_uri=http://localhost:{port}/callback"
    webbrowser.open(full_url)

    server.handle_request()
    server.server_close()

    return captured["data"]


def check_auth_status() -> tuple[bool, str]:
    """Check if user is logged in."""
    session = AlterisSession()
    if not session.load_from_keychain():
        return False, "Not logged in. Run: alteris-listener login"

    token = session.get_id_token()
    if not token:
        return False, "Session expired. Run: alteris-listener login"

    return True, f"Logged in as {session.email} (uid: {session.uid})"
