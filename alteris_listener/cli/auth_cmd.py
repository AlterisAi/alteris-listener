"""CLI commands for Alteris authentication."""

from __future__ import annotations

import logging
from typing import Optional

import click
from rich.console import Console

from alteris_listener.api.config import load_api_config, save_api_config
from alteris_listener.api.session import AlterisSession, check_auth_status, login_via_browser

console = Console()
logger = logging.getLogger(__name__)


@click.command("login")
@click.option("--api-url", default=None, help="Alteris API URL (saved to config)")
def login(api_url: Optional[str]):
    """Log in to Alteris to sync local results to the cloud.

    Opens your browser for Google sign-in, then stores credentials
    securely in macOS Keychain.

    On first use, you may need to set the API URL:
        alteris-listener login --api-url https://api.alteris.ai
    """
    config = load_api_config()

    if api_url:
        config["api_url"] = api_url
        save_api_config(config)

    base_url = config.get("api_url", "")
    auth_endpoint = config.get("auth_endpoint", "/cli_auth")
    auth_url = f"{base_url}{auth_endpoint}"

    if not base_url or "YOUR-PROJECT" in base_url:
        console.print("[red]API URL not configured.[/red]")
        console.print("Run: alteris-listener login --api-url https://your-api-url")
        return

    console.print("[bold]Opening browser for Alteris login...[/bold]")
    console.print("[dim]Waiting for authentication (timeout: 2 minutes)...[/dim]")

    result = login_via_browser(auth_url)

    if not result or not result.get("id_token"):
        console.print("[red]Login failed or timed out.[/red]")
        return

    session = AlterisSession()
    session.login_with_tokens(
        id_token=result["id_token"],
        refresh_token=result.get("refresh_token", ""),
        uid=result.get("uid", ""),
        email=result.get("email", ""),
    )

    save_api_config(config)

    console.print(f"[green]✓[/green] Logged in as [bold]{session.email}[/bold]")
    console.print(f"[dim]UID: {session.uid}[/dim]")
    console.print()
    console.print("Use [cyan]--upload[/cyan] with run-query to sync results to Alteris.")


@click.command("logout")
def logout():
    """Log out of Alteris and remove stored credentials."""
    session = AlterisSession()
    session.logout()
    console.print("[green]✓[/green] Logged out. Local credentials removed.")


@click.command("whoami")
def whoami():
    """Check current Alteris login status."""
    authenticated, detail = check_auth_status()
    if authenticated:
        console.print(f"[green]✓[/green] {detail}")
    else:
        console.print(f"[yellow]✗[/yellow] {detail}")
