"""Alteris Listener CLI — query local Mac data sources with LLMs."""

import subprocess

import click
from rich.console import Console

from alteris_listener.cli.ask_cmd import ask
from alteris_listener.cli.auth_cmd import login, logout, whoami
from alteris_listener.cli.query_cmd import run_query

console = Console()


@click.group()
def cli():
    """Alteris Listener — query local Mac data with LLMs."""
    pass


@cli.command("set-key")
@click.argument("provider", type=click.Choice(["gemini", "claude", "slack"]))
@click.option("--key", prompt=True, hide_input=True, help="API key")
def set_key(provider, key):
    """Store an API key in macOS Keychain.

    Examples:

        alteris-listener set-key gemini

        alteris-listener set-key claude
    """
    service = "alteris-listener"
    account = provider
    subprocess.run(
        ["security", "delete-generic-password", "-a", account, "-s", service],
        capture_output=True,
    )
    result = subprocess.run(
        ["security", "add-generic-password", "-a", account, "-s", service, "-w", key],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        console.print(f"[green]✓[/green] {provider} API key stored in Keychain")
    else:
        console.print(f"[red]Failed to store key:[/red] {result.stderr}")


cli.add_command(run_query)
cli.add_command(ask)
cli.add_command(login)
cli.add_command(logout)
cli.add_command(whoami)


if __name__ == "__main__":
    cli()
