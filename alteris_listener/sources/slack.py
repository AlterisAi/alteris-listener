"""Slack message reader — fetches recent messages via the Slack API.

Uses a bot token stored in Keychain or environment.
"""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from alteris_listener.sources.base import Message

logger = logging.getLogger(__name__)

KEYCHAIN_SERVICE = "alteris-listener"


def _get_slack_token() -> Optional[str]:
    """Get Slack bot token from env or Keychain."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    if token:
        return token

    for account, service in [
        ("slack", KEYCHAIN_SERVICE),
        ("alteris-listener", "alteris-listener-slack"),
    ]:
        result = subprocess.run(
            ["security", "find-generic-password", "-a", account, "-s", service, "-w"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()

    return None


def check_slack_available() -> tuple[bool, str]:
    """Check if Slack token is configured."""
    token = _get_slack_token()
    if not token:
        return False, (
            "Slack token not found. Either:\n"
            "  1. Run: alteris-listener set-key slack\n"
            "  2. Or:  export SLACK_BOT_TOKEN='xoxb-...'"
        )
    return True, "Slack token found"


def _get_user_map(client) -> Dict[str, str]:
    """Build user ID -> display name map."""
    user_map = {}
    try:
        resp = client.users_list()
        if resp["ok"]:
            for member in resp["members"]:
                uid = member["id"]
                profile = member.get("profile", {})
                name = (
                    profile.get("display_name")
                    or profile.get("real_name")
                    or member.get("name", uid)
                )
                user_map[uid] = name
    except Exception as exc:
        logger.warning("Failed to fetch Slack users: %s", exc)

    return user_map


def _get_channels(client, types: str = "public_channel,private_channel") -> List[Dict]:
    """Get list of channels the bot is a member of."""
    channels = []
    try:
        cursor = None
        while True:
            kwargs = {"types": types, "limit": 200, "exclude_archived": True}
            if cursor:
                kwargs["cursor"] = cursor

            resp = client.conversations_list(**kwargs)
            if not resp["ok"]:
                break

            for ch in resp["channels"]:
                if ch.get("is_member", False):
                    channels.append(ch)

            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

    except Exception as exc:
        logger.warning("Failed to list Slack channels: %s", exc)

    return channels


def read_recent_slack_messages(
    hours: int = 24,
    limit: int = 50,
    channel_filter: Optional[List[str]] = None,
) -> List[Message]:
    """Read recent Slack messages.

    Args:
        hours: Look back this many hours.
        limit: Max messages per channel.
        channel_filter: If set, only read from these channel names.
    """
    try:
        from slack_sdk import WebClient
        from slack_sdk.errors import SlackApiError
    except ImportError:
        logger.error("Install slack-sdk: pip install slack-sdk")
        return []

    token = _get_slack_token()
    if not token:
        logger.error("No Slack token available")
        return []

    client = WebClient(token=token)
    user_map = _get_user_map(client)

    channels = _get_channels(client)
    if channel_filter:
        filter_set = {name.lower().lstrip("#") for name in channel_filter}
        channels = [ch for ch in channels if ch["name"].lower() in filter_set]

    if not channels:
        logger.info("No accessible Slack channels found")
        return []

    oldest = str((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp())
    messages = []

    for channel in channels:
        ch_id = channel["id"]
        ch_name = channel.get("name", ch_id)

        try:
            resp = client.conversations_history(
                channel=ch_id,
                oldest=oldest,
                limit=limit,
            )

            if not resp["ok"]:
                continue

            for msg in resp["messages"]:
                subtype = msg.get("subtype", "")
                if subtype in ("channel_join", "channel_leave", "bot_message"):
                    continue

                user_id = msg.get("user", "unknown")
                sender = user_map.get(user_id, user_id)
                text = msg.get("text", "")
                ts_float = float(msg.get("ts", 0))
                ts = datetime.fromtimestamp(ts_float, tz=timezone.utc)
                thread_ts = msg.get("thread_ts")

                for uid, uname in user_map.items():
                    text = text.replace(f"<@{uid}>", f"@{uname}")

                messages.append(Message(
                    source="slack",
                    sender=sender,
                    recipient=f"#{ch_name}",
                    subject=f"#{ch_name}",
                    body=text,
                    timestamp=ts,
                    thread_id=thread_ts or msg.get("ts"),
                    metadata={
                        "channel_id": ch_id,
                        "channel_name": ch_name,
                        "has_thread": bool(thread_ts),
                        "reply_count": msg.get("reply_count", 0),
                    },
                ))

        except SlackApiError as exc:
            logger.warning("Failed to read #%s: %s", ch_name, exc.response["error"])

    messages.sort(key=lambda m: m.timestamp, reverse=True)
    return messages


def read_slack_thread(channel_id: str, thread_ts: str, limit: int = 50) -> List[Message]:
    """Read all messages in a Slack thread."""
    try:
        from slack_sdk import WebClient
        from slack_sdk.errors import SlackApiError
    except ImportError:
        return []

    token = _get_slack_token()
    if not token:
        return []

    client = WebClient(token=token)
    user_map = _get_user_map(client)

    try:
        resp = client.conversations_replies(
            channel=channel_id,
            ts=thread_ts,
            limit=limit,
        )

        if not resp["ok"]:
            return []

        messages = []
        for msg in resp["messages"]:
            user_id = msg.get("user", "unknown")
            sender = user_map.get(user_id, user_id)
            text = msg.get("text", "")
            ts_float = float(msg.get("ts", 0))
            ts = datetime.fromtimestamp(ts_float, tz=timezone.utc)

            for uid, uname in user_map.items():
                text = text.replace(f"<@{uid}>", f"@{uname}")

            messages.append(Message(
                source="slack",
                sender=sender,
                recipient="thread",
                subject="",
                body=text,
                timestamp=ts,
                thread_id=thread_ts,
            ))

        return messages

    except SlackApiError as exc:
        logger.error("Failed to read Slack thread: %s", exc.response["error"])
        return []
