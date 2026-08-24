import json

import requests

from utils.logging_config import get_logger

log = get_logger(__name__)


def _load_discord_config():
    with open("context/discord.json", "r") as discord_file:
        return json.load(discord_file)


def send_discord_message(channel: str, message: str):
    try:
        config = _load_discord_config()
    except Exception as e:
        log.error("Failed to load context/discord.json: %s", e)
        return {"status_code": None, "ok": False}
    if channel not in config:
        log.error(
            "Unknown Discord channel %r (configured: %s)",
            channel,
            ", ".join(sorted(config)) or "none",
        )
        return {"status_code": None, "ok": False}
    webhook_url = config[channel]["webhook_url"]
    try:
        response = requests.post(webhook_url, json={"content": message}, timeout=10)
    except Exception as e:
        log.error("Discord webhook request to #%s failed: %s", channel, e)
        return {"status_code": None, "ok": False}
    if not response.ok:
        log.error(
            "Discord webhook #%s returned %s: %s",
            channel,
            response.status_code,
            response.text[:200],
        )
    else:
        log.info("Posted %d chars to Discord #%s", len(message), channel)
    return {"status_code": response.status_code, "ok": response.ok}
