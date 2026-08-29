"""
Discord: the outbound webhook poster used by scheduled jobs, plus the gateway bot that lets
Max be talked to from a phone. The bot listens to every (non-bot) message it can see and
replies with the same agent loop the web UI drives — thoughts and tool calls as `-#` subtext,
then the reply itself.
"""

import asyncio
import json
import os

import discord
import requests
from dotenv import load_dotenv

from utils.logging_config import get_logger

load_dotenv("secrets/.env")

log = get_logger(__name__)

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
# Discord rejects any message body over 2000 characters.
DISCORD_MAX_CHARS = 2000
# Tool arguments can be a whole bash script — enough to identify the call is plenty.
TOOL_ARGS_PREVIEW_CHARS = 300


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


def _chunks(text, limit=DISCORD_MAX_CHARS):
    """Split text into message-sized pieces, preferring line then word boundaries."""
    text = text.strip()
    while text:
        if len(text) <= limit:
            yield text
            return
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        yield text[:cut].rstrip()
        text = text[cut:].lstrip()


def _as_subtext(chunk):
    """Discord's `-#` subtext is per-line, so every line needs the marker."""
    return "\n".join(
        line if line.startswith("-# ") else f"-# {line}" for line in chunk.splitlines()
    )


async def _send(channel, text, subtext=False):
    if not subtext:
        for chunk in _chunks(text):
            await channel.send(chunk)
        return
    # Mark up front so the markers count toward the limit, then re-mark the one line a chunk
    # can split mid-way — hence the 3 characters held back for it.
    marked = "\n".join(f"-# {line}" for line in text.strip().splitlines() if line.strip())
    for chunk in _chunks(marked, DISCORD_MAX_CHARS - len("-# ")):
        await channel.send(_as_subtext(chunk))


async def _respond(message, generation):
    """Relay one generation's events into the channel as they arrive."""
    from utils.webui import iter_events

    thinking = ""
    content = ""

    async def flush_thinking():
        nonlocal thinking
        if thinking.strip():
            await _send(message.channel, thinking, subtext=True)
        thinking = ""

    async for event in iter_events(generation):
        if event["type"] == "thinking":
            thinking += event["delta"]
        elif event["type"] == "content":
            content += event["delta"]
        elif event["type"] == "tool_call":
            await flush_thinking()
            arguments = json.dumps(event.get("arguments") or {})[:TOOL_ARGS_PREVIEW_CHARS]
            await _send(message.channel, f"🔧 {event['name']}({arguments})", subtext=True)
        elif event["type"] == "error":
            await _send(message.channel, f"⚠️ {event['message']}", subtext=True)

    await flush_thinking()
    await _send(message.channel, content or "_(no reply)_")


class _MaxClient(discord.Client):
    async def on_ready(self):
        log.info("Discord bot connected as %s", self.user)

    async def on_message(self, message):
        # Ignore ourselves, other bots, and the job webhook's own reminder posts.
        if message.author.bot or message.webhook_id:
            return
        text = (message.content or "").strip()
        if not text:
            # Empty content on a real message means the privileged Message Content intent
            # is off in the Discord developer portal.
            log.warning("Discord message from %s had no readable content", message.author)
            return
        log.info("Discord message from %s: %d chars", message.author, len(text))

        from utils.webui import generation_in_flight, start_generation

        if generation_in_flight():
            await _send(message.channel, "Max is mid-reply — send that again in a moment.", subtext=True)
            return
        generation = start_generation(f"[USER MESSAGE SOURCE: DISCORD]\n{text}")
        try:
            async with message.channel.typing():
                await _respond(message, generation)
        except Exception:
            log.exception("Discord reply failed")


def start_bot():
    """Launch the gateway alongside the API; returns the task, or None if unconfigured."""
    if not DISCORD_BOT_TOKEN:
        log.warning("DISCORD_BOT_TOKEN not set — Discord bot not started")
        return None
    intents = discord.Intents.default()
    intents.message_content = True
    client = _MaxClient(intents=intents)
    return asyncio.create_task(client.start(DISCORD_BOT_TOKEN))
