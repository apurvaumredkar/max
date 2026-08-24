import json
import requests


def _load_discord_config():
    with open("context/discord.json", "r") as discord_file:
        return json.load(discord_file)


def send_discord_message(channel: str, message: str):
    config = _load_discord_config()
    webhook_url = config[channel]["webhook_url"]
    response = requests.post(webhook_url, json={"content": message})
    return {"status_code": response.status_code, "ok": response.ok}
