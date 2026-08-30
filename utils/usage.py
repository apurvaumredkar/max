"""
Passive token accounting: what every completion cost, and a plain daily report of it.

Nothing here is a tool and nothing here calls a model — usage is appended by the two fallback
wrappers in utils/inference.py as a side effect of inference that already happened, and the
report is assembled by string formatting, not generated. Deliberately importing nothing from
inference or agent so it stays at the bottom of the dependency order and can run as a standalone
CLI from cron.

    uv run python utils/usage.py                 # yesterday's report to stdout
    uv run python utils/usage.py --date today
    uv run python utils/usage.py --post          # ...and send it to Discord
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import requests

# Run either as a module (`python -m utils.usage`) or as a script (`python utils/usage.py`) —
# the latter needs the repo root on the path before `utils.` resolves. Same shim as utils/web.py.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_config import get_logger

log = get_logger(__name__)

USAGE_PATH = "context/usage.jsonl"
DISCORD_CONFIG_PATH = "context/discord.json"
REPORT_CHANNEL = "usage_report"
DISCORD_MAX_CHARS = 2000


def record(model_key, provider, model, prompt_tokens, completion_tokens):
    """Append one completion's token usage. Never raises — accounting must not break a reply."""
    if not (prompt_tokens or completion_tokens):
        return
    now = datetime.now()
    entry = {
        "timestamp": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "model_key": model_key,
        # Resolved by the caller, which still has the model's entry in hand: a model retired from
        # Ollama or the OpenRouter catalog would otherwise become unlabelable by report time.
        "provider": provider,
        "model": model,
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
    }
    try:
        with open(USAGE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log.warning("Failed to record token usage: %s", e)


def _load(date):
    if not os.path.exists(USAGE_PATH):
        return []
    entries = []
    try:
        with open(USAGE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("date") == date:
                    entries.append(entry)
    except Exception as e:
        log.error("Failed to read %s: %s", USAGE_PATH, e)
    return entries


def build_report(date):
    entries = _load(date)
    heading = datetime.strptime(date, "%Y-%m-%d").strftime("%A, %d %b %Y")
    if not entries:
        return f"**Token usage — {heading}**\nNo model calls."

    # Only providers and models that actually ran appear, so the report carries no empty rows.
    by_provider = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    total_in = total_out = total_calls = 0
    for entry in entries:
        prompt = entry.get("prompt_tokens", 0)
        completion = entry.get("completion_tokens", 0)
        row = by_provider[entry.get("provider") or "unknown"][
            entry.get("model") or "unknown"
        ]
        row[0] += prompt
        row[1] += completion
        row[2] += 1
        total_in += prompt
        total_out += completion
        total_calls += 1

    lines = [
        f"**Token usage — {heading}**",
        f"{total_in:,} in · {total_out:,} out · {total_in + total_out:,} total "
        f"({total_calls} call{'s' if total_calls != 1 else ''})",
    ]
    for provider in sorted(
        by_provider, key=lambda p: -sum(r[0] + r[1] for r in by_provider[p].values())
    ):
        models = by_provider[provider]
        lines.append(f"\n__{provider}__")
        for model in sorted(models, key=lambda m: -(models[m][0] + models[m][1])):
            prompt, completion, calls = models[model]
            lines.append(f"`{model}` — {prompt:,} in · {completion:,} out ({calls})")
    return "\n".join(lines)


def _webhook_url():
    try:
        with open(DISCORD_CONFIG_PATH, "r", encoding="utf-8") as f:
            return (json.load(f).get(REPORT_CHANNEL) or {}).get("webhook_url")
    except Exception as e:
        log.error("Failed to read %s: %s", DISCORD_CONFIG_PATH, e)
        return None


def post(text):
    url = _webhook_url()
    if not url:
        log.error("No %r webhook configured in %s", REPORT_CHANNEL, DISCORD_CONFIG_PATH)
        return False
    if len(text) > DISCORD_MAX_CHARS:
        text = text[: DISCORD_MAX_CHARS - 4] + "\n…"
    try:
        response = requests.post(url, json={"content": text}, timeout=15)
        response.raise_for_status()
    except Exception as e:
        log.error("Failed to post usage report to Discord: %s", e)
        return False
    return True


if __name__ == "__main__":
    # Same reason as utils/web.py: logs default to stdout for the journal, but stdout here is
    # the report itself — send log records to stderr instead.
    import logging

    from utils.logging_config import DATE_FORMAT, LOG_FORMAT

    root = logging.getLogger()
    root.handlers.clear()
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(stderr_handler)

    parser = argparse.ArgumentParser(description="Daily token usage report")
    parser.add_argument(
        "--date",
        default="yesterday",
        help="YYYY-MM-DD, or 'today'/'yesterday' (default: yesterday, since the report fires "
        "at local midnight for the day that just ended)",
    )
    parser.add_argument(
        "--post", action="store_true", help="send the report to the Discord webhook"
    )
    args = parser.parse_args()

    if args.date == "today":
        date = datetime.now().strftime("%Y-%m-%d")
    elif args.date == "yesterday":
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        date = datetime.strptime(args.date, "%Y-%m-%d").strftime("%Y-%m-%d")

    report = build_report(date)
    print(report)
    if args.post and not post(report):
        sys.exit(1)
