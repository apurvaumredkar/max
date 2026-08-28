from ollama import Client
from ollama._utils import convert_function_to_tool
from openai import AsyncOpenAI
import os
import subprocess
import shlex
import requests
from dotenv import load_dotenv
from datetime import datetime
import json
import uuid
from fastapi import APIRouter, Body
from utils.discord_functions import send_discord_message
from utils.logging_config import get_logger

load_dotenv("secrets/.env")

log = get_logger(__name__)

JOB_TRIGGER_URL = "http://localhost/max/job-trigger"

# Selectable inference backends — the UI's model selector picks a key from here per request.
# Each is an OpenAI-compatible endpoint; base_url/api_key come from secrets/.env so credentials
# stay out of source. The Ollama host's own entries are discovered dynamically (see
# discover_ollama_models below) rather than hardcoded, so newly pulled models show up without
# a code change.
STATIC_MODEL_OPTIONS = {
    "openrouter-nemotron": {
        "label": "OpenRouter · Nemotron 3 Ultra",
        "group": "OpenRouter",
        "model_label": "Nemotron 3 Ultra",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "base_url": os.getenv("INFERENCE_HOST_URL"),
        "api_key": os.getenv("INFERENCE_API_KEY"),
        # The :free tier's context window (the paid tier is capped at 512,288).
        "context_length": 1_000_000,
    },
}
DEFAULT_MODEL_KEY = "openrouter-nemotron"

# Default when OpenRouter's /models doesn't report a context length for some entry.
_OPENROUTER_DEFAULT_CONTEXT_LENGTH = 8192


def discover_openrouter_models():
    """
    Query OpenRouter's /models endpoint for the full catalog, so the picker offers every model
    available there instead of just the hardcoded Nemotron entry. The Nemotron model id itself is
    skipped since STATIC_MODEL_OPTIONS already covers it (and must, so a picker default survives
    even if this request fails). Unreachable/errors are logged and skipped — the static entry
    still works.
    """
    base_url = os.getenv("INFERENCE_HOST_URL")
    api_key = os.getenv("INFERENCE_API_KEY")
    if not base_url:
        return {}
    static_model_id = STATIC_MODEL_OPTIONS["openrouter-nemotron"]["model"]
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    options = {}
    try:
        response = requests.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=10)
        response.raise_for_status()
        models = response.json().get("data", [])
    except Exception as e:
        log.error("Failed to list OpenRouter models from %s: %s", base_url, e)
        return options

    for entry in models:
        model_id = entry.get("id")
        if not model_id or model_id == static_model_id:
            continue
        name = entry.get("name") or model_id
        options[f"openrouter:{model_id}"] = {
            "label": f"OpenRouter · {name}",
            "group": "OpenRouter",
            "model_label": name,
            "model": model_id,
            "base_url": base_url,
            "api_key": api_key,
            "context_length": entry.get("context_length") or _OPENROUTER_DEFAULT_CONTEXT_LENGTH,
        }
    return options

# Two Tailscale-reachable Ollama hosts, keyed by provider id. "work" needs no API key —
# it's plain `ollama serve` reached over Tailscale's own network-level security.
TAILSCALE_OLLAMA_PROVIDERS = {
    "home": {
        "label": "Home",
        "host_url": os.getenv("TAILSCALE_OLLAMA_HOME_HOST_URL"),
        "api_key": os.getenv("TAILSCALE_OLLAMA_HOME_API_KEY"),
    },
    "work": {
        "label": "Work (MacBook Pro)",
        "host_url": os.getenv("TAILSCALE_OLLAMA_WORK_HOST_URL"),
        "api_key": os.getenv("TAILSCALE_OLLAMA_WORK_API_KEY"),
    },
}

# Default when Ollama's /api/tags doesn't report a model's context length (some families omit it).
_OLLAMA_DEFAULT_CONTEXT_LENGTH = 8192


def _ollama_model_context_length(host, headers, name, fallback):
    """
    /api/tags's details.context_length is only populated for a handful of custom models
    (ones whose Modelfile sets it explicitly) — most models, including every one on the
    work MacBook (MLX/safetensors format), omit it there entirely. /api/show's model_info
    always has it, keyed as "<architecture>.context_length" (e.g. "gemma4.context_length"),
    so that's the reliable source; /api/tags's field (when present) only saves this call.
    """
    try:
        response = requests.post(f"{host}/api/show", headers=headers, json={"model": name}, timeout=5)
        response.raise_for_status()
        model_info = response.json().get("model_info", {})
        for key, value in model_info.items():
            if key.endswith(".context_length"):
                return value
    except Exception as e:
        log.error("Failed to fetch context length for %s from %s: %s", name, host, e)
    return fallback


def discover_ollama_models():
    """
    Query each Tailscale Ollama host's native API (not the OpenAI-compatible /v1 route it's
    otherwise used through) for every model currently pulled there, so the model picker always
    reflects what's actually available instead of a hardcoded model name. A provider that's
    unreachable is logged and skipped — the static options and the other provider still work.
    """
    options = {}
    for provider_id, provider in TAILSCALE_OLLAMA_PROVIDERS.items():
        host = (provider["host_url"] or "").removesuffix("/v1").rstrip("/")
        if not host:
            continue
        headers = {"Authorization": f"Bearer {provider['api_key']}"} if provider["api_key"] else {}
        try:
            response = requests.get(f"{host}/api/tags", headers=headers, timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
        except Exception as e:
            log.error("Failed to list Ollama models from %s (%s): %s", host, provider_id, e)
            continue

        for entry in models:
            name = entry.get("model") or entry.get("name")
            if not name:
                continue
            fallback = entry.get("details", {}).get("context_length") or _OLLAMA_DEFAULT_CONTEXT_LENGTH
            options[f"tailscale-ollama-{provider_id}:{name}"] = {
                "label": f"Ollama · {provider['label']} · {name}",
                "group": provider["label"],
                "model_label": name,
                "model": name,
                "base_url": provider["host_url"],
                "api_key": provider["api_key"],
                "context_length": _ollama_model_context_length(host, headers, name, fallback),
            }
    return options


MODEL_OPTIONS = {**STATIC_MODEL_OPTIONS, **discover_ollama_models(), **discover_openrouter_models()}


def refresh_model_options():
    """Re-query the Ollama hosts and OpenRouter so newly available models show up without a restart."""
    global MODEL_OPTIONS
    MODEL_OPTIONS = {**STATIC_MODEL_OPTIONS, **discover_ollama_models(), **discover_openrouter_models()}
    return MODEL_OPTIONS


def _model_choice(model_key):
    """Resolve a UI model key to a client + model id + context window, falling back to the default."""
    key = model_key if model_key in MODEL_OPTIONS else DEFAULT_MODEL_KEY
    opts = MODEL_OPTIONS[key]
    # AsyncOpenAI's own validation rejects a falsy api_key outright (it doesn't just get sent
    # and ignored) — a provider like the work MacBook's plain `ollama serve`, with no auth at
    # all, still needs some non-empty placeholder to satisfy the client constructor.
    client = AsyncOpenAI(base_url=opts["base_url"], api_key=opts["api_key"] or "ollama")
    return client, opts["model"], opts["context_length"]


CONFIG_PATH = "context/config.json"


def _load_config():
    """Small persisted runtime config (currently just the last model used). {} if missing/invalid."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def get_default_model_key():
    """
    The model key to use when a request doesn't name one: whatever last successfully generated
    a reply (persisted in context/config.json), so the picker — and scheduled jobs, which have
    no UI to pick from — pick up where the user left off instead of always landing back on
    DEFAULT_MODEL_KEY.
    """
    last_key = _load_config().get("last_model_key")
    return last_key if last_key in MODEL_OPTIONS else DEFAULT_MODEL_KEY


def _remember_model_key(model_key):
    """Persist the model that just successfully generated a reply as the new default."""
    config = _load_config()
    if config.get("last_model_key") == model_key:
        return
    config["last_model_key"] = model_key
    _save_config(config)


def _model_fallback_candidates(preferred_key):
    """
    Ordered model keys to try: the requested one first, then every other configured model —
    so a dead client (bad credentials, host unreachable, rate limited, model removed from
    Ollama) doesn't fail the whole turn when another configured model could serve it.
    """
    ordered = [preferred_key] if preferred_key in MODEL_OPTIONS else []
    ordered += [key for key in MODEL_OPTIONS if key not in ordered]
    return ordered


async def _complete_with_fallback(model_key, messages):
    """
    Call the chat-completions endpoint for model_key; on failure, retry with the next
    configured model instead of failing the turn. Returns (response, resolved_model_key) —
    resolved_model_key may differ from model_key if a fallback was needed, and becomes the
    new remembered default on success.
    """
    candidates = _model_fallback_candidates(model_key)
    last_error = None
    for i, candidate_key in enumerate(candidates):
        client, model_name, context_length = _model_choice(candidate_key)
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=TOOL_SCHEMAS,
                extra_body={"options": {"num_ctx": context_length}, "keep_alive": -1},
            )
        except Exception as e:
            last_error = e
            log.error(
                "Model %s failed (%s: %s)%s",
                candidate_key,
                type(e).__name__,
                e,
                "" if i + 1 == len(candidates) else " — falling back to next model",
            )
            continue
        if candidate_key != model_key:
            log.warning("Fell back from model %s to %s", model_key, candidate_key)
        _remember_model_key(candidate_key)
        return response, candidate_key
    raise last_error


async def _stream_with_fallback(model_key, messages):
    """
    Streaming counterpart to _complete_with_fallback. Only falls back if the failure happens
    before any chunk of this round has been produced — a stream that dies partway through
    can't be safely resumed on a different model without duplicating or losing output already
    sent to the client. Yields (resolved_model_key, chunk) pairs; resolved_model_key is stable
    across a round's yields and becomes the new remembered default on success.
    """
    candidates = _model_fallback_candidates(model_key)
    last_error = None
    for i, candidate_key in enumerate(candidates):
        client, model_name, context_length = _model_choice(candidate_key)
        try:
            stream = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=TOOL_SCHEMAS,
                stream=True,
                extra_body={"options": {"num_ctx": context_length}, "keep_alive": -1},
            )
            aiter = stream.__aiter__()
            try:
                first_chunk = await aiter.__anext__()
            except StopAsyncIteration:
                if candidate_key != model_key:
                    log.warning("Fell back from model %s to %s", model_key, candidate_key)
                _remember_model_key(candidate_key)
                return
        except Exception as e:
            last_error = e
            log.error(
                "Model %s failed to start a response (%s: %s)%s",
                candidate_key,
                type(e).__name__,
                e,
                "" if i + 1 == len(candidates) else " — falling back to next model",
            )
            continue
        if candidate_key != model_key:
            log.warning("Fell back from model %s to %s", model_key, candidate_key)
        _remember_model_key(candidate_key)
        yield candidate_key, first_chunk
        async for chunk in aiter:
            yield candidate_key, chunk
        return
    raise last_error


embed_client = Client(host="http://localhost:11434")
EMBED_MODEL = "all-minilm"

router = APIRouter()


def _load_persona():
    try:
        with open("context/SYSTEM.md", "r", encoding="utf-8") as system_file:
            return system_file.read()
    except Exception as e:
        log.error("Failed to load persona from context/SYSTEM.md: %s", e)


def _load_history():
    d = datetime.today().strftime("%Y%m%d")
    filename = f"chats/{d}.jsonl"
    if not os.path.exists(filename):
        try:
            flags = os.O_CREAT | os.O_RDWR
            fd = os.open(filename, flags)
            os.close(fd)
        except Exception as e:
            log.error("Failed to create chat log %s: %s", filename, e)
    else:
        try:
            with open(filename, "r") as chats:
                turns = [json.loads(i) for i in chats.readlines()]
                return [turn for turn in turns if turn.get("source") != "job"]
        except Exception as e:
            log.error("Failed to read today's history %s: %s", filename, e)
    return []


def _load_all_history(exclude_files=()):
    if not os.path.isdir("chats"):
        return []
    turns = []
    for filename in sorted(os.listdir("chats")):
        if not filename.endswith(".jsonl") or filename in exclude_files:
            continue
        try:
            with open(f"chats/{filename}", "r") as chats:
                turns.extend(json.loads(line) for line in chats.readlines())
        except Exception as e:
            log.error("Failed to read history file chats/%s: %s", filename, e)
    return turns


def _load_past_history():
    today_file = f"{datetime.today().strftime('%Y%m%d')}.jsonl"
    return _load_all_history(exclude_files={today_file})


def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _search_history(query: str, top_k: int = 5):
    """
    Semantically search prior days' conversation history (not today's) for turns relevant to a
    query. Use this when the user references something from an earlier conversation that isn't
    in your current context.

    Args:
        query: What to search for, described in natural language.
        top_k: Maximum number of matching turns to return.
    """
    past_turns = _load_past_history()
    if not past_turns:
        return {"results": []}
    contents = [turn["content"] for turn in past_turns]
    query_embedding = embed_client.embed(model=EMBED_MODEL, input=query).embeddings[0]
    turn_embeddings = embed_client.embed(model=EMBED_MODEL, input=contents).embeddings
    scored = sorted(
        zip(past_turns, turn_embeddings),
        key=lambda pair: _cosine_similarity(query_embedding, pair[1]),
        reverse=True,
    )
    return {
        "results": [
            {
                "role": turn["role"],
                "content": turn["content"],
                "timestamp": turn.get("timestamp"),
            }
            for turn, _ in scored[:top_k]
        ]
    }


def _save_turn(turn: dict):
    d = datetime.today().strftime("%Y%m%d")
    filename = f"chats/{d}.jsonl"
    # Stable per-turn id so the UI can delete a specific turn. Line numbers won't do —
    # they shift as soon as anything above is removed.
    turn = {
        "id": turn.get("id") or uuid.uuid4().hex,
        **turn,
        "timestamp": datetime.now().strftime("%B %d, %Y %I:%M %p"),
    }
    try:
        with open(filename, "a") as chats:
            chats.write(json.dumps(turn) + "\n")
    except Exception as e:
        log.error("Failed to append turn to %s: %s", filename, e)
    return turn


def _execute_bash(command: str):
    """
    Execute a bash shell command and return its output.

    Args:
        command: The bash command to execute.
    """
    log.info("Executing bash: %s", command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning(
            "Bash exited %s: %s", result.returncode, (result.stderr or "").strip()[:300]
        )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


_READ_DEFAULT_LIMIT = 300  # lines returned by default
_READ_MAX_CHARS = 8000  # hard backstop regardless of line count (e.g. one huge minified line)


def _read_file(path: str, offset: int = 1, limit: int = _READ_DEFAULT_LIMIT):
    """
    Read a slice of a text file's contents — the token-efficient alternative to `cat` via
    _execute_bash, which dumps the entire file regardless of size. Returns up to `limit` lines
    starting at `offset`, plus the file's total line count, so you know whether to page further
    with a later offset.

    Args:
        path: Path to the file to read (absolute, or relative to the working directory).
        offset: 1-indexed line number to start reading from. Defaults to the start of the file.
        limit: Maximum number of lines to return. Defaults to 300 — raise it only for a file you
            know is short, otherwise page through a long one with successive offsets.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        return {"error": f"{type(e).__name__}: {e}"}

    total_lines = len(lines)
    start = max(offset, 1)
    # start past the end of the file means nothing to return — end_line should reflect that
    # (an empty range) rather than being clamped back to total_lines, which would put it before
    # start_line and misreport what was actually read.
    end = start - 1 if start > total_lines else min(start + max(limit, 1) - 1, total_lines)
    content = "".join(lines[start - 1 : end])

    char_truncated = len(content) > _READ_MAX_CHARS
    if char_truncated:
        content = content[:_READ_MAX_CHARS]

    return {
        "path": path,
        "total_lines": total_lines,
        "start_line": start,
        "end_line": end,
        "content": content,
        "truncated": end < total_lines or char_truncated,
    }


def _load_jobs():
    filename = "context/jobs.json"
    if not os.path.exists(filename):
        return {"scheduled": [], "cron": []}
    try:
        with open(filename, "r") as jobs_file:
            return json.load(jobs_file)
    except Exception as e:
        log.error("Failed to load jobs file %s: %s", filename, e)
        return {"scheduled": [], "cron": []}


def _save_jobs(jobs: dict):
    filename = "context/jobs.json"
    try:
        with open(filename, "w") as jobs_file:
            json.dump(jobs, jobs_file, indent=2)
    except Exception as e:
        log.error("Failed to write jobs file %s: %s", filename, e)


MANAGED_BEGIN = "# BEGIN max-agent jobs (managed — edited via the web UI)"
MANAGED_END = "# END max-agent jobs"


def _sync_crontab():
    """
    Regenerate the managed block of the crontab from jobs.json.

    jobs.json is the source of truth; the crontab is derived. Appending lines per job (the
    old approach) let the two drift — deleting a job left its cron line behind forever.
    Anything outside the managed markers is another tool's and is preserved verbatim.
    """
    existing = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    lines = existing.splitlines()

    preserved, inside, adopted = [], False, 0
    for line in lines:
        if line.strip() == MANAGED_BEGIN:
            inside = True
            continue
        if line.strip() == MANAGED_END:
            inside = False
            continue
        if inside:
            continue
        # Legacy lines from before the managed block existed: these were appended per job
        # and never removed, so they duplicate what jobs.json now generates. Drop them —
        # jobs.json is the source of truth. Unrelated crontab entries are preserved.
        if JOB_TRIGGER_URL in line:
            adopted += 1
            continue
        preserved.append(line)
    if adopted:
        log.info("Dropped %d legacy unmanaged job-trigger line(s)", adopted)

    jobs = _load_jobs()
    managed = [MANAGED_BEGIN]
    for job in jobs.get("cron", []):
        managed.append(
            f"{job['schedule']} "
            f"{_job_trigger_command(job['prompt'], job.get('channel', 'reminders'))}"
        )
    for job in jobs.get("scheduled", []):
        schedule = _run_at_to_cron(job["run_at"])
        if not schedule:
            continue
        managed.append(
            f"{schedule} "
            f"{_job_trigger_command(job['prompt'], job.get('channel', 'reminders'))}"
        )
    managed.append(MANAGED_END)

    while preserved and not preserved[-1].strip():
        preserved.pop()
    updated = "\n".join(preserved + managed) + "\n"

    result = subprocess.run(
        ["crontab", "-"], input=updated, capture_output=True, text=True
    )
    if result.returncode != 0:
        log.error("Failed to sync crontab: %s", result.stderr.strip())
        return False
    log.info(
        "Crontab synced: %d cron + %d scheduled job(s)",
        len(jobs.get("cron", [])),
        len(jobs.get("scheduled", [])),
    )
    return True


def _run_at_to_cron(run_at: str):
    """Convert "YYYY-MM-DD HH:MM" to a 5-field cron expression, or None if unparseable."""
    try:
        dt = datetime.strptime(run_at, "%Y-%m-%d %H:%M")
    except ValueError:
        log.error("Unparseable run_at %r", run_at)
        return None
    return f"{dt.minute} {dt.hour} {dt.day} {dt.month} *"


SYSTEM_TRIGGER_PREFIX = "[THIS IS AN AUTOMATED SYSTEM TRIGGER]"


def _job_trigger_command(prompt: str, channel: str):
    # The prompt arrives as a user turn when the job fires, so mark it: it is the scheduler
    # nudging Max, not Apurva speaking.
    payload = shlex.quote(
        json.dumps(
            {"prompt": f"{SYSTEM_TRIGGER_PREFIX} {prompt}", "channel": channel}
        )
    )
    return (
        f"curl -s -X POST {JOB_TRIGGER_URL} "
        f"-H 'Content-Type: application/json' -d {payload}"
    )


def _cron(name: str, schedule: str, prompt: str, channel: str = "reminders"):
    """
    Schedule a recurring reminder. At each firing, the prompt is fed back to you and you generate
    a fresh message in the moment, posted to the chat UI and to a Discord channel.

    Args:
        name: A short, human-readable name for the job.
        schedule: A standard 5-field cron expression, e.g. "0 9 * * *" for daily at 9am.
        prompt: An instruction to your future self describing what to say and why, not a
            literal message — e.g. "Remind Apurva to stretch; they mentioned a stiff back."
        channel: The Discord channel key, as configured in context/discord.json. Defaults to
            "reminders" — only use another value if the user names a different configured channel.
    """
    jobs = _load_jobs()
    jobs["cron"].append(
        {
            "id": uuid.uuid4().hex,
            "name": name,
            "schedule": schedule,
            "prompt": prompt,
            "channel": channel,
        }
    )
    _save_jobs(jobs)
    _sync_crontab()
    log.info("Cron job created: %r schedule=%r channel=%s", name, schedule, channel)
    return {"status": "created", "name": name, "schedule": schedule, "prompt": prompt}


def _schedule(name: str, run_at: str, prompt: str, channel: str = "reminders"):
    """
    Schedule a one-time reminder. When it fires, the prompt is fed back to you and you generate a
    fresh message in the moment, posted to the chat UI and to a Discord channel.

    Args:
        name: A short, human-readable name for the job.
        run_at: The date and time to run the job, in "YYYY-MM-DD HH:MM" 24-hour format.
        prompt: An instruction to your future self describing what to say and why, not a
            literal message — e.g. "Remind Apurva their dentist appointment is in an hour."
        channel: The Discord channel key, as configured in context/discord.json. Defaults to
            "reminders" — only use another value if the user names a different configured channel.
    """
    datetime.strptime(run_at, "%Y-%m-%d %H:%M")  # validate early
    jobs = _load_jobs()
    jobs["scheduled"].append(
        {
            "id": uuid.uuid4().hex,
            "name": name,
            "run_at": run_at,
            "prompt": prompt,
            "channel": channel,
        }
    )
    _save_jobs(jobs)
    _sync_crontab()
    log.info("Scheduled job created: %r run_at=%s channel=%s", name, run_at, channel)
    return {"status": "created", "name": name, "run_at": run_at, "prompt": prompt}


def _jobs_system_message():
    """
    Render the currently installed jobs as a system message. Active context is only today's chat,
    so without this Max cannot see reminders it set on an earlier day and creates duplicates.
    """
    jobs = _load_jobs()
    lines = []
    for job in jobs.get("cron", []):
        lines.append(
            f"- [recurring] {job['name']} — cron \"{job['schedule']}\" "
            f"→ #{job.get('channel', 'reminders')}: {job['prompt']}"
        )
    for job in jobs.get("scheduled", []):
        lines.append(
            f"- [one-time] {job['name']} — at {job['run_at']} "
            f"→ #{job.get('channel', 'reminders')}: {job['prompt']}"
        )
    if not lines:
        body = "There are currently no reminders or scheduled jobs set."
    else:
        body = "Reminders and scheduled jobs you have already set:\n" + "\n".join(lines)
    return {
        "role": "system",
        "content": (
            f"{body}\n\n"
            "This list is the full set of jobs across all days — your chat context only covers "
            "today. Check it before creating a new job: if an equivalent one already exists, say "
            "so instead of creating a duplicate."
        ),
    }


SKILLS_DIR = "context/skills"


def _parse_skill_frontmatter(content):
    """Extract `name`/`description` from a skill file's leading YAML frontmatter, if present."""
    if not content.startswith("---"):
        return None, None
    end = content.find("\n---", 3)
    if end == -1:
        return None, None
    name = description = None
    for line in content[3:end].splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip().strip('"').strip("'")
        if key == "name":
            name = value
        elif key == "description":
            description = value
    return name, description


def _load_skills():
    """List available playbooks (context/skills/*.md) with their name/description parsed
    from frontmatter."""
    skills = []
    if not os.path.isdir(SKILLS_DIR):
        return skills
    for filename in sorted(os.listdir(SKILLS_DIR)):
        if not filename.endswith(".md"):
            continue
        path = os.path.join(SKILLS_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            log.error("Failed to read skill file %s: %s", path, e)
            continue
        name, description = _parse_skill_frontmatter(content)
        skills.append(
            {"filename": filename, "name": name or filename[:-3], "description": description or ""}
        )
    return skills


def _skills_system_message():
    """
    Render available playbooks as a system message — just their name/description, not their
    full contents. Each entry is cheap (a couple lines); the full file is only worth reading
    once a description actually matches what's being asked, via `_execute_bash`.
    """
    skills = _load_skills()
    if not skills:
        body = "There are currently no playbooks in context/skills/."
    else:
        lines = [f"- `{s['filename']}` — {s['name']}: {s['description']}" for s in skills]
        body = "Available playbooks:\n" + "\n".join(lines)
    return {
        "role": "system",
        "content": (
            f"{body}\n\n"
            "A playbook is a markdown file with step-by-step instructions for handling a "
            "specific kind of task. If one's description matches what the user is asking for, "
            "read it in full first with `_execute_bash` (e.g. `cat context/skills/<filename>`) "
            "before acting on it — don't guess its contents from the name/description alone."
        ),
    }


def sync_crontab_on_startup():
    """
    Bring the crontab in line with jobs.json at boot.

    Historically jobs were appended to the crontab and never removed, so the two drifted.
    Syncing here means a deleted job's cron line can't outlive it.
    """
    return _sync_crontab()


AVAILABLE_TOOLS = {
    "_execute_bash": _execute_bash,
    "_read_file": _read_file,
    "_cron": _cron,
    "_schedule": _schedule,
    "_search_history": _search_history,
}

# Built from the same docstrings above, in OpenAI's tool-schema format.
TOOL_SCHEMAS = [
    convert_function_to_tool(func).model_dump(exclude_none=True)
    for func in AVAILABLE_TOOLS.values()
]


async def _generate_reply(user_input: str, save_user_turn: bool = True, model_key: str = None):
    model_key = model_key if model_key in MODEL_OPTIONS else get_default_model_key()
    log.info("Generating reply (non-streaming, save_user_turn=%s, model=%s)", save_user_turn, model_key)
    persona = _load_persona()
    if save_user_turn:
        _save_turn({"role": "user", "content": user_input})
    history = _load_history()
    context = [{"role": turn["role"], "content": turn["content"]} for turn in history]
    messages = [
        {"role": "system", "content": persona},
        _jobs_system_message(),
        _skills_system_message(),
        *context,
    ]
    if not save_user_turn:
        messages.append({"role": "user", "content": user_input})
    tool_calls_log = []

    while True:
        response, model_key = await _complete_with_fallback(model_key, messages)
        message = response.choices[0].message
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                **(
                    {
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ]
                    }
                    if message.tool_calls
                    else {}
                ),
            }
        )
        if not message.tool_calls:
            break
        for tool_call in message.tool_calls:
            function_to_call = AVAILABLE_TOOLS[tool_call.function.name]
            arguments = json.loads(tool_call.function.arguments)
            log.info("Tool call: %s(%s)", tool_call.function.name, arguments)
            try:
                output = function_to_call(**arguments)
            except Exception as e:
                log.exception("Tool %s failed", tool_call.function.name)
                output = {"error": f"{type(e).__name__}: {e}"}
            tool_calls_log.append(
                {
                    "name": tool_call.function.name,
                    "arguments": arguments,
                    "output": output,
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(output),
                }
            )

    return _save_turn(
        {
            "role": "assistant",
            "content": message.content,
            "thinking": getattr(message, "reasoning", None),
            "tool_calls": tool_calls_log,
            "source": "job",
        }
    )


async def _generate_reply_stream(user_input: str, model_key: str = None):
    """Yield event dicts; the caller serializes them for the wire."""
    model_key = model_key if model_key in MODEL_OPTIONS else get_default_model_key()
    log.info("Chat request: %d chars (model=%s)", len(user_input), model_key)
    persona = _load_persona()
    _save_turn({"role": "user", "content": user_input})
    history = _load_history()
    context = [{"role": turn["role"], "content": turn["content"]} for turn in history]
    messages = [
        {"role": "system", "content": persona},
        _jobs_system_message(),
        _skills_system_message(),
        *context,
    ]
    tool_calls_log = []
    content = ""
    thinking = ""

    while True:
        tool_call_chunks = {}
        round_content = ""
        round_thinking = ""
        async for resolved_key, chunk in _stream_with_fallback(model_key, messages):
            model_key = resolved_key
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning", None)
            if reasoning:
                round_thinking += reasoning
                yield {"type": "thinking", "delta": reasoning}
            if delta.content:
                round_content += delta.content
                yield {"type": "content", "delta": delta.content}
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    entry = tool_call_chunks.setdefault(
                        tc_delta.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if tc_delta.id:
                        entry["id"] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        entry["name"] = tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        entry["arguments"] += tc_delta.function.arguments

        content += round_content
        thinking += round_thinking
        tool_calls = (
            [tool_call_chunks[i] for i in sorted(tool_call_chunks.keys())]
            if tool_call_chunks
            else None
        )
        messages.append(
            {
                "role": "assistant",
                "content": round_content,
                **(
                    {
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"],
                                },
                            }
                            for tc in tool_calls
                        ]
                    }
                    if tool_calls
                    else {}
                ),
            }
        )
        if not tool_calls:
            break
        for tc in tool_calls:
            function_to_call = AVAILABLE_TOOLS[tc["name"]]
            arguments = json.loads(tc["arguments"])
            log.info("Tool call: %s(%s)", tc["name"], arguments)
            try:
                output = function_to_call(**arguments)
            except Exception as e:
                log.exception("Tool %s failed", tc["name"])
                output = {"error": f"{type(e).__name__}: {e}"}
            tool_call_record = {
                "name": tc["name"],
                "arguments": arguments,
                "output": output,
            }
            tool_calls_log.append(tool_call_record)
            yield {"type": "tool_call", **tool_call_record}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(output),
                }
            )

    turn = _save_turn(
        {
            "role": "assistant",
            "content": content,
            "thinking": thinking,
            "tool_calls": tool_calls_log,
        }
    )
    yield {"type": "done", "turn": turn}


@router.post("/job-trigger")
async def job_trigger(prompt: str = Body(embed=True), channel: str = Body(embed=True)):
    log.info("Job fired → channel=%s prompt=%r", channel, prompt[:120])
    turn = await _generate_reply(prompt, save_user_turn=False)
    result = send_discord_message(channel, turn["content"])
    if result.get("ok"):
        log.info("Job message posted to Discord #%s", channel)
    else:
        log.error(
            "Discord post to #%s failed (status=%s)", channel, result.get("status_code")
        )
    return turn
