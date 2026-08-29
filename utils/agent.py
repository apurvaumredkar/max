from ollama._utils import convert_function_to_tool
import os
import subprocess
import shlex
from datetime import datetime
import json
import uuid
from fastapi import APIRouter, Body
from utils import inference
from utils.discord_functions import send_discord_message
# MODEL_OPTIONS is deliberately *not* imported by name — refresh_model_options() rebinds it, and
# a from-import here would freeze the startup snapshot. Reach it as inference.MODEL_OPTIONS.
from utils.inference import (
    EMBED_MODEL,
    _complete_with_fallback,
    _stream_with_fallback,
    embed_client,
    get_default_model_key,
)
from utils.logging_config import get_logger

log = get_logger(__name__)

JOB_TRIGGER_URL = "http://localhost/max/job-trigger"


router = APIRouter()


def _load_persona():
    try:
        with open("context/SYSTEM.md", "r", encoding="utf-8") as system_file:
            return system_file.read()
    except Exception as e:
        log.error("Failed to load persona from context/SYSTEM.md: %s", e)


def _load_history():
    # Blank lines are skipped rather than parsed: a single trailing newline (a hand-edited
    # file, a partial write) used to raise JSONDecodeError here and silently return [] —
    # losing the whole day's context with nothing but a log line to show for it.
    filename = f"chats/{datetime.today().strftime('%Y%m%d')}.jsonl"
    try:
        with open(filename, "r") as chats:
            turns = [json.loads(line) for line in chats if line.strip()]
    except FileNotFoundError:
        return []
    except Exception as e:
        log.error("Failed to read today's history %s: %s", filename, e)
        return []
    return [turn for turn in turns if turn.get("source") != "job"]


def _load_all_history(exclude_files=()):
    if not os.path.isdir("chats"):
        return []
    turns = []
    for filename in sorted(os.listdir("chats")):
        if not filename.endswith(".jsonl") or filename in exclude_files:
            continue
        try:
            with open(f"chats/{filename}", "r") as chats:
                turns.extend(json.loads(line) for line in chats if line.strip())
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


def _prune_expired_scheduled(jobs: dict):
    """
    Drop one-time jobs whose run_at has passed, returning the ones removed.

    Not just tidiness: `_run_at_to_cron` emits no year field, so a past one-time job keeps a live
    crontab line that fires again on the same date every year. It also stays in the {{JOBS}}
    listing, where Max reads it as a reminder still pending.

    Expiry is by *minute*, not instant: a job is pruned only once the clock has moved to a later
    minute than its run_at, so a sync landing in the same minute the job is due can never remove
    its cron line before it fires. Unparseable run_at values are left alone — `_run_at_to_cron`
    already logs those, and dropping a job we failed to read would be silent data loss.
    """
    current_minute = datetime.now().replace(second=0, microsecond=0)
    kept, removed = [], []
    for job in jobs.get("scheduled", []):
        try:
            due = datetime.strptime(job["run_at"], "%Y-%m-%d %H:%M")
        except (ValueError, KeyError):
            kept.append(job)
            continue
        (removed if due < current_minute else kept).append(job)
    jobs["scheduled"] = kept
    return removed


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
    expired = _prune_expired_scheduled(jobs)
    if expired:
        _save_jobs(jobs)
        log.info(
            "Pruned %d expired scheduled job(s): %s",
            len(expired),
            ", ".join(f"{j.get('name', '?')} @ {j.get('run_at', '?')}" for j in expired),
        )

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


def _jobs_listing():
    """The bare job list — the value of the {{JOBS}} persona variable."""
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
        return "There are currently no reminders or scheduled jobs set."
    return "Reminders and scheduled jobs you have already set:\n" + "\n".join(lines)


def _jobs_system_message():
    """
    Fallback wrapper for {{JOBS}} when SYSTEM.md doesn't place it. Active context is only today's
    chat, so without this Max can't see reminders it set on an earlier day and creates duplicates.
    """
    return {
        "role": "system",
        "content": (
            f"{_jobs_listing()}\n\n"
            "This list is the full set of jobs across all days — your chat context only covers "
            "today. Check it before creating a new job: if an equivalent one already exists, say "
            "so instead of creating a duplicate."
        ),
    }


CONTEXT_DIR = "context"
SKILLS_DIR = "context/skills"


def _parse_frontmatter(content):
    """Extract `name`/`description` from a markdown file's leading YAML frontmatter, if present."""
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
    """
    List available playbooks (context/skills/**.md) with their name/description parsed from
    frontmatter. Walks the tree rather than listing one level: the web UI's Playbooks panel
    walks it (webui._list_md_files), so a playbook saved into a subdirectory there would
    otherwise be editable in the UI and invisible to the model. `filename` is the path
    relative to SKILLS_DIR, which is how the UI addresses these files too.
    """
    skills = []
    if not os.path.isdir(SKILLS_DIR):
        return skills
    for root, _dirs, filenames in os.walk(SKILLS_DIR):
        for filename in sorted(filenames):
            if not filename.endswith(".md"):
                continue
            path = os.path.join(root, filename)
            rel_path = os.path.relpath(path, SKILLS_DIR).replace(os.sep, "/")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                log.error("Failed to read skill file %s: %s", path, e)
                continue
            name, description = _parse_frontmatter(content)
            skills.append(
                {
                    "filename": rel_path,
                    "name": name or rel_path[:-3],
                    "description": description or "",
                }
            )
    return sorted(skills, key=lambda s: s["filename"])


def _load_context_files():
    """
    List the context files Max can read (context/**.md) with their name/description parsed from
    frontmatter. Discovered by walking the tree, so a file added through the UI is visible on the
    next turn without a code change. Skips context/skills/ (listed separately as playbooks) and
    SYSTEM.md (this persona, already in context).
    """
    files = []
    if not os.path.isdir(CONTEXT_DIR):
        return files
    for root, dirs, filenames in os.walk(CONTEXT_DIR):
        rel_root = os.path.relpath(root, CONTEXT_DIR)
        if rel_root == ".":
            dirs[:] = [d for d in dirs if d != "skills"]
        for filename in sorted(filenames):
            if not filename.endswith(".md") or (rel_root == "." and filename == "SYSTEM.md"):
                continue
            rel_path = filename if rel_root == "." else os.path.join(rel_root, filename)
            rel_path = rel_path.replace(os.sep, "/")
            try:
                with open(os.path.join(root, filename), "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                log.error("Failed to read context file %s: %s", rel_path, e)
                continue
            name, description = _parse_frontmatter(content)
            files.append(
                {
                    "path": f"{CONTEXT_DIR}/{rel_path}",
                    "name": name or rel_path[:-3],
                    "description": description or "",
                }
            )
    return sorted(files, key=lambda f: f["path"])


def _context_files_listing():
    """
    The context file tree — paths and one-line descriptions, not contents. The value of the
    {{CONTEXT_FILES}} persona variable. The index is cheap enough to carry every turn; a file is
    only worth reading in full once its description matches what's being asked.
    """
    files = _load_context_files()
    if not files:
        return "There are currently no context files in context/."
    lines = [f"- `{f['path']}` — {f['name']}: {f['description']}" for f in files]
    return "Available context files:\n" + "\n".join(lines)


def _context_files_system_message():
    """Fallback wrapper for {{CONTEXT_FILES}} when SYSTEM.md doesn't place it."""
    return {
        "role": "system",
        "content": (
            f"{_context_files_listing()}\n\n"
            "These hold durable background — facts about Apurva, the people in his life, project "
            "notes — that isn't in today's conversation. None of them are loaded automatically. "
            "When one's description covers what you need, read it with `_read_file` before "
            "answering rather than guessing from its name, and prefer it over `_search_history` "
            "for background that isn't tied to a specific past conversation."
        ),
    }


def _skills_listing():
    """
    Available playbooks — name/description only, not contents. The value of the {{PLAYBOOKS}}
    persona variable.
    """
    skills = _load_skills()
    if not skills:
        return "There are currently no playbooks in context/skills/."
    lines = [f"- `{s['filename']}` — {s['name']}: {s['description']}" for s in skills]
    return "Available playbooks:\n" + "\n".join(lines)


def _skills_system_message():
    """Fallback wrapper for {{PLAYBOOKS}} when SYSTEM.md doesn't place it."""
    return {
        "role": "system",
        "content": (
            f"{_skills_listing()}\n\n"
            "A playbook is a markdown file with step-by-step instructions for handling a "
            "specific kind of task. If one's description matches what the user is asking for, "
            "read it in full first with `_read_file` before acting on it — don't guess its "
            "contents from the name/description alone."
        ),
    }


# SYSTEM.md may place any of these inline as `{{NAME}}`, so the persona controls *where* live
# state appears in its own narrative instead of it arriving as a detached block after the prompt.
PERSONA_VARIABLES = {
    "JOBS": _jobs_listing,
    "PLAYBOOKS": _skills_listing,
    "CONTEXT_FILES": _context_files_listing,
}


def _render_persona():
    """Expand any {{VARIABLE}} SYSTEM.md places; returns the text and which names it used."""
    persona = _load_persona() or ""
    inlined = set()
    for name, build in PERSONA_VARIABLES.items():
        placeholder = "{{" + name + "}}"
        if placeholder in persona:
            persona = persona.replace(placeholder, build())
            inlined.add(name)
    return persona, inlined


def system_messages():
    """
    The system half of every turn: the rendered persona, plus a fallback block for each variable
    SYSTEM.md didn't place itself. A variable inlined in the persona is *not* also appended —
    that double-injection is exactly the redundancy the placeholders exist to remove.
    """
    persona, inlined = _render_persona()
    messages = [{"role": "system", "content": persona}]
    fallbacks = {
        "JOBS": _jobs_system_message,
        "PLAYBOOKS": _skills_system_message,
        "CONTEXT_FILES": _context_files_system_message,
    }
    for name, build in fallbacks.items():
        if name not in inlined:
            messages.append(build())
    return messages


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
    model_key = model_key if model_key in inference.MODEL_OPTIONS else get_default_model_key()
    log.info("Generating reply (non-streaming, save_user_turn=%s, model=%s)", save_user_turn, model_key)
    if save_user_turn:
        _save_turn({"role": "user", "content": user_input})
    history = _load_history()
    context = [{"role": turn["role"], "content": turn["content"]} for turn in history]
    messages = [*system_messages(), *context]
    if not save_user_turn:
        messages.append({"role": "user", "content": user_input})
    tool_calls_log = []

    while True:
        response, model_key = await _complete_with_fallback(model_key, messages, TOOL_SCHEMAS)
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
            # Lookup and arg-parsing sit inside the try on purpose: a hallucinated tool name
            # (KeyError) and truncated/malformed arguments (JSONDecodeError) are routine with
            # small local models, and outside the try they'd kill the whole generation instead
            # of being handed back to the model as a tool error it can recover from.
            arguments = {}
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
                log.info("Tool call: %s(%s)", tool_call.function.name, arguments)
                output = AVAILABLE_TOOLS[tool_call.function.name](**arguments)
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
            # `or ""` matters: message.content is None when the final round is tool-calls
            # only, and send_discord_message would POST {"content": null} → HTTP 400.
            "content": message.content or "",
            "thinking": getattr(message, "reasoning", None),
            "tool_calls": tool_calls_log,
            "source": "job",
        }
    )


async def _generate_reply_stream(user_input: str, model_key: str = None):
    """Yield event dicts; the caller serializes them for the wire."""
    model_key = model_key if model_key in inference.MODEL_OPTIONS else get_default_model_key()
    log.info("Chat request: %d chars (model=%s)", len(user_input), model_key)
    _save_turn({"role": "user", "content": user_input})
    history = _load_history()
    context = [{"role": turn["role"], "content": turn["content"]} for turn in history]
    messages = [*system_messages(), *context]
    tool_calls_log = []
    content = ""
    thinking = ""

    while True:
        tool_call_chunks = {}
        round_content = ""
        round_thinking = ""
        async for resolved_key, chunk in _stream_with_fallback(model_key, messages, TOOL_SCHEMAS):
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
            # Inside the try for the same reason as the non-streaming loop above — an unknown
            # tool name or unparseable arguments must reach the model as an error, not kill
            # the detached generation task mid-reply.
            arguments = {}
            try:
                arguments = json.loads(tc["arguments"] or "{}")
                log.info("Tool call: %s(%s)", tc["name"], arguments)
                output = AVAILABLE_TOOLS[tc["name"]](**arguments)
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
    # model_key is whatever _stream_with_fallback last resolved to, which may not be what the
    # caller asked for. The client pins its picker in localStorage, so without this it keeps
    # showing (and sizing its context meter against) a model that isn't answering any more.
    yield {"type": "done", "turn": turn, "model": model_key}


@router.post("/job-trigger")
async def job_trigger(prompt: str = Body(embed=True), channel: str = Body(embed=True)):
    log.info("Job fired → channel=%s prompt=%r", channel, prompt[:120])
    turn = await _generate_reply(prompt, save_user_turn=False)
    # The job that just fired is now expired; the reply took long enough that the minute has
    # almost always rolled over, so this is what actually removes it in practice.
    _sync_crontab()
    result = send_discord_message(channel, turn["content"])
    if result.get("ok"):
        log.info("Job message posted to Discord #%s", channel)
    else:
        log.error(
            "Discord post to #%s failed (status=%s)", channel, result.get("status_code")
        )
    return turn
