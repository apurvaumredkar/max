from ollama import Client
from ollama._utils import convert_function_to_tool
from openai import AsyncOpenAI
import os
import subprocess
import shlex
from dotenv import load_dotenv
from datetime import datetime
import json
from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse
from utils.discord_functions import send_discord_message
from utils.logging_config import get_logger

load_dotenv("secrets/.env")

log = get_logger(__name__)

JOB_TRIGGER_URL = "http://localhost/max/job-trigger"

# OpenAI-compatible client — INFERENCE_HOST_URL/INFERENCE_API_KEY in secrets/.env point this at
# whichever provider is active (Ollama, OpenRouter, etc). Swap MODEL below to match.
openai_client = AsyncOpenAI(
    base_url=os.getenv("INFERENCE_HOST_URL"),
    api_key=os.getenv("INFERENCE_API_KEY"),
)
embed_client = Client(host="http://localhost:11434")
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
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
    turn = {**turn, "timestamp": datetime.now().strftime("%B %d, %Y %I:%M %p")}
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


def _install_crontab_line(line: str):
    existing = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    updated = existing + line + "\n"
    result = subprocess.run(
        ["crontab", "-"], input=updated, capture_output=True, text=True
    )
    if result.returncode != 0:
        log.error("Failed to install crontab line %r: %s", line, result.stderr.strip())
    else:
        log.info("Installed crontab line: %s", line)


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
    command = _job_trigger_command(prompt, channel)
    _install_crontab_line(f"{schedule} {command}")
    jobs = _load_jobs()
    jobs["cron"].append(
        {"name": name, "schedule": schedule, "prompt": prompt, "channel": channel}
    )
    _save_jobs(jobs)
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
    dt = datetime.strptime(run_at, "%Y-%m-%d %H:%M")
    schedule = f"{dt.minute} {dt.hour} {dt.day} {dt.month} *"
    command = _job_trigger_command(prompt, channel)
    _install_crontab_line(f"{schedule} {command}")
    jobs = _load_jobs()
    jobs["scheduled"].append(
        {"name": name, "run_at": run_at, "prompt": prompt, "channel": channel}
    )
    _save_jobs(jobs)
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


@router.get("/history")
async def history():
    return _load_all_history()


@router.get("/jobs")
async def jobs():
    return _load_jobs()


AVAILABLE_TOOLS = {
    "_execute_bash": _execute_bash,
    "_cron": _cron,
    "_schedule": _schedule,
    "_search_history": _search_history,
}

# Built from the same docstrings above, in OpenAI's tool-schema format.
TOOL_SCHEMAS = [
    convert_function_to_tool(func).model_dump(exclude_none=True)
    for func in AVAILABLE_TOOLS.values()
]


async def _generate_reply(user_input: str, save_user_turn: bool = True):
    log.info("Generating reply (non-streaming, save_user_turn=%s)", save_user_turn)
    persona = _load_persona()
    if save_user_turn:
        _save_turn({"role": "user", "content": user_input})
    history = _load_history()
    context = [{"role": turn["role"], "content": turn["content"]} for turn in history]
    messages = [
        {"role": "system", "content": persona},
        _jobs_system_message(),
        *context,
    ]
    if not save_user_turn:
        messages.append({"role": "user", "content": user_input})
    tool_calls_log = []

    while True:
        response = await openai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            extra_body={"options": {"num_ctx": 262144}, "keep_alive": -1},
        )
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


async def _generate_reply_stream(user_input: str):
    log.info("Chat request: %d chars", len(user_input))
    persona = _load_persona()
    _save_turn({"role": "user", "content": user_input})
    history = _load_history()
    context = [{"role": turn["role"], "content": turn["content"]} for turn in history]
    messages = [
        {"role": "system", "content": persona},
        _jobs_system_message(),
        *context,
    ]
    tool_calls_log = []
    content = ""
    thinking = ""

    while True:
        tool_call_chunks = {}
        round_content = ""
        round_thinking = ""
        stream = await openai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            stream=True,
            extra_body={"options": {"num_ctx": 262144}, "keep_alive": -1},
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning", None)
            if reasoning:
                round_thinking += reasoning
                yield json.dumps({"type": "thinking", "delta": reasoning}) + "\n"
            if delta.content:
                round_content += delta.content
                yield json.dumps({"type": "content", "delta": delta.content}) + "\n"
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
            yield json.dumps({"type": "tool_call", **tool_call_record}) + "\n"
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
    yield json.dumps({"type": "done", "turn": turn}) + "\n"


@router.post("/chat")
async def chat(user_input: str = Body(embed=True)):
    return StreamingResponse(
        _generate_reply_stream(user_input), media_type="application/x-ndjson"
    )


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
