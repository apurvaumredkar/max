"""
Web-UI-serving endpoints for the Max chat frontend (web/app.js).

Everything here exists purely to support the browser UI — jobs CRUD, context-file CRUD,
chat history, model listing, the context-usage estimator, and the streaming-chat
reattach/replay machinery. None of it is invoked by the LLM's tool-calling loop; that lives
in utils/agent.py, which this module imports from for the handful of things (job storage,
crontab sync, persona/history loading) that both sides need. Model/backend config comes from
utils/inference.py directly.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime

import psutil
import requests
from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse

from utils import agent, inference
from utils.agent import (
    TOOL_SCHEMAS,
    _generate_reply_stream,
    _jobs_system_message,
    _load_all_history,
    _load_history,
    _load_jobs,
    _save_jobs,
    _skills_system_message,
    _sync_crontab,
)
from utils.inference import get_default_model_key
from utils.logging_config import get_logger

log = get_logger(__name__)

router = APIRouter()


# --- Chat history ---


def _rewrite_turns(path, turns):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as tmp:
        for turn in turns:
            tmp.write(json.dumps(turn) + "\n")
    os.replace(tmp_path, path)


def _delete_turn(turn_id: str):
    if not os.path.isdir("chats"):
        return False
    for filename in sorted(os.listdir("chats")):
        if not filename.endswith(".jsonl"):
            continue
        path = f"chats/{filename}"
        try:
            with open(path, "r") as chats:
                turns = [json.loads(line) for line in chats if line.strip()]
        except Exception as e:
            log.error("Failed to read %s while deleting turn: %s", path, e)
            continue
        remaining = [turn for turn in turns if turn.get("id") != turn_id]
        if len(remaining) == len(turns):
            continue
        try:
            _rewrite_turns(path, remaining)
        except Exception as e:
            log.error("Failed to rewrite %s while deleting turn: %s", path, e)
            return False
        log.info("Deleted turn %s from %s", turn_id, path)
        return True
    log.warning("Delete requested for unknown turn id %s", turn_id)
    return False


@router.get("/history")
async def history():
    return _load_all_history()


@router.delete("/history/{turn_id}")
async def delete_history_turn(turn_id: str):
    deleted = _delete_turn(turn_id)
    return {"deleted": deleted, "id": turn_id}


# --- Jobs CRUD ---


def _find_job(jobs, job_id):
    for kind in ("cron", "scheduled"):
        for index, job in enumerate(jobs.get(kind, [])):
            if job.get("id") == job_id:
                return kind, index
    return None, None


def _validate_job(kind, payload):
    if not (payload.get("name") or "").strip():
        return "Name is required"
    if not (payload.get("prompt") or "").strip():
        return "Prompt is required"
    if kind == "cron":
        schedule = (payload.get("schedule") or "").strip()
        if len(schedule.split()) != 5:
            return "Schedule must be a 5-field cron expression"
    else:
        try:
            datetime.strptime((payload.get("run_at") or "").strip(), "%Y-%m-%d %H:%M")
        except ValueError:
            return "run_at must be 'YYYY-MM-DD HH:MM'"
    return None


@router.get("/jobs")
async def jobs():
    return _load_jobs()


@router.post("/jobs")
async def create_job(payload: dict = Body(...)):
    kind = "scheduled" if payload.get("run_at") else "cron"
    error = _validate_job(kind, payload)
    if error:
        log.warning("Rejected job create: %s", error)
        return {"ok": False, "error": error}
    job = {
        "id": uuid.uuid4().hex,
        "name": payload["name"].strip(),
        "prompt": payload["prompt"].strip(),
        "channel": (payload.get("channel") or "reminders").strip(),
    }
    if kind == "cron":
        job["schedule"] = payload["schedule"].strip()
    else:
        job["run_at"] = payload["run_at"].strip()
    jobs = _load_jobs()
    jobs.setdefault(kind, []).append(job)
    _save_jobs(jobs)
    if not _sync_crontab():
        return {"ok": False, "error": "Job saved but crontab sync failed"}
    log.info("Job created via UI: %r (%s)", job["name"], kind)
    return {"ok": True, "job": job}


@router.put("/jobs/{job_id}")
async def update_job(job_id: str, payload: dict = Body(...)):
    jobs = _load_jobs()
    kind, index = _find_job(jobs, job_id)
    if kind is None:
        log.warning("Update requested for unknown job id %s", job_id)
        return {"ok": False, "error": "Job not found"}
    # A job can switch kind — a recurring reminder becoming a one-off, or vice versa.
    new_kind = "scheduled" if payload.get("run_at") else "cron"
    error = _validate_job(new_kind, payload)
    if error:
        log.warning("Rejected job update: %s", error)
        return {"ok": False, "error": error}
    job = {
        "id": job_id,
        "name": payload["name"].strip(),
        "prompt": payload["prompt"].strip(),
        "channel": (payload.get("channel") or "reminders").strip(),
    }
    if new_kind == "cron":
        job["schedule"] = payload["schedule"].strip()
    else:
        job["run_at"] = payload["run_at"].strip()
    jobs[kind].pop(index)
    jobs.setdefault(new_kind, []).append(job)
    _save_jobs(jobs)
    if not _sync_crontab():
        return {"ok": False, "error": "Job saved but crontab sync failed"}
    log.info("Job updated via UI: %r (%s)", job["name"], new_kind)
    return {"ok": True, "job": job}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    jobs = _load_jobs()
    kind, index = _find_job(jobs, job_id)
    if kind is None:
        log.warning("Delete requested for unknown job id %s", job_id)
        return {"ok": False, "error": "Job not found"}
    removed = jobs[kind].pop(index)
    _save_jobs(jobs)
    if not _sync_crontab():
        return {"ok": False, "error": "Job removed but crontab sync failed"}
    log.info("Job deleted via UI: %r (%s)", removed.get("name"), kind)
    return {"ok": True, "deleted": removed}


# --- Context files ---

CONTEXT_DIR = "context"
SKILLS_DIR = "context/skills"


def _estimate_tokens(text):
    return max(1, round(len(text or "") / 4))


def _md_path_in(directory, filename):
    if not filename.endswith(".md") or "\\" in filename:
        return None
    if any(part in ("", ".", "..") for part in filename.split("/")):
        return None
    path = os.path.normpath(os.path.join(directory, filename))
    root = os.path.normpath(directory)
    if path != root and not path.startswith(root + os.sep):
        return None
    return path


def _list_md_files(directory, exclude_dirs=()):
    if not os.path.isdir(directory):
        return []
    results = []
    for root, dirs, files in os.walk(directory):
        rel_root = os.path.relpath(root, directory)
        if rel_root == ".":
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for filename in files:
            if not filename.endswith(".md"):
                continue
            rel_path = filename if rel_root == "." else os.path.join(rel_root, filename)
            results.append(rel_path.replace(os.sep, "/"))
    return sorted(results)


@router.get("/context-files")
async def list_context_files():
    return {"files": _list_md_files(CONTEXT_DIR, exclude_dirs={"skills"})}


@router.get("/context-files/{filename:path}")
async def get_context_file(filename: str):
    path = _md_path_in(CONTEXT_DIR, filename)
    if not path or not os.path.isfile(path):
        return {"ok": False, "error": "File not found"}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"ok": True, "filename": filename, "content": content, "tokens": _estimate_tokens(content)}


@router.put("/context-files/{filename:path}")
async def save_context_file(filename: str, payload: dict = Body(...)):
    path = _md_path_in(CONTEXT_DIR, filename)
    if not path:
        return {"ok": False, "error": "Invalid filename"}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    content = payload.get("content", "")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    log.info("Context file saved via UI: %s", filename)
    return {"ok": True}


# --- Skill files (playbooks) ---


@router.get("/skills-files")
async def list_skill_files():
    return {"files": _list_md_files(SKILLS_DIR)}


@router.get("/skills-files/{filename:path}")
async def get_skill_file(filename: str):
    path = _md_path_in(SKILLS_DIR, filename)
    if not path or not os.path.isfile(path):
        return {"ok": False, "error": "File not found"}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"ok": True, "filename": filename, "content": content, "tokens": _estimate_tokens(content)}


@router.put("/skills-files/{filename:path}")
async def save_skill_file(filename: str, payload: dict = Body(...)):
    path = _md_path_in(SKILLS_DIR, filename)
    if not path:
        return {"ok": False, "error": "Invalid filename"}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    content = payload.get("content", "")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    log.info("Skill file saved via UI: %s", filename)
    return {"ok": True}


# --- Model / context info ---


# NOTE: the three handlers below are deliberately `def`, not `async def`. Each does blocking
# I/O (requests to Tailscale Ollama hosts / OpenRouter, or a full walk of context/ + chats/).
# FastAPI runs a plain `def` handler in its threadpool; as `async def` they ran ON the event
# loop, so an unreachable Tailscale peer's 3s connect timeout froze the chat stream, the TTS
# socket, the Discord gateway and the log SSE along with it. Do not re-add `async` here.
@router.get("/models")
def models_list():
    options = inference.refresh_model_options()
    return {
        "default": get_default_model_key(),
        "options": [
            {
                "key": key,
                "label": opts["label"],
                "group": opts.get("group", opts["label"]),
                "model_label": opts.get("model_label", opts["label"]),
                "context_length": opts["context_length"],
            }
            for key, opts in options.items()
        ],
    }


# --- System monitor ---


def _tailscale_ollama_reachable(provider_id):
    provider = inference.TAILSCALE_OLLAMA_PROVIDERS[provider_id]
    host = (provider["host_url"] or "").removesuffix("/v1").rstrip("/")
    if not host:
        return False
    headers = {"Authorization": f"Bearer {provider['api_key']}"} if provider["api_key"] else {}
    try:
        response = requests.get(f"{host}/api/tags", headers=headers, timeout=3)
        return response.ok
    except Exception:
        return False


@router.get("/system-status")
def system_status():
    vm = psutil.virtual_memory()
    return {
        "ollama": {
            provider_id: _tailscale_ollama_reachable(provider_id)
            for provider_id in inference.TAILSCALE_OLLAMA_PROVIDERS
        },
        "ram": {
            "used_mb": round((vm.total - vm.available) / (1024 * 1024)),
            "total_mb": round(vm.total / (1024 * 1024)),
            "percent": vm.percent,
        },
    }


def _context_blocks():
    history = _load_history()
    persona, inlined = agent._render_persona()
    blocks = [
        {
            "key": "persona",
            "label": "Persona",
            # Say so when SYSTEM.md placed the live blocks itself — otherwise the persona looks
            # inexplicably large and the missing rows look like a bug.
            "detail": (
                f"context/SYSTEM.md · {{{{{'}}, {{'.join(sorted(inlined))}}}}} expanded inline"
                if inlined
                else "context/SYSTEM.md"
            ),
            "content": persona,
        },
    ]
    # A variable SYSTEM.md inlined is already counted inside the persona — listing it again would
    # double-count it in the meter.
    if "JOBS" not in inlined:
        blocks.append({
            "key": "jobs",
            "label": "Jobs",
            "detail": "context/jobs.json",
            "content": _jobs_system_message()["content"],
        })
    if "PLAYBOOKS" not in inlined:
        blocks.append({
            "key": "skills",
            "label": "Playbooks",
            "detail": "listing of context/skills/",
            "content": _skills_system_message()["content"],
        })
    if "CONTEXT_FILES" not in inlined:
        blocks.append({
            "key": "context_files",
            "label": "Context files",
            "detail": "listing of context/**.md",
            "content": agent._context_files_system_message()["content"],
        })
    blocks += [
        {
            "key": "history",
            "label": "Today's chat",
            "detail": f"{len(history)} turns",
            "content": "".join(turn.get("content") or "" for turn in history),
        },
        {
            "key": "tools",
            "label": "Tool schemas",
            "detail": f"{len(TOOL_SCHEMAS)} tools",
            "content": json.dumps(TOOL_SCHEMAS),
        },
    ]
    return blocks


@router.get("/context-usage")
def context_usage(model: str = None):
    key = model if model in inference.MODEL_OPTIONS else get_default_model_key()
    context_length = inference.MODEL_OPTIONS[key]["context_length"]
    blocks = [
        {
            "key": b["key"],
            "label": b["label"],
            "detail": b["detail"],
            "tokens": _estimate_tokens(b["content"]),
        }
        for b in _context_blocks()
    ]
    used_tokens = sum(b["tokens"] for b in blocks)
    return {
        "model": key,
        "context_length": context_length,
        "used_tokens": used_tokens,
        "percent_remaining": max(0, round(100 * (1 - used_tokens / context_length))),
        "blocks": blocks,
    }


@router.get("/persona-variables")
async def persona_variables():
    return {name: build() for name, build in agent.PERSONA_VARIABLES.items()}


@router.get("/injected-context")
async def injected_context(key: str):
    for block in _context_blocks():
        if block["key"] == key:
            return {**block, "tokens": _estimate_tokens(block["content"])}
    return {"error": f"Unknown block {key!r}"}


# --- Streaming chat ---


class _Generation:
    """
    A single in-flight reply, decoupled from the HTTP request that started it.

    Generation runs as a background task writing events into a buffer, so closing or
    reloading the page doesn't cancel it — the turn still completes and is saved. A client
    that (re)connects replays the buffer and then follows live.
    """

    def __init__(self, user_input, model_key=None):
        self.user_input = user_input
        self.model_key = model_key if model_key in inference.MODEL_OPTIONS else get_default_model_key()
        self.events = []
        self.done = False
        self.task = None
        self._waiters = []

    def emit(self, event):
        self.events.append(event)
        for waiter in self._waiters:
            if not waiter.done():
                waiter.set_result(None)
        self._waiters.clear()

    def finish(self):
        self.done = True
        self.emit({"type": "_eof"})

    async def wait_for_event(self, seen):
        if seen < len(self.events) or self.done:
            return
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        await waiter

    async def run(self):
        try:
            async for event in _generate_reply_stream(self.user_input, self.model_key):
                self.emit(event)
        except asyncio.CancelledError:
            log.warning("Generation cancelled")
            self.emit({"type": "error", "message": "Generation cancelled"})
            raise
        except Exception as e:
            log.exception("Generation failed")
            self.emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            self.finish()


# At most one generation at a time — the agent is single-user and each turn depends on the
# history the previous one wrote.
_current_generation = None


async def iter_events(generation):
    seen = 0
    while True:
        await generation.wait_for_event(seen)
        while seen < len(generation.events):
            event = generation.events[seen]
            seen += 1
            if event.get("type") == "_eof":
                return
            yield event
        if generation.done and seen >= len(generation.events):
            return


async def _replay(generation):
    async for event in iter_events(generation):
        yield json.dumps(event) + "\n"


def generation_in_flight():
    return bool(_current_generation and not _current_generation.done)


def start_generation(user_input, model_key=None):
    global _current_generation
    if generation_in_flight():
        log.info("Request arrived while a generation is in flight — attaching to it")
        return _current_generation
    generation = _Generation(user_input, model_key)
    _current_generation = generation
    # A bare task already outlives the request that started it: the disconnect cancels
    # the StreamingResponse, not this task. (Wrapping it in shield() was a bug —
    # create_task() needs a coroutine and shield() returns a Future.)
    generation.task = asyncio.create_task(generation.run())
    return generation


@router.post("/chat")
async def chat(user_input: str = Body(embed=True), model: str = Body(default=None, embed=True)):
    generation = start_generation(user_input, model)
    return StreamingResponse(_replay(generation), media_type="application/x-ndjson")


@router.get("/chat/active")
async def chat_active():
    generation = _current_generation
    if not generation or generation.done:
        return {"active": False}
    return {"active": True}


@router.get("/chat/attach")
async def chat_attach():
    generation = _current_generation
    if not generation or generation.done:
        return {"active": False}
    log.info("Client reattached to in-flight generation")
    return StreamingResponse(_replay(generation), media_type="application/x-ndjson")
