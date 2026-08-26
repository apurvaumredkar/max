"""
Web-UI-serving endpoints for the Max chat frontend (web/app.js).

Everything here exists purely to support the browser UI — jobs CRUD, context-file CRUD,
chat history, model listing, the context-usage estimator, and the streaming-chat
reattach/replay machinery. None of it is invoked by the LLM's tool-calling loop; that lives
in utils/agent.py, which this module imports from for the handful of things (job storage,
crontab sync, model config, persona/history loading) that both sides need.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime

from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse

from utils.agent import (
    DEFAULT_MODEL_KEY,
    MODEL_OPTIONS,
    TOOL_SCHEMAS,
    _generate_reply_stream,
    _jobs_system_message,
    _load_all_history,
    _load_history,
    _load_jobs,
    _load_persona,
    _save_jobs,
    _sync_crontab,
)
from utils.logging_config import get_logger

log = get_logger(__name__)

router = APIRouter()


# --- Chat history ---


def backfill_turn_ids():
    """
    Give any pre-existing turn an id, rewriting the file in place.

    Turns written before ids existed can't be addressed for deletion, so stamp them once
    on startup. Returns the number of turns updated.
    """
    if not os.path.isdir("chats"):
        return 0
    updated = 0
    for filename in sorted(os.listdir("chats")):
        if not filename.endswith(".jsonl"):
            continue
        path = f"chats/{filename}"
        try:
            with open(path, "r") as chats:
                turns = [json.loads(line) for line in chats if line.strip()]
        except Exception as e:
            log.error("Failed to read %s for id backfill: %s", path, e)
            continue
        missing = [turn for turn in turns if not turn.get("id")]
        if not missing:
            continue
        for turn in missing:
            turn["id"] = uuid.uuid4().hex
        try:
            _rewrite_turns(path, turns)
            updated += len(missing)
            log.info("Backfilled %d turn ids in %s", len(missing), path)
        except Exception as e:
            log.error("Failed to backfill ids in %s: %s", path, e)
    return updated


def _rewrite_turns(path, turns):
    """Atomically replace a chat file's contents — write a temp file, then rename over."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as tmp:
        for turn in turns:
            tmp.write(json.dumps(turn) + "\n")
    os.replace(tmp_path, path)


def _delete_turn(turn_id: str):
    """
    Remove the turn with this id from whichever chat file holds it.

    Returns True if a turn was deleted. Rewrites atomically so a crash mid-write can't
    truncate the day's history.
    """
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


def backfill_job_ids():
    """Stamp ids on jobs written before ids existed, so the UI can address them."""
    jobs = _load_jobs()
    updated = 0
    for kind in ("cron", "scheduled"):
        for job in jobs.get(kind, []):
            if not job.get("id"):
                job["id"] = uuid.uuid4().hex
                updated += 1
    if updated:
        _save_jobs(jobs)
        log.info("Backfilled %d job id(s)", updated)
    return updated


def _find_job(jobs, job_id):
    """Return (kind, index) for a job id, or (None, None)."""
    for kind in ("cron", "scheduled"):
        for index, job in enumerate(jobs.get(kind, [])):
            if job.get("id") == job_id:
                return kind, index
    return None, None


def _validate_job(kind, payload):
    """Return an error string if the payload is unusable, else None."""
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


def _estimate_tokens(text):
    """Rough token estimate (~4 chars/token) — no tokenizer for these models is available locally."""
    return max(1, round(len(text or "") / 4))


def _context_md_path(filename):
    """Resolve a filename to a .md path inside CONTEXT_DIR, or None if unsafe/invalid."""
    if not filename.endswith(".md") or "/" in filename or "\\" in filename or filename in (".", ".."):
        return None
    path = os.path.join(CONTEXT_DIR, filename)
    if os.path.dirname(path) != CONTEXT_DIR:
        return None
    return path


@router.get("/context-files")
async def list_context_files():
    try:
        names = sorted(f for f in os.listdir(CONTEXT_DIR) if f.endswith(".md"))
    except OSError as e:
        log.error("Failed to list context files: %s", e)
        return {"files": []}
    return {"files": names}


@router.get("/context-files/{filename}")
async def get_context_file(filename: str):
    path = _context_md_path(filename)
    if not path or not os.path.isfile(path):
        return {"ok": False, "error": "File not found"}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"ok": True, "filename": filename, "content": content, "tokens": _estimate_tokens(content)}


@router.put("/context-files/{filename}")
async def save_context_file(filename: str, payload: dict = Body(...)):
    path = _context_md_path(filename)
    if not path:
        return {"ok": False, "error": "Invalid filename"}
    content = payload.get("content", "")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    log.info("Context file saved via UI: %s", filename)
    return {"ok": True}


# --- Model / context info ---


@router.get("/models")
async def models_list():
    return {
        "default": DEFAULT_MODEL_KEY,
        "options": [
            {"key": key, "label": opts["label"], "context_length": opts["context_length"]}
            for key, opts in MODEL_OPTIONS.items()
        ],
    }


@router.get("/context-usage")
async def context_usage(model: str = DEFAULT_MODEL_KEY):
    key = model if model in MODEL_OPTIONS else DEFAULT_MODEL_KEY
    context_length = MODEL_OPTIONS[key]["context_length"]
    persona = _load_persona() or ""
    jobs_text = _jobs_system_message()["content"]
    history = _load_history()
    history_text = "".join(turn.get("content") or "" for turn in history)
    tool_schemas_text = json.dumps(TOOL_SCHEMAS)
    used_tokens = _estimate_tokens(persona + jobs_text + history_text + tool_schemas_text)
    return {
        "model": key,
        "context_length": context_length,
        "used_tokens": used_tokens,
        "percent_remaining": max(0, round(100 * (1 - used_tokens / context_length))),
    }


# --- Streaming chat ---


class _Generation:
    """
    A single in-flight reply, decoupled from the HTTP request that started it.

    Generation runs as a background task writing events into a buffer, so closing or
    reloading the page doesn't cancel it — the turn still completes and is saved. A client
    that (re)connects replays the buffer and then follows live.
    """

    def __init__(self, user_input, model_key=DEFAULT_MODEL_KEY):
        self.user_input = user_input
        self.model_key = model_key
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
        """Block until there are events past index `seen`, or generation is done."""
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


async def _replay(generation):
    """Stream a generation's events to one client, from the beginning, then live."""
    seen = 0
    while True:
        await generation.wait_for_event(seen)
        while seen < len(generation.events):
            event = generation.events[seen]
            seen += 1
            if event.get("type") == "_eof":
                return
            yield json.dumps(event) + "\n"
        if generation.done and seen >= len(generation.events):
            return


@router.post("/chat")
async def chat(user_input: str = Body(embed=True), model: str = Body(default=DEFAULT_MODEL_KEY, embed=True)):
    global _current_generation
    if _current_generation and not _current_generation.done:
        log.info("Chat request arrived while a generation is in flight — attaching to it")
        return StreamingResponse(
            _replay(_current_generation), media_type="application/x-ndjson"
        )
    generation = _Generation(user_input, model)
    _current_generation = generation
    # A bare task already outlives the request that started it: the disconnect cancels
    # the StreamingResponse, not this task. (Wrapping it in shield() was a bug —
    # create_task() needs a coroutine and shield() returns a Future.)
    generation.task = asyncio.create_task(generation.run())
    return StreamingResponse(_replay(generation), media_type="application/x-ndjson")


@router.get("/chat/active")
async def chat_active():
    """
    Report whether a reply is still being generated, so a freshly loaded page can
    reattach to it instead of missing the rest of the response.
    """
    generation = _current_generation
    if not generation or generation.done:
        return {"active": False}
    return {"active": True}


@router.get("/chat/attach")
async def chat_attach():
    """Reattach to the in-flight generation, replaying everything emitted so far."""
    generation = _current_generation
    if not generation or generation.done:
        return {"active": False}
    log.info("Client reattached to in-flight generation")
    return StreamingResponse(_replay(generation), media_type="application/x-ndjson")
