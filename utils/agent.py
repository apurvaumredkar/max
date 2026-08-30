from ollama._utils import convert_function_to_tool
import inspect
import os
import subprocess
import shlex
import sqlite3
import hashlib
import math
import re
from collections import Counter
from array import array
from datetime import datetime
import json
import uuid
from fastapi import APIRouter, Body
from utils import inference, mcp_client
from utils.discord_functions import send_discord_message
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


EMBED_CACHE_PATH = "context/embeddings.sqlite3"
# all-minilm silently truncates at 512 tokens, so a long turn's tail would never be searchable.
# ~1200 characters stays inside that window even for token-dense text (code, JSON tool output).
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150


def _chunk_text(text):
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= CHUNK_CHARS:
        return [text]
    step = CHUNK_CHARS - CHUNK_OVERLAP
    return [text[start : start + CHUNK_CHARS] for start in range(0, len(text), step)]


def _open_embed_cache():
    db = sqlite3.connect(EMBED_CACHE_PATH)
    db.execute(
        "CREATE TABLE IF NOT EXISTS embeddings ("
        "key TEXT NOT NULL, model TEXT NOT NULL, vector BLOB NOT NULL, "
        "PRIMARY KEY (key, model))"
    )
    return db


# Cache keys are a hash of the text itself, so an edited or deleted turn needs no invalidation
# and only genuinely new text ever reaches the embedding model.
def _embed_texts(texts):
    if not texts:
        return []
    keys = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    wanted = dict(zip(keys, texts))
    cached = {}
    try:
        db = _open_embed_cache()
    except Exception as e:
        log.warning("Embedding cache unavailable (%s); embedding everything", e)
        db = None
    if db is not None:
        try:
            unique = list(wanted)
            for start in range(0, len(unique), 500):
                batch = unique[start : start + 500]
                placeholders = ",".join("?" * len(batch))
                rows = db.execute(
                    f"SELECT key, vector FROM embeddings WHERE model = ? AND key IN ({placeholders})",
                    (EMBED_MODEL, *batch),
                ).fetchall()
                for key, blob in rows:
                    cached[key] = array("f", blob).tolist()
        except Exception as e:
            log.warning("Failed to read embedding cache: %s", e)

    missing = [(key, text) for key, text in wanted.items() if key not in cached]
    if missing:
        fresh = embed_client.embed(
            model=EMBED_MODEL, input=[text for _, text in missing]
        ).embeddings
        for (key, _), vector in zip(missing, fresh):
            cached[key] = list(vector)
        if db is not None:
            try:
                db.executemany(
                    "INSERT OR REPLACE INTO embeddings (key, model, vector) VALUES (?, ?, ?)",
                    [
                        (key, EMBED_MODEL, array("f", cached[key]).tobytes())
                        for key, _ in missing
                    ],
                )
                db.commit()
            except Exception as e:
                log.warning("Failed to write embedding cache: %s", e)
        log.info(
            "Embedded %s new chunk(s), %s served from cache",
            len(missing),
            len(cached) - len(missing),
        )
    if db is not None:
        db.close()
    return [cached[key] for key in keys]


_TOKEN = re.compile(r"[a-z0-9']+")
# "max" is the agent's own name and appears throughout the corpus, so it carries no signal and
# would otherwise make every greeting look like a lexical match.
_STOPWORDS = set(
    "max the a an and or but if of to in on at for with is are was were be been am i you it this "
    "that these those my your we us our do does did what when how why who can could would should "
    "will just about from as by not no yes me he she they them there here so up out get got have "
    "has had like more some any than then too very".split()
)
BM25_K1 = 1.5
BM25_B = 0.75
# Semantic similarity alone can't find a literal term the embedding blurs (a proper noun, a
# filename); lexical alone can't match a paraphrase. Weighted evenly, each covers the other's miss.
SEMANTIC_WEIGHT = 0.5


def _tokenize(text):
    return [
        word
        for word in _TOKEN.findall((text or "").lower())
        if len(word) > 2 and word not in _STOPWORDS
    ]


def _bm25_scores(query_tokens, documents):
    if not query_tokens or not documents:
        return [0.0] * len(documents)
    total = len(documents)
    doc_freq = Counter()
    for document in documents:
        doc_freq.update(set(document))
    avg_len = sum(len(d) for d in documents) / total or 1.0
    idf = {
        word: math.log(1 + (total - n + 0.5) / (n + 0.5)) for word, n in doc_freq.items()
    }
    scores = []
    for document in documents:
        freqs = Counter(document)
        length = len(document) or 1
        score = 0.0
        for word in set(query_tokens):
            freq = freqs.get(word, 0)
            if not freq or word not in idf:
                continue
            score += (
                idf[word]
                * freq
                * (BM25_K1 + 1)
                / (freq + BM25_K1 * (1 - BM25_B + BM25_B * length / avg_len))
            )
        scores.append(score)
    return scores


def _rank_past_turns(query):
    past_turns = _load_past_history()
    chunks = [
        (turn, chunk)
        for turn in past_turns
        for chunk in _chunk_text(turn.get("content"))
    ]
    if not chunks:
        return []
    query_embedding = _embed_texts([query])[0]
    chunk_embeddings = _embed_texts([chunk for _, chunk in chunks])
    documents = [_tokenize(chunk) for _, chunk in chunks]
    lexical = _bm25_scores(_tokenize(query), documents)
    # BM25 has no fixed range, so it's scaled against the best hit for this query before being
    # blended with cosine similarity, which is already 0-1.
    best_lexical = max(lexical) or 1.0
    # A turn scores as its best-matching chunk, so a long turn ranks on whichever part is
    # actually relevant rather than on an average diluted by everything else it contains.
    best = {}
    for (turn, _), vector, lex in zip(chunks, chunk_embeddings, lexical):
        turn_id = turn.get("id") or id(turn)
        score = SEMANTIC_WEIGHT * _cosine_similarity(query_embedding, vector) + (
            1 - SEMANTIC_WEIGHT
        ) * (lex / best_lexical)
        if score > best.get(turn_id, (None, -1.0))[1]:
            best[turn_id] = (turn, score)
    return sorted(best.values(), key=lambda pair: pair[1], reverse=True)


def _search_history(query: str, top_k: int = 5):
    """
    Semantically search prior days' conversation history (not today's) for turns relevant to a
    query. Use this when the user references something from an earlier conversation that isn't
    in your current context.

    Args:
        query: What to search for, described in natural language.
        top_k: Maximum number of matching turns to return.
    """
    scored = _rank_past_turns(query)
    if not scored:
        return {"results": []}
    return {
        "results": [
            {
                "turn_id": turn.get("id"),
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
_READ_MAX_CHARS = (
    8000  # hard backstop regardless of line count (e.g. one huge minified line)
)


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
    end = (
        start - 1
        if start > total_lines
        else min(start + max(limit, 1) - 1, total_lines)
    )
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
            ", ".join(
                f"{j.get('name', '?')} @ {j.get('run_at', '?')}" for j in expired
            ),
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
        json.dumps({"prompt": f"{SYSTEM_TRIGGER_PREFIX} {prompt}", "channel": channel})
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
    # Same schedule and same prompt is the same job however it's named — a near-duplicate name
    # (" Hydration Reminder" against "Hydration Reminder") is exactly how a job once cloned
    # itself into firing twice an hour.
    existing = next(
        (
            job
            for job in jobs["cron"]
            if job.get("schedule") == schedule
            and (job.get("prompt") or "").strip() == (prompt or "").strip()
        ),
        None,
    )
    if existing:
        log.info(
            "Cron job %r already exists as %r (%s) — not creating a duplicate",
            name,
            existing.get("name"),
            existing.get("id"),
        )
        return {
            "status": "already_exists",
            "name": existing.get("name"),
            "schedule": schedule,
            "prompt": existing.get("prompt"),
        }
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
    # Same guard as _cron: same time and same prompt is the same reminder, whatever it's called.
    existing = next(
        (
            job
            for job in jobs["scheduled"]
            if job.get("run_at") == run_at
            and (job.get("prompt") or "").strip() == (prompt or "").strip()
        ),
        None,
    )
    if existing:
        log.info(
            "Scheduled job %r already exists as %r (%s) — not creating a duplicate",
            name,
            existing.get("name"),
            existing.get("id"),
        )
        return {
            "status": "already_exists",
            "name": existing.get("name"),
            "run_at": run_at,
            "prompt": existing.get("prompt"),
        }
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
    files = []
    if not os.path.isdir(CONTEXT_DIR):
        return files
    for root, dirs, filenames in os.walk(CONTEXT_DIR):
        rel_root = os.path.relpath(root, CONTEXT_DIR)
        if rel_root == ".":
            dirs[:] = [d for d in dirs if d != "skills"]
        for filename in sorted(filenames):
            if not filename.endswith(".md") or (
                rel_root == "." and filename == "SYSTEM.md"
            ):
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
    files = _load_context_files()
    if not files:
        return "There are currently no context files in context/."
    lines = [f"- `{f['path']}` — {f['name']}: {f['description']}" for f in files]
    return "Available context files:\n" + "\n".join(lines)


def _context_files_system_message():
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
    skills = _load_skills()
    if not skills:
        return "There are currently no playbooks in context/skills/."
    lines = [f"- `{s['filename']}` — {s['name']}: {s['description']}" for s in skills]
    return "Available playbooks:\n" + "\n".join(lines)


def _skills_system_message():
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


# Retrieval runs on every turn rather than waiting for the model to call `_search_history`, so
# recall doesn't depend on the model realising it has forgotten something. These constants are
# heuristics hand-tuned against the real corpus, not derived — revisit them if recall starts
# injecting noise or missing obvious references.
RECALL_MIN_SCORE = 0.40
RECALL_TOP_K = 3
RECALL_MAX_CHARS = 1500
# Short turns ("hi", "thanks", "ok") match old greetings almost perfectly yet carry nothing worth
# recalling, so they're excluded on content rather than on score.
RECALL_MIN_TURN_CHARS = 200

# What the last generated turn actually recalled, so the context meter can report it without
# re-running retrieval on every poll — and reports what was really injected, not an approximation.
last_recalled_message = None


def _recalled_history_system_message(query):
    global last_recalled_message
    if not (query or "").strip():
        return None
    try:
        scored = _rank_past_turns(query)
    except Exception as e:
        # Recall is an enhancement; a dead embedding host must not take the whole turn down.
        log.warning("Automatic history recall failed: %s", e)
        return None
    hits = [
        (turn, score)
        for turn, score in scored
        if score >= RECALL_MIN_SCORE
        and len(turn.get("content") or "") >= RECALL_MIN_TURN_CHARS
    ][:RECALL_TOP_K]
    if not hits:
        last_recalled_message = None
        return None
    log.info(
        "Recalled %s past turn(s) for this message (top score %.2f)",
        len(hits),
        hits[0][1],
    )
    excerpts = []
    for turn, score in hits:
        content = turn.get("content") or ""
        if len(content) > RECALL_MAX_CHARS:
            content = content[:RECALL_MAX_CHARS] + " […]"
        excerpts.append(
            f"- [{turn.get('timestamp', 'unknown date')}] {turn.get('role')} "
            f"(relevance {score:.2f}):\n{content}"
        )
    last_recalled_message = {
        "role": "system",
        "content": (
            "Possibly relevant excerpts from earlier conversations, retrieved automatically "
            "because they resemble the user's latest message:\n\n"
            + "\n\n".join(excerpts)
            + "\n\nThese are excerpts, not the current conversation, and the match is by "
            "similarity alone — some may be irrelevant. Use them only where they genuinely "
            "apply, cite the date when you rely on one, and call `_search_history` yourself if "
            "you need more or the excerpt is cut off."
        ),
    }
    return last_recalled_message


# SYSTEM.md may place any of these inline as `{{NAME}}`, so the persona controls *where* live
# state appears in its own narrative instead of it arriving as a detached block after the prompt.
PERSONA_VARIABLES = {
    "JOBS": _jobs_listing,
    "PLAYBOOKS": _skills_listing,
    "CONTEXT_FILES": _context_files_listing,
}


def _render_persona():
    persona = _load_persona() or ""
    inlined = set()
    for name, build in PERSONA_VARIABLES.items():
        placeholder = "{{" + name + "}}"
        if placeholder in persona:
            persona = persona.replace(placeholder, build())
            inlined.add(name)
    return persona, inlined


def system_messages(query=None):
    # `query` is the user's message on the interactive path. The job path passes nothing on
    # purpose — see _generate_reply.
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
    recalled = _recalled_history_system_message(query)
    if recalled:
        messages.append(recalled)
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


def tool_schemas():
    """Native tools plus whatever the configured MCP servers are currently offering.

    A function rather than a constant because mcp_client.refresh() rebinds its list — the same
    trap as inference.MODEL_OPTIONS, and the loops and the context meter must all see a refresh.
    """
    return [*TOOL_SCHEMAS, *mcp_client.TOOL_SCHEMAS]


async def _call_tool(name, arguments):
    """Dispatch one tool call to a native function or, failing that, an MCP server."""
    if name in AVAILABLE_TOOLS:
        output = AVAILABLE_TOOLS[name](**arguments)
        # Native tools are sync today; awaiting when needed lets an async one be registered
        # without touching either loop.
        return await output if inspect.isawaitable(output) else output
    if mcp_client.handles(name):
        return await mcp_client.call(name, arguments)
    raise KeyError(name)


async def _generate_reply(
    user_input: str, save_user_turn: bool = True, model_key: str = None
):
    model_key = (
        model_key if model_key in inference.MODEL_OPTIONS else get_default_model_key()
    )
    log.info(
        "Generating reply (non-streaming, save_user_turn=%s, model=%s)",
        save_user_turn,
        model_key,
    )
    if save_user_turn:
        _save_turn({"role": "user", "content": user_input})
    history = _load_history()
    context = [{"role": turn["role"], "content": turn["content"]} for turn in history]
    # No automatic recall here. A job's prompt is a self-contained instruction to future Max, not
    # a user asking about the past — and with no real user turn to anchor them, recalled excerpts
    # read as the most recent instruction. A hydration reminder firing recalled the conversation
    # that first set it up and re-created the job, which is the same failure _jobs_system_message
    # exists to prevent. The interactive path passes its query; this one deliberately doesn't.
    messages = [*system_messages(), *context]
    if not save_user_turn:
        messages.append({"role": "user", "content": user_input})
    tool_calls_log = []

    while True:
        response, model_key = await _complete_with_fallback(
            model_key, messages, tool_schemas()
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
            # Lookup and arg-parsing sit inside the try on purpose: a hallucinated tool name
            # (KeyError) and truncated/malformed arguments (JSONDecodeError) are routine with
            # small local models, and outside the try they'd kill the whole generation instead
            # of being handed back to the model as a tool error it can recover from.
            arguments = {}
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
                log.info("Tool call: %s(%s)", tool_call.function.name, arguments)
                output = await _call_tool(tool_call.function.name, arguments)
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
    model_key = (
        model_key if model_key in inference.MODEL_OPTIONS else get_default_model_key()
    )
    log.info("Chat request: %d chars (model=%s)", len(user_input), model_key)
    _save_turn({"role": "user", "content": user_input})
    history = _load_history()
    context = [{"role": turn["role"], "content": turn["content"]} for turn in history]
    messages = [*system_messages(user_input), *context]
    tool_calls_log = []
    content = ""
    thinking = ""

    while True:
        tool_call_chunks = {}
        round_content = ""
        round_thinking = ""
        async for resolved_key, chunk in _stream_with_fallback(
            model_key, messages, tool_schemas()
        ):
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
                output = await _call_tool(tc["name"], arguments)
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
