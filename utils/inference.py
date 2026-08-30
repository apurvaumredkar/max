"""
Inference backends: which models exist, which one a request resolves to, and the clients that
actually call them.

Everything here is model plumbing, deliberately kept out of utils/agent.py so that module is
just the agentic loop and its tools. Nothing in here knows what a tool or a turn is — the two
fallback wrappers take the tool schemas as an argument rather than importing them, which is
what keeps the dependency one-way (agent imports inference, never the reverse).

MODEL_OPTIONS is rebound by refresh_model_options(), so other modules must reach it through
the module (`inference.MODEL_OPTIONS`) — a `from ... import MODEL_OPTIONS` captures a snapshot
that a later refresh won't update.
"""

import json
import os

import requests
from dotenv import load_dotenv
from ollama import Client
from openai import AsyncOpenAI

from utils import usage
from utils.logging_config import get_logger

load_dotenv("secrets/.env")

log = get_logger(__name__)


_OPENROUTER_DEFAULT_CONTEXT_LENGTH = 8192


def discover_openrouter_models():
    base_url = os.getenv("INFERENCE_HOST_URL")
    api_key = os.getenv("INFERENCE_API_KEY")
    if not base_url:
        return {}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    options = {}
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/models", headers=headers, timeout=10
        )
        response.raise_for_status()
        models = response.json().get("data", [])
    except Exception as e:
        log.error("Failed to list OpenRouter models from %s: %s", base_url, e)
        return options

    for entry in models:
        model_id = entry.get("id")
        if not model_id:
            continue
        name = entry.get("name") or model_id
        options[f"openrouter:{model_id}"] = {
            "label": f"OpenRouter · {name}",
            "group": "OpenRouter",
            "model_label": name,
            "model": model_id,
            "base_url": base_url,
            "api_key": api_key,
            "context_length": entry.get("context_length")
            or _OPENROUTER_DEFAULT_CONTEXT_LENGTH,
        }
    return options


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
    try:
        response = requests.post(
            f"{host}/api/show", headers=headers, json={"model": name}, timeout=5
        )
        response.raise_for_status()
        model_info = response.json().get("model_info", {})
        for key, value in model_info.items():
            if key.endswith(".context_length"):
                return value
    except Exception as e:
        log.error("Failed to fetch context length for %s from %s: %s", name, host, e)
    return fallback


def discover_ollama_models():
    options = {}
    for provider_id, provider in TAILSCALE_OLLAMA_PROVIDERS.items():
        host = (provider["host_url"] or "").removesuffix("/v1").rstrip("/")
        if not host:
            continue
        headers = (
            {"Authorization": f"Bearer {provider['api_key']}"}
            if provider["api_key"]
            else {}
        )
        try:
            response = requests.get(f"{host}/api/tags", headers=headers, timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
        except Exception as e:
            log.error(
                "Failed to list Ollama models from %s (%s): %s", host, provider_id, e
            )
            continue

        for entry in models:
            name = entry.get("model") or entry.get("name")
            if not name:
                continue
            fallback = (
                entry.get("details", {}).get("context_length")
                or _OLLAMA_DEFAULT_CONTEXT_LENGTH
            )
            options[f"tailscale-ollama-{provider_id}:{name}"] = {
                "label": f"Ollama · {provider['label']} · {name}",
                "group": provider["label"],
                "model_label": name,
                "model": name,
                "base_url": provider["host_url"],
                "api_key": provider["api_key"],
                "context_length": _ollama_model_context_length(
                    host, headers, name, fallback
                ),
            }
    return options


MODEL_OPTIONS = {**discover_ollama_models(), **discover_openrouter_models()}


def refresh_model_options():
    global MODEL_OPTIONS
    MODEL_OPTIONS = {**discover_ollama_models(), **discover_openrouter_models()}
    return MODEL_OPTIONS


def _default_model_key():
    if not MODEL_OPTIONS:
        raise RuntimeError(
            "No inference backends discovered — every Ollama host and OpenRouter were "
            "unreachable at startup. Call refresh_model_options() once one is back."
        )
    return next(iter(MODEL_OPTIONS))


def _model_choice(model_key):
    key = model_key if model_key in MODEL_OPTIONS else _default_model_key()
    opts = MODEL_OPTIONS[key]
    client = AsyncOpenAI(base_url=opts["base_url"], api_key=opts["api_key"] or "ollama")
    return client, opts["model"], opts["context_length"]


def _record_usage(model_key, response_usage):
    if not response_usage:
        return
    opts = MODEL_OPTIONS.get(model_key, {})
    usage.record(
        model_key,
        opts.get("group") or "unknown",
        opts.get("model_label") or opts.get("model") or model_key,
        getattr(response_usage, "prompt_tokens", 0),
        getattr(response_usage, "completion_tokens", 0),
    )


CONFIG_PATH = "context/config.json"


def _load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def get_default_model_key():
    last_key = _load_config().get("last_model_key")
    return last_key if last_key in MODEL_OPTIONS else _default_model_key()


def _remember_model_key(model_key):
    config = _load_config()
    if config.get("last_model_key") == model_key:
        return
    config["last_model_key"] = model_key
    _save_config(config)


def _is_free(model_key):
    return not model_key.startswith("openrouter:") or model_key.endswith(":free")


def _model_fallback_candidates(preferred_key):
    ordered = [preferred_key] if preferred_key in MODEL_OPTIONS else []
    ordered += [key for key in MODEL_OPTIONS if key not in ordered and _is_free(key)]
    return ordered


async def _complete_with_fallback(model_key, messages, tools):
    candidates = _model_fallback_candidates(model_key)
    last_error = None
    for i, candidate_key in enumerate(candidates):
        client, model_name, context_length = _model_choice(candidate_key)
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
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
        _record_usage(candidate_key, getattr(response, "usage", None))
        return response, candidate_key
    raise last_error


def _absorb_usage(model_key, chunk, recorded):
    """Record a streamed chunk's usage; True if the chunk was usage-only and must not be yielded.

    `recorded` is a one-element list guarding against a provider that repeats cumulative usage on
    every chunk — only the last report for a stream counts, and double-counting would silently
    inflate the daily report.
    """
    chunk_usage = getattr(chunk, "usage", None)
    if chunk_usage:
        recorded[0] = (model_key, chunk_usage)
    return chunk_usage is not None and not getattr(chunk, "choices", None)


async def _stream_with_fallback(model_key, messages, tools):
    candidates = _model_fallback_candidates(model_key)
    last_error = None
    for i, candidate_key in enumerate(candidates):
        client, model_name, context_length = _model_choice(candidate_key)
        try:
            stream = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                stream=True,
                stream_options={"include_usage": True},
                extra_body={"options": {"num_ctx": context_length}, "keep_alive": -1},
            )
            aiter = stream.__aiter__()
            try:
                first_chunk = await aiter.__anext__()
            except StopAsyncIteration:
                if candidate_key != model_key:
                    log.warning(
                        "Fell back from model %s to %s", model_key, candidate_key
                    )
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
        recorded = [None]
        if not _absorb_usage(candidate_key, first_chunk, recorded):
            yield candidate_key, first_chunk
        async for chunk in aiter:
            # The usage chunk arrives last and carries no choices, so it's held back rather than
            # handed to the agent loops, which index choices[0].
            if not _absorb_usage(candidate_key, chunk, recorded):
                yield candidate_key, chunk
        if recorded[0]:
            _record_usage(*recorded[0])
        return
    raise last_error


embed_client = Client(host="http://localhost:11434")
EMBED_MODEL = "all-minilm"
