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

from utils.logging_config import get_logger

load_dotenv("secrets/.env")

log = get_logger(__name__)


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
    # Only the hand-declared entries and the Tailscale Ollama hosts are fallback candidates.
    # The discovered OpenRouter catalog is ~400 models and only a handful are :free, so
    # walking it meant a single rate-limited free-tier failure — the *normal* failure here —
    # could silently land the turn on a billed model, and _remember_model_key would then
    # persist it as the default for every later request, scheduled jobs included. It also
    # made a total outage take hundreds of sequential API calls to give up on.
    ordered += [
        key
        for key in MODEL_OPTIONS
        if key not in ordered and not key.startswith("openrouter:")
    ]
    return ordered


async def _complete_with_fallback(model_key, messages, tools):
    """
    Call the chat-completions endpoint for model_key; on failure, retry with the next
    configured model instead of failing the turn. `tools` is the caller's tool schemas, passed
    in rather than imported so this module stays independent of the agent loop's tool registry.
    Returns (response, resolved_model_key) —
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
        return response, candidate_key
    raise last_error


async def _stream_with_fallback(model_key, messages, tools):
    """
    Streaming counterpart to _complete_with_fallback. Only falls back if the failure happens
    before any chunk of this round has been produced — a stream that dies partway through
    can't be safely resumed on a different model without duplicating or losing output already
    sent to the client. `tools` is the caller's tool schemas, as in _complete_with_fallback.
    Yields (resolved_model_key, chunk) pairs; resolved_model_key is stable
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
                tools=tools,
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
