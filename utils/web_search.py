"""
Web search via the Gemini API's Grounding with Google Search tool.

Standalone helper, like google_drive.py and spotify.py — Max invokes it through
`_execute_bash` rather than as a registered tool:

    python utils/web_search.py "who won the 2026 world cup"
    python utils/web_search.py "latest on the liverpool transfer window" --sources-only
    python utils/web_search.py "python 3.13 release notes" --resolve-urls

Requires GEMINI_API_KEY in secrets/.env.

Why grounding rather than a plain completion: the response carries a
`groundingMetadata` block alongside the prose, giving the actual sources the answer
was built from — so the output can be cited web data instead of an unattributable
model summary. Three pieces are useful:

  groundingChunks    — the sources: {"web": {"uri": ..., "title": ...}}
  groundingSupports  — which span of the answer came from which chunk(s), via
                       {"segment": {"startIndex","endIndex","text"},
                        "groundingChunkIndices": [...]}
  webSearchQueries   — the queries Gemini actually ran

Source URIs are vertexaisearch.cloud.google.com redirects, not publisher URLs.
They resolve fine in a browser; --resolve-urls follows them to the real
destination at the cost of one HEAD request per source.
"""

import argparse
import json
import os
import subprocess
import sys

from dotenv import load_dotenv

# Run either as a module (`python -m utils.web_search`) or as a script
# (`python utils/web_search.py`) — the latter needs the repo root on the path
# before `utils.` resolves.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_config import get_logger

load_dotenv("secrets/.env")

log = get_logger(__name__)

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
# Grounding is supported across the 2.5+ line; flash keeps latency and cost sane
# for what is effectively a search call.
MODEL = "gemini-2.5-flash"
TIMEOUT_SECONDS = 60


def _api_key():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set in secrets/.env")
    return key


def _post(payload, model=MODEL):
    """
    POST to generateContent with curl.

    The key goes in the x-goog-api-key header — passing it as a ?key= query
    parameter is rejected as invalid by this endpoint.
    """
    url = f"{API_BASE}/{model}:generateContent"
    result = subprocess.run(
        [
            "curl",
            "-sS",
            "--max-time",
            str(TIMEOUT_SECONDS),
            "-X",
            "POST",
            url,
            "-H",
            f"x-goog-api-key: {_api_key()}",
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
        ],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed ({result.returncode}): {result.stderr.strip()}")
    try:
        body = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"Non-JSON response: {result.stdout[:300]}")
    if "error" in body:
        raise RuntimeError(f"Gemini API error: {body['error'].get('message')}")
    return body


def _resolve(uri):
    """Follow a grounding redirect to the publisher URL, or return it unchanged."""
    result = subprocess.run(
        ["curl", "-sS", "-o", "/dev/null", "-w", "%{url_effective}",
         "-I", "-L", "--max-redirs", "5", "--max-time", "15", uri],
        capture_output=True,
        text=True,
    )
    resolved = result.stdout.strip()
    return resolved if result.returncode == 0 and resolved else uri


def search(query, model=MODEL, resolve_urls=False):
    """
    Search the web and return the grounded answer with its sources.

    Args:
        query: What to search for, in natural language.
        model: Gemini model to use; must support the google_search tool.
        resolve_urls: Follow each source's redirect to the publisher URL.

    Returns a dict with:
        answer   — the prose answer
        sources  — [{index, title, uri}], the pages the answer drew on
        citations— [{text, sources: [index]}], which span came from which source
        queries  — the search queries Gemini ran
    """
    log.info("Web search: %r", query)
    payload = {
        "contents": [{"parts": [{"text": query}]}],
        "tools": [{"google_search": {}}],
    }
    body = _post(payload, model=model)

    candidates = body.get("candidates") or []
    if not candidates:
        log.warning("No candidates returned for %r", query)
        return {"answer": "", "sources": [], "citations": [], "queries": []}
    candidate = candidates[0]

    answer = "".join(
        part.get("text", "")
        for part in candidate.get("content", {}).get("parts", [])
    ).strip()

    metadata = candidate.get("groundingMetadata") or {}
    chunks = metadata.get("groundingChunks") or []

    sources = []
    for index, chunk in enumerate(chunks):
        web = chunk.get("web") or {}
        uri = web.get("uri", "")
        sources.append(
            {
                "index": index,
                "title": web.get("title", ""),
                "uri": _resolve(uri) if (resolve_urls and uri) else uri,
            }
        )

    citations = [
        {
            "text": (support.get("segment") or {}).get("text", ""),
            "sources": support.get("groundingChunkIndices", []),
        }
        for support in (metadata.get("groundingSupports") or [])
    ]

    if not sources:
        # Grounding is a tool the model chooses to use; it can answer from its own
        # weights and return no metadata at all.
        log.warning("Answer for %r came back ungrounded (no sources)", query)

    log.info("Web search returned %d source(s) for %r", len(sources), query)
    return {
        "answer": answer,
        "sources": sources,
        "citations": citations,
        "queries": metadata.get("webSearchQueries") or [],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search the web via Gemini grounding; prints JSON."
    )
    parser.add_argument("query", help="What to search for")
    parser.add_argument("--model", default=MODEL, help=f"Model (default: {MODEL})")
    parser.add_argument(
        "--sources-only",
        action="store_true",
        help="Print just the sources, omitting the prose answer",
    )
    parser.add_argument(
        "--resolve-urls",
        action="store_true",
        help="Follow redirects to publisher URLs (one request per source)",
    )
    args = parser.parse_args()

    # Logs go to stdout under systemd (so the journal picks them up), but stdout here
    # is the JSON payload Max parses — send log records to stderr instead.
    import logging

    from utils.logging_config import DATE_FORMAT, LOG_FORMAT

    root = logging.getLogger()
    root.handlers.clear()
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(stderr_handler)

    try:
        result = search(args.query, model=args.model, resolve_urls=args.resolve_urls)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        raise SystemExit(1)

    if args.sources_only:
        print(json.dumps({"sources": result["sources"], "queries": result["queries"]}, indent=2))
    else:
        print(json.dumps(result, indent=2))
