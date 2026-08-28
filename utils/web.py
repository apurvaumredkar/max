"""
Web search (Gemini API's Grounding with Google Search) and page fetching.

Standalone helper, like google_drive.py and spotify.py — Max invokes it through
`_execute_bash` rather than as a registered tool:

    python utils/web.py --search "who won the 2026 world cup"
    python utils/web.py --search "latest on the liverpool transfer window" --sources-only
    python utils/web.py --search "python 3.13 release notes" --json
    python utils/web.py --fetch "https://example.com/some-article"
    python utils/web.py --fetch "https://example.com/some-article" --json

--search output is markdown by default: the answer with inline [1][2] citation markers,
then a numbered source list. Publisher URLs are resolved by default so the links are
quotable. Pass --json for the raw structure, --raw-urls to skip redirect resolution.
Requires GEMINI_API_KEY in secrets/.env.

--fetch downloads one page and strips it down to its readable text (script/style/markup
removed) — for reading a specific URL's content, as opposed to --search's "find and
answer from the web" grounding. No API key needed; --json wraps the same result in JSON
instead of printing it directly.

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
import re
import subprocess
import sys
from html.parser import HTMLParser
from urllib.parse import urlparse

from dotenv import load_dotenv

# Run either as a module (`python -m utils.web`) or as a script
# (`python utils/web.py`) — the latter needs the repo root on the path
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


def _domain(uri):
    """Best-effort hostname for display, minus the www."""
    match = re.match(r"https?://([^/]+)", uri or "")
    return match.group(1).replace("www.", "") if match else ""


def format_result(result, include_answer=True):
    """
    Render a search result as markdown for an LLM to read.

    Raw JSON reads badly here: source URLs are ~200-character redirect blobs that
    swamp the payload, and citations point at sources by bare index. This inlines
    those indices as [1][2] markers right after the sentences they support, then
    lists the numbered sources underneath — the shape a model already knows how to
    read and cite from.
    """
    lines = []
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    citations = result.get("citations", [])

    if include_answer and answer:
        lines.append(_annotate(answer, citations))
        lines.append("")

    if sources:
        lines.append("## Sources")
        for source in sources:
            title = source.get("title") or _domain(source.get("uri", "")) or "source"
            lines.append(f"[{source['index'] + 1}] {title} — {source.get('uri', '')}")
    else:
        lines.append("## Sources")
        lines.append(
            "(none — this answer was NOT grounded in search results; "
            "treat it as unverified model output)"
        )

    queries = result.get("queries") or []
    if queries:
        lines.append("")
        lines.append(f"Searched for: {'; '.join(queries)}")

    return "\n".join(lines).strip()


def _annotate(answer, citations):
    """
    Append [n] markers to the answer at the end of each cited span.

    Offsets from the API are byte offsets into the UTF-8 answer, and inserting
    shifts everything after — so work back-to-front over the encoded bytes.
    """
    insertions = {}
    encoded = answer.encode("utf-8")
    for citation in citations:
        text = citation.get("text") or ""
        if not text:
            continue
        needle = text.encode("utf-8")
        position = encoded.find(needle)
        if position < 0:
            continue
        end = position + len(needle)
        marks = "".join(f"[{index + 1}]" for index in citation.get("sources", []))
        insertions.setdefault(end, "")
        insertions[end] += marks

    for end in sorted(insertions, reverse=True):
        encoded = encoded[:end] + insertions[end].encode("utf-8") + encoded[end:]
    return encoded.decode("utf-8", errors="replace")


# --- Page fetching ---

FETCH_TIMEOUT_SECONDS = 20
FETCH_MAX_CHARS = 8000
FETCH_USER_AGENT = "Mozilla/5.0 (compatible; MaxAgent/1.0)"

# Tags whose text content isn't real page content — code/styling, or the site chrome
# (nav menus, headers/footers, forms, icon SVGs) rather than the article/body text itself.
_SKIP_TAGS = {
    "script", "style", "noscript", "template",
    "nav", "header", "footer", "aside", "form", "button", "svg", "iframe",
}
# Block-level tags: a line break is inserted at each boundary so the extracted text keeps
# roughly the page's paragraph/heading/list structure instead of running everything together.
_BLOCK_TAGS = {
    "p", "div", "br", "li", "tr", "section", "article", "header", "footer",
    "blockquote", "pre", "h1", "h2", "h3", "h4", "h5", "h6",
}


class _TextExtractor(HTMLParser):
    """Strips a page down to its visible text — no tags, no script/style content — while
    keeping a `<title>` and rough line breaks at block-tag boundaries for readability."""

    def __init__(self):
        super().__init__()
        self._chunks = []
        self._title_chunks = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data):
        if self._skip_depth:
            return
        (self._title_chunks if self._in_title else self._chunks).append(data)

    def text(self):
        raw = "".join(self._chunks)
        # Collapse runs of horizontal whitespace within a line, drop blank lines, but keep
        # the line breaks the block tags inserted.
        lines = (re.sub(r"[ \t]+", " ", line).strip() for line in raw.splitlines())
        return "\n".join(line for line in lines if line)

    def title(self):
        return "".join(self._title_chunks).strip()


def fetch(url, max_chars=FETCH_MAX_CHARS):
    """
    Fetch a web page and return its readable text — script/style/markup stripped out —
    instead of raw HTML, so an LLM can read it as plain context.

    Args:
        url: The page to fetch; must start with http:// or https://.
        max_chars: Maximum characters of extracted text to return. The page is truncated
            (not the fetch itself) beyond this so one large page can't blow out the context
            it's being read into.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL must start with http:// or https://: {url!r}")

    log.info("Web fetch: %s", url)
    result = subprocess.run(
        [
            "curl", "-sS", "-L",
            "--max-time", str(FETCH_TIMEOUT_SECONDS),
            "--max-redirs", "5",
            "-A", FETCH_USER_AGENT,
            url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed ({result.returncode}): {result.stderr.strip()[:300]}")

    parser = _TextExtractor()
    parser.feed(result.stdout)
    content = parser.text()
    truncated = len(content) > max_chars

    log.info("Web fetch of %s returned %d chars (truncated=%s)", url, len(content), truncated)
    return {
        "url": url,
        "title": parser.title(),
        "content": content[:max_chars],
        "truncated": truncated,
    }


def format_fetch_result(result):
    """Render a fetch result as markdown for an LLM to read."""
    lines = [f"# {result['title'] or result['url']}", f"Source: {result['url']}", ""]
    lines.append(result["content"])
    if result.get("truncated"):
        lines.append("\n(truncated — page content is longer than what's shown here)")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search the web (Gemini grounding) or fetch a page's readable text. "
        "Prints markdown by default."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--search", metavar="QUERY", help="Search the web via Gemini grounding")
    mode.add_argument("--fetch", metavar="URL", help="Fetch a page and extract its readable text")
    parser.add_argument("--model", default=MODEL, help=f"--search only: model (default: {MODEL})")
    parser.add_argument(
        "--sources-only",
        action="store_true",
        help="--search only: print just the sources, omitting the prose answer",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON instead of markdown",
    )
    parser.add_argument(
        "--raw-urls",
        action="store_true",
        help="--search only: keep the grounding redirect URLs instead of resolving to publisher links",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=FETCH_MAX_CHARS,
        help=f"--fetch only: max characters of page text to return (default: {FETCH_MAX_CHARS})",
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
        if args.search is not None:
            # Redirect blobs are unreadable and unquotable, so resolve by default and
            # make keeping them the opt-in.
            result = search(args.search, model=args.model, resolve_urls=not args.raw_urls)
        else:
            result = fetch(args.fetch, max_chars=args.max_chars)
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"{'Search' if args.search is not None else 'Fetch'} failed: {e}")
        raise SystemExit(1)

    if args.json:
        if args.search is not None and args.sources_only:
            print(
                json.dumps(
                    {"sources": result["sources"], "queries": result["queries"]},
                    indent=2,
                )
            )
        else:
            print(json.dumps(result, indent=2))
    elif args.search is not None:
        print(format_result(result, include_answer=not args.sources_only))
    else:
        print(format_fetch_result(result))
