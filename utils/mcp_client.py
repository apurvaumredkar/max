"""
General-purpose MCP connector: any MCP server's tools, offered to Max as its own tools.

Servers are declared in context/mcp.json and discovered at startup, so attaching one is a config
change rather than a code change — the same shape as MODEL_OPTIONS, jobs.json and the context
files. Discovered tools are namespaced `<server>__<tool>` and appended to the agent loops' schema
list; the loops dispatch anything they don't recognise back here.

Sessions are opened per call rather than held open. An MCP session is an async context manager
bound to the task that entered it, and a long-lived one inside a detached generation task tends to
end in cancel-scope errors on client disconnect — the very thing webui's _Generation is built to
survive. Reconnecting costs a handshake (tens of ms against a local server) and buys a connector
where one dead server cannot wedge a turn.

    uv run python utils/mcp_client.py --list            # servers and their tools
    uv run python utils/mcp_client.py --call maps__places_geocode --args '{"address": "..."}'
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager

import httpx

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from utils.logging_config import get_logger

log = get_logger(__name__)

# The SDK logs a session id and a reconnect notice per connection at INFO. With a session per
# call that is one or two lines of noise per tool call in the journal.
logging.getLogger("mcp.client").setLevel(logging.WARNING)

CONFIG_PATH = "context/mcp.json"
# A server that is down must not stall a turn: the model is waiting on this call.
CONNECT_TIMEOUT_SECONDS = 10
CALL_TIMEOUT_SECONDS = 60
# `.` and `-` are not safe in a tool name for every model's function-calling grammar.
NAME_SEPARATOR = "__"

# Populated by refresh(); read through the module (mcp_client.TOOL_SCHEMAS) rather than imported,
# for the same reason as inference.MODEL_OPTIONS — refresh() rebinds it.
TOOL_SCHEMAS = []
_TOOL_ROUTES = {}


def _load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("servers", {})
    except FileNotFoundError:
        return {}
    except Exception as e:
        log.error("Failed to read %s: %s", CONFIG_PATH, e)
        return {}


@asynccontextmanager
async def _session(server):
    transport = (server.get("transport") or "http").lower()
    if transport in ("http", "streamable-http"):
        url = server.get("url")
        if not url:
            raise ValueError("http server is missing 'url'")
        http_client = httpx.AsyncClient(
            headers=server.get("headers") or {},
            timeout=httpx.Timeout(CALL_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS),
        )
        async with http_client:
            async with streamable_http_client(url, http_client=http_client) as (
                read,
                write,
                *_,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
    elif transport == "stdio":
        command = server.get("command")
        if not command:
            raise ValueError("stdio server is missing 'command'")
        params = StdioServerParameters(
            command=command,
            args=server.get("args") or [],
            # Inherit the parent environment so a server can read its own credentials, with the
            # config's env layered on top.
            env={**os.environ, **(server.get("env") or {})},
            cwd=server.get("cwd"),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    else:
        raise ValueError(f"unknown transport {transport!r}")


def _openai_schema(server_name, tool):
    schema = tool.input_schema or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": f"{server_name}{NAME_SEPARATOR}{tool.name}",
            "description": (tool.description or "").strip() or tool.name,
            "parameters": schema,
        },
    }


async def refresh():
    """Re-discover every enabled server's tools. One unreachable server never blocks the others."""
    global TOOL_SCHEMAS, _TOOL_ROUTES
    schemas, routes = [], {}
    for name, server in _load_config().items():
        if not server.get("enabled", True):
            continue
        try:
            async with asyncio.timeout(CONNECT_TIMEOUT_SECONDS + CALL_TIMEOUT_SECONDS):
                async with _session(server) as session:
                    tools = (await session.list_tools()).tools
        except Exception as e:
            log.error("MCP server %r unavailable (%s: %s)", name, type(e).__name__, e)
            continue
        # Every tool a server offers is sent to the model on every turn, so a big server is a
        # standing context cost. An optional allowlist attaches a server without paying for all
        # of it; omit it to take everything.
        allowed = server.get("tools")
        skipped = 0
        for tool in tools:
            if allowed and tool.name not in allowed:
                skipped += 1
                continue
            schemas.append(_openai_schema(name, tool))
            routes[f"{name}{NAME_SEPARATOR}{tool.name}"] = (name, tool.name)
        log.info(
            "MCP server %r: %d tool(s)%s",
            name,
            len(tools) - skipped,
            f" ({skipped} not in allowlist)" if skipped else "",
        )
    TOOL_SCHEMAS, _TOOL_ROUTES = schemas, routes
    return TOOL_SCHEMAS


def handles(tool_name):
    return tool_name in _TOOL_ROUTES


def _flatten(result):
    """MCP returns a list of content blocks; the agent loops want something JSON-serialisable."""
    if getattr(result, "structured_content", None):
        return result.structured_content
    parts = []
    for block in result.content or []:
        text = getattr(block, "text", None)
        if text is not None:
            try:
                parts.append(json.loads(text))
            except (json.JSONDecodeError, TypeError):
                parts.append(text)
        elif getattr(block, "resource", None) is not None:
            parts.append(getattr(block.resource, "uri", str(block.resource)))
        else:
            parts.append(getattr(block, "type", "unknown"))
    if not parts:
        return {"ok": True}
    return parts[0] if len(parts) == 1 else parts


async def call(tool_name, arguments):
    route = _TOOL_ROUTES.get(tool_name)
    if not route:
        raise KeyError(tool_name)
    server_name, remote_name = route
    server = _load_config().get(server_name) or {}
    async with asyncio.timeout(CONNECT_TIMEOUT_SECONDS + CALL_TIMEOUT_SECONDS):
        async with _session(server) as session:
            result = await session.call_tool(remote_name, arguments or {})
    payload = _flatten(result)
    # is_error is the server reporting a tool-level failure; surface it as the loops' error shape
    # so the model can retry or explain rather than treating the message as a result.
    if getattr(result, "is_error", False):
        return {"error": payload if isinstance(payload, str) else json.dumps(payload)}
    return payload


if __name__ == "__main__":
    import logging

    from utils.logging_config import DATE_FORMAT, LOG_FORMAT

    root = logging.getLogger()
    root.handlers.clear()
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(stderr_handler)

    parser = argparse.ArgumentParser(description="Inspect and exercise configured MCP servers")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="list every server's tools")
    mode.add_argument("--call", metavar="TOOL", help="call one tool, e.g. maps__places_geocode")
    parser.add_argument("--args", default="{}", help="--call only: JSON arguments object")
    args = parser.parse_args()

    async def main():
        await refresh()
        if args.list:
            if not TOOL_SCHEMAS:
                print("No MCP tools discovered.")
                return
            for schema in TOOL_SCHEMAS:
                fn = schema["function"]
                description = " ".join((fn["description"] or "").split())
                print(f"{fn['name']}\n    {description[:140]}")
            print(f"\n{len(TOOL_SCHEMAS)} tool(s).")
        else:
            print(json.dumps(await call(args.call, json.loads(args.args)), indent=2, default=str))

    asyncio.run(main())
