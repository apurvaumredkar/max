"""
Live service logs for the web UI's Logs tab.

Streams `journalctl -u max-agent -f` over Server-Sent Events. The `max` user is in the `adm`
group, so the journal is readable without sudo.
"""

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from utils.logging_config import get_logger

log = get_logger(__name__)

router = APIRouter()

SERVICE = "max-agent"
BACKFILL_LINES = 200


async def _journal_lines(request: Request, lines: int):
    """
    Yield SSE events from journalctl --follow, stopping when the client disconnects.

    Backfills the last `lines` entries first so the tab isn't empty on open, then follows.
    """
    process = await asyncio.create_subprocess_exec(
        "journalctl",
        "-u",
        SERVICE,
        "-n",
        str(lines),
        "-f",
        "--no-pager",
        "-o",
        "short-iso",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    log.info("Log stream opened (pid=%s, backfill=%d)", process.pid, lines)

    async def _terminate():
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
        log.info("Log stream closed (pid=%s)", process.pid)

    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                raw = await asyncio.wait_for(process.stdout.readline(), timeout=15)
            except asyncio.TimeoutError:
                # Keep the connection alive through idle periods; comments are ignored by
                # EventSource but keep proxies from timing the stream out.
                yield ": keepalive\n\n"
                continue
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if line:
                yield f"data: {json.dumps({'line': line})}\n\n"
    finally:
        # A disconnected client usually leaves this generator to be garbage-collected rather
        # than resumed, so `finally` may run without a live event loop. Shield the cleanup on
        # the running loop when there is one; otherwise fall back to killing outright, so a
        # closed tab never leaves a journalctl -f behind.
        try:
            asyncio.get_running_loop().create_task(_terminate())
        except RuntimeError:
            if process.returncode is None:
                process.kill()
            log.info("Log stream closed (pid=%s, no loop)", process.pid)


@router.get("/logs/stream")
async def stream_logs(request: Request, lines: int = BACKFILL_LINES):
    lines = max(1, min(lines, 2000))
    return StreamingResponse(
        _journal_lines(request, lines),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
