import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from utils import agent, discord_functions, logs, mcp_client, spotify, tts, webui
from utils.logging_config import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Max agent started")
    try:
        agent._sync_crontab()
    except Exception:
        log.exception("Crontab sync failed at startup; continuing without it")
    try:
        await mcp_client.refresh()
    except Exception:
        log.exception("MCP discovery failed at startup; continuing without MCP tools")
    discord_task = discord_functions.start_bot()
    yield
    if discord_task:
        discord_task.cancel()
    log.info("Max agent shutting down")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent.router, prefix="/max", tags=["Max Agent"])
app.include_router(webui.router, prefix="/max", tags=["Web UI"])
app.include_router(spotify.router, prefix="/max", tags=["Spotify"])
app.include_router(logs.router, prefix="/max", tags=["Logs"])
app.include_router(tts.router, prefix="/max", tags=["TTS"])
app.mount("/assets", StaticFiles(directory="web/assets"), name="assets")
app.mount("/static", StaticFiles(directory="web"), name="static")


def _asset_version(path):
    try:
        return str(int(os.path.getmtime(path)))
    except OSError:
        return "0"


@app.get("/")
async def read_index():
    # Stamp ?v=<mtime> onto the css/js links so browsers pick up edits immediately
    # instead of holding a stale copy.
    with open("web/index.html", "r", encoding="utf-8") as index_file:
        html = index_file.read()
    html = html.replace(
        "/static/style.css", f"/static/style.css?v={_asset_version('web/style.css')}"
    ).replace("/static/app.js", f"/static/app.js?v={_asset_version('web/app.js')}")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@app.get("/favicon.ico")
async def favicon():
    return FileResponse("web/assets/max.png")


@app.get("/callback")
async def spotify_callback(code: str):
    log.info("Spotify OAuth callback received")
    spotify.exchange_code_for_token(code)
    return RedirectResponse("/")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=80)
