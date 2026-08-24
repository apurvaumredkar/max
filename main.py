from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from utils import agent, spotify

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent.router, prefix="/max", tags=["Max Chat Webhook"])
app.include_router(spotify.router, prefix="/max", tags=["Spotify"])
app.mount("/assets", StaticFiles(directory="web/assets"), name="assets")
app.mount("/static", StaticFiles(directory="web"), name="static")


@app.get("/")
async def read_index():
    return FileResponse("web/index.html")


@app.get("/callback")
async def spotify_callback(code: str):
    spotify.exchange_code_for_token(code)
    return RedirectResponse("http://max/")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=80)
