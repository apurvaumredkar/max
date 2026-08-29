"""
TTS router backed by the dockerized Kokoro-82M service (voice/tts/). Not an agent tool — it
powers the web UI's "Voice" toggle, which speaks the assistant's streamed replies. Needs the
container running (scripts/setup-tts.sh). Overrides in secrets/.env: TTS_SERVICE_URL, TTS_VOICE.
"""

import os

import requests
from dotenv import load_dotenv
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from utils.logging_config import get_logger

load_dotenv("secrets/.env")

log = get_logger(__name__)

router = APIRouter()

TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://127.0.0.1:8880")
TTS_VOICE = os.getenv("TTS_VOICE", "am_echo")

# Pooled so a sentence-per-message stream reuses one connection instead of a handshake each.
_session = requests.Session()


def synthesize(text: str, voice: str | None = None, speed: float = 1.0) -> bytes:
    """Call the Kokoro TTS container and return WAV audio bytes."""
    response = _session.post(
        f"{TTS_SERVICE_URL}/synthesize",
        json={"text": text, "voice": voice or TTS_VOICE, "speed": speed},
        timeout=30,
    )
    response.raise_for_status()
    return response.content


@router.websocket("/tts/stream")
async def stream_speak(websocket: WebSocket):
    """
    Persistent WS for the Voice toggle: the client pushes one {"text": ...} per sentence as
    soon as it's segmented, so synthesis of the next overlaps playback of the current one.
    """
    await websocket.accept()
    log.info("TTS stream connected")
    try:
        while True:
            message = await websocket.receive_json()
            text = (message.get("text") or "").strip()
            if not text:
                continue
            log.info("TTS stream: synthesizing %d chars: %r", len(text), text[:80])
            try:
                audio = await run_in_threadpool(synthesize, text, message.get("voice"))
            except Exception as e:
                log.error("TTS stream synthesis failed: %s", e)
                # A JSON text frame tells the client this chunk failed; audio is binary frames.
                await websocket.send_json({"error": str(e)})
                continue
            await websocket.send_bytes(audio)
            log.info("TTS stream: sent %d bytes", len(audio))
    except WebSocketDisconnect:
        log.info("TTS stream disconnected")
