"""
Standalone TTS helper backed by the dockerized Kokoro-82M service (voice/tts/). Not wired
into the agent's tool-calling loop — this powers the web UI's "Voice" toggle, which speaks
the assistant's streamed replies. Can also be invoked directly for manual testing:

    python utils/tts.py --say "hello there"

Requires the TTS container to be running (see scripts/setup-tts.sh). Optional overrides in
secrets/.env: TTS_SERVICE_URL (default http://127.0.0.1:8880), TTS_VOICE (default am_echo).
"""

import argparse
import os

import requests
from dotenv import load_dotenv
from fastapi import APIRouter, Body, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from utils.logging_config import get_logger

load_dotenv("secrets/.env")

log = get_logger(__name__)

router = APIRouter()

TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://127.0.0.1:8880")
TTS_VOICE = os.getenv("TTS_VOICE", "am_echo")

# One pooled session rather than a bare requests.post per call: the WS path synthesizes a
# sentence at a time, so a fresh connection (and its TCP handshake) was being set up and torn
# down for every sentence of every spoken reply. Keep-alive reuses one.
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


@router.post("/tts/speak")
async def route_speak(text: str = Body(embed=True), voice: str | None = Body(default=None, embed=True)):
    audio = await run_in_threadpool(synthesize, text, voice)
    return Response(content=audio, media_type="audio/wav")


@router.websocket("/tts/stream")
async def stream_speak(websocket: WebSocket):
    """
    Persistent WS for the Voice toggle: the client pushes {"text": ..., "voice": ...} messages
    for each sentence as soon as it's segmented, without waiting for earlier audio to finish
    playing. Keeping one connection open — rather than a POST per sentence gated on playback
    finishing — lets synthesis of the next sentence run while the current one is still playing,
    instead of only starting once the client goes idle again (which was the source of the
    audible gap between every sentence in the old fetch-per-sentence design).
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
                # Tell the client instead of just dropping the chunk silently — it distinguishes
                # a JSON text frame (error) from binary audio frames by event.data's type.
                await websocket.send_json({"error": str(e)})
                continue
            await websocket.send_bytes(audio)
            log.info("TTS stream: sent %d bytes", len(audio))
    except WebSocketDisconnect:
        log.info("TTS stream disconnected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--say", metavar="TEXT", help="Synthesize text and write it to out.wav")
    parser.add_argument("--voice", default=None, help="Override the default voice")
    parser.add_argument("--out", default="out.wav", help="Output WAV path (default: out.wav)")
    args = parser.parse_args()

    if args.say:
        audio = synthesize(args.say, voice=args.voice)
        with open(args.out, "wb") as f:
            f.write(audio)
        print(f"wrote {len(audio)} bytes to {args.out}")
    else:
        parser.print_help()
