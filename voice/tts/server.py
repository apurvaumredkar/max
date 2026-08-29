"""
Minimal FastAPI wrapper around kokoro-onnx (Kokoro-82M, fp32 ONNX, CUDA execution provider).
Baked into the TTS container by voice/tts/Dockerfile; utils/tts.py talks to it over HTTP.
"""

import io
import logging
import os
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as rt
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from kokoro_onnx import Kokoro
from pydantic import BaseModel

log = logging.getLogger("tts")
logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/kokoro-v1.0.onnx")
VOICES_PATH = os.environ.get("VOICES_PATH", "/models/voices-v1.0.bin")
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "am_echo")
ONNX_PROVIDER = os.environ.get("ONNX_PROVIDER", "CUDAExecutionProvider")
# Single-stream service with GPU-side heavy ops; one thread per core is pure overhead here.
INTRA_OP_THREADS = int(os.environ.get("ORT_INTRA_OP_THREADS", "2"))

kokoro: Kokoro | None = None
# Cached style vectors: passing the ndarray to create() skips a 510KB NpzFile inflate per call.
_styles: dict[str, np.ndarray] = {}


def _build_session() -> rt.InferenceSession:
    """Build the ORT session ourselves — kokoro-onnx's create_session() sets no options at all."""
    options = rt.SessionOptions()
    options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = INTRA_OP_THREADS
    options.inter_op_num_threads = 1
    options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    # Nothing of consequence runs on the CPU EP here — the arena is pure overhead.
    options.enable_cpu_mem_arena = False

    if ONNX_PROVIDER == "CUDAExecutionProvider":
        cuda_options = {
            # Default kNextPowerOfTwo rounds each arena growth up and holds it for the process.
            "arena_extend_strategy": "kSameAsRequested",
            # EXHAUSTIVE re-benchmarks convs per distinct input shape; token length is dynamic.
            "cudnn_conv_algo_search": "HEURISTIC",
            "cudnn_conv_use_max_workspace": "0",
            "do_copy_in_default_stream": "1",
        }
        providers = [("CUDAExecutionProvider", cuda_options)]
    else:
        providers = [ONNX_PROVIDER]

    try:
        return rt.InferenceSession(MODEL_PATH, sess_options=options, providers=providers)
    except Exception as error:  # noqa: BLE001 — never fail to boot over a tuning option
        log.warning("Tuned session failed (%s); falling back to stock options", error)
        return rt.InferenceSession(MODEL_PATH, providers=[ONNX_PROVIDER])


def _style(voice: str) -> np.ndarray:
    style = _styles.get(voice)
    if style is None:
        if voice not in kokoro.voices:
            raise HTTPException(status_code=400, detail=f"unknown voice {voice!r}")
        style = np.asarray(kokoro.voices[voice])
        _styles[voice] = style
    return style


@asynccontextmanager
async def lifespan(app: FastAPI):
    global kokoro
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOICES_PATH):
        raise RuntimeError(
            f"Model files missing ({MODEL_PATH}, {VOICES_PATH}) — run download_models.sh "
            "before starting the container."
        )
    session = _build_session()
    log.info("onnxruntime providers in use: %s", session.get_providers())
    kokoro = Kokoro.from_session(session, VOICES_PATH)
    # Warm up now so the first real request doesn't pay espeak-ng load + CUDA first-conv setup.
    try:
        kokoro.create("Ready.", voice=_style(DEFAULT_VOICE), speed=1.0, lang="en-us", trim=False)
    except Exception as error:  # noqa: BLE001 — warm-up is best effort
        log.warning("Warm-up synthesis failed: %s", error)
    log.info("Kokoro loaded (default voice %s)", DEFAULT_VOICE)
    yield


app = FastAPI(lifespan=lifespan)


class SpeakRequest(BaseModel):
    text: str
    voice: str = DEFAULT_VOICE
    speed: float = 1.0
    lang: str = "en-us"


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": kokoro is not None}


@app.post("/synthesize")
async def synthesize(req: SpeakRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    # trim=True (the default) crashes on some short sentences — it reduces over an empty array.
    samples, sample_rate = kokoro.create(
        req.text, voice=_style(req.voice), speed=req.speed, lang=req.lang, trim=False
    )
    # Canary for the all-NaN/all-zero audio the old fp16 export produced; unseen on fp32.
    if not np.isfinite(samples).all() or not samples.any():
        log.error("Synthesis produced unusable audio for %r", req.text[:120])
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return Response(content=buffer.getvalue(), media_type="audio/wav")
