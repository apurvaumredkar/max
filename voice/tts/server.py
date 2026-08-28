"""
Minimal FastAPI wrapper around kokoro-onnx (Kokoro-82M, fp32 ONNX, CUDA execution provider).

Not part of the main app — this is what voice/tts/Dockerfile bakes into the TTS container.
The main app talks to this over HTTP via utils/tts.py; see voice/tts/README for the full
picture (Dockerfile, download_models.sh, docker-compose.yml).
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
# The Orin has 6 cores, but this is a single-stream service whose heavy ops run on the GPU;
# onnxruntime otherwise spins up one intra-op thread per core, each with its own allocator,
# for CPU-side glue work that doesn't parallelize usefully at this graph size.
INTRA_OP_THREADS = int(os.environ.get("ORT_INTRA_OP_THREADS", "2"))

kokoro: Kokoro | None = None
# Style vectors, resolved once per voice name. kokoro-onnx keeps `self.voices` as a lazy
# numpy NpzFile, so `Kokoro.create(..., voice="am_echo")` re-inflates a 510x1x256 float32
# array (510KB, ~1.6ms) out of the zip on *every* request. Passing the ndarray straight in
# — `create()` accepts either a name or an array — skips that entirely.
_styles: dict[str, np.ndarray] = {}


def _build_session() -> rt.InferenceSession:
    """
    Build the onnxruntime session ourselves instead of letting kokoro-onnx's
    `create_session()` do it, so we can pass memory/latency options it never sets.

    kokoro_onnx.session.create_session() calls `rt.InferenceSession(path, providers=[...])`
    with no SessionOptions and no provider options at all, which on this box means:
      - a CPU arena that grows and is never returned to the OS, on top of the CUDA arena;
      - one intra-op thread per core (6), each with its own allocator;
      - CUDA's default `cudnn_conv_algo_search=EXHAUSTIVE`, which benchmarks every
        convolution algorithm the first time it sees a given input shape and allocates
        large scratch buffers to do it. Kokoro's token input is *dynamic length*, so every
        new phoneme count is a new shape and pays that search again;
      - `cudnn_conv_use_max_workspace=1`, reserving the largest workspace each conv could
        want rather than a sufficient one.

    `Kokoro.from_session()` is the library's supported hook for handing it a prebuilt
    session, so this stays on the public API.

    Any option name onnxruntime rejects would otherwise take the container down at startup,
    so an unrecognised option falls back to the stock session rather than failing to boot.
    """
    options = rt.SessionOptions()
    options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = INTRA_OP_THREADS
    options.inter_op_num_threads = 1
    options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    # Nothing of consequence runs on the CPU EP here — the arena is pure overhead.
    options.enable_cpu_mem_arena = False

    if ONNX_PROVIDER == "CUDAExecutionProvider":
        cuda_options = {
            # Default kNextPowerOfTwo rounds every arena growth up to the next power of
            # two, so a 300MB need takes 512MB and holds it for the process's life.
            "arena_extend_strategy": "kSameAsRequested",
            # See above: EXHAUSTIVE re-benchmarks per distinct input shape.
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


def _is_bad_audio(samples: np.ndarray) -> bool:
    return not np.isfinite(samples).all() or not samples.any()


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
    # Resolve the default voice and run one throwaway synthesis now, so the first real
    # request doesn't pay for espeak-ng's data load, the tokenizer warm-up, or CUDA's
    # first-conv setup while a user is waiting on it.
    _style(DEFAULT_VOICE)
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
    # trim=True (kokoro-onnx's default) crashes on some short/simple sentences — its
    # pause-insertion step reduces over a zero-size "quiet frames" array. trim=False sidesteps
    # the bug entirely and just leaves in a bit of natural leading/trailing silence.
    style = _style(req.voice)
    samples, sample_rate = kokoro.create(
        req.text, voice=style, speed=req.speed, lang=req.lang, trim=False
    )
    # The fp16 export this used to run deterministically produced all-NaN/all-zero audio for a
    # large fraction of real sentences (confirmed on both CPU and CUDA — a numerical stability
    # issue in the model itself, not an execution-provider bug), which needed a CPU-retry
    # workaround. Switching to fp32 (see download_models.sh/Dockerfile) measured faster on CPU
    # and equally fast on CUDA, with zero failures across everything tested — so the retry
    # machinery was removed rather than kept as speculative insurance. This check stays only so
    # a future regression is visible in the journal instead of just a silent gap in a reply.
    if _is_bad_audio(samples):
        log.error("Synthesis produced unusable audio for %r", req.text[:120])
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return Response(content=buffer.getvalue(), media_type="audio/wav")
