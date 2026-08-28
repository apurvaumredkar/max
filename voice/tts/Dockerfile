# Kokoro-82M (ONNX, fp32, CUDA) text-to-speech microservice.
#
# onnxruntime-gpu has no official aarch64+CUDA pip wheel, so this builds on top of
# dustynv/onnxruntime (the jetson-containers project), which already ships a working
# onnxruntime-gpu wheel with the CUDA execution provider for Jetson/L4T. Model weights are
# NOT baked into the image — see download_models.sh and docker-compose.yml, which bind-mount
# them into /models at runtime.
#
# Tag chosen to match this host's L4T R36.4.7 (JetPack 6.2.1, driver-reported CUDA 12.6):
# dustynv only publishes r36.4.0-family tags (no plain "r36.4.0"), and 1.20.2-r36.4.0 was
# built against CUDA 12.6 — the cu128-24.04 variant targets a newer CUDA than this driver
# reports supporting, so it's avoided here.
FROM dustynv/onnxruntime:1.20.2-r36.4.0

RUN apt-get update && \
    apt-get install -y --no-install-recommends espeak-ng && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# The base image points PIP_INDEX_URL at pypi.jetson-ai-lab.dev (a Jetson-specific wheel
# mirror for CUDA-heavy packages like onnxruntime itself) which isn't reachable from this
# network and doesn't carry these pure-Python packages anyway — use plain PyPI for them.
#
# IMPORTANT: the base image's onnxruntime is registered under the pip distribution name
# "onnxruntime-gpu" (not "onnxruntime" — `pip show onnxruntime` reports "not found" even
# though `import onnxruntime` works and lists CUDAExecutionProvider/TensorrtExecutionProvider).
# A plain `pip install kokoro-onnx` can't see that as satisfying its `onnxruntime>=1.20.1`
# dependency, and silently pulls the generic CPU-only PyPI wheel over it — quietly losing GPU
# acceleration. So kokoro-onnx is installed with --no-deps, and its actual other dependencies
# (everything except onnxruntime) are installed explicitly.
#
# Also leave numpy untouched: the base image's onnxruntime pybind11 binding is compiled
# against NumPy 1.x's C-API and segfaults/ImportErrors under NumPy 2.x. kokoro-onnx declares
# numpy>=2.0.2 but is pure Python (no compiled numpy usage of its own), so it runs fine on
# the base image's numpy 1.26 despite the (unenforced, since we use --no-deps) constraint.
RUN pip install --no-cache-dir --index-url https://pypi.org/simple --no-deps kokoro-onnx && \
    pip install --no-cache-dir --index-url https://pypi.org/simple \
        "espeakng-loader>=0.2.4" "phonemizer>=3.4.0" \
        fastapi "uvicorn[standard]" soundfile

WORKDIR /app
COPY server.py /app/server.py

# Model precision: fp32, not fp16 or int8. int8 needs ConvInteger/MatMulInteger kernels this
# onnxruntime build doesn't implement on EITHER execution provider. fp16 runs but is
# numerically unstable — deterministically produces all-NaN output for a large fraction of
# real sentences on both CPU and CUDA (confirmed independently, not shape/length dependent —
# e.g. "The quick brown fox jumps over the lazy dog." reliably failed, "Hello there." never
# did). fp32 measured faster than fp16 on CPU and equally fast on CUDA (RTF ~0.13-0.24), with
# zero failures across everything tested — a strictly better choice on this hardware, not a
# speed/reliability tradeoff.
#
# kokoro-onnx auto-detects an installed "onnxruntime-gpu" distribution and, left to its own
# defaults, requests *every* available provider including TensorRT. TensorRT's EP fails hard
# during session init on this model export (SplitToSequence has no static shape — a known
# TensorRT/dynamic-shape limitation, not something ORT falls back gracefully from). Force
# plain CUDA instead.
ENV MODEL_DIR=/models \
    MODEL_PATH=/models/kokoro-v1.0.onnx \
    VOICES_PATH=/models/voices-v1.0.bin \
    DEFAULT_VOICE=am_echo \
    ONNX_PROVIDER=CUDAExecutionProvider

EXPOSE 8880

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8880"]
