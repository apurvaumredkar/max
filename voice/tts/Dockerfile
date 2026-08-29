# Kokoro-82M (ONNX, fp32, CUDA) text-to-speech microservice.
#
# Base image: dustynv/onnxruntime ships the only working aarch64+CUDA onnxruntime-gpu wheel
# for Jetson/L4T. Tag matches this host's L4T R36.4.7 / CUDA 12.6.
# Weights are not baked in — download_models.sh + docker-compose.yml mount them at /models.
FROM dustynv/onnxruntime:1.20.2-r36.4.0

RUN apt-get update && \
    apt-get install -y --no-install-recommends espeak-ng && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --index-url: the base image points pip at a Jetson wheel mirror unreachable from here.
# --no-deps for kokoro-onnx: the base's onnxruntime is registered as "onnxruntime-gpu", so pip
# doesn't see its `onnxruntime>=1.20.1` as satisfied and silently installs a CPU-only wheel over
# it. Its other deps are listed explicitly; numpy is left alone (the base's ORT binding is built
# against numpy 1.x and breaks under 2.x, and kokoro-onnx is pure Python so 1.26 is fine).
RUN pip install --no-cache-dir --index-url https://pypi.org/simple --no-deps kokoro-onnx && \
    pip install --no-cache-dir --index-url https://pypi.org/simple \
        "espeakng-loader>=0.2.4" "phonemizer>=3.4.0" \
        fastapi "uvicorn[standard]" soundfile

WORKDIR /app
COPY server.py /app/server.py

# fp32, not fp16 (deterministically emits all-NaN audio for many real sentences on both EPs) or
# int8 (needs ConvInteger/MatMulInteger kernels this build lacks on both EPs). fp32 also measured
# faster on CPU and equal on CUDA (RTF ~0.13-0.24).
# Force plain CUDA: kokoro-onnx otherwise also requests TensorRT, whose EP fails session init on
# this export (SplitToSequence has no static shape) rather than falling back.
ENV MODEL_DIR=/models \
    MODEL_PATH=/models/kokoro-v1.0.onnx \
    VOICES_PATH=/models/voices-v1.0.bin \
    DEFAULT_VOICE=am_echo \
    ONNX_PROVIDER=CUDAExecutionProvider

EXPOSE 8880

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8880"]
