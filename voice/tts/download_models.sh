#!/usr/bin/env bash
# Idempotent: downloads the Kokoro-82M fp32 ONNX model + voice pack into voice/tts/models/,
# which docker-compose.yml bind-mounts into the container at /models. Run once per clone
# (or whenever the models are missing) via scripts/setup-tts.sh.
#
# fp32, not fp16 or int8: int8 needs ConvInteger/MatMulInteger kernels this onnxruntime build
# doesn't implement on EITHER execution provider (confirmed: NOT_IMPLEMENTED on CPU too, not
# just CUDA). fp16 does run, but is numerically unstable — it deterministically produces
# all-NaN output for a large fraction of real sentences, on both CPU and CUDA, and measured
# CPU inference *slower* than fp32 despite the smaller file (extra fp16<->fp32 cast overhead
# with no offsetting benefit on this hardware/build). fp32 measured faster on CPU and just as
# fast on CUDA (RTF ~0.13-0.24), with zero silent/NaN failures across everything tested.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
RELEASE_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"

mkdir -p "$MODELS_DIR"

fetch() {
    local filename="$1"
    local dest="$MODELS_DIR/$filename"
    if [ -f "$dest" ]; then
        echo "already present: $filename"
        return
    fi
    echo "downloading $filename..."
    curl -L --retry 5 --retry-delay 2 --retry-all-errors --fail \
        -o "$dest" "$RELEASE_URL/$filename"
}

fetch "kokoro-v1.0.onnx"
fetch "voices-v1.0.bin"

echo "models ready in $MODELS_DIR"
