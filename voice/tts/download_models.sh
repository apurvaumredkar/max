#!/usr/bin/env bash
# Idempotent: downloads the Kokoro-82M fp32 ONNX model + voice pack into voice/tts/models/,
# bind-mounted into the container at /models. Run via scripts/setup-tts.sh.
# fp32, not fp16 (all-NaN output on many sentences) or int8 (kernels this ORT build lacks).
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
