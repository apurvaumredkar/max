#!/usr/bin/env bash
# One-time (or re-run anytime) setup for the Kokoro TTS Docker service:
# downloads model weights, then builds and starts the container.
#
#   bash scripts/setup-tts.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TTS_DIR="$REPO_ROOT/voice/tts"

bash "$TTS_DIR/download_models.sh"

docker compose -f "$TTS_DIR/docker-compose.yml" up -d --build

echo "waiting for the TTS service to become healthy..."
for _ in $(seq 1 60); do
    if curl -sf http://127.0.0.1:8880/health > /dev/null; then
        echo "TTS service is up: http://127.0.0.1:8880"
        exit 0
    fi
    sleep 2
done

echo "TTS service did not become healthy in time — check: docker logs max-kokoro-tts" >&2
exit 1
