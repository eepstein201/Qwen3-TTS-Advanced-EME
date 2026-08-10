#!/usr/bin/env bash
# CUDA Docker E2E gate -- the Docker-specific surface that the native Colab
# path (colab_notebook.ipynb) does NOT cover:
#   * `docker build -f Dockerfile` succeeds on a real nvidia/cuda base
#   * the container ENTRYPOINT (`tts server start --public`) boots the server
#   * the GPU is visible inside the container (nvidia-container-toolkit works)
#   * /health responds (the Dockerfile HEALTHCHECK target)
#   * the Dockerfile's symlinked UserFiles paths resolve (config / voice_prompts)
#   * the declared HF cache volume mounts
#
# This is a MANUAL / pre-release gate: it builds the ~3.5GB production image and
# downloads models on first run, so it is intentionally NOT wired into per-push
# CI. Generation *correctness* is already covered by the native Colab/CUDA path;
# this gate only covers what only a container can exercise.
#
# Run on any CUDA host with Docker + nvidia-container-toolkit:
#   bash scripts/validate_docker_cuda.sh
# In Colab Pro (CUDA + Docker available):
#   !bash scripts/validate_docker_cuda.sh
#
# Env:
#   TTS_TEST_TOKEN  optional bearer token; if set, also hits an authed endpoint
#   IMAGE_TAG       image tag to build (default qwen3-tts:cuda)
#   READY_TIMEOUT   seconds to wait for /health (default 900; first run downloads models)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

IMAGE_TAG="${IMAGE_TAG:-qwen3-tts:cuda}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"
CONTAINER="qwen3-tts-cuda-gate"
HEALTH_URL="http://127.0.0.1:5123/health"

log()  { printf '\033[1;34m[cuda-gate]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[cuda-gate FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

cleanup() {
  if docker inspect "$CONTAINER" >/dev/null 2>&1; then
    log "removing container $CONTAINER"
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# --- preflight -----------------------------------------------------------------
log "preflight: docker daemon"
docker info >/dev/null 2>&1 || fail "docker daemon is not running"

log "preflight: host GPU (nvidia-smi)"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found -- run this on a CUDA host"
nvidia-smi -L >/dev/null 2>&1 || fail "nvidia-smi -L failed -- no GPU detected on the host"

log "preflight: GPU passthrough into a container (nvidia-container-toolkit)"
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi -L >/dev/null 2>&1 \
  || fail "docker cannot see the GPU -- install nvidia-container-toolkit on the host"

# --- build the production image ------------------------------------------------
log "building production image $IMAGE_TAG (pulls nvidia/cuda + apt + pip deps)..."
docker build -f Dockerfile -t "$IMAGE_TAG" .

# --- run with GPU + the HF cache volume ----------------------------------------
# Config and voice_prompts are intentionally NOT bind-mounted: the image bakes
# config.json in and symlinks the UserFiles paths, and this gate verifies those
# Dockerfile-declared symlinks resolve. Only the HF cache is mounted, so models
# are not re-downloaded on every run.
log "starting container $CONTAINER (GPU, port 5123, HF cache volume)..."
docker run -d --name "$CONTAINER" \
  --gpus all \
  -p 127.0.0.1:5123:5123 \
  -e TTS_DISABLE_RATE_LIMITING=1 \
  -v qwen3-tts-hf-cache:/root/.cache/huggingface \
  "$IMAGE_TAG" >/dev/null

# --- wait for /health (fail fast if the container dies) ------------------------
log "waiting for $HEALTH_URL (up to ${READY_TIMEOUT}s; first run downloads models)..."
ok=""
i=0
while [ "$i" -lt "$READY_TIMEOUT" ]; do
  if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then ok=1; break; fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != "true" ]; then
    docker logs "$CONTAINER" --tail 60 >&2 || true
    fail "container exited before /health came up (logs above)"
  fi
  i=$((i + 1))
  sleep 1
done
if [ -z "$ok" ]; then
  docker logs "$CONTAINER" --tail 60 >&2 || true
  fail "timed out waiting for /health after ${READY_TIMEOUT}s (logs above)"
fi
log "/health returned 200"

# --- Docker-specific checks ----------------------------------------------------
log "check: GPU visible inside the container"
docker exec "$CONTAINER" nvidia-smi -L >/dev/null \
  || fail "nvidia-smi not visible inside the container (GPU passthrough failed)"

log "check: symlinked config resolves (/root/Qwen3-TTS_UserFiles/config.json)"
docker exec "$CONTAINER" test -f /root/Qwen3-TTS_UserFiles/config.json \
  || fail "config.json not reachable at the Dockerfile's symlinked UserFiles path"

log "check: symlinked voice_prompts resolves (/root/Qwen3-TTS_UserFiles/voice_prompts)"
docker exec "$CONTAINER" test -d /root/Qwen3-TTS_UserFiles/voice_prompts \
  || fail "voice_prompts dir not reachable at the symlinked UserFiles path"

log "check: declared HF cache volume mounted"
docker exec "$CONTAINER" test -d /root/.cache/huggingface \
  || fail "/root/.cache/huggingface volume not mounted"

# --- optional authed smoke -----------------------------------------------------
if [ -n "${TTS_TEST_TOKEN:-}" ]; then
  log "smoke: authed GET /models (TTS_TEST_TOKEN set)"
  curl -sf http://127.0.0.1:5123/models \
    -H "Authorization: Bearer $TTS_TEST_TOKEN" >/dev/null \
    || fail "authed /models returned non-200 (token or server issue)"
  log "authed /models OK"
else
  log "skip: authed smoke (set TTS_TEST_TOKEN to enable) -- generation correctness is covered by the native Colab path"
fi

log "ALL CHECKS PASSED -- production CUDA image builds and serves."
