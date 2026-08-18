#!/bin/bash
# Validate the suite inside a hardened Linux container.
# Run from anywhere; resolves the project root itself.
#
# A Linux container shares the host kernel: this proves LINUX behavior only.
# It does not validate macOS, Windows, or CUDA.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Docker Desktop ships its credential helper outside the default PATH on macOS.
if [ -d /Applications/Docker.app/Contents/Resources/bin ]; then
    PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
    export PATH
fi

COMPOSE="docker compose -p qwen3-tts-test -f docker-compose.test.yml"

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running. Start Docker Desktop and re-run."
    exit 1
fi

echo "=== Validating compose model ==="
$COMPOSE config --quiet

echo "=== Building test image (arch: $(uname -m)) ==="
$COMPOSE build suite

echo "=== Static gates ==="
$COMPOSE run --rm -T gates

echo "=== Full non-E2E suite ==="
$COMPOSE run --rm -T suite

echo "=== Batch runner ==="
$COMPOSE run --rm -T batches

$COMPOSE down --remove-orphans

echo "=== Docker validation complete ==="
