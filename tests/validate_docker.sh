#!/bin/bash
# Validate tests in Docker container.
# Run from project root. Uses Dockerfile.test (multi-stage, non-root testuser).

set -e

# Resolve project root regardless of where this is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=== Docker Test Validation ==="

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running. Start Docker Desktop and re-run."
    exit 1
fi

echo "Building test image with Dockerfile.test ..."
docker build -f Dockerfile.test -t qwen3-tts:test .

echo "Running batch suite inside container ..."
docker run --rm qwen3-tts:test python tests/run_batches.py

echo "=== Docker validation complete ==="
