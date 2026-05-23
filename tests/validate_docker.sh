#!/bin/bash
# Validate tests in Docker container
# Run from project root

set -e

echo "=== Docker Test Validation ==="
echo "Building test image..."

docker build -t qwen3-tts-test - <<'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY qwen3_tts/ qwen3_tts/
COPY tests/ tests/

# Install test dependencies
RUN pip install --upgrade pip && \
    pip install -e ".[test]" --quiet

# Run all test batches
RUN python tests/run_batches.py

CMD ["python", "-m", "unittest", "discover", "-v", "tests/"]
EOF

echo "Running tests..."
docker run --rm qwen3-tts-test

echo "=== Docker validation complete ==="
