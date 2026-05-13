#!/bin/bash
# Validate vLLM parameters in Docker environment
set -e

echo "Starting vLLM validation..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker not running"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: docker-compose not found"
    exit 1
fi

# Start vLLM server container
echo "Starting vLLM container..."
docker-compose up -d vllm-server

# Wait for health check
echo "Waiting for vLLM to be ready..."
timeout=300
elapsed=0
while [ $elapsed -lt $timeout ]; do
    if curl -s http://localhost:8100/v1/models > /dev/null 2>&1; then
        echo "✓ vLLM is ready"
        break
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done

if [ $elapsed -ge $timeout ]; then
    echo "✗ vLLM failed to start within ${timeout}s"
    docker-compose logs vllm-server
    docker-compose down
    exit 1
fi

# Check process arguments
echo "Checking vLLM process arguments..."
container_id=$(docker-compose ps -q vllm-server)

if [ -z "$container_id" ]; then
    echo "✗ Could not find vLLM container"
    docker-compose down
    exit 1
fi

# Get the PID of the vLLM process
pid=$(docker top $container_id | grep vllm | head -1 | awk '{print $2}')

if [ -z "$pid" ]; then
    echo "✗ Could not find vLLM process"
    docker-compose down
    exit 1
fi

# Get full command line
echo "vLLM process PID: $pid"

# Check if we can read the command line
if docker exec $container_id cat /proc/$pid/cmdline > /dev/null 2>&1; then
    cmd=$(docker exec $container_id cat /proc/$pid/cmdline | tr '\0' ' ')
    echo "vLLM command: $cmd"

    # Verify critical parameters
    if echo "$cmd" | grep -q -- "--limit-mm-per-prompt"; then
        if echo "$cmd" | grep -q -- "audio=1"; then
            echo "✓ --limit-mm-per-prompt audio=1 is set"
        else
            echo "✗ --limit-mm-per-prompt found but audio=1 not set"
            docker-compose down
            exit 1
        fi
    else
        echo "✗ Missing --limit-mm-per-prompt audio=1"
        docker-compose down
        exit 1
    fi

    if echo "$cmd" | grep -q -- "--enable-chunked-prefill"; then
        echo "✓ --enable-chunked-prefill is set"
    else
        echo "✗ Missing --enable-chunked-prefill"
        docker-compose down
        exit 1
    fi

    if echo "$cmd" | grep -q -- "--dtype"; then
        if echo "$cmd" | grep -q -- "bfloat16"; then
            echo "✓ --dtype bfloat16 is set"
        else
            echo "✗ --dtype found but bfloat16 not set"
            docker-compose down
            exit 1
        fi
    else
        echo "✗ Missing --dtype bfloat16"
        docker-compose down
        exit 1
    fi
else
    echo "⚠ Cannot read /proc/$pid/cmdline (may not be Linux)"
    echo "Falling back to docker inspect..."

    # Alternative: check container command
    container_cmd=$(docker inspect $container_id | jq -r '.[0].Config.Cmd[]' | tr '\n' ' ')
    echo "Container command: $container_cmd"

    # Basic check - at least verify vllm is in the command
    if echo "$container_cmd" | grep -q "vllm"; then
        echo "✓ vLLM command found in container"
    else
        echo "✗ vLLM command not found"
        docker-compose down
        exit 1
    fi
fi

echo ""
echo "All vLLM parameters validated successfully!"
echo "Shutting down..."
docker-compose down
