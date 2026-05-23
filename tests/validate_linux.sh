#!/bin/bash
# Validate tests on Linux (CPU or GPU/CUDA)
# Run from project root on Linux machine

set -e

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_PATH="/tmp/qwen3_tts_test_$$"

echo "=== Linux Test Validation ==="
echo "Python version: $PYTHON_VERSION"
echo "venv path: $VENV_PATH"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python$PYTHON_VERSION -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

# Install test dependencies
echo "Installing test dependencies..."
pip install --upgrade pip --quiet
pip install -e ".[test]" --quiet

# Check for CUDA if available
echo ""
echo "Checking GPU/CUDA availability..."
python -c "
import sys
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('PyTorch not installed (CPU-only tests will run)')
except Exception as e:
    print(f'GPU check error: {e}')
"

# Run all test batches
echo ""
echo "Running all test batches..."
python tests/run_batches.py

# Cleanup
echo ""
echo "Cleaning up..."
deactivate
rm -rf "$VENV_PATH"

echo "=== Linux validation complete ==="
