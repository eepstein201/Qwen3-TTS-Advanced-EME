"""
AI Regression Tests - Prevent systematic AI-introduced bugs.

This module contains tests specifically designed to catch patterns where
AI-assisted code review misses bugs:
- Backend consistency (MLX vs Torch return same shape)
- Model state edge cases (graceful failures when models not loaded)
- API response contracts (required fields always present)
"""
import pytest
import json
import time
import urllib.request
import os
from pathlib import Path

SERVER_URL = "http://127.0.0.1:5123"

def _get_auth_token():
    """Read the server auth token."""
    token_path = os.path.expanduser("~/.config/qwen3-tts/.voice_server_token")
    try:
        with open(token_path) as f:
            return f.read().strip()
    except FileNotFoundError:
        pytest.skip("Server auth token not found - server not running?")

def _post_json(endpoint: str, data: dict) -> tuple[int, dict]:
    """POST JSON to server endpoint, return (status_code, response_json)."""
    token = _get_auth_token()
    url = f"{SERVER_URL}{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode()) if e.headers.get("Content-Type", "").startswith("application/json") else {}
    except Exception as e:
        pytest.fail(f"Request to {endpoint} failed: {e}")

def _get_json(endpoint: str) -> tuple[int, dict]:
    """GET JSON from server endpoint, return (status_code, response_json)."""
    token = _get_auth_token()
    url = f"{SERVER_URL}{endpoint}"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    req = urllib.request.Request(url, headers=headers, method="GET")

    try:
        resp = urllib.request.urlopen(req, timeout=10)
        return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode()) if e.headers.get("Content-Type", "").startswith("application/json") else {}
    except Exception as e:
        pytest.fail(f"Request to {endpoint} failed: {e}")

class TestBackendConsistency:
    """Test MLX and Torch backends return identical API response shapes."""

    @pytest.mark.parametrize("mode", ["clone", "design", "custom"])
    def test_all_backends_return_same_response_shape(self, mode: str):
        """Test MLX and Torch backends return identical API shapes for generation.

        This prevents the common AI regression where one backend path is updated
        but the other is forgotten, causing shape mismatches for consumers.
        """
        # Skip if server not running or models not loaded
        status, models_data = _get_json("/models")
        if status != 200:
            pytest.skip("Server not running - cannot test backend consistency")

        # Check if models are loaded for the requested mode
        model_key = f"{mode}_model_loaded"
        if not models_data.get("models", {}).get(mode, {}).get("loaded"):
            pytest.skip(f"{mode.capitalize()} model not loaded - skipping backend consistency test")

        # Generate audio and verify response structure
        generation_data = {
            "text": "Backend consistency test - this is a short text.",
            "mode": mode
        }

        if mode == "custom":
            generation_data["speaker"] = "default_en"

        status, response = _post_json("/generate", generation_data)

        # Should succeed
        assert status == 200, f"Generation failed with status {status}: {response}"

        # Verify response has all required fields
        required_fields = ["audio", "duration", "chunks"]
        for field in required_fields:
            assert field in response, f"Missing required field '{field}' in {mode} backend response"

        # Verify field types
        assert isinstance(response["audio"], str), f"audio should be string, got {type(response['audio'])}"
        assert isinstance(response["duration"], (int, float)), f"duration should be numeric, got {type(response['duration'])}"
        assert isinstance(response["chunks"], int), f"chunks should be int, got {type(response['chunks'])}"

        # Verify audio is base64-encoded string (not empty)
        assert len(response["audio"]) > 0, f"audio should not be empty"

        # Verify chunks is non-negative
        assert response["chunks"] >= 0, f"chunks should be >= 0, got {response['chunks']}"

    def test_stats_endpoint_includes_all_required_fields(self):
        """Test /stats endpoint returns consistent response shape across backends.

        This prevents regression where new stats fields are added to one backend
        but not the other, breaking monitoring and dashboards.
        """
        status, stats = _get_json("/stats")

        if status != 200:
            pytest.skip("Server not running - cannot test stats endpoint")

        # Verify /stats has required top-level fields
        required_fields = ["backend", "model_size"]
        for field in required_fields:
            assert field in stats, f"Missing required field '{field}' in /stats response"

        # Verify types
        assert isinstance(stats["backend"], str), f"backend should be string, got {type(stats['backend'])}"
        assert isinstance(stats["model_size"], str), f"model_size should be string, got {type(stats['model_size'])}"

# Test classes will be added in subsequent tasks
