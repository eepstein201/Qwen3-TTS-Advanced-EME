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

# Test classes will be added in subsequent tasks
