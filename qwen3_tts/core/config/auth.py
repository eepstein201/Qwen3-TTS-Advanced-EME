#!/usr/bin/env python3
"""Server auth token read helpers.

No torch, numpy, or heavy imports.

TOKEN_FILE / _LEGACY_TOKEN_FILE (defined in paths.py) are resolved via a
lazy per-call import from ``qwen3_tts.core.config`` (the package facade) —
see qwen3_tts/core/config/__init__.py for the rationale.
"""

import os


def read_auth_token():
    """Read the server auth token from TOKEN_FILE.

    Falls back to legacy path (~/.voice_server_token) with a deprecation warning.

    Returns:
        The token string, or None if file doesn't exist.
    """
    from qwen3_tts.core.config import _LEGACY_TOKEN_FILE, TOKEN_FILE

    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            return f.read().strip()
    # Backward compat: check legacy location
    if os.path.exists(_LEGACY_TOKEN_FILE):
        import logging

        logging.getLogger("tts").warning(
            "Reading auth token from legacy path %s — "
            "restart the server to migrate to %s",
            _LEGACY_TOKEN_FILE,
            TOKEN_FILE,
        )
        with open(_LEGACY_TOKEN_FILE) as f:
            return f.read().strip()
    return None


def auth_headers():
    """Return HTTP headers dict with Bearer auth token, or empty dict."""
    from qwen3_tts.core.config import read_auth_token as _read_auth_token

    token = _read_auth_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}
