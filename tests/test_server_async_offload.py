"""Blocking prompt-endpoint handlers must run off the FastAPI event loop.

The prompt endpoints (/prompts, /delete-prompt, /rename-prompt, /preview-prompt,
/prompt-details) call synchronous handlers that perform filesystem I/O
(os.listdir/remove/rename, save_config, reading .wav files). Running those
directly inside an async endpoint blocks the event loop and stalls in-flight
streaming responses. They must be dispatched via ``asyncio.to_thread``, matching
the pattern used by /transcribe and /create-voice-prompt. (/load-model's
handler is async since the #192 warm-up serialization and is awaited directly.)

See docs/reviews/e2e-review-2026-07-01.md (Phase 5 FastAPI review, HIGH).
"""

import asyncio

import pytest

try:
    from unittest.mock import patch

    from fastapi.testclient import TestClient  # noqa: F401

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False


def _make_offload_recorder(response):
    """Return (state, handler) where handler records its event-loop context.

    ``asyncio.get_running_loop()`` succeeds only when called on the thread that
    runs the event loop. A handler dispatched via ``asyncio.to_thread`` runs on a
    worker thread, where it raises ``RuntimeError`` -> off the loop.
    """
    state = {"called": False, "off_loop": None}

    def handler(*args, **kwargs):
        state["called"] = True
        try:
            asyncio.get_running_loop()
            state["off_loop"] = False
        except RuntimeError:
            state["off_loop"] = True
        return response

    return state, handler


# (app symbol, client method, path, request kwargs, mocked handler response)
# The mocked responses must satisfy each route's response_model contract
# (GEN-2) or FastAPI raises ResponseValidationError before the assertions run.
_PROMPT_ENDPOINTS = [
    (
        "handle_list_prompts",
        "get",
        "/prompts",
        {},
        {"prompts": [], "total": 0, "offset": 0, "limit": 0},
    ),
    (
        "handle_delete_prompt",
        "post",
        "/delete-prompt",
        {"json": {"name": "x"}},
        {"status": "deleted", "name": "x", "files_removed": []},
    ),
    (
        "handle_rename_prompt",
        "post",
        "/rename-prompt",
        {"json": {"old_name": "x", "new_name": "y"}},
        {
            "status": "renamed",
            "old_name": "x",
            "new_name": "y",
            "files_renamed": [],
        },
    ),
    ("handle_preview_prompt", "get", "/preview-prompt?name=x", {}, {"ok": True}),
    (
        "handle_prompt_details",
        "get",
        "/prompt-details?name=x",
        {},
        {
            "name": "x",
            "formats": [".wav"],
            "size_bytes": 1,
            "created": 0.0,
            "is_default": False,
        },
    ),
]


@pytest.mark.skipif(not HAS_FASTAPI, reason="requires fastapi")
@pytest.mark.parametrize("symbol,method,path,kwargs,response", _PROMPT_ENDPOINTS)
def test_prompt_handler_runs_off_event_loop(
    fastapi_client, symbol, method, path, kwargs, response
):
    """Each prompt endpoint must dispatch its handler off the event loop thread."""
    state, handler = _make_offload_recorder(response)
    with patch(f"qwen3_tts.server.app.{symbol}", handler):
        getattr(fastapi_client, method)(path, **kwargs)
    assert state["called"], f"{symbol} was never invoked (request rejected before handler)"
    assert state["off_loop"] is True, f"{symbol} ran ON the event loop; expected asyncio.to_thread offload"
