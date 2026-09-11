"""Blocking prompt-endpoint handlers must run off the FastAPI event loop.

The prompt endpoints (/prompts, /delete-prompt, /rename-prompt, /preview-prompt,
/prompt-details) call synchronous handlers that perform filesystem I/O
(os.listdir/remove/rename, save_config, reading .wav files). Running those
directly inside an async endpoint blocks the event loop and stalls in-flight
streaming responses. They must be dispatched via ``asyncio.to_thread``.
(/load-model, /transcribe and /create-voice-prompt handlers are async since
the #192 serialization work and are awaited directly.)

See docs/reviews/e2e-review-2026-07-01.md (Phase 5 FastAPI review, HIGH).
"""

import asyncio
import unittest

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


@unittest.skipUnless(HAS_FASTAPI, "requires fastapi")
class TestStatsModelsOffload(unittest.TestCase):
    """The /stats and /models handlers must run off the event loop thread.

    Both are sync handlers whose first call lazily imports the engine package
    (voice_prompt_cache_info / get_asr_model_info) and, for /stats, probes
    torch/MLX memory — blocking work that stalls in-flight streaming
    responses. They are the last sync model/stats siblings still invoked
    directly by the async route wrappers; the /load-model-era offload
    convention (asyncio.to_thread, e.g. app_models.py save_config /
    unload_model_cleanup) applies. Asserted at the source level per the
    test_generation_offload.py convention (driving /stats through the app
    requires the conftest client fixture, which the unittest batch runner
    cannot fire).
    """

    def _app_source(self) -> str:
        import inspect

        from qwen3_tts.server import app as app_module

        return inspect.getsource(app_module)

    def test_stats_handler_is_offloaded(self):
        src = self._app_source()
        self.assertRegex(
            src,
            r"asyncio\.to_thread\(\s*handle_stats\b",
            "the /stats route must dispatch handle_stats via asyncio.to_thread",
        )

    def test_list_models_handler_is_offloaded(self):
        src = self._app_source()
        self.assertRegex(
            src,
            r"asyncio\.to_thread\(\s*handle_list_models\b",
            "the /models route must dispatch handle_list_models via "
            "asyncio.to_thread",
        )
