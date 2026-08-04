"""H4: batch ``/generate`` finally must respect generation-state ownership.

``handle_generate`` (the batch ``/generate`` path) used to unconditionally
reset ``generation_state`` in its ``finally`` block — even when every item
was a cache hit (no inference ran, so the batch never set ``active``) or
when a concurrent streaming request had since taken ownership of
``generation_state``.  This clobbered an in-flight stream's ``active`` /
``generation_id``, making the stream look inactive to ``/queue-status``
and ``/cancel-generation``.

The fix mirrors the streaming-path guard at ``app_generation.py:729``:
stamp a per-batch ``batch_gen_id`` and only reset ``generation_state`` in
the finally when this batch still owns it (``generation_id == batch_gen_id``).

Run: python -m unittest tests.test_batch_generation_state_ownership -v
"""

import asyncio
import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

try:
    import numpy as np  # noqa: F401
    import soundfile  # noqa: F401

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False  # noqa: F811

_skip = unittest.skipUnless(_HAS_DEPS, "requires numpy + soundfile")

_APP_GENERATION = "qwen3_tts.server.app_generation"
_ENGINE = "qwen3_tts.core.engine"


def _make_state():
    """Minimal app.state for exercising ``handle_generate``.

    Mirrors ``_setup_fastapi_app_state`` but uses a SimpleNamespace so the
    test is self-contained (no real FastAPI app / lifespan).
    """
    state = SimpleNamespace()
    state.auth_token = "test_token"  # nosec B105
    state.models = {
        "clone": MagicMock(),
        "design": MagicMock(),
        "custom": MagicMock(),
    }
    state.model_load_times = {}
    state.model_load_errors = {"clone": None, "design": None, "custom": None}
    state.generation_state = {
        "active": False,
        "start_time": 0.0,
        "text_length": 0,
        "mode": "",
        "batch_index": 0,
        "batch_total": 0,
        "chunk_index": 0,
        "chunk_total": 0,
        "generation_id": None,
        "cancelled": False,
    }
    state.request_queue = set()
    state.request_queue_lock = threading.Lock()
    # generation_lock is an asyncio context manager; a no-op AsyncMock
    # mirrors _setup_fastapi_app_state from voice_test_helpers.
    _glock = AsyncMock()
    _glock.__aenter__.return_value = None
    _glock.__aexit__.return_value = None
    state.generation_lock = _glock
    state.pending_requests = []
    state.last_activity = 0
    state.models_loaded = threading.Event()
    state.models_loaded.set()
    state.gen_cache = {}
    state.gen_cache_lock = threading.Lock()
    state.inference_lock = asyncio.Lock()
    state.eta_cache = {"median_rate": None, "last_updated": 0}
    state.eta_cache_lock = threading.Lock()
    state.shutdown_timer = None
    state.server_config = {
        "security": {"max_text_length": 50000, "max_batch_size": 20},
        "auto_shutdown_minutes": 0,
        "vllm": {"enabled": False, "fallback_to_torch": True},
    }
    return state


def _make_request(state):
    """Build a MagicMock request whose ``app.state`` points to *state*."""
    request = MagicMock()
    request.app.state = state
    request.headers = {"accept": "application/json"}
    return request


def _prime_cache_for_design(text, state):
    """Prime ``gen_cache`` so *text* in design mode is a pre-lock cache hit.

    Returns the temp file path (caller cleans up).
    """
    from qwen3_tts.server.validation import _gen_cache_key

    gen_params = {
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.05,
        "max_new_tokens": 2048,
    }
    cache_key = _gen_cache_key(
        text,
        "design",
        gen_params,
        prompt_file=None,
        voice_description="friendly",
        speaker=None,
        instruct="",
        language="English",
        x_vector_only_mode=False,
        max_chunk_chars=None,
        seed_lock_chunks=False,
    )
    cache_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    cache_file.write(b"fake-audio-data")
    cache_file.close()
    state.gen_cache[cache_key] = {
        "main_file": cache_file.name,
        "sample_rate": 24000,
        "timestamp": time.time(),
        "chunks": 1,
        "seed": 42,
    }
    return cache_file.name


@_skip
class TestBatchGenerationStateOwnership(unittest.TestCase):
    """The batch ``/generate`` finally must only reset state it owns."""

    def setUp(self):
        # Each test gets a fresh event loop via asyncio.run; nothing to clean
        # between tests beyond what tearDown handles.
        pass

    def tearDown(self):
        # Nothing global to reset — each test builds its own state.
        pass

    # -- Test 1: all-cache-hit must not clobber an in-flight stream --------

    def test_all_cache_hit_does_not_clobber_inflight_stream(self):
        """An all-cache-hit batch never runs inference, so it must NOT reset
        ``generation_state``.  Pre-fix the finally unconditionally resets
        ``active=False``, clobbering a concurrent stream that set
        ``active=True, generation_id="streamXYZ"``.
        """

        async def run():
            from qwen3_tts.server.app_generation import handle_generate
            from qwen3_tts.server.validation import GenerateRequest

            state = _make_state()
            # Simulate a concurrent stream owning generation_state.
            state.generation_state["active"] = True
            state.generation_state["generation_id"] = "streamXYZ"

            cache_path = _prime_cache_for_design("cached text", state)

            req = GenerateRequest(
                text="cached text", mode="design", voice_description="friendly"
            )
            request = _make_request(state)
            try:
                with patch(
                    f"{_APP_GENERATION}._check_memory_available",
                    return_value=(True, 4096),
                ), patch(
                    "qwen3_tts.server.validation._validate_generation_request"
                ):
                    result = await handle_generate(
                        request=request,
                        state=state,
                        req=req,
                        security={"max_text_length": 50000, "max_batch_size": 20},
                        config_provider=None,
                    )
                # Sanity: the batch returned a cache hit.
                self.assertIn("results", result)
                self.assertEqual(len(result["results"]), 1)

                # The concurrent stream's state must survive.
                self.assertTrue(
                    state.generation_state["active"],
                    "All-cache-hit batch clobbered active=True of an "
                    "in-flight stream.",
                )
                self.assertEqual(
                    state.generation_state["generation_id"],
                    "streamXYZ",
                    "All-cache-hit batch clobbered generation_id of an "
                    "in-flight stream.",
                )
            finally:
                if os.path.exists(cache_path):
                    os.unlink(cache_path)

        asyncio.run(run())

    # -- Test 2: batch stamps a non-None generation_id ---------------------

    def test_batch_sets_generation_id(self):
        """When the batch reaches the inference path it must stamp a
        non-None ``generation_id`` onto ``generation_state`` so the finally
        can check ownership.  Captured from inside the mocked inference
        callback (before the finally resets it).
        """

        async def run():
            from qwen3_tts.server.app_generation import handle_generate
            from qwen3_tts.server.validation import GenerateRequest

            state = _make_state()
            captured_gen_id = [None]
            fake_wav = np.zeros(500, dtype=np.float32)

            def mock_inference(model, text, **kwargs):
                # Snapshot the generation_id that was set just before
                # inference — this is the batch's ownership stamp.
                captured_gen_id[0] = state.generation_state.get("generation_id")
                return fake_wav, 24000

            req = GenerateRequest(text="hello world", mode="design",
                                  voice_description="friendly")
            request = _make_request(state)
            with patch(
                f"{_APP_GENERATION}._check_memory_available",
                return_value=(True, 4096),
            ), patch(
                "qwen3_tts.server.validation._validate_generation_request"
            ), patch(
                f"{_ENGINE}.run_inference", side_effect=mock_inference
            ):
                result = await handle_generate(
                    request=request,
                    state=state,
                    req=req,
                    security={"max_text_length": 50000, "max_batch_size": 20},
                    config_provider=None,
                )
            self.assertIn("results", result)
            self.assertIsNotNone(
                captured_gen_id[0],
                "Batch did not stamp a generation_id before inference; "
                "the finally ownership guard cannot work without it.",
            )

        asyncio.run(run())

    # -- Test 3: finally resets ONLY when this batch still owns state ------

    def test_batch_finally_resets_only_own_state(self):
        """When a concurrent stream has overwritten ``generation_id`` during
        this batch's inference, the batch's finally must NOT reset state
        (the stream owns it now).  We simulate the takeover from inside the
        mocked inference callback, then assert the foreign id survives.
        """

        async def run():
            from qwen3_tts.server.app_generation import handle_generate
            from qwen3_tts.server.validation import GenerateRequest

            state = _make_state()
            fake_wav = np.zeros(500, dtype=np.float32)

            def mock_inference(model, text, **kwargs):
                # Simulate a concurrent stream taking over generation_state
                # mid-batch (overwriting the batch's own generation_id).
                state.generation_state["generation_id"] = "FOREIGN_STREAM"
                state.generation_state["active"] = True
                return fake_wav, 24000

            req = GenerateRequest(text="hello world", mode="design",
                                  voice_description="friendly")
            request = _make_request(state)
            with patch(
                f"{_APP_GENERATION}._check_memory_available",
                return_value=(True, 4096),
            ), patch(
                "qwen3_tts.server.validation._validate_generation_request"
            ), patch(
                f"{_ENGINE}.run_inference", side_effect=mock_inference
            ):
                result = await handle_generate(
                    request=request,
                    state=state,
                    req=req,
                    security={"max_text_length": 50000, "max_batch_size": 20},
                    config_provider=None,
                )
            self.assertIn("results", result)

            # The foreign stream's state must survive the batch's finally.
            self.assertEqual(
                state.generation_state["generation_id"],
                "FOREIGN_STREAM",
                "Batch finally clobbered a foreign generation_id.",
            )
            self.assertTrue(
                state.generation_state["active"],
                "Batch finally clobbered a foreign stream's active=True.",
            )

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
