#!/usr/bin/env python3
"""Step 2 (Lane H) -- /generate and /generate-stream load models on demand.

``design``/``custom`` default to ``load_at_startup: false``; the documented
contract ("one click away", RUNBOOK / CLAUDE.md) extends to the API: POST
/generate with an unloaded model LOADS it instead of 503ing
``model_not_loaded``. Before this, only the UI's manual Load Model button
attempted the load.

Contract pinned by these tests:

  * /generate and /generate-stream with mode=design (and custom) on an empty
    slot load the model through /load-model's per-load-record owner
    (``model_loading.load_model_deduped`` -- claim/attach dedup + the
    #192 leaf-locked warm-up) and then generate: 200, exactly one load,
    claim released.
  * A failed on-demand load surfaces /load-model's sanitized failure shape
    (classified ``load_failed`` 500, ``_recover_from_failed_load`` having
    reclaimed the slot) -- never a bare 500, never a raw traceback.
  * The residual ``model_not_loaded`` (slot emptied again mid-flight, e.g.
    an /unload-model racing the finished load) points at POST /load-model /
    the Manage Models tab (``recovery: "load_model"``), not at a restart.
  * Two simultaneous first-requests for the same unloaded model do not
    double-load -- the claim/attach coalescing works through the /generate
    path exactly as it does through /load-model (#214 item 3).
  * An on-demand load initiated while ANOTHER mode's generation is in
    flight does not deadlock on the inference_lock + MODEL_LOAD_LOCK
    interaction: weight construction runs unlocked (it completes while the
    sibling generation still holds inference_lock), the design warm-up
    waits for the lock as a leaf, and both requests complete. 4B's campaign
    exercised only SERIAL load/unload cycling; this pins the interleaving
    constructible without a live GPU.

No GPU, models, or running server required -- ``load_model`` /
``run_inference`` are patched at the ``qwen3_tts.core.engine`` facade
(handlers import function-locally, so the patched attribute is what they
resolve at call time).

Run: pytest tests/test_generate_on_demand_load.py -v --tb=short
"""

import asyncio
import os
import struct
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    import numpy as np
    import soundfile  # noqa: F401

    _HAS_NUMPY_SF = True
except ImportError:
    _HAS_NUMPY_SF = False

try:
    import fastapi  # noqa: F401

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

_skip_direct = unittest.skipUnless(_HAS_NUMPY_SF, "requires numpy + soundfile")
_skip_client = unittest.skipUnless(
    _HAS_NUMPY_SF and _HAS_FASTAPI, "requires numpy + soundfile + fastapi"
)

_ENGINE = "qwen3_tts.core.engine"
_APP_GENERATION = "qwen3_tts.server.app_generation"
_WARMUP_DISABLED = f"{_ENGINE}.model_loader._warmup_disabled"

_SECURITY = {"max_text_length": 50000, "max_batch_size": 20}


# ---------------------------------------------------------------------------
# Direct-drive helpers (shape from tests/test_issue214_unload_queued_window.py)
# ---------------------------------------------------------------------------


def _make_state(**overrides):
    """Complete app.state stand-in (conftest-shaped).

    handle_generate reads state attributes directly, so the state must be
    fully populated -- a bare object reds for the wrong reason
    (AttributeError, not the missing lock).
    """
    state = SimpleNamespace()
    state.auth_token = "test_token"  # nosec B105
    state.models = {"clone": None, "design": None, "custom": None}
    state.model_load_times = {}
    state.model_load_errors = {"clone": None, "design": None, "custom": None}
    state.model_loads = {"clone": None, "design": None, "custom": None}
    state.model_config_epoch = 0
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
    state.generation_lock = asyncio.Lock()
    state.pending_requests = []
    state.pending_lock = asyncio.Lock()
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
    state.vllm_adapter = None
    state.vllm_client = None
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _make_request(state):
    """A REAL starlette Request whose ``app.state`` points at *state*."""
    from starlette.datastructures import Address
    from starlette.requests import Request

    scope = {
        "type": "http",
        "app": SimpleNamespace(state=state),
        "headers": [],
        "path": "/generate",
        "method": "POST",
        "client": Address("127.0.0.1", 51000),
        "query_string": b"",
    }
    return Request(scope)


def _cleanup_gen_cache_files(state):
    """Unlink the empty cache files handle_generate's cache-write step
    creates (NamedTemporaryFile is real even with soundfile.write patched).
    Mirrors the precedent cleanup in tests/test_batch_generation_state_ownership.py."""
    for entry in list(getattr(state, "gen_cache", {}).values()):
        main_file = entry.get("main_file") or entry.get("file")
        if main_file and os.path.exists(main_file):
            try:
                os.remove(main_file)
            except OSError:
                pass
    state.gen_cache.clear()


# ---------------------------------------------------------------------------
# Endpoint-level tests (real app + TestClient, mocked engine)
# ---------------------------------------------------------------------------


@_skip_client
class TestGenerateOnDemandLoad(unittest.TestCase):
    """POST /generate and /generate-stream must LOAD an unloaded model."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        from tests.voice_test_helpers import _make_test_client

        cls.app = app
        cls.client = _make_test_client(
            app,
            server_config={
                "security": {"max_text_length": 10000, "max_batch_size": 20},
                "auto_shutdown_minutes": 0,
            },
        )
        app.state.models_loaded.set()
        cls.auth = {"Authorization": "Bearer test_token"}

    def setUp(self):
        # The app singleton persists across tests in-process: reset the
        # slots and the per-load-record table so each test starts cold.
        self.app.state.models = {"clone": None, "design": None, "custom": None}
        self.app.state.model_loads = {"clone": None, "design": None, "custom": None}
        self.app.state.model_config_epoch = 0
        self.app.state.model_load_times = {}
        self.app.state.model_load_errors = {
            "clone": None,
            "design": None,
            "custom": None,
        }
        with self.app.state.gen_cache_lock:
            self.app.state.gen_cache.clear()
        # The streaming handler reads the pending-request registry (queue
        # headers); the shared helper does not create it. Fresh per test so
        # the asyncio.Lock never crosses a TestClient portal loop.
        self.app.state.pending_requests = []
        self.app.state.pending_lock = asyncio.Lock()

    def tearDown(self):
        _cleanup_gen_cache_files(self.app.state)

    def test_generate_design_loads_model_on_demand(self):
        """/generate with mode=design on an empty slot loads then generates."""
        sentinel = MagicMock(name="design-weights")
        load_calls = []

        def _load(model_type, warmup=False):
            load_calls.append((model_type, warmup))
            return sentinel

        fake_wav = np.zeros(100, dtype="float32")
        with (
            patch(f"{_ENGINE}.load_model", side_effect=_load),
            patch(f"{_ENGINE}.run_inference", return_value=(fake_wav, 24000)),
            patch(
                f"{_APP_GENERATION}._check_memory_available",
                return_value=(True, 10000),
            ),
            # Keep this test focused on the load path; the racing test below
            # exercises the warm-up's lock interaction.
            patch(_WARMUP_DISABLED, return_value=True),
            patch("soundfile.write"),
            patch(
                f"{_ENGINE}.audio_processing.calculate_waveform_peaks",
                return_value=[0.1] * 10,
            ),
        ):
            resp = self.client.post(
                "/generate",
                json={
                    "texts": ["on demand design hello"],
                    "mode": "design",
                    "voice_description": "friendly",
                },
                headers=self.auth,
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(
            load_calls,
            [("design", False)],
            f"expected exactly one on-demand design load, saw {load_calls!r}",
        )
        self.assertIs(
            self.app.state.models["design"],
            sentinel,
            "the loaded weights must be installed in the slot",
        )
        self.assertIsNone(
            self.app.state.model_loads["design"], "claim must release after the load"
        )
        results = resp.json()["results"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["sample_rate"], 24000)

    def test_generate_custom_loads_model_on_demand(self):
        """/generate with mode=custom on an empty slot loads then generates."""
        sentinel = MagicMock(name="custom-weights")
        load_calls = []

        def _load(model_type, warmup=False):
            load_calls.append((model_type, warmup))
            return sentinel

        fake_wav = np.zeros(100, dtype="float32")
        with (
            patch(f"{_ENGINE}.load_model", side_effect=_load),
            patch(f"{_ENGINE}.run_inference", return_value=(fake_wav, 24000)),
            patch(
                f"{_APP_GENERATION}._check_memory_available",
                return_value=(True, 10000),
            ),
            patch("soundfile.write"),
            patch(
                f"{_ENGINE}.audio_processing.calculate_waveform_peaks",
                return_value=[0.1] * 10,
            ),
        ):
            resp = self.client.post(
                "/generate",
                json={
                    "texts": ["on demand custom hello"],
                    "mode": "custom",
                    "speaker": "Ryan",
                },
                headers=self.auth,
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(load_calls, [("custom", False)])
        self.assertIs(self.app.state.models["custom"], sentinel)
        self.assertIsNone(self.app.state.model_loads["custom"])

    def test_generate_stream_design_loads_model_on_demand(self):
        """/generate-stream with mode=design on an empty slot loads then streams."""
        sentinel = MagicMock(name="design-weights")
        load_calls = []

        def _load(model_type, warmup=False):
            load_calls.append((model_type, warmup))
            return sentinel

        chunk = np.array([0.1, 0.2, 0.3], dtype="float32")

        def fake_stream(**kwargs):
            yield chunk, 24000

        with (
            patch(f"{_ENGINE}.load_model", side_effect=_load),
            patch(f"{_ENGINE}.run_inference_streaming", side_effect=fake_stream),
            patch(
                f"{_APP_GENERATION}._check_memory_available",
                return_value=(True, 10000),
            ),
            patch(_WARMUP_DISABLED, return_value=True),
        ):
            resp = self.client.post(
                "/generate-stream",
                json={
                    "text": "on demand stream hello",
                    "mode": "design",
                    "voice_description": "friendly",
                },
                headers=self.auth,
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(load_calls, [("design", False)])
        self.assertIs(self.app.state.models["design"], sentinel)
        self.assertIsNone(self.app.state.model_loads["design"])
        data = resp.content
        self.assertGreater(len(data), 8)
        sr, length = struct.unpack("<II", data[:8])
        self.assertEqual(sr, 24000)
        self.assertEqual(length, len(chunk) * 4)

    def test_generate_on_demand_load_failure_returns_sanitized_shape(self):
        """A failed on-demand load returns /load-model's classified shape
        (``load_failed`` 500 with sanitized detail), never a bare 500 --
        and the slot stays empty with the error recorded."""
        load_calls = []

        def _boom(model_type, warmup=False):
            load_calls.append(model_type)
            raise RuntimeError("cold load failed under /Users/alice/weights")

        with (
            patch(f"{_ENGINE}.load_model", side_effect=_boom),
            patch(
                f"{_APP_GENERATION}._check_memory_available",
                return_value=(True, 10000),
            ),
            patch(_WARMUP_DISABLED, return_value=True),
        ):
            resp = self.client.post(
                "/generate",
                json={
                    "texts": ["doomed load"],
                    "mode": "design",
                    "voice_description": "friendly",
                },
                headers=self.auth,
            )
        self.assertEqual(resp.status_code, 500, resp.text)
        detail = resp.json()["detail"]
        self.assertIsInstance(detail, dict)
        self.assertEqual(detail.get("error"), "load_failed")
        self.assertEqual(detail.get("recovery"), "restart")
        self.assertIn(
            "<path>",
            str(detail.get("detail", "")),
            f"the raw failure path must be sanitized away: {detail!r}",
        )
        self.assertNotIn("/Users/alice", str(detail))
        self.assertIsNone(self.app.state.models["design"])
        self.assertTrue(self.app.state.model_load_errors["design"])
        self.assertIsNone(self.app.state.model_loads["design"])


# ---------------------------------------------------------------------------
# Direct-drive tests: concurrency and the residual paths
# ---------------------------------------------------------------------------


@_skip_direct
class TestOnDemandLoadConcurrency(unittest.TestCase):
    """claim/attach coalescing and lock ordering through the /generate path."""

    def test_two_concurrent_first_requests_load_design_once(self):
        """Two simultaneous first-requests for the same unloaded model must
        not double-load (#214 item 3 through the /generate route): the
        duplicate attaches to the in-flight record and waits."""
        from qwen3_tts.server.app_generation import handle_generate
        from qwen3_tts.server.validation import GenerateRequest

        state = _make_state()
        calls = []
        release = threading.Event()
        sentinel = MagicMock(name="design-weights")
        timer = threading.Timer(0.3, release.set)
        self.addCleanup(timer.cancel)
        timer.start()

        def _load(model_type, warmup=False):
            calls.append(model_type)
            if len(calls) == 1:
                release.wait(timeout=10)
            return sentinel

        fake_wav = np.zeros(100, dtype="float32")

        async def _one(i):
            req = GenerateRequest(
                texts=[f"concurrent first-request {i}"],
                mode="design",
                voice_description="friendly",
            )
            return await handle_generate(
                request=_make_request(state),
                state=state,
                req=req,
                security=_SECURITY,
                config_provider=None,
            )

        async def _scenario():
            return await asyncio.wait_for(
                asyncio.gather(_one(1), _one(2), return_exceptions=True), timeout=20
            )

        try:
            with (
                patch(f"{_ENGINE}.load_model", side_effect=_load),
                patch(f"{_ENGINE}.run_inference", return_value=(fake_wav, 24000)),
                patch(
                    f"{_APP_GENERATION}._check_memory_available",
                    return_value=(True, 4096),
                ),
                patch(_WARMUP_DISABLED, return_value=True),
                patch("soundfile.write"),
                patch(
                    f"{_ENGINE}.audio_processing.calculate_waveform_peaks",
                    return_value=[0.1] * 10,
                ),
            ):
                first, second = asyncio.run(_scenario())
        finally:
            _cleanup_gen_cache_files(state)

        for label, result in (("first", first), ("second", second)):
            self.assertNotIsInstance(
                result,
                Exception,
                f"the {label} concurrent request failed: {result!r}",
            )
            self.assertEqual(len(result["results"]), 1)
        self.assertEqual(
            calls,
            ["design"],
            f"two simultaneous first-requests must not double-load, saw {calls!r}",
        )
        self.assertIs(state.models["design"], sentinel)
        self.assertIsNone(state.model_loads["design"], "claim must release")

    def test_on_demand_load_races_inflight_generation_without_deadlock(self):
        """An on-demand design load initiated while a custom generation is
        in flight must not deadlock on inference_lock + MODEL_LOAD_LOCK.

        Expected interleaving (asserted via ordered markers): the custom
        generation acquires inference_lock and parks inside run_inference;
        the design request's WEIGHT CONSTRUCTION completes while that lock
        is still held (it never needs it); the design WARM-UP then waits
        for the lock as a leaf and runs only after the custom generation
        released it; the design generation runs last. A lock-ordering
        deadlock surfaces as the wait_for timeout instead of a hang."""
        from qwen3_tts.server.app_generation import handle_generate
        from qwen3_tts.server.validation import GenerateRequest

        state = _make_state()
        state.models["custom"] = MagicMock(name="custom-weights")
        markers = []
        warmup_saw_lock = {}
        load_calls = []
        gen_release = threading.Event()
        timer = threading.Timer(0.4, gen_release.set)
        self.addCleanup(timer.cancel)
        timer.start()

        def _load(model_type, warmup=False):
            load_calls.append(model_type)
            markers.append("design_load")
            return MagicMock(name="design-weights-loaded")

        def _warmup(model, model_type, backend):
            markers.append("design_warmup")
            warmup_saw_lock["value"] = state.inference_lock.locked()

        def _run_inference(**kwargs):
            if kwargs.get("mode") == "custom":
                markers.append("custom_inference_start")
                gen_release.wait(timeout=10)
                markers.append("custom_inference_end")
            else:
                markers.append("design_inference")
            return np.zeros(100, dtype="float32"), 24000

        async def _custom():
            req = GenerateRequest(
                texts=["inflight custom generation"], mode="custom", speaker="Ryan"
            )
            return await handle_generate(
                request=_make_request(state),
                state=state,
                req=req,
                security=_SECURITY,
                config_provider=None,
            )

        async def _design():
            req = GenerateRequest(
                texts=["queued design generation"],
                mode="design",
                voice_description="friendly",
            )
            return await handle_generate(
                request=_make_request(state),
                state=state,
                req=req,
                security=_SECURITY,
                config_provider=None,
            )

        async def _scenario():
            return await asyncio.wait_for(
                asyncio.gather(_custom(), _design(), return_exceptions=True),
                timeout=30,
            )

        try:
            with (
                patch(f"{_ENGINE}.load_model", side_effect=_load),
                patch(f"{_ENGINE}.run_inference", side_effect=_run_inference),
                patch(
                    f"{_APP_GENERATION}._check_memory_available",
                    return_value=(True, 4096),
                ),
                patch(_WARMUP_DISABLED, return_value=False),
                patch(
                    f"{_ENGINE}.model_loader._warmup_model", side_effect=_warmup
                ),
                patch("soundfile.write"),
                patch(
                    f"{_ENGINE}.audio_processing.calculate_waveform_peaks",
                    return_value=[0.1] * 10,
                ),
            ):
                custom_result, design_result = asyncio.run(_scenario())
        finally:
            _cleanup_gen_cache_files(state)

        for label, result in (
            ("custom", custom_result),
            ("design", design_result),
        ):
            self.assertNotIsInstance(
                result,
                Exception,
                f"the {label} request failed (deadlock surfaces as a "
                f"timeout error): {result!r}",
            )
            self.assertEqual(len(result["results"]), 1)
        self.assertEqual(load_calls, ["design"])
        self.assertTrue(
            warmup_saw_lock.get("value"),
            "the design warm-up must run serialized under inference_lock",
        )
        self.assertLess(
            markers.index("custom_inference_start"),
            markers.index("design_load"),
            "the load must start while the other mode's generation is in flight",
        )
        self.assertLess(
            markers.index("design_load"),
            markers.index("custom_inference_end"),
            "weight construction must not wait for inference_lock -- it "
            "completes while the sibling generation still holds it",
        )
        self.assertLess(
            markers.index("custom_inference_end"),
            markers.index("design_warmup"),
            "the warm-up must wait for the in-flight generation's lock release",
        )
        self.assertLess(
            markers.index("design_warmup"),
            markers.index("design_inference"),
        )

    def test_post_load_unload_race_points_at_load_model(self):
        """The residual model_not_loaded (slot emptied again after the load
        returned, e.g. an /unload-model racing it) must point the caller at
        POST /load-model / the Manage Models tab, not at a restart."""
        from fastapi import HTTPException

        from qwen3_tts.server.app_generation import handle_generate
        from qwen3_tts.server.validation import GenerateRequest

        state = _make_state()
        req = GenerateRequest(
            texts=["unload race"], mode="design", voice_description="friendly"
        )
        deduped_calls = []

        async def _fake_deduped(state_, model_type, request=None):
            deduped_calls.append(model_type)
            # "Loaded" -- but an unload raced past the assign and the slot
            # is empty again by the time the handler re-reads it.
            return {"status": "loaded", "model": model_type}

        async def _scenario():
            return await handle_generate(
                request=_make_request(state),
                state=state,
                req=req,
                security=_SECURITY,
                config_provider=None,
            )

        with (
            patch(
                f"{_APP_GENERATION}._check_memory_available",
                return_value=(True, 4096),
            ),
            patch(
                "qwen3_tts.server.model_loading.load_model_deduped",
                side_effect=_fake_deduped,
            ),
        ):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(_scenario())

        self.assertEqual(deduped_calls, ["design"])
        self.assertEqual(ctx.exception.status_code, 503)
        detail = ctx.exception.detail
        self.assertIsInstance(detail, dict)
        self.assertEqual(detail.get("error"), "model_not_loaded")
        self.assertEqual(
            detail.get("recovery"),
            "load_model",
            f"recovery must point at the manual load, got {detail!r}",
        )
        self.assertIn(
            "/load-model",
            str(detail.get("detail", "")),
            f"the message must name POST /load-model, got {detail!r}",
        )


if __name__ == "__main__":
    unittest.main()
