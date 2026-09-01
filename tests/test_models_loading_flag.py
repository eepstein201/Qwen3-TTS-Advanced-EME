#!/usr/bin/env python3
"""Phase 2c — server-side `loading: bool` on /models, from the record table.

Phase 1b introduced the `loading` field backed by a write-only
``state.models_loading`` flag dict. Phase 2c (#214 item 3) deleted that dict:
the per-load record table (``state.model_loads``, owned by
``model_loading.claim_model_load`` / ``release_model_load``) is the single
source of truth, and ``handle_list_models`` derives the display flag from it.
The wire field is UNCHANGED — no client notices.

RED tests assert the contract:
  * /models entry includes "loading"
  * value tracks an in-flight record for that model type
  * a DONE record never reads as loading (release clears the slot, but a
    lingering done record must not lie either)
  * `loaded` and `loading` are mutually exclusive
  * handle_load_model's in-flight record is visible while it owns the load

No GPU, models, or running server required.

Run: pytest tests/test_models_loading_flag.py -v --tb=short
"""

import asyncio
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

try:
    import fastapi  # noqa: F401

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = unittest.skipUnless(HAS_DEPS, "requires fastapi")


class _FakeRecord:
    """Stand-in for model_loading._LoadRecord — only `done` is consulted."""

    def __init__(self, done=False):
        self.done = threading.Event()
        if done:
            self.done.set()


def _make_state(in_flight=(), loaded_map=None):
    """Build a MagicMock app.state with the fields handle_list_models reads.

    ``in_flight`` names model types with a live (undone) record in
    ``model_loads``.
    """
    state = MagicMock()
    state.models = {"clone": None, "design": None, "custom": None}
    if loaded_map:
        for k, v in loaded_map.items():
            state.models[k] = v  # truthy = loaded
    state.model_loads = {
        "clone": _FakeRecord() if "clone" in in_flight else None,
        "design": _FakeRecord() if "design" in in_flight else None,
        "custom": _FakeRecord() if "custom" in in_flight else None,
    }
    state.model_load_times = {}
    state.model_load_errors = {"clone": None, "design": None, "custom": None}
    return state


@_skip
class TestModelsEndpointLoadingField(unittest.TestCase):
    """handle_list_models must derive `loading` from the record table."""

    def _server_config(self):
        return {"models": {}}

    def test_loading_field_present_in_entry(self):
        from qwen3_tts.server.app_models import handle_list_models

        state = _make_state()
        result = handle_list_models(state, self._server_config())

        for model_type in ("clone", "design", "custom"):
            self.assertIn(
                "loading",
                result["models"][model_type],
                f"/models[{model_type}] missing 'loading' key",
            )

    def test_loading_true_when_record_in_flight(self):
        from qwen3_tts.server.app_models import handle_list_models

        state = _make_state(in_flight=("clone",))
        result = handle_list_models(state, self._server_config())

        self.assertTrue(result["models"]["clone"]["loading"])
        self.assertFalse(result["models"]["design"]["loading"])
        self.assertFalse(result["models"]["custom"]["loading"])

    def test_loading_false_when_no_records(self):
        from qwen3_tts.server.app_models import handle_list_models

        state = _make_state()
        result = handle_list_models(state, self._server_config())

        for model_type in ("clone", "design", "custom"):
            self.assertFalse(
                result["models"][model_type]["loading"],
                f"{model_type} reported loading=True with an empty record table",
            )

    def test_loading_false_when_already_loaded(self):
        """If a model is loaded, it cannot also be loading (mutex)."""
        from qwen3_tts.server.app_models import handle_list_models

        state = _make_state(in_flight=("clone",), loaded_map={"clone": object()})
        result = handle_list_models(state, self._server_config())

        self.assertTrue(result["models"]["clone"]["loaded"])
        self.assertFalse(
            result["models"]["clone"]["loading"],
            "loaded and loading both True — must be mutually exclusive",
        )

    def test_loading_false_when_record_done(self):
        """A DONE record must never read as loading, even if the slot
        somehow still holds it (release normally clears the slot)."""
        from qwen3_tts.server.app_models import handle_list_models

        state = _make_state()
        state.model_loads["clone"] = _FakeRecord(done=True)
        result = handle_list_models(state, self._server_config())

        self.assertFalse(
            result["models"]["clone"]["loading"],
            "a finished load's record must not render as loading",
        )

    def test_loading_false_when_model_loads_attr_missing(self):
        """Backwards-compat: no record table at all must read as not loading.

        This guards hand-built test states (and any partial deployment)
        against AttributeError — the DISPLAY field fails open; only the
        mutual-exclusion gate itself is fail-closed.
        """
        from qwen3_tts.server.app_models import handle_list_models

        state = MagicMock()
        state.models = {"clone": None, "design": None, "custom": None}
        # Deliberately don't set model_loads — simulate missing attr.
        del state.model_loads
        state.model_load_times = {}
        state.model_load_errors = {"clone": None, "design": None, "custom": None}

        result = handle_list_models(state, self._server_config())

        for model_type in ("clone", "design", "custom"):
            self.assertFalse(
                result["models"][model_type]["loading"],
                f"{model_type} reported loading=True when model_loads attr missing",
            )


@_skip
class TestHandleLoadModelDrivesRecords(unittest.TestCase):
    """handle_load_model's claim must be visible to /models while in flight."""

    def test_record_in_flight_during_load_then_cleared(self):
        """While load_model runs, model_loads[type] holds a live record and
        /models reports loading=True; after, the slot is cleared."""
        from qwen3_tts.server.app_models import handle_list_models, handle_load_model

        state = MagicMock()
        state.models = {"clone": None, "design": None, "custom": None}
        state.model_loads = {"clone": None, "design": None, "custom": None}
        state.model_config_epoch = 0
        state.model_load_errors = {"clone": None, "design": None, "custom": None}
        state.model_load_times = {}
        state.inference_lock = asyncio.Lock()

        release = threading.Event()
        self.addCleanup(release.set)
        observed = {}

        def fake_load_model(model_type, warmup=False):
            observed["record_during"] = state.model_loads.get(model_type)
            observed["loading_during"] = handle_list_models(
                state, {"models": {}}
            )["models"][model_type]["loading"]
            release.wait(timeout=10)
            return MagicMock()

        req = MagicMock()
        req.model_type = "clone"

        def _run():
            asyncio.run(handle_load_model(state, req))

        with (
            patch("qwen3_tts.core.engine.load_model", side_effect=fake_load_model),
            patch(
                "qwen3_tts.core.config.get_model_info",
                return_value={"name": "qwen3-tts-clone"},
            ),
        ):
            worker = threading.Thread(target=_run, daemon=True)
            worker.start()
            self.addCleanup(worker.join, timeout=5)
            for _ in range(500):
                if "loading_during" in observed:
                    break
                time.sleep(0.01)

            self.assertIsNotNone(
                observed.get("record_during"),
                "no live record in model_loads during the load — /models "
                "has no source of truth to read (Phase 2c)",
            )
            self.assertFalse(
                observed.get("record_during").done.is_set(),
                "the in-flight record must not be done while the load runs",
            )
            self.assertTrue(
                observed.get("loading_during"),
                "/models must report loading=True while a record is in flight",
            )

            release.set()

        for _ in range(500):
            if state.model_loads["clone"] is None:
                break
            time.sleep(0.01)
        self.assertIsNone(
            state.model_loads["clone"],
            "claim slot not cleared after the load completed",
        )

    def test_record_cleared_on_exception(self):
        """If load_model raises, the claim slot must still be released."""
        from qwen3_tts.server.app_models import handle_load_model

        state = MagicMock()
        state.models = {"clone": None, "design": None, "custom": None}
        state.model_loads = {"clone": None, "design": None, "custom": None}
        state.model_config_epoch = 0
        state.model_load_errors = {"clone": None, "design": None, "custom": None}
        state.model_load_times = {}
        state.inference_lock = asyncio.Lock()

        def boom(model_type, warmup=False):
            raise RuntimeError("boom")

        req = MagicMock()
        req.model_type = "clone"

        import fastapi

        with (
            patch("qwen3_tts.core.engine.load_model", side_effect=boom),
            patch(
                "qwen3_tts.core.config.get_model_info",
                return_value={"name": "qwen3-tts-clone"},
            ),
        ):
            with self.assertRaises(fastapi.HTTPException):
                asyncio.run(handle_load_model(state, req))

        self.assertIsNone(
            state.model_loads["clone"],
            "claim slot not released after RuntimeError (release must run "
            "in finally — a leak wedges this model type into 870s->503)",
        )


@_skip
class TestAppStateInitializesModelLoads(unittest.TestCase):
    """Lifespan setup must initialize app.state.model_loads + epoch."""

    def test_lifespan_creates_record_table(self):
        """Source-level check: app_lifespan initializes the record table."""
        import inspect

        from qwen3_tts.server import app_lifespan

        src = inspect.getsource(app_lifespan)
        self.assertIn(
            "model_loads",
            src,
            "app_lifespan.py does not initialize state.model_loads anywhere",
        )
        self.assertIn(
            "model_config_epoch",
            src,
            "app_lifespan.py does not initialize state.model_config_epoch",
        )


if __name__ == "__main__":
    unittest.main()
