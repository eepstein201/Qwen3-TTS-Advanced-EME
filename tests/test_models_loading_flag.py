#!/usr/bin/env python3
"""Phase 1b — Server-side `loading: bool` field on /models endpoint.

Phase 1a's `poll_model_loading_state()` reads `info.get("loading")` from the
/models response, but the server never emits this field. This is dead code in
real operation. Phase 1b makes the field real so UI progress polling can work.

RED tests assert the contract Phase 1a already depends on:
  * `/models` entry includes "loading" key
  * value tracks `state.models_loading[model_type]`
  * `loaded` and `loading` are mutually exclusive
  * `handle_load_model` sets/clears the flag (finally semantics)

No GPU, models, or running server required.

Run: pytest tests/test_models_loading_flag.py -v --tb=short
"""

import asyncio
import unittest
from unittest.mock import MagicMock, patch

try:
    import fastapi  # noqa: F401
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = unittest.skipUnless(HAS_DEPS, "requires fastapi")


def _make_state(loading_map=None, loaded_map=None):
    """Build a MagicMock app.state with the fields handle_list_models reads."""
    state = MagicMock()
    state.models = {"clone": None, "design": None, "custom": None}
    if loaded_map:
        for k, v in loaded_map.items():
            state.models[k] = v  # truthy = loaded
    state.models_loading = (
        {"clone": False, "design": False, "custom": False}
        if loading_map is None
        else dict(loading_map)
    )
    state.model_load_times = {}
    state.model_load_errors = {"clone": None, "design": None, "custom": None}
    return state


@_skip
class TestModelsEndpointLoadingField(unittest.TestCase):
    """handle_list_models must include `loading: bool` in each model entry."""

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

    def test_loading_true_when_state_says_so(self):
        from qwen3_tts.server.app_models import handle_list_models

        state = _make_state(loading_map={"clone": True, "design": False, "custom": False})
        result = handle_list_models(state, self._server_config())

        self.assertTrue(result["models"]["clone"]["loading"])
        self.assertFalse(result["models"]["design"]["loading"])
        self.assertFalse(result["models"]["custom"]["loading"])

    def test_loading_false_when_state_default(self):
        from qwen3_tts.server.app_models import handle_list_models

        state = _make_state()  # all False
        result = handle_list_models(state, self._server_config())

        for model_type in ("clone", "design", "custom"):
            self.assertFalse(
                result["models"][model_type]["loading"],
                f"{model_type} reported loading=True when state is default",
            )

    def test_loading_false_when_already_loaded(self):
        """If a model is loaded, it cannot also be loading (mutex)."""
        from qwen3_tts.server.app_models import handle_list_models

        # Loaded but stale loading flag — handler must reconcile.
        state = _make_state(
            loaded_map={"clone": object()},
            loading_map={"clone": True, "design": False, "custom": False},
        )
        result = handle_list_models(state, self._server_config())

        self.assertTrue(result["models"]["clone"]["loaded"])
        self.assertFalse(
            result["models"]["clone"]["loading"],
            "loaded and loading both True — must be mutually exclusive",
        )

    def test_loading_false_when_models_loading_attr_missing(self):
        """Backwards-compat: if state.models_loading doesn't exist, default to False.

        This guards against partial deployments where lifespan hasn't initialized
        the new attribute yet (e.g., tests that build state by hand).
        """
        from qwen3_tts.server.app_models import handle_list_models

        state = MagicMock()
        state.models = {"clone": None, "design": None, "custom": None}
        # Deliberately don't set models_loading — simulate missing attr.
        del state.models_loading
        state.model_load_times = {}
        state.model_load_errors = {"clone": None, "design": None, "custom": None}

        result = handle_list_models(state, self._server_config())

        for model_type in ("clone", "design", "custom"):
            self.assertFalse(
                result["models"][model_type]["loading"],
                f"{model_type} reported loading=True when models_loading attr missing",
            )


@_skip
class TestHandleLoadModelTracksLoadingFlag(unittest.TestCase):
    """handle_load_model must flip loading flag to True before, False after."""

    def _make_state(self):
        state = MagicMock()
        state.models = MagicMock()
        state.models.get.return_value = None  # not loaded
        state.models_loading = {"clone": False, "design": False, "custom": False}
        state.model_load_errors = {"clone": None, "design": None, "custom": None}
        state.model_load_times = {}
        return state

    def test_loading_set_true_during_load_then_false(self):
        """During load_model() call, models_loading[type] is True; cleared after."""
        from qwen3_tts.server.app_models import handle_load_model

        state = self._make_state()
        observed_during_load = {}

        def fake_load_model(model_type, **kwargs):  # kwargs: warmup (#192 split)
            observed_during_load["loading"] = state.models_loading.get(model_type)
            return MagicMock()

        req = MagicMock()
        req.model_type = "clone"

        with patch("qwen3_tts.core.engine.load_model", side_effect=fake_load_model), \
             patch("qwen3_tts.core.config.get_model_info", return_value={"name": "qwen3-tts-clone"}):
            asyncio.run(handle_load_model(state, req))

        self.assertTrue(
            observed_during_load.get("loading"),
            "models_loading['clone'] was not True during load_model() call",
        )
        self.assertFalse(
            state.models_loading["clone"],
            "models_loading['clone'] not reset to False after load completed",
        )

    def test_loading_cleared_on_exception(self):
        """If load_model raises, models_loading[type] must still flip back to False."""
        from qwen3_tts.server.app_models import handle_load_model

        state = self._make_state()
        observed = {}

        def boom(model_type, **kwargs):  # kwargs: warmup (#192 split)
            # Capture the flag at the moment load_model is invoked. If the handler
            # didn't set it to True first, this test passes vacuously — we want
            # it to fail loudly until the implementation tracks loading.
            observed["loading_at_call"] = state.models_loading.get(model_type)
            raise RuntimeError("boom")

        req = MagicMock()
        req.model_type = "clone"

        with patch("qwen3_tts.core.engine.load_model", side_effect=boom), \
             patch("qwen3_tts.core.config.get_model_info", return_value={"name": "qwen3-tts-clone"}), \
             patch("qwen3_tts.server.app_models._error_response") as mock_err:
            mock_err.return_value = None  # don't re-raise; let handler return
            asyncio.run(handle_load_model(state, req))

        self.assertTrue(
            observed.get("loading_at_call"),
            "models_loading['clone'] was not True at the moment load_model raised",
        )
        self.assertFalse(
            state.models_loading["clone"],
            "models_loading['clone'] not reset to False after RuntimeError",
        )


@_skip
class TestAppStateInitializesModelsLoading(unittest.TestCase):
    """Lifespan setup must initialize app.state.models_loading."""

    def test_lifespan_creates_models_loading_dict(self):
        """After app.state setup, models_loading is a dict with all model types as False."""
        # We can't easily run the full lifespan, but we can check the source-level
        # initialization site exists by importing and inspecting setup_app_state.
        import inspect

        from qwen3_tts.server import app_lifespan

        # Look for any function/code that sets models_loading.
        src = inspect.getsource(app_lifespan)
        self.assertIn(
            "models_loading",
            src,
            "app_lifespan.py does not initialize state.models_loading anywhere",
        )


if __name__ == "__main__":
    unittest.main()
