#!/usr/bin/env python3
"""PRF-5: defensive recovery after a failed model swap.

A failed load leaves partially-allocated backend memory behind. Upstream
mlx-audio #827 reports Base cloning running ~2.4x slower afterwards, and the
server has a known-red "dies under repeated load/unload". The load handler
recorded the error but never ran the backend cleanup the unload path uses, so
the degraded state persisted for the life of the process.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/test_model_swap_recovery.py -v

No GPU, models, or running server required.
"""

import unittest
from unittest.mock import MagicMock, patch

try:
    import pytest
    HAS_PYTEST = True
except ImportError:  # pragma: no cover
    HAS_PYTEST = False

    class _DummyMarker:
        def __call__(self, func):
            return func

        def __getattr__(self, name):
            return self

    class _DummyMark:
        def __getattr__(self, name):
            return _DummyMarker()

    class _DummyPytest:
        mark = _DummyMark()

    pytest = _DummyPytest()

try:
    from fastapi import HTTPException
    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False
    HTTPException = Exception

_skip = unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")


def _make_state():
    """A state double whose mutable maps are real containers."""
    state = MagicMock()
    state.models = {"clone": None, "design": None, "custom": None}
    state.models_loading = {"clone": False, "design": False, "custom": False}
    state.model_load_errors = {"clone": None, "design": None, "custom": None}
    # Stale timing left over from a previous successful load of this type.
    state.model_load_times = {"clone": 12.3}
    return state


def _req(model_type="clone"):
    req = MagicMock()
    req.model_type = model_type
    return req


def _load_failing(exc):
    """Patch context: load_model raises, model info resolves."""
    return (
        patch("qwen3_tts.core.engine.load_model", side_effect=exc),
        patch(
            "qwen3_tts.core.config.get_model_info",
            return_value={"name": "qwen3-tts-clone"},
        ),
    )


@_skip
@pytest.mark.unit
class TestFailedLoadTriggersRecovery(unittest.TestCase):
    """A failed swap must reclaim backend memory, not just record the error."""

    def test_runtime_error_runs_backend_cleanup(self):
        from qwen3_tts.server.app_models import handle_load_model

        state = _make_state()
        load_p, info_p = _load_failing(RuntimeError("CUDA out of memory"))
        with load_p, info_p, patch(
            "qwen3_tts.core.engine.unload_model_cleanup"
        ) as cleanup:
            with self.assertRaises(HTTPException):
                handle_load_model(state, _req())

        cleanup.assert_called_once()

    def test_import_error_runs_backend_cleanup(self):
        from qwen3_tts.server.app_models import handle_load_model

        state = _make_state()
        load_p, info_p = _load_failing(ImportError("no mlx_audio"))
        with load_p, info_p, patch(
            "qwen3_tts.core.engine.unload_model_cleanup"
        ) as cleanup:
            with self.assertRaises(HTTPException):
                handle_load_model(state, _req())

        cleanup.assert_called_once()

    def test_unexpected_error_runs_backend_cleanup(self):
        from qwen3_tts.server.app_models import handle_load_model

        state = _make_state()
        load_p, info_p = _load_failing(KeyError("weird"))
        with load_p, info_p, patch(
            "qwen3_tts.core.engine.unload_model_cleanup"
        ) as cleanup:
            with self.assertRaises(HTTPException):
                handle_load_model(state, _req())

        cleanup.assert_called_once()

    def test_successful_load_does_not_run_cleanup(self):
        """Recovery is for failures only — don't churn memory on the happy path."""
        from qwen3_tts.server.app_models import handle_load_model

        state = _make_state()
        with patch("qwen3_tts.core.engine.load_model", return_value=MagicMock()), patch(
            "qwen3_tts.core.config.get_model_info",
            return_value={"name": "qwen3-tts-clone"},
        ), patch("qwen3_tts.core.engine.unload_model_cleanup") as cleanup:
            result = handle_load_model(state, _req())

        self.assertEqual(result["status"], "loaded")
        cleanup.assert_not_called()


@_skip
@pytest.mark.unit
class TestFailedLoadLeavesConsistentState(unittest.TestCase):
    """After a failed swap /models must not describe a half-loaded model."""

    def _run_failed_load(self, state):
        from qwen3_tts.server.app_models import handle_load_model

        load_p, info_p = _load_failing(RuntimeError("CUDA out of memory"))
        with load_p, info_p, patch("qwen3_tts.core.engine.unload_model_cleanup"):
            with self.assertRaises(HTTPException):
                handle_load_model(state, _req())

    def test_model_slot_is_none(self):
        state = _make_state()
        self._run_failed_load(state)
        self.assertIsNone(state.models["clone"])

    def test_stale_load_time_is_dropped(self):
        """A leftover load_time would make /models report a healthy model."""
        state = _make_state()
        self._run_failed_load(state)
        self.assertNotIn("clone", state.model_load_times)

    def test_loading_flag_cleared(self):
        state = _make_state()
        self._run_failed_load(state)
        self.assertFalse(state.models_loading["clone"])

    def test_error_is_recorded(self):
        state = _make_state()
        self._run_failed_load(state)
        self.assertIn("out of memory", state.model_load_errors["clone"])


@_skip
@pytest.mark.unit
class TestRecoveryIsNonFatal(unittest.TestCase):
    """Recovery must never replace the error the caller needs to see."""

    def test_cleanup_failure_does_not_mask_load_error(self):
        from qwen3_tts.server.app_models import handle_load_model

        state = _make_state()
        load_p, info_p = _load_failing(RuntimeError("CUDA out of memory"))
        with load_p, info_p, patch(
            "qwen3_tts.core.engine.unload_model_cleanup",
            side_effect=RuntimeError("cleanup blew up"),
        ):
            with self.assertRaises(HTTPException) as ctx:
                handle_load_model(state, _req())

        # The surfaced failure is the load failure, not the cleanup failure.
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertNotIn("cleanup blew up", str(ctx.exception.detail))

    def test_cleanup_failure_still_clears_loading_flag(self):
        from qwen3_tts.server.app_models import handle_load_model

        state = _make_state()
        load_p, info_p = _load_failing(RuntimeError("boom"))
        with load_p, info_p, patch(
            "qwen3_tts.core.engine.unload_model_cleanup",
            side_effect=RuntimeError("cleanup blew up"),
        ):
            with self.assertRaises(HTTPException):
                handle_load_model(state, _req())

        self.assertFalse(state.models_loading["clone"])


if __name__ == "__main__":
    unittest.main()
