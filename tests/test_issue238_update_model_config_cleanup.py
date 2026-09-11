#!/usr/bin/env python3
"""Issue #238 + master-plan finding 3e/M8 -- the /update-model-config cleanup gap.

``handle_update_model_config`` (qwen3_tts/server/app_models.py) unloads all
three model slots (nulls them under ``generation_lock``), bumps
``model_config_epoch`` under ``MODEL_LOAD_LOCK``, and clears the generation
cache -- but it NEVER calls ``unload_model_cleanup()`` and NEVER pops the
entries it orphaned from ``state.model_load_times``. Two distinct defects,
one root cause: the handler reimplemented the unload sequence from
``handle_unload_model`` and skipped two of its steps.

User-visible symptom (#238): after ``/update-model-config``, ``/models``
keeps reporting the OLD weights' ``load_time_sec`` for models that are no
longer loaded -- ``handle_list_models`` reads
``state.model_load_times.get(model_type)`` (app_models.py:154), so a slot
nulled to ``None`` still advertises a stale load time.

Contract pinned by these tests:

  * After a successful ``handle_update_model_config``, ``state.model_load_times``
    holds NO entry for "clone", "design", or "custom" -- the stale load-time
    record for the unloaded weights is gone for every model.
  * ``handle_update_model_config`` invokes ``unload_model_cleanup()`` exactly
    once with no arguments. The patch seam is load-bearing: patch the FACADE
    attribute ``qwen3_tts.core.engine.unload_model_cleanup``, NOT
    ``qwen3_tts.core.engine.asr.unload_model_cleanup``. The fix will import
    the name function-locally (``from qwen3_tts.core.engine import
    unload_model_cleanup``, the same pattern as ``handle_unload_model`` at
    app_models.py:260), and a function-local import binds the name at call
    time -- so the facade attribute is what the handler will resolve.
    Patching the ``asr`` submodule attribute would give a false RED/GREEN.

Both tests fail against the intentionally UNFIXED handler (RED); the fix is
the next dispatch.

No GPU, models, or running server required: the state stub mirrors the
existing direct-handler fixtures (``_make_state`` in
tests/test_issue214_unload_asr_race.py and tests/test_fastapi_app_ext2.py),
``save_config`` is patched at its module seam
(``qwen3_tts.server.app_models.save_config``, used unqualified in the
handler), and ``config_fn`` returns a config without ``audio_loader`` so the
handler's loader-sync branch is inert.

Run: pytest tests/test_issue238_update_model_config_cleanup.py -v --tb=short
"""

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import fastapi  # noqa: F401

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = unittest.skipUnless(HAS_DEPS, "requires fastapi")


def _make_state() -> SimpleNamespace:
    """Build a minimal app.state stand-in for handle_update_model_config.

    Mirrors the existing direct-handler stubs: real locks, real dicts,
    truthy sentinels for loaded slots. The handler touches exactly these
    attributes: ``generation_lock`` (``async with``), ``models`` (all three
    slots nulled), ``model_config_epoch`` (bumped under MODEL_LOAD_LOCK),
    and ``gen_cache``/``gen_cache_lock`` (cleared). ``model_load_times`` is
    the field the UNFIXED handler forgets.
    """
    state = SimpleNamespace()
    state.models = {
        "clone": object(),  # truthy sentinel: a loaded model slot
        "design": object(),
        "custom": object(),
    }
    state.model_load_times = {"clone": 12.34, "design": 3.21, "custom": 1.23}
    state.model_config_epoch = 0
    state.generation_lock = asyncio.Lock()  # real: the handler does `async with`
    state.gen_cache = {}
    state.gen_cache_lock = threading.Lock()
    return state


def _run_update(state: SimpleNamespace) -> dict:
    """Run handle_update_model_config(model_size="0.6B") on ``state``.

    ``save_config`` is patched at its module seam so the config write never
    touches disk; ``config_fn`` returns a config without ``audio_loader`` so
    the loader-sync branch is inert.
    """
    from qwen3_tts.server.app_models import handle_update_model_config
    from qwen3_tts.server.validation import UpdateModelConfigRequest

    req = UpdateModelConfigRequest(model_size="0.6B")
    with patch("qwen3_tts.server.app_models.save_config"):
        return asyncio.run(
            handle_update_model_config(state, req, lambda: {"advanced": {}})
        )


@_skip
class TestUpdateModelConfigPopsStaleLoadTimes(unittest.TestCase):
    """#238: the stale load_time_sec record must not survive the update."""

    def test_update_model_config_pops_stale_load_times_for_all_models(self):
        """After handle_update_model_config, model_load_times holds no
        entry for any of clone/design/custom."""
        state = _make_state()

        result = _run_update(state)

        # Sanity: the handler ran its happy path, so the assertion below
        # fails because of the missing cleanup, not a handler error.
        self.assertEqual(result["status"], "config_updated")
        for name in ("clone", "design", "custom"):
            self.assertNotIn(
                name,
                state.model_load_times,
                f"stale load time for the unloaded '{name}' weights "
                f"survived /update-model-config (issue #238); "
                f"model_load_times={state.model_load_times!r}",
            )


@_skip
class TestUpdateModelConfigInvokesUnloadCleanup(unittest.TestCase):
    """3e/M8: unload_model_cleanup() must actually run on config update."""

    def test_update_model_config_invokes_unload_model_cleanup(self):
        """handle_update_model_config calls the engine facade's
        unload_model_cleanup() exactly once with no arguments."""
        state = _make_state()

        # FACADE attribute, not the asr submodule attribute -- see the
        # module docstring for why this seam is load-bearing.
        with patch("qwen3_tts.core.engine.unload_model_cleanup") as mock_cleanup:
            result = _run_update(state)

        self.assertEqual(result["status"], "config_updated")
        mock_cleanup.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
