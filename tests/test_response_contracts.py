"""GEN-2: Pydantic response contracts for server routes.

Each test hits a route through the FastAPI TestClient and validates the JSON
body against the route's ``response_model``. ``response_model`` FILTERS any
field the model does not declare, so these tests exist to catch a model that
silently drops a field the handler emits (the assertion on key fields plus
``model_validate`` catches both shape drift and filtering).

Deliberately untyped routes (see validation.py GEN-2 block): /generate-stream
and /ws (binary/WebSocket frames), /preview-prompt and /shutdown (non-JSON
Response objects).

Run: conda run -n qwen3-tts-mlx python -m pytest tests/test_response_contracts.py -v
"""

import os
import time
import unittest
from unittest.mock import MagicMock, patch

try:
    from fastapi.testclient import TestClient

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = unittest.skipUnless(HAS_DEPS, "requires fastapi")


@_skip
class _ContractTestBase(unittest.TestCase):
    """Shared TestClient harness with app.state snapshot/restore."""

    def setUp(self):
        from qwen3_tts.server.app import app
        from tests.conftest import _init_app_state, _restore_app_state, _save_app_state

        self._restore_app_state = _restore_app_state
        self._original_state = _save_app_state(app)
        _init_app_state(app, auth_token="test_token")
        app.state.models_loaded.set()  # simulate ready
        app.state.server_config = {
            "security": {"max_text_length": 50000, "max_batch_size": 20},
            "auto_shutdown_minutes": 0,
        }
        self.app = app
        self.client = TestClient(app, raise_server_exceptions=False)
        self.auth = {"Authorization": "Bearer test_token"}
        self._reset_limiters()

    def tearDown(self):
        self._restore_app_state(self.app, self._original_state)

    def _reset_limiters(self):
        """Reset in-process rate limiters (house pattern; unittest ignores the
        pytest autouse fixture in conftest.py)."""
        for attr in ("limiter", "limiter_global", "limiter_hybrid", "limiter_ip", "limiter_token"):
            limiter = getattr(self.app.state, attr, None)
            if limiter is not None and hasattr(limiter, "reset"):
                limiter.reset()


@_skip
class TestPublicStatusContracts(_ContractTestBase):
    """G1: /ready, /queue-status, /generation-status, /stats, /models."""

    def test_ready_matches_contract(self):
        from qwen3_tts.server.validation import ReadyResponse

        resp = self.client.get("/ready")
        self.assertEqual(resp.status_code, 200, resp.text)
        model = ReadyResponse.model_validate(resp.json())
        self.assertEqual(model.status, "ready")

    def test_queue_status_matches_contract(self):
        from qwen3_tts.server.validation import QueueStatusResponse

        resp = self.client.get("/queue-status")
        self.assertEqual(resp.status_code, 200, resp.text)
        model = QueueStatusResponse.model_validate(resp.json())
        self.assertIsInstance(model.queue_length, int)
        self.assertIsInstance(model.active, bool)

    def test_generation_status_matches_contract(self):
        from qwen3_tts.server.validation import GenerationStatusResponse

        resp = self.client.get("/generation-status")
        self.assertEqual(resp.status_code, 200, resp.text)
        model = GenerationStatusResponse.model_validate(resp.json())
        self.assertFalse(model.active)
        self.assertFalse(model.cancelled)

    def test_generation_status_active_matches_contract(self):
        import time as _time

        from qwen3_tts.server.validation import GenerationStatusResponse

        gs = self.app.state.generation_state
        gs["active"] = True
        gs["start_time"] = _time.time() - 5
        try:
            resp = self.client.get("/generation-status")
            model = GenerationStatusResponse.model_validate(resp.json())
            self.assertTrue(model.active)
            self.assertIsNotNone(model.elapsed_sec)
        finally:
            gs["active"] = False
            gs["start_time"] = 0.0

    def test_stats_matches_contract(self):
        from qwen3_tts.server.validation import StatsResponse

        resp = self.client.get("/stats", headers=self.auth)
        self.assertEqual(resp.status_code, 200, resp.text)
        model = StatsResponse.model_validate(resp.json())
        self.assertEqual(model.status, "ok")
        self.assertFalse(model.generation_health.degraded)
        # Backend-conditional fields stay present-but-null on the other backend.
        self.assertIn("backend", resp.json())

    def test_stats_omits_keys_the_handler_did_not_emit(self):
        """exclude_unset fidelity: optional keys absent from the handler dict
        must stay absent (no additive nulls) — e.g. CUDA stats on a machine
        with no CUDA, or MLX stats when the backend import failed."""
        resp = self.client.get("/stats", headers=self.auth)
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertNotIn("cuda_memory_allocated_mb", data)
        self.assertNotIn("cuda_memory_reserved_mb", data)

    def test_models_matches_contract(self):
        from qwen3_tts.server.validation import ModelsResponse

        resp = self.client.get("/models", headers=self.auth)
        self.assertEqual(resp.status_code, 200, resp.text)
        model = ModelsResponse.model_validate(resp.json())
        self.assertIn("clone", model.models)
        entry = model.models["clone"]
        # Key fields must survive response_model filtering (the silent-data-loss
        # failure mode these tests exist to catch).
        for field in ("loaded", "loading", "description", "memory_mb", "repo_id",
                      "load_at_startup", "load_time_sec"):
            self.assertIn(field, resp.json()["models"]["clone"])
        self.assertFalse(entry.loaded)


@_skip
class TestGenerationContracts(_ContractTestBase):
    """G2: /generate (typed since #176-era GenerateResponse) and /transcribe."""

    def _mocked_generate_patches(self):
        """Patches that make /generate return 200 without touching models."""
        import numpy as np

        return (
            patch(
                "qwen3_tts.server.app_generation._check_memory_available",
                return_value=(True, 4000),
            ),
            patch(
                "qwen3_tts.core.engine.load_voice_prompt",
                return_value=MagicMock(),
            ),
            patch(
                "qwen3_tts.core.engine.run_inference",
                return_value=(np.zeros(4800, dtype=np.float32), 24000),
            ),
            patch(
                "qwen3_tts.core.engine.audio_processing.calculate_waveform_peaks",
                return_value=[0.1] * 500,
            ),
            patch("soundfile.write"),
        )

    def tearDown(self):
        # Remove cache temp files the generation handler created.
        for entry in list(getattr(self.app.state, "gen_cache", {}).values()):
            main_file = entry.get("main_file")
            if main_file and os.path.exists(main_file):
                try:
                    os.remove(main_file)
                except OSError:
                    pass
        self.app.state.gen_cache = {}
        super().tearDown()

    def test_generate_matches_contract(self):
        from qwen3_tts.server.validation import GenerateResponse

        self.app.state.models["clone"] = MagicMock()
        p1, p2, p3, p4, p5 = self._mocked_generate_patches()
        with p1, p2, p3, p4, p5:
            resp = self.client.post(
                "/generate",
                json={"text": "Hello contracts", "mode": "clone",
                      "prompt_file": "voice.wav"},
                headers=self.auth,
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        payload = resp.json()
        model = GenerateResponse.model_validate(payload)
        self.assertEqual(len(model.results), 1)
        # The top-level cancelled flag must survive response_model filtering
        # too. Every server-side test calls handle_generate directly, which
        # bypasses response_model entirely — so without this, a future
        # response_model_exclude_defaults/_exclude_none would silently drop the
        # flag on the wire and clients would go back to indexing results[0]
        # on a short batch.
        self.assertIn("cancelled", payload)
        self.assertIs(payload["cancelled"], False)
        row = payload["results"][0]
        # Every GenerateResult field must survive response_model filtering.
        for field in ("index", "audio_base64", "sample_rate", "peaks",
                      "chunks", "seed"):
            self.assertIn(field, row)
        self.assertEqual(row["sample_rate"], 24000)
        self.assertIsInstance(row["peaks"], list)
        self.assertIsInstance(row["chunks"], int)
        self.assertIsInstance(row["seed"], int)

    def test_transcribe_matches_contract(self):
        from qwen3_tts.server.validation import TranscribeResponse

        with (
            patch("qwen3_tts.core.engine.is_asr_loaded", return_value=True),
            patch(
                "qwen3_tts.core.engine.transcribe_audio",
                return_value="hello world",
            ),
        ):
            resp = self.client.post(
                "/transcribe",
                json={"audio_base64": "aGVsbG8=", "language": "en"},
                headers=self.auth,
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        model = TranscribeResponse.model_validate(resp.json())
        self.assertEqual(model.transcript, "hello world")


@_skip
class TestOpsContracts(_ContractTestBase):
    """G3: model/ASR ops, config updates, and prompt CRUD routes."""

    _TMP_BASE = "zz_contract_tmp_prompt"

    def _make_prompt_file(self, base=_TMP_BASE, ext=".wav"):
        """Create a throwaway prompt file in the real VOICE_PROMPTS_DIR."""
        from qwen3_tts.core.config import VOICE_PROMPTS_DIR

        path = os.path.join(str(VOICE_PROMPTS_DIR), f"{base}{ext}")

        def _safe_remove(p=path):
            if os.path.exists(p):
                os.remove(p)

        # CI's clean runner home has no ~/Qwen3-TTS_UserFiles yet (config
        # resolves home-relative), so the prompts dir may not exist there.
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"RIFF-contract-test")
        self.addCleanup(_safe_remove)
        return base

    def test_load_model_matches_contract(self):
        from qwen3_tts.server.validation import ModelOpResponse

        self.app.state.models["clone"] = MagicMock()
        resp = self.client.post(
            "/load-model", json={"model_type": "clone"}, headers=self.auth
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        model = ModelOpResponse.model_validate(resp.json())
        self.assertEqual(model.status, "already_loaded")
        self.assertEqual(model.model, "clone")

    def test_unload_model_matches_contract(self):
        from qwen3_tts.server.validation import ModelOpResponse

        self.app.state.models["design"] = MagicMock()
        with patch("qwen3_tts.core.engine.unload_model_cleanup"):
            resp = self.client.post(
                "/unload-model", json={"model_type": "design"}, headers=self.auth
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        model = ModelOpResponse.model_validate(resp.json())
        self.assertEqual(model.status, "unloaded")
        self.assertEqual(model.model, "design")

    def test_update_model_config_matches_contract(self):
        from qwen3_tts.server.validation import UpdateModelConfigResponse

        with patch(
            "qwen3_tts.server.app._get_app_config", return_value={}
        ), patch("qwen3_tts.server.app_models.save_config"):
            resp = self.client.post(
                "/update-model-config",
                json={"model_size": "0.6B"},
                headers=self.auth,
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        model = UpdateModelConfigResponse.model_validate(resp.json())
        self.assertEqual(model.status, "config_updated")
        self.assertEqual(model.changes, ["model_size=0.6B"])
        self.assertIn("models_unloaded", resp.json())

    def test_update_startup_config_matches_contract(self):
        from qwen3_tts.server.validation import UpdateStartupConfigResponse

        with patch(
            "qwen3_tts.server.app._get_app_config", return_value={}
        ), patch("qwen3_tts.server.app_models.save_config"):
            resp = self.client.post(
                "/update-startup-config",
                json={"clone": False},
                headers=self.auth,
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        model = UpdateStartupConfigResponse.model_validate(resp.json())
        self.assertEqual(model.status, "updated")
        self.assertEqual(model.changes, ["clone=off"])

    def test_load_asr_matches_contract(self):
        from qwen3_tts.server.validation import LoadAsrResponse

        with patch("qwen3_tts.core.engine.is_asr_loaded", return_value=True):
            resp = self.client.post("/load-asr", headers=self.auth)
        self.assertEqual(resp.status_code, 200, resp.text)
        model = LoadAsrResponse.model_validate(resp.json())
        self.assertEqual(model.status, "already_loaded")

    def test_unload_asr_matches_contract(self):
        from qwen3_tts.server.validation import UnloadAsrResponse

        with patch("qwen3_tts.core.engine.unload_asr_model"):
            resp = self.client.post("/unload-asr", headers=self.auth)
        self.assertEqual(resp.status_code, 200, resp.text)
        model = UnloadAsrResponse.model_validate(resp.json())
        self.assertEqual(model.status, "unloaded")

    def test_prompts_matches_contract(self):
        from qwen3_tts.server.validation import PromptsListResponse

        resp = self.client.get("/prompts", headers=self.auth)
        self.assertEqual(resp.status_code, 200, resp.text)
        model = PromptsListResponse.model_validate(resp.json())
        self.assertEqual(model.total, len(model.prompts))

    def test_prompt_details_all_matches_contract(self):
        from qwen3_tts.server.validation import PromptDetailsResponse

        resp = self.client.get("/prompt-details", headers=self.auth)
        self.assertEqual(resp.status_code, 200, resp.text)
        model = PromptDetailsResponse.model_validate(resp.json())
        self.assertIsInstance(model.prompts, list)

    def test_prompt_details_single_matches_contract(self):
        from qwen3_tts.server.validation import PromptInfo

        base = self._make_prompt_file()
        resp = self.client.get(
            "/prompt-details", params={"name": base}, headers=self.auth
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        model = PromptInfo.model_validate(resp.json())
        self.assertEqual(model.name, base)
        self.assertEqual(model.formats, [".wav"])
        self.assertGreater(model.size_bytes, 0)
        self.assertFalse(model.is_default)

    def test_delete_prompt_matches_contract(self):
        from qwen3_tts.server.validation import DeletePromptResponse

        base = self._make_prompt_file()
        with patch("qwen3_tts.server.app._get_app_config", return_value={}):
            resp = self.client.post(
                "/delete-prompt", json={"name": base}, headers=self.auth
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        model = DeletePromptResponse.model_validate(resp.json())
        self.assertEqual(model.status, "deleted")
        self.assertEqual(model.files_removed, [f"{base}.wav"])

    def test_rename_prompt_matches_contract(self):
        from qwen3_tts.server.validation import RenamePromptResponse

        old = self._make_prompt_file()
        new_base = f"{self._TMP_BASE}_renamed"
        renamed_path = os.path.join(
            str(_VOICE_PROMPTS_DIR()), f"{new_base}.wav"
        )
        self.addCleanup(
            lambda: os.path.exists(renamed_path) and os.remove(renamed_path)
        )
        with patch("qwen3_tts.server.app._get_app_config", return_value={}):
            resp = self.client.post(
                "/rename-prompt",
                json={"old_name": old, "new_name": new_base},
                headers=self.auth,
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        model = RenamePromptResponse.model_validate(resp.json())
        self.assertEqual(model.status, "renamed")
        self.assertEqual(model.old_name, old)
        self.assertEqual(model.new_name, new_base)
        self.assertEqual(model.files_renamed, [f"{new_base}.wav"])

    def test_preview_prompt_returns_audio_bytes(self):
        """Deliberately untyped: /preview-prompt returns a FileResponse, so the
        contract is the media type, not a JSON model."""
        base = self._make_prompt_file()
        resp = self.client.get(
            "/preview-prompt", params={"name": base}, headers=self.auth
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertTrue(
            resp.headers["content-type"].startswith("audio/wav"),
            resp.headers["content-type"],
        )


def _VOICE_PROMPTS_DIR():
    from qwen3_tts.core.config import VOICE_PROMPTS_DIR

    return VOICE_PROMPTS_DIR


@_skip
class TestOpenApiContract(unittest.TestCase):
    """G4: the OpenAPI spec generates cleanly with every response contract.

    /shutdown intentionally carries no response_model — it returns a plain
    Response (BackgroundTask + SIGTERM per #176), which a JSON schema cannot
    describe.
    """

    def test_openapi_generates_cleanly(self):
        import qwen3_tts.server.app as app_mod

        spec = app_mod.app.openapi()  # must not raise on any response model
        schemas = spec["components"]["schemas"]
        for name in (
            "ModelsResponse",
            "StatsResponse",
            "ReadyResponse",
            "QueueStatusResponse",
            "GenerationStatusResponse",
            "GenerateResponse",
            "TranscribeResponse",
            "ModelOpResponse",
            "UpdateModelConfigResponse",
            "UpdateStartupConfigResponse",
            "PromptsListResponse",
            "PromptDetailsResponse",
            "DeletePromptResponse",
            "RenamePromptResponse",
        ):
            self.assertIn(name, schemas)

    def test_openapi_union_route_uses_anyof(self):
        """/prompt-details returns two disjoint shapes — the union must land
        in the spec as anyOf, not silently collapse to one member."""
        import qwen3_tts.server.app as app_mod

        spec = app_mod.app.openapi()
        resp = spec["paths"]["/prompt-details"]["get"]["responses"]["200"]
        self.assertIn("anyOf", resp["content"]["application/json"]["schema"])


@_skip
class TestLoadModelDedupContracts(_ContractTestBase):
    """Phase 2c (#214 item 3): the ``deduped`` / ``warmup_failed`` fields.

    ``test_load_model_matches_contract`` presets ``models["clone"]`` and can
    only ever reach ``already_loaded`` — these are NEW round-trips, not
    extensions of it. Raw-payload assertions (``resp.json()[...]``) on the
    coalesced shape, because ``model_validate`` alone would mask an
    ASGI-layer field-stripping bug behind the Pydantic default
    (``response_model_exclude_unset=True`` omits unset fields).
    """

    def test_load_model_deduped_absent_on_already_loaded(self):
        """Characterization (GREEN-on-first-run): no attach, no field."""
        from qwen3_tts.server.validation import ModelOpResponse

        self.app.state.models["clone"] = MagicMock()
        resp = self.client.post(
            "/load-model", json={"model_type": "clone"}, headers=self.auth
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["status"], "already_loaded")
        self.assertNotIn(
            "deduped",
            body,
            "already_loaded never attaches — the field must stay unset so "
            "exclude_unset omits it (no client-visible wire drift)",
        )
        # Pydantic defaults exist for programmatic consumers of the model.
        model = ModelOpResponse.model_validate(body)
        self.assertFalse(model.deduped)
        self.assertFalse(model.warmup_failed)

    def test_load_model_deduped_true_on_coalesced_round_trip(self):
        """The real proof: a second POST attaches to the in-flight load and
        the RAW response carries ``deduped: true``."""
        import threading

        calls = []
        release = threading.Event()
        sentinel = MagicMock()
        timer = threading.Timer(0.3, release.set)
        self.addCleanup(timer.cancel)
        timer.start()

        def _load(model_type, warmup=False):
            calls.append(model_type)
            if len(calls) == 1:
                release.wait(timeout=10)
            return sentinel

        box = {}
        # Explicit reset: _restore_app_state skips keys whose saved value is
        # None, so a prior test's model entry would leak in here.
        self.app.state.models["clone"] = None

        def _first_post():
            box["first"] = self.client.post(
                "/load-model", json={"model_type": "clone"}, headers=self.auth
            )

        first = threading.Thread(target=_first_post, daemon=True)
        try:
            with (
                patch("qwen3_tts.core.engine.load_model", side_effect=_load),
                patch(
                    "qwen3_tts.core.config.get_model_info",
                    return_value={"name": "qwen3-tts-clone"},
                ),
            ):
                # Start INSIDE the patch context — outside it, the thread can
                # win the race and hit the real engine (a multi-GB load).
                first.start()
                for _ in range(500):
                    if self.app.state.model_loads["clone"] is not None:
                        break
                    time.sleep(0.01)
                second = self.client.post(
                    "/load-model", json={"model_type": "clone"}, headers=self.auth
                )
                first.join(timeout=10)
        finally:
            release.set()

        self.assertEqual(calls, ["clone"], "one construction for two POSTs")
        self.assertEqual(second.status_code, 200, second.text)
        self.assertIsNotNone(box.get("first"), "first POST never completed")
        self.assertEqual(box["first"].status_code, 200, box["first"].text)
        self.assertEqual(
            box["first"].json().get("deduped"),
            None,
            "the owner must not claim deduped (exclude_unset omits it)",
        )
        self.assertEqual(
            second.json().get("deduped"),
            True,
            "the attaching duplicate's raw payload must carry deduped: true",
        )
        self.assertEqual(second.json().get("status"), "loaded")

    def test_load_model_warmup_failed_field_contract(self):
        """W1: warm-up throws -> 200 with ``warmup_failed: true``, model kept."""
        sentinel = MagicMock()
        cleanup_calls = []
        self.app.state.models["design"] = None

        with (
            patch(
                "qwen3_tts.core.engine.load_model", return_value=sentinel
            ),
            patch(
                "qwen3_tts.core.engine.model_loader._warmup_disabled",
                return_value=False,
            ),
            patch(
                "qwen3_tts.core.engine.model_loader._warmup_model",
                side_effect=RuntimeError("warmup boom"),
            ),
            patch(
                "qwen3_tts.core.engine.unload_model_cleanup",
                side_effect=lambda: cleanup_calls.append(1),
            ),
        ):
            resp = self.client.post(
                "/load-model", json={"model_type": "design"}, headers=self.auth
            )

        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json().get("status"), "loaded")
        self.assertEqual(
            resp.json().get("warmup_failed"),
            True,
            "raw payload must carry warmup_failed: true (W1)",
        )
        self.assertIs(
            self.app.state.models["design"],
            sentinel,
            "the model must stay assigned after a warm-up failure",
        )
        self.assertEqual(cleanup_calls, [], "no recovery cleanup may run")


if __name__ == "__main__":
    unittest.main()
