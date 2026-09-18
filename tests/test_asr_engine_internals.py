#!/usr/bin/env python3
"""Failure-arm tests for the engine ASR module internals (Step 4B.1 item 4).

``core/engine/asr.py`` was the lowest-coverage risk-bearing module (62%): the
endpoint tests patch the engine facade, so the module's own bodies never ran —
only the no-op paths were covered. These tests drive the REAL code with
module-global stubbing, restoring every mutation via ``addCleanup``
registered BEFORE the mutation (standing project discipline — a failed
assertion must not leak stubs into the next test).

Covers:
- ``unload_asr_model`` — the #214-item-2 real body: both model globals nulled
  under ``_asr_lock``; ``gc.collect()`` BEFORE ``torch.cuda.empty_cache()``
  (order matters — empty_cache before gc releases nothing); the torch branch
  fires only when a torch model was actually held.
- ``_ensure_asr_torch_loaded`` — pipeline construction with the
  device-name→device mapping (cuda→0, mps→"mps", cpu→-1), the
  already-loaded fast path, and the transformers-ImportError re-raise.
- ``_ensure_mlx_whisper_processor`` — the HF-download shim arms: snapshot
  lookup failure (warn + return), per-file fetch-and-copy of missing
  processor files, and the per-file skip-on-error loop.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import qwen3_tts.core.engine.asr as asr_module

VP = "qwen3_tts.core.engine.asr"


class _AsrGlobalRestoreMixin(unittest.TestCase):
    """Restore the ASR module globals no matter how the test exits."""

    def setUp(self):
        for name in ("_asr_model_mlx", "_asr_model_torch"):
            original = getattr(asr_module, name, None)
            self.addCleanup(setattr, asr_module, name, original)
            setattr(asr_module, name, None)


class TestUnloadAsrModel(_AsrGlobalRestoreMixin):
    def _fake_torch(self, cuda_available):
        fake = MagicMock(name="torch")
        fake.cuda.is_available.return_value = cuda_available
        return fake

    def test_unloads_both_models_then_collects_then_clears_cuda_cache(self):
        asr_module._asr_model_mlx = object()
        asr_module._asr_model_torch = object()
        fake_torch = self._fake_torch(True)
        with patch.dict(sys.modules, {"torch": fake_torch}), patch(
            "gc.collect"
        ) as mock_gc:
            asr_module.unload_asr_model()
        self.assertIsNone(asr_module._asr_model_mlx)
        self.assertIsNone(asr_module._asr_model_torch)
        mock_gc.assert_called_once()
        fake_torch.cuda.empty_cache.assert_called_once()

    def test_mlx_only_unload_never_clears_cuda_cache(self):
        asr_module._asr_model_mlx = object()
        fake_torch = self._fake_torch(True)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            asr_module.unload_asr_model()
        self.assertIsNone(asr_module._asr_model_mlx)
        fake_torch.cuda.empty_cache.assert_not_called()

    def test_cuda_unavailable_skips_empty_cache(self):
        asr_module._asr_model_torch = object()
        fake_torch = self._fake_torch(False)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            asr_module.unload_asr_model()
        fake_torch.cuda.empty_cache.assert_not_called()


class TestEnsureAsrTorchLoaded(_AsrGlobalRestoreMixin):
    def test_already_loaded_returns_without_building(self):
        existing = object()
        asr_module._asr_model_torch = existing
        with patch("transformers.pipeline") as mock_pipeline:
            asr_module._ensure_asr_torch_loaded()
        mock_pipeline.assert_not_called()
        self.assertIs(asr_module._asr_model_torch, existing)

    def _assert_device(self, device_name, expected_device):
        with patch("transformers.pipeline") as mock_pipeline, patch(
            "qwen3_tts.core.config.get_device", return_value=device_name
        ):
            asr_module._ensure_asr_torch_loaded()
        self.assertIs(asr_module._asr_model_torch, mock_pipeline.return_value)
        self.assertEqual(mock_pipeline.call_args.kwargs["device"], expected_device)

    def test_device_mapping_cuda_mps_cpu(self):
        for device_name, expected in (("cuda", 0), ("mps", "mps"), ("cpu", -1)):
            with self.subTest(device_name=device_name):
                asr_module._asr_model_torch = None
                self._assert_device(device_name, expected)

    def test_transformers_import_error_is_rewritten(self):
        with patch(
            "transformers.pipeline", side_effect=ImportError("nope")
        ):
            with self.assertRaises(ImportError) as ctx:
                asr_module._ensure_asr_torch_loaded()
        self.assertIn("ASR transcription requires transformers", str(ctx.exception))
        self.assertIsNone(asr_module._asr_model_torch)


class TestEnsureMlxWhisperProcessor(unittest.TestCase):
    """The HF snapshot shim: fetch missing processor files, skip on error."""

    def _fake_hub(self, snapshot_dir, download_results):
        """Fake huggingface_hub module; download_results maps filename->exc-or-path."""
        hub = MagicMock(name="huggingface_hub")
        hub.snapshot_download.return_value = str(snapshot_dir)

        def _download(repo_id, filename, revision):
            result = download_results[filename]
            if isinstance(result, Exception):
                raise result
            return str(result)

        hub.hf_hub_download.side_effect = _download
        return hub

    def _snapshot_dir(self):
        d = Path(tempfile.mkdtemp(prefix="asr_whisper_snapshot_"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        return d

    def test_snapshot_failure_warns_and_returns(self):
        hub = MagicMock(name="huggingface_hub")
        hub.snapshot_download.side_effect = RuntimeError("hub down")
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            asr_module._ensure_mlx_whisper_processor()
        hub.hf_hub_download.assert_not_called()

    def test_missing_files_fetched_and_copied(self):
        snapshot = self._snapshot_dir()  # empty: every processor file missing
        src = Path(tempfile.mkdtemp(prefix="asr_whisper_src_"))
        self.addCleanup(shutil.rmtree, src, ignore_errors=True)
        src_files = {}
        for name in asr_module._WHISPER_PROCESSOR_FILES:
            f = src / name
            f.write_text("stub")
            src_files[name] = f

        hub = self._fake_hub(snapshot, src_files)
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            asr_module._ensure_mlx_whisper_processor()

        self.assertEqual(
            hub.hf_hub_download.call_count, len(asr_module._WHISPER_PROCESSOR_FILES)
        )
        for name in asr_module._WHISPER_PROCESSOR_FILES:
            self.assertTrue(
                (snapshot / name).exists(), f"{name} not copied into snapshot"
            )

    def test_per_file_failure_skips_and_continues(self):
        snapshot = self._snapshot_dir()
        src = Path(tempfile.mkdtemp(prefix="asr_whisper_src_"))
        self.addCleanup(shutil.rmtree, src, ignore_errors=True)
        names = list(asr_module._WHISPER_PROCESSOR_FILES)
        results = {names[0]: RuntimeError("404")}
        for name in names[1:]:
            f = src / name
            f.write_text("stub")
            results[name] = f

        hub = self._fake_hub(snapshot, results)
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            asr_module._ensure_mlx_whisper_processor()  # must not raise

        self.assertFalse((snapshot / names[0]).exists())
        for name in names[1:]:
            self.assertTrue((snapshot / name).exists())


if __name__ == "__main__":
    unittest.main()
