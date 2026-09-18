#!/usr/bin/env python3
"""Failure-arm tests for engine voice_prompt.py (Step 4B.1 item 5).

Covers the arms the #236 dedicated tests never reach:

- ``save_voice_prompt_mlx`` — the non-atomic pair guard: a ``.txt``-write
  failure must remove the just-written ``.wav`` (an orphaned ``.wav`` would
  make the MLX ``/prompts`` listing show a prompt that cannot load) and
  re-raise as a plain ``RuntimeError`` so the handler's 400
  ``invalid_audio`` clause cannot swallow a server-side fault.
- ``_load_pt_safe`` — the corrupted-``.pt`` fallback: the path-traversal
  ValueError when the realpath escapes ``voice_prompts/``, and the
  rebuild-hint RuntimeError for a prompt that IS in the directory (hint
  text depends on whether a sibling ``.wav`` exists).
- ``migrate_orphan_mlx_prompts`` — orphan ``.wav``+``.txt`` pairs without
  ``.pt``: migration via the lazily-imported builders, per-prompt failure
  isolation (one failure must not abort the loop), and the no-orphans
  fast path that never loads the clone model.

torch is faked via ``patch.dict(sys.modules)`` so these run in torchless CI.
"""

import math
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import qwen3_tts.core.engine.voice_prompt as vp_module

VP = "qwen3_tts.core.engine.voice_prompt"


class _PromptsDirMixin(unittest.TestCase):
    """Point VOICE_PROMPTS_DIR (imported name) at a fresh tmpdir."""

    def setUp(self):
        self.prompts_dir = Path(tempfile.mkdtemp(prefix="vp_failure_arms_"))
        self.addCleanup(tempfile.TemporaryDirectory, dir=str(self.prompts_dir))
        original = vp_module.VOICE_PROMPTS_DIR
        self.addCleanup(setattr, vp_module, "VOICE_PROMPTS_DIR", original)
        vp_module.VOICE_PROMPTS_DIR = self.prompts_dir


def _make_wav(path, seconds=1.0, rate=24_000):
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(n):
            sample = int(0.4 * 32_767 * math.sin(2 * math.pi * 440 * i / rate))
            frames += struct.pack("<h", sample)
        w.writeframes(bytes(frames))


class TestSaveVoicePromptMlxRollback(_PromptsDirMixin):
    def test_txt_failure_removes_orphan_wav_and_raises(self):
        ref_dir = Path(tempfile.mkdtemp(prefix="vp_ref_", dir=tempfile.gettempdir()))
        self.addCleanup(tempfile.TemporaryDirectory, dir=str(ref_dir))
        ref = ref_dir / "ref.wav"
        _make_wav(ref)

        # A DIRECTORY at the .txt path makes open(txt, "w") fail with
        # IsADirectoryError — a store-phase fault, not bad client audio.
        (self.prompts_dir / "orphan.txt").mkdir()

        with self.assertRaises(RuntimeError) as ctx:
            vp_module.save_voice_prompt_mlx("orphan", str(ref), "transcript")
        self.assertIn("failed to store voice prompt", str(ctx.exception))
        # The pair guard: no orphaned .wav may survive the .txt failure.
        self.assertFalse((self.prompts_dir / "orphan.wav").exists())


class TestLoadPtSafe(_PromptsDirMixin):
    def _fake_torch(self, load_result=None, load_exc=None):
        fake = MagicMock(name="torch")
        if load_exc is not None:
            fake.load.side_effect = load_exc
        else:
            fake.load.return_value = load_result
        return fake

    def test_success_returns_loaded_object(self):
        sentinel = object()
        with patch.dict(sys.modules, {"torch": self._fake_torch(load_result=sentinel)}):
            result = vp_module._load_pt_safe(
                str(self.prompts_dir / "p.pt"), "p.pt", "cpu"
            )
        self.assertIs(result, sentinel)

    def test_escape_from_prompts_dir_is_refused(self):
        outside = Path(tempfile.mkdtemp(prefix="vp_outside_"))
        self.addCleanup(tempfile.TemporaryDirectory, dir=str(outside))
        # qwen_tts is absent in this env, so the add_safe_globals ImportError
        # arm also runs on the way in — harmless here.
        with patch.dict(
            sys.modules, {"torch": self._fake_torch(load_exc=RuntimeError("corrupt"))}
        ):
            with self.assertRaises(ValueError) as ctx:
                vp_module._load_pt_safe(str(outside / "evil.pt"), "evil.pt", "cpu")
        self.assertIn("outside voice_prompts/ directory", str(ctx.exception))

    def test_corrupt_in_dir_hints_rebuild_when_wav_exists(self):
        _make_wav(self.prompts_dir / "p.wav")
        with patch.dict(
            sys.modules, {"torch": self._fake_torch(load_exc=RuntimeError("corrupt"))}
        ):
            with self.assertRaises(RuntimeError) as ctx:
                vp_module._load_pt_safe(
                    str(self.prompts_dir / "p.pt"), "p.pt", "cpu"
                )
        self.assertIn("tts voice rebuild p", str(ctx.exception))

    def test_corrupt_in_dir_hints_recreate_when_no_wav(self):
        with patch.dict(
            sys.modules, {"torch": self._fake_torch(load_exc=ValueError("bad pickle"))}
        ):
            with self.assertRaises(RuntimeError) as ctx:
                vp_module._load_pt_safe(
                    str(self.prompts_dir / "q.pt"), "q.pt", "cpu"
                )
        self.assertIn("tts voice create", str(ctx.exception))


class TestMigrateOrphanMlxPrompts(_PromptsDirMixin):
    def _orphan(self, base, transcript="hello"):
        _make_wav(self.prompts_dir / f"{base}.wav")
        (self.prompts_dir / f"{base}.txt").write_text(transcript)

    def test_no_orphans_never_loads_model(self):
        (self.prompts_dir / "already.pt").write_bytes(b"x")
        (self.prompts_dir / "already.wav")  # .pt exists → skipped regardless
        with patch(
            "qwen3_tts.core.engine.model_loader.load_model"
        ) as mock_load:
            migrated = vp_module.migrate_orphan_mlx_prompts()
        self.assertEqual(migrated, 0)
        mock_load.assert_not_called()

    def test_orphan_is_migrated_and_saved(self):
        self._orphan("orphan1")
        fake_torch = MagicMock(name="torch")

        def _save(obj, path):
            Path(path).write_bytes(b"saved")

        fake_torch.save.side_effect = _save
        with patch.dict(sys.modules, {"torch": fake_torch}), patch(
            f"{VP}.load_audio_for_cloning", return_value=([0.1] * 100, 24_000)
        ) as mock_load_audio, patch(
            "qwen3_tts.core.engine.inference.create_voice_prompt",
            return_value=object(),
        ) as mock_create, patch(
            "qwen3_tts.core.engine.model_loader.load_model"
        ) as mock_load_model:
            migrated = vp_module.migrate_orphan_mlx_prompts()

        self.assertEqual(migrated, 1)
        self.assertTrue((self.prompts_dir / "orphan1.pt").exists())
        mock_load_model.assert_called_once_with("clone")  # lazy, only when needed
        mock_load_audio.assert_called_once()
        self.assertEqual(mock_create.call_args.kwargs.get("x_vector_only_mode"), False)

    def test_one_failure_does_not_abort_the_loop(self):
        self._orphan("bad")
        self._orphan("good")

        def _create(model, audio, sr, transcript, x_vector_only_mode=False):
            if "bad" in str(getattr(_create, "current_wav", "")):
                raise RuntimeError("nope")
            return object()

        fake_torch = MagicMock(name="torch")
        fake_torch.save.side_effect = lambda obj, path: Path(path).write_bytes(b"s")
        with patch.dict(sys.modules, {"torch": fake_torch}), patch(
            f"{VP}.load_audio_for_cloning",
            side_effect=lambda wav: (
                setattr(_create, "current_wav", wav) or ([0.1] * 100, 24_000)
            ),
        ), patch(
            "qwen3_tts.core.engine.inference.create_voice_prompt",
            side_effect=_create,
        ), patch(
            "qwen3_tts.core.engine.model_loader.load_model"
        ):
            migrated = vp_module.migrate_orphan_mlx_prompts()

        self.assertEqual(migrated, 1)  # "good" migrated; "bad" warned and skipped
        self.assertTrue((self.prompts_dir / "good.pt").exists())
        self.assertFalse((self.prompts_dir / "bad.pt").exists())


if __name__ == "__main__":
    unittest.main()
