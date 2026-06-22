"""Tests for audio pipeline — inference configuration behavior."""
import inspect

import numpy as np


class TestMetalRetrySilenceGap:
    """Tests for R-27: Metal retry path should not use hardcoded silence gap."""

    def test_metal_retry_does_not_hardcode_silence(self):
        """Metal retry sub-chunk concatenation must not use sr * 0.1 literal.

        The silence gap should come from config (generation.silence_gap_seconds),
        not be hardcoded to 0.1 seconds.
        """
        from qwen3_tts.core.engine import inference

        source = inspect.getsource(inference._run_inference_single)
        assert "sr * 0.1" not in source, (
            "Metal retry still uses hardcoded 0.1s silence — should read from config"
        )


class TestLUFSNormalizationToggle:
    """Tests for R-23: LUFS normalization opt-in via generation.lufs_normalize."""

    def test_run_inference_applies_lufs_when_enabled(self, monkeypatch):
        """When generation.lufs_normalize=True, process_audio is called with lufs_target."""
        from unittest.mock import MagicMock, patch

        from qwen3_tts.core.engine import inference

        fake_audio = np.zeros(24000, dtype=np.float32)
        fake_sr = 24000

        monkeypatch.setattr(
            inference, "_run_inference_single",
            lambda *a, **kw: (fake_audio, fake_sr),
        )
        monkeypatch.setattr(
            inference._DEFAULT_CONFIG_LOADER, "load",
            lambda: {"generation": {"lufs_normalize": True, "silence_gap_seconds": 0.0}},
        )

        with patch("qwen3_tts.core.engine.inference.process_audio",
                   return_value=(fake_audio, fake_sr)) as mock_pa:
            inference.run_inference(
                model=MagicMock(), text="hello world", mode="clone",
                gen_params={"temperature": 0.7, "top_k": 50, "top_p": 0.95, "repetition_penalty": 1.05},
            )
            mock_pa.assert_called_once()
            _, kwargs = mock_pa.call_args
            assert kwargs.get("lufs_target") is not None

    def test_run_inference_skips_lufs_by_default(self, monkeypatch):
        """When generation.lufs_normalize is absent/false, process_audio is not called."""
        from unittest.mock import MagicMock, patch

        from qwen3_tts.core.engine import inference

        fake_audio = np.zeros(24000, dtype=np.float32)
        fake_sr = 24000

        monkeypatch.setattr(
            inference, "_run_inference_single",
            lambda *a, **kw: (fake_audio, fake_sr),
        )
        monkeypatch.setattr(
            inference._DEFAULT_CONFIG_LOADER, "load",
            lambda: {"generation": {"silence_gap_seconds": 0.0}},
        )

        with patch("qwen3_tts.core.engine.inference.process_audio") as mock_pa:
            inference.run_inference(
                model=MagicMock(), text="hello world", mode="clone",
                gen_params={"temperature": 0.7, "top_k": 50, "top_p": 0.95, "repetition_penalty": 1.05},
            )
            mock_pa.assert_not_called()


class TestLufsTargetConfigurable:
    """R-23: lufs_target is read from generation.lufs_target (default -16.0)."""

    def test_custom_lufs_target_propagates(self):
        from unittest.mock import patch

        import numpy as np
        captured = {}

        def fake_process_audio(audio, sr, **kwargs):
            captured.update(kwargs)
            return audio, sr

        with patch(
            "qwen3_tts.core.engine.inference.process_audio",
            side_effect=fake_process_audio,
        ), patch(
            "qwen3_tts.core.engine.inference.load_config",
            return_value={
                "generation": {
                    "lufs_normalize": True,
                    "lufs_target": -19.0,
                    "silence_gap_seconds": 0.0,
                }
            },
        ):
            from qwen3_tts.core.engine.inference import _maybe_apply_lufs
            wav = np.zeros(1000, dtype=np.float32)
            _maybe_apply_lufs(wav, 24000)

        assert captured.get("lufs_target") == -19.0

    def test_default_lufs_target_is_minus_16(self):
        from unittest.mock import patch

        import numpy as np
        captured = {}

        def fake_process_audio(audio, sr, **kwargs):
            captured.update(kwargs)
            return audio, sr

        with patch(
            "qwen3_tts.core.engine.inference.process_audio",
            side_effect=fake_process_audio,
        ), patch(
            "qwen3_tts.core.engine.inference.load_config",
            return_value={"generation": {"lufs_normalize": True}},
        ):
            from qwen3_tts.core.engine.inference import _maybe_apply_lufs
            wav = np.zeros(1000, dtype=np.float32)
            _maybe_apply_lufs(wav, 24000)

        assert captured.get("lufs_target") == -16.0
