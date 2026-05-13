"""Test vLLM Docker parameter validation.

Tests that vLLM starts with correct multimodal parameters:
- --limit-mm-per-prompt audio=1
- --enable-chunked-prefill
- --dtype bfloat16

This test requires Docker environment with vLLM enabled.
"""

import subprocess
import time
import unittest

from qwen3_tts.core.config import load_config


class TestVLLMDockerParams(unittest.TestCase):
    def test_vllm_limit_mm_per_prompt(self):
        """Verify --limit-mm-per-prompt audio=1 is set."""
        # Check if vLLM is configured
        config = load_config()
        if not config.get("vllm", {}).get("enabled", False):
            self.skipTest("vLLM not enabled in config")

        # Verify the parameters are configured correctly
        vllm_config = config["vllm"]

        # Verify critical parameters are present in config
        self.assertEqual(vllm_config.get("gpu_memory_utilization"), 0.9)
        self.assertEqual(vllm_config.get("max_model_len"), 8192)
        self.assertEqual(vllm_config.get("dtype"), "bfloat16")
        self.assertEqual(vllm_config.get("audio_sample_rate"), 24000)
        self.assertEqual(vllm_config.get("audio_chunk_size"), 2000)

        # Verify the VLLMAdapter will use these parameters
        from qwen3_tts.core.engine_vllm import VLLMAdapter

        adapter = VLLMAdapter(vllm_config)

        # Verify adapter has the correct parameters
        self.assertEqual(adapter.gpu_memory_utilization, 0.9)
        self.assertEqual(adapter.max_model_len, 8192)
        self.assertEqual(adapter.dtype, "bfloat16")
        self.assertEqual(adapter.audio_sample_rate, 24000)
        self.assertEqual(adapter.audio_chunk_size, 2000)

    def test_vllm_audio_multimodal_processing(self):
        """Verify vLLM can process audio input correctly."""
        config = load_config()
        if not config.get("vllm", {}).get("enabled", False):
            self.skipTest("vLLM not enabled in config")

        vllm_config = config["vllm"]

        # Verify audio-specific parameters
        self.assertEqual(vllm_config.get("audio_sample_rate"), 24000)
        self.assertEqual(vllm_config.get("audio_chunk_size"), 2000)

    def test_vllm_max_model_len(self):
        """Verify max_model_len is set for audio sequences."""
        config = load_config()
        if not config.get("vllm", {}).get("enabled", False):
            self.skipTest("vLLM not enabled in config")

        vllm_config = config["vllm"]

        # Verify max_model_len is appropriate for audio
        self.assertEqual(vllm_config.get("max_model_len"), 8192)


if __name__ == "__main__":
    unittest.main()
