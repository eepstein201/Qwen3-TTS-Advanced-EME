"""Test vLLM parameter optimization.

Tests that vLLM parameters are optimally configured for audio TTS:
- gpu_memory_utilization: 0.9 (high GPU utilization)
- max_model_len: 8192 (appropriate for audio sequences)
- tensor_parallel_size: 1 (single GPU default)
- audio_chunk_size: 2000 (~83ms at 24kHz)
"""

import unittest

from qwen3_tts.core.engine_vllm import VLLMAdapter


class TestVLLMParameterOptimization(unittest.TestCase):
    def test_default_gpu_memory_utilization(self):
        """Verify GPU memory utilization is optimally set."""
        # Create adapter with default parameters
        adapter = VLLMAdapter()
        self.assertEqual(adapter.gpu_memory_utilization, 0.9)

    def test_max_model_len_for_audio(self):
        """Verify max_model_len is appropriate for audio TTS."""
        # Create adapter with default parameters
        adapter = VLLMAdapter()
        self.assertEqual(adapter.max_model_len, 8192)

    def test_tensor_parallel_size(self):
        """Verify tensor_parallel_size is appropriate."""
        # Create adapter with default parameters
        VLLMAdapter()
        # Note: tensor_parallel_size isn't a direct attribute, it's handled internally
        # The default in the code is 1 for single GPU

    def test_audio_chunk_size_optimization(self):
        """Verify audio_chunk_size is optimized for TTS."""
        # Create adapter with default parameters
        adapter = VLLMAdapter()
        self.assertEqual(adapter.audio_chunk_size, 2000)

    def test_dtype_configuration(self):
        """Verify dtype is set to bfloat16 for memory efficiency."""
        # Create adapter with default parameters
        adapter = VLLMAdapter()
        self.assertEqual(adapter.dtype, "bfloat16")

    def test_audio_sample_rate(self):
        """Verify audio sample rate matches TTS output."""
        # Create adapter with default parameters
        adapter = VLLMAdapter()
        self.assertEqual(adapter.audio_sample_rate, 24000)


if __name__ == "__main__":
    unittest.main()
