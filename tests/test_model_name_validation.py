"""VLLMAdapter model_name validation (closes CodeQL py/command-line-injection).

model_name flows into the vLLM subprocess command list; while Popen uses a list
(no shell), an unvalidated value could inject a leading-dash flag. Validate it to
an HF-repo-id / local-path shape and reject shell metacharacters and leading dashes.
"""
import unittest

from qwen3_tts.core.engine_vllm import VLLMAdapter


class TestModelNameValidation(unittest.TestCase):
    def test_valid_hf_id(self):
        adapter = VLLMAdapter(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        self.assertEqual(adapter.model_name, "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    def test_valid_local_path(self):
        adapter = VLLMAdapter(model_name="/models/qwen3-tts")
        self.assertEqual(adapter.model_name, "/models/qwen3-tts")

    def test_valid_relative_path(self):
        adapter = VLLMAdapter(model_name="./models/qwen3-tts")
        self.assertEqual(adapter.model_name, "./models/qwen3-tts")

    def test_rejects_leading_dash(self):
        with self.assertRaises(ValueError):
            VLLMAdapter(model_name="--config-path=/etc/evil")

    def test_rejects_shell_metacharacters(self):
        for bad in ("model; rm -rf /", "model$(whoami)", "model`id`", "model|cat", "a b"):
            with self.assertRaises(ValueError):
                VLLMAdapter(model_name=bad)

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            VLLMAdapter(model_name="")


if __name__ == "__main__":
    unittest.main()
