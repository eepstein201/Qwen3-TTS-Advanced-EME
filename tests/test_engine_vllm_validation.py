"""Tests for VLLMAdapter parameter validation (SEC-3)."""

import sys
import types
import unittest

# Ensure httpx is importable (VLLMAdapter imports it at module scope).
if "httpx" not in sys.modules:
    sys.modules["httpx"] = types.ModuleType("httpx")
    sys.modules["httpx"].AsyncClient = type("AsyncClient", (), {})
    sys.modules["httpx"].Timeout = type("Timeout", (), {"__init__": lambda *a, **kw: None})
    sys.modules["httpx"].Limits = type("Limits", (), {"__init__": lambda *a, **kw: None})
    sys.modules["httpx"].RequestError = type("RequestError", (Exception,), {})
    sys.modules["httpx"].HTTPStatusError = type("HTTPStatusError", (Exception,), {})

from qwen3_tts.core.engine_vllm import VLLMAdapter


class TestMmProcessorNameValidation(unittest.TestCase):
    """mm_processor_name must match the same regex as model_name."""

    def _make(self, mm_processor_name: str) -> VLLMAdapter:
        return VLLMAdapter(mm_processor_name=mm_processor_name)

    def test_valid_hf_id(self):
        adapter = self._make("Qwen/Qwen2-Audio-7B-Instruct")
        self.assertEqual(adapter.mm_processor_name, "Qwen/Qwen2-Audio-7B-Instruct")

    def test_valid_local_path(self):
        adapter = self._make("./local-model")
        self.assertEqual(adapter.mm_processor_name, "./local-model")

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            self._make("")

    def test_rejects_leading_dash(self):
        with self.assertRaises(ValueError):
            self._make("--malicious-flag")

    def test_rejects_shell_metacharacters(self):
        for bad in ["name; rm -rf /", "name && echo pwned", "name$(cmd)", "name|cat"]:
            with self.assertRaises(ValueError, msg=f"Should reject: {bad!r}"):
                self._make(bad)

    def test_rejects_spaces(self):
        with self.assertRaises(ValueError):
            self._make("has space")


if __name__ == "__main__":
    unittest.main()
