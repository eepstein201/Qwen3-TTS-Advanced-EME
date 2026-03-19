"""Tests validating FastAPI/vLLM decoupling constraints.

Ensures app.py does not directly import heavy inference libraries
at module scope, keeping the FastAPI server lightweight.
"""

import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDecoupledInference(unittest.TestCase):
    """Verify FastAPI server doesn't import heavy libs at module scope."""

    def _get_top_level_imports(self, filepath):
        """Extract all top-level import names from a Python file."""
        with open(filepath) as f:
            tree = ast.parse(f.read())

        imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split(".")[0])
        return imports

    def test_app_py_no_torch_at_module_scope(self):
        """app.py must not import torch at module scope."""
        imports = self._get_top_level_imports("qwen3_tts/server/app.py")
        self.assertNotIn("torch", imports,
                         "app.py must not import torch at module scope")

    def test_app_py_no_vllm_at_module_scope(self):
        """app.py must not import vllm at module scope."""
        imports = self._get_top_level_imports("qwen3_tts/server/app.py")
        self.assertNotIn("vllm", imports,
                         "app.py must not import vllm at module scope")

    def test_app_py_no_transformers_at_module_scope(self):
        """app.py must not import transformers at module scope."""
        imports = self._get_top_level_imports("qwen3_tts/server/app.py")
        self.assertNotIn("transformers", imports,
                         "app.py must not import transformers at module scope")

    def test_app_py_no_mlx_at_module_scope(self):
        """app.py must not import mlx at module scope."""
        imports = self._get_top_level_imports("qwen3_tts/server/app.py")
        self.assertNotIn("mlx", imports,
                         "app.py must not import mlx at module scope")

    def test_engine_vllm_exists(self):
        """engine_vllm.py should exist as a separate adapter module."""
        self.assertTrue(
            os.path.exists("qwen3_tts/core/engine_vllm.py"),
            "engine_vllm.py must exist as a separate inference adapter",
        )


if __name__ == "__main__":
    unittest.main()
