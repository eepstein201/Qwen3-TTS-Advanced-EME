"""Tests for package metadata and entry points.

Validates that the pure Python package (v3.0) is correctly configured.
"""

import subprocess
import sys
import unittest

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Dummy decorator for when pytest is not available
    class _DummyMarkerFunc:
        """Represents a marker function like skipif that takes condition and returns decorator."""
        def __init__(self, name=None):
            self._name = name
        def __call__(self, condition, **kwargs):
            # skipif, etc. take condition as first arg, return a decorator
            return lambda f: f
    class _DummyMarker:
        def __call__(self, func):
            return func
        def __getattr__(self, name):
            # Return special function for skipif, otherwise return a callable marker
            if name == 'skipif':
                return _DummyMarkerFunc(name)
            return _DummyMarkerFunc(name)
        @property
        def unit(self):
            return self
    class _DummyMark:
        def __getattr__(self, name):
            return _DummyMarkerFunc()
    class _DummyPytest:
        mark = _DummyMark()
    pytest = _DummyPytest()



class TestPackageMetadata(unittest.TestCase):
    """Test package metadata and entry points."""

    def test_package_has_name_and_version(self):
        """Package should have correct name and version."""
        import importlib.metadata
        metadata = importlib.metadata.metadata('qwen3-tts')
        self.assertEqual(metadata['Name'], 'qwen3-tts')
        # Version should be 3.0.0 or higher
        version = metadata['Version']
        major = int(version.split('.')[0])
        self.assertGreaterEqual(major, 3)

    def test_entry_point_exists(self):
        """The 'tts' command should be registered as an entry point."""
        import importlib.metadata
        eps = importlib.metadata.entry_points()
        if hasattr(eps, 'select'):
            # Python 3.10+
            console_scripts = eps.select(group='console_scripts')
        else:
            console_scripts = eps.get('console_scripts', [])
        entry_names = [ep.name for ep in console_scripts]
        self.assertIn('tts', entry_names)

    def test_tts_command_help(self):
        """The 'tts' command should respond to --help."""
        result = subprocess.run(
            [sys.executable, '-m', 'qwen3_tts.cli', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn('Qwen3-TTS', result.stdout)

    def test_core_dependencies_importable(self):
        """Core dependencies should be importable."""
        # Core deps (always required)
        import click
        self.assertTrue(hasattr(click, 'Command'))

    def test_server_dep_group(self):
        """Server dependencies should be listed in optional-dependencies."""
        import importlib.metadata
        metadata = importlib.metadata.metadata('qwen3-tts')
        deps = metadata.get_all('Requires-Dist') or []
        # FastAPI should be in extras 'server' (note: metadata uses single quotes)
        server_deps = [d for d in deps if "extra == 'server'" in d]
        self.assertTrue(any('fastapi' in d.lower() for d in server_deps),
                       "fastapi should be in server dependencies")

    def test_torch_dep_group(self):
        """Torch dependencies should be listed in optional-dependencies."""
        import importlib.metadata
        metadata = importlib.metadata.metadata('qwen3-tts')
        deps = metadata.get_all('Requires-Dist') or []
        torch_deps = [d for d in deps if "extra == 'torch'" in d]
        self.assertTrue(any('torch' in d.lower() or 'qwen-tts' in d.lower() for d in torch_deps),
                       "torch or qwen-tts should be in torch dependencies")

    def test_mlx_dep_group(self):
        """MLX dependencies should be listed in optional-dependencies."""
        import importlib.metadata
        metadata = importlib.metadata.metadata('qwen3-tts')
        deps = metadata.get_all('Requires-Dist') or []
        mlx_deps = [d for d in deps if "extra == 'mlx'" in d]
        self.assertTrue(any('mlx' in d.lower() for d in mlx_deps),
                       "mlx should be in MLX dependencies")


if __name__ == '__main__':
    unittest.main()
