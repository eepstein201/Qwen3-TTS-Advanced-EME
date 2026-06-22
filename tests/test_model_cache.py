"""Tests for model_cache.py module."""
import pathlib
import unittest
from datetime import datetime
from unittest import mock


class TestSharedFormatSize(unittest.TestCase):
    """Tests that _format_size lives in qwen3_tts.tools._shared."""

    def test_import_from_shared(self):
        """_format_size is importable from the shared module."""
        from qwen3_tts.tools._shared import _format_size
        self.assertTrue(callable(_format_size))

    def test_shared_bytes(self):
        from qwen3_tts.tools._shared import _format_size
        self.assertEqual(_format_size(500), "500.0 B")

    def test_shared_kilobytes(self):
        from qwen3_tts.tools._shared import _format_size
        self.assertEqual(_format_size(1024), "1.0 KB")

    def test_shared_megabytes(self):
        from qwen3_tts.tools._shared import _format_size
        self.assertEqual(_format_size(1024 * 1024), "1.0 MB")

    def test_shared_gigabytes(self):
        from qwen3_tts.tools._shared import _format_size
        self.assertEqual(_format_size(1024 * 1024 * 1024), "1.0 GB")

    def test_shared_terabytes(self):
        from qwen3_tts.tools._shared import _format_size
        self.assertEqual(_format_size(1024 * 1024 * 1024 * 1024), "1.0 TB")


class TestFormatSize(unittest.TestCase):
    """Tests for _format_size function."""

    def test_formats_bytes(self):
        """Formats bytes correctly."""
        from qwen3_tts.tools.model_cache import _format_size
        self.assertEqual(_format_size(500), "500.0 B")

    def test_formats_kilobytes(self):
        """Formats kilobytes correctly."""
        from qwen3_tts.tools.model_cache import _format_size
        self.assertEqual(_format_size(1024), "1.0 KB")
        self.assertEqual(_format_size(2048), "2.0 KB")

    def test_formats_megabytes(self):
        """Formats megabytes correctly."""
        from qwen3_tts.tools.model_cache import _format_size
        self.assertEqual(_format_size(1024 * 1024), "1.0 MB")

    def test_formats_gigabytes(self):
        """Formats gigabytes correctly."""
        from qwen3_tts.tools.model_cache import _format_size
        self.assertEqual(_format_size(1024 * 1024 * 1024), "1.0 GB")


class TestGetModelDirSize(unittest.TestCase):
    """Tests for _get_model_dir_size function."""

    def test_returns_zero_for_nonexistent_dir(self):
        """Returns 0 if directory doesn't exist."""
        from qwen3_tts.tools.model_cache import _get_model_dir_size

        mock_path = mock.MagicMock(spec=pathlib.Path)
        mock_path.is_dir.return_value = False
        result = _get_model_dir_size(mock_path)
        self.assertEqual(result, 0)

    def test_returns_zero_for_empty_dir(self):
        """Returns 0 for empty directory."""
        from qwen3_tts.tools.model_cache import _get_model_dir_size

        mock_path = mock.MagicMock(spec=pathlib.Path)
        mock_path.is_dir.return_value = True
        mock_path.rglob.return_value = []
        result = _get_model_dir_size(mock_path)
        self.assertEqual(result, 0)

    def test_sums_file_sizes(self):
        """Sums sizes of all files in directory."""
        from qwen3_tts.tools.model_cache import _get_model_dir_size

        mock_path = mock.MagicMock(spec=pathlib.Path)
        mock_path.is_dir.return_value = True

        # Create mock files
        mock_file1 = mock.MagicMock()
        mock_file1.is_file.return_value = True
        mock_file1.stat.return_value.st_size = 1000

        mock_file2 = mock.MagicMock()
        mock_file2.is_file.return_value = True
        mock_file2.stat.return_value.st_size = 2000

        mock_path.rglob.return_value = [mock_file1, mock_file2]
        result = _get_model_dir_size(mock_path)
        self.assertEqual(result, 3000)


class TestGetModelInfo(unittest.TestCase):
    """Tests for _get_model_info function."""

    def test_parses_torch_model(self):
        """Correctly parses PyTorch model info."""
        from qwen3_tts.tools.model_cache import _get_model_info

        mock_path = mock.MagicMock(spec=pathlib.Path)
        mock_path.name = "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"

        with mock.patch('qwen3_tts.tools.model_cache._get_model_dir_size', return_value=3500000000):
            with mock.patch('qwen3_tts.tools.model_cache._get_model_access_time', return_value=datetime.now()):
                result = _get_model_info(mock_path)
                self.assertEqual(result["backend"], "torch")
                self.assertEqual(result["model_type"], "clone")
                self.assertEqual(result["model_size"], "1.7B")

    def test_parses_small_model(self):
        """Correctly parses 0.6B model size."""
        from qwen3_tts.tools.model_cache import _get_model_info

        mock_path = mock.MagicMock(spec=pathlib.Path)
        mock_path.name = "models--Qwen--Qwen3-TTS-12Hz-0.6B-Base"

        with mock.patch('qwen3_tts.tools.model_cache._get_model_dir_size', return_value=2000000000):
            with mock.patch('qwen3_tts.tools.model_cache._get_model_access_time', return_value=datetime.now()):
                result = _get_model_info(mock_path)
                self.assertEqual(result["model_size"], "0.6B")


class TestListModels(unittest.TestCase):
    """Tests for list_models function."""

    def test_returns_empty_when_cache_missing(self):
        """Returns empty list when cache doesn't exist."""
        from qwen3_tts.tools.model_cache import list_models

        with mock.patch('qwen3_tts.tools.model_cache.HF_CACHE') as mock_cache:
            mock_cache.exists.return_value = False
            result = list_models()
            self.assertEqual(result, [])

    def test_returns_empty_when_no_tts_models(self):
        """Returns empty list when no TTS models found."""
        from qwen3_tts.tools.model_cache import list_models

        with mock.patch('qwen3_tts.tools.model_cache.HF_CACHE') as mock_cache:
            mock_cache.exists.return_value = True
            mock_dir = mock.MagicMock()
            mock_dir.name = "models--other--model"
            mock_dir.is_dir.return_value = True
            mock_cache.iterdir.return_value = [mock_dir]
            result = list_models()
            self.assertEqual(result, [])

    def test_returns_tts_models(self):
        """Returns list of TTS models."""
        from qwen3_tts.tools.model_cache import list_models

        with mock.patch('qwen3_tts.tools.model_cache.HF_CACHE') as mock_cache:
            mock_cache.exists.return_value = True

            mock_dir = mock.MagicMock()
            mock_dir.name = "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"
            mock_dir.is_dir.return_value = True

            mock_cache.iterdir.return_value = [mock_dir]

            with mock.patch('qwen3_tts.tools.model_cache._get_model_info') as mock_info:
                mock_info.return_value = {"name": mock_dir.name, "last_access": datetime.now()}
                result = list_models()
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0]["name"], "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base")


class TestGetTotalSize(unittest.TestCase):
    """Tests for get_total_size function."""

    def test_returns_zero_when_no_models(self):
        """Returns 0 when no models cached."""
        from qwen3_tts.tools.model_cache import get_total_size

        with mock.patch('qwen3_tts.tools.model_cache.list_models', return_value=[]):
            result = get_total_size()
            self.assertEqual(result, 0)

    def test_sums_model_sizes(self):
        """Sums sizes of all models."""
        from qwen3_tts.tools.model_cache import get_total_size

        models = [
            {"size_bytes": 1000},
            {"size_bytes": 2000},
            {"size_bytes": 3000},
        ]

        with mock.patch('qwen3_tts.tools.model_cache.list_models', return_value=models):
            result = get_total_size()
            self.assertEqual(result, 6000)


class TestModelPrefixes(unittest.TestCase):
    """Tests for model prefix constants."""

    def test_torch_prefixes_exist(self):
        """Torch model prefixes are defined."""
        from qwen3_tts.tools.model_cache import _TORCH_MODEL_PREFIXES
        self.assertGreater(len(_TORCH_MODEL_PREFIXES), 0)
        self.assertTrue(any("Qwen--" in p for p in _TORCH_MODEL_PREFIXES))

    def test_mlx_prefixes_exist(self):
        """MLX model prefixes are defined."""
        from qwen3_tts.tools.model_cache import _MLX_MODEL_PREFIXES
        self.assertGreater(len(_MLX_MODEL_PREFIXES), 0)
        self.assertTrue(any("mlx-community" in p for p in _MLX_MODEL_PREFIXES))

    def test_model_aliases_exist(self):
        """Model aliases are defined."""
        from qwen3_tts.tools.model_cache import _MODEL_ALIASES
        self.assertIn("Qwen3-TTS-12Hz-1.7B-Base", _MODEL_ALIASES)
        self.assertEqual(_MODEL_ALIASES["Qwen3-TTS-12Hz-1.7B-Base"]["torch"], "clone")
