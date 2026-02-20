#!/usr/bin/env python3
"""Tests for flash-attn pre-built wheel URL construction logic.

The URL-building logic from colab_notebook.ipynb Cell 1 is tested here
in isolation. No GPU, Colab, or actual download required.

Run: python -m pytest tests/test_flash_attn_install.py -v
"""
import re
import unittest


def _build_flash_attn_wheel_url(flash_attn_version, cuda_ver_str, torch_ver_str, py_ver_str):
    """
    Build the Dao-AILab GitHub releases URL for a pre-built flash-attn wheel.

    Args:
        flash_attn_version: e.g. "2.7.4"
        cuda_ver_str:       e.g. "124"  (CUDA 12.4 -> "12.4" -> strip dot -> "124")
        torch_ver_str:      e.g. "260"  (torch 2.6.0 -> digits only -> "260")
        py_ver_str:         e.g. "cp312"

    Returns:
        Full URL string, or None if any input is empty.
    """
    if not all([flash_attn_version, cuda_ver_str, torch_ver_str, py_ver_str]):
        return None
    whl = (
        f"flash_attn-{flash_attn_version}"
        f"+cu{cuda_ver_str}torch{torch_ver_str}cxx11abiFALSE"
        f"-{py_ver_str}-{py_ver_str}-linux_x86_64.whl"
    )
    base = "https://github.com/Dao-AILab/flash-attention/releases/download"
    return f"{base}/v{flash_attn_version}/{whl}"


def _normalize_cuda_version(cuda_string):
    """'12.4.0' or '12.4' -> '124'"""
    parts = cuda_string.split(".")
    return parts[0] + parts[1]  # major + minor only


def _normalize_torch_version(torch_string):
    """'2.6.0+cu124' -> '260'"""
    clean = re.sub(r'[^0-9.]', '', torch_string.split('+')[0])
    parts = clean.split('.')[:3]
    while len(parts) < 3:
        parts.append('0')
    return ''.join(parts[:3])


class TestFlashAttnWheelUrlConstruction(unittest.TestCase):

    def test_typical_l4_environment_url(self):
        """L4 GPU: CUDA 12.4, torch 2.6.0, Python 3.12 produces a valid URL."""
        url = _build_flash_attn_wheel_url("2.7.4", "124", "260", "cp312")
        self.assertIn("Dao-AILab/flash-attention", url)
        self.assertIn("v2.7.4", url)
        self.assertIn("cu124", url)
        self.assertIn("torch260", url)
        self.assertIn("cp312", url)
        self.assertIn("linux_x86_64", url)
        self.assertTrue(url.endswith(".whl"))

    def test_a100_environment_url(self):
        """A100: CUDA 12.1, torch 2.5.1, Python 3.11."""
        url = _build_flash_attn_wheel_url("2.7.4", "121", "251", "cp311")
        self.assertIn("cu121", url)
        self.assertIn("torch251", url)
        self.assertIn("cp311", url)

    def test_missing_version_returns_none(self):
        """Empty flash_attn_version returns None."""
        self.assertIsNone(_build_flash_attn_wheel_url("", "124", "260", "cp312"))

    def test_missing_cuda_returns_none(self):
        self.assertIsNone(_build_flash_attn_wheel_url("2.7.4", "", "260", "cp312"))

    def test_url_includes_cxx11abi_false(self):
        """Colab standard requires cxx11abiFALSE in wheel filename."""
        url = _build_flash_attn_wheel_url("2.7.4", "124", "260", "cp312")
        self.assertIn("cxx11abiFALSE", url)

    def test_cuda_normalization_with_patch(self):
        """CUDA '12.4.0' normalizes to '124'."""
        self.assertEqual(_normalize_cuda_version("12.4.0"), "124")

    def test_cuda_normalization_without_patch(self):
        """CUDA '12.1' normalizes to '121'."""
        self.assertEqual(_normalize_cuda_version("12.1"), "121")

    def test_torch_normalization_strips_cu_suffix(self):
        """torch '2.6.0+cu124' normalizes to '260'."""
        self.assertEqual(_normalize_torch_version("2.6.0+cu124"), "260")

    def test_torch_normalization_clean_version(self):
        """torch '2.5.1' normalizes to '251'."""
        self.assertEqual(_normalize_torch_version("2.5.1"), "251")

    def test_torch_normalization_short_version(self):
        """torch '2.6' normalizes to '260' (pads minor)."""
        self.assertEqual(_normalize_torch_version("2.6"), "260")


if __name__ == "__main__":
    unittest.main()
