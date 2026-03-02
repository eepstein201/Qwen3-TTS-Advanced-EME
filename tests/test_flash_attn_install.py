#!/usr/bin/env python3
"""Tests for flash-attn pre-built wheel installation logic.

The wheel selection logic from colab_notebook.ipynb Cell 1 is tested here
in isolation. No GPU, Colab, or actual download required.

NOTE: The notebook's inline code produces 4-tuples (cu_num, th_num, url, name)
because it needs the download URL for pip install. The test helpers below use
simplified 3-tuples (cu_num, th_num, name) since the selection/sort logic only
operates on indices [0] and [1]. Behavior is identical.

Run: python -m pytest tests/test_flash_attn_install.py -v
"""
import re
import unittest

import pytest


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


def _select_best_wheel(candidates, cu_installed, th_installed):
    """
    Select compatible wheels sorted by preference (highest CUDA, then highest torch).

    Args:
        candidates:    list of (cu_num, th_num, name) tuples
        cu_installed:  numeric CUDA version of the runtime (e.g. 128 for CUDA 12.8)
        th_installed:  numeric torch version of the runtime (e.g. 2100 for torch 2.10.0)

    Returns:
        Ordered list of (cu_num, th_num, name) — best match first.
        If no candidates have versions <= installed, returns ALL candidates
        sorted highest-first as a last-resort fallback.
    """
    if not candidates:
        return []
    compat = [c for c in candidates if c[0] <= cu_installed and c[1] <= th_installed]
    if not compat:
        return sorted(candidates, key=lambda x: (x[0], x[1]), reverse=True)
    return sorted(compat, key=lambda x: (x[0], x[1]), reverse=True)


def _parse_wheel_candidates(asset_names, py_str):
    """
    Match flash-attn wheel filenames and extract (cu_num, th_num, name) tuples.

    Args:
        asset_names: list of filename strings from a GitHub release
        py_str:      e.g. "cp312"

    Returns:
        List of (cu_num: int, th_num: int, name: str) for matching wheels.
    """
    pattern = re.compile(
        r"flash_attn-[\d.]+\+cu(\d+)torch(\d+)cxx11abiFALSE"
        r"-(" + py_str + r")-\3-linux_x86_64\.whl"
    )
    candidates = []
    for name in asset_names:
        m = pattern.match(name)
        if m:
            candidates.append((int(m.group(1)), int(m.group(2)), name))
    return candidates


def _normalize_cuda_to_numeric(cuda_string):
    """'12.8' or '12.4.0' -> 128 or 124 (major*10 + minor, as int).

    Assumes CUDA minor versions are single-digit (0-9), which has been true
    for all NVIDIA releases through CUDA 12.x. If NVIDIA ever ships a 2-digit
    minor (e.g. 12.10), this formula and the wheel-tag parsing would need to
    switch to major*100 + minor.
    """
    parts = cuda_string.split(".")
    return int(parts[0]) * 10 + int(parts[1])


def _normalize_torch_version(torch_string):
    """'2.6.0+cu124' -> '260'"""
    clean = re.sub(r'[^0-9.]', '', torch_string.split('+')[0])
    parts = clean.split('.')[:3]
    while len(parts) < 3:
        parts.append('0')
    return ''.join(parts[:3])


@pytest.mark.unit
class TestFlashAttnWheelUrlConstruction(unittest.TestCase):
    """Unit tests for flash-attn wheel URL construction."""

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

    def test_torch_normalization_two_digit_minor(self):
        """torch '2.10.0+cu128' normalizes to '2100' (not '210' or error)."""
        self.assertEqual(_normalize_torch_version("2.10.0+cu128"), "2100")

    def test_cuda_numeric_128(self):
        """CUDA '12.8' -> 128 (numeric for <= comparison with wheel tags)."""
        self.assertEqual(_normalize_cuda_to_numeric("12.8"), 128)

    def test_cuda_numeric_124(self):
        """CUDA '12.4.0' -> 124."""
        self.assertEqual(_normalize_cuda_to_numeric("12.4.0"), 124)

    def test_cuda_numeric_121(self):
        """CUDA '12.1' -> 121."""
        self.assertEqual(_normalize_cuda_to_numeric("12.1"), 121)

    def test_cuda_numeric_118(self):
        """CUDA '11.8' -> 118."""
        self.assertEqual(_normalize_cuda_to_numeric("11.8"), 118)


# Realistic wheel filenames for test fixtures
_SAMPLE_ASSETS = [
    "flash_attn-2.7.4+cu124torch260cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    "flash_attn-2.7.4+cu124torch251cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    "flash_attn-2.7.4+cu121torch260cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    "flash_attn-2.7.4+cu124torch260cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
    "flash_attn-2.7.4+cu121torch251cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
    "flash_attn-2.7.4.tar.gz",          # source tarball — should be ignored
    "checksums.txt",                      # non-wheel — should be ignored
]


@pytest.mark.unit
class TestParseWheelCandidates(unittest.TestCase):

    def test_filters_to_matching_python_version(self):
        """Only wheels for cp312 are returned when py_str='cp312'."""
        result = _parse_wheel_candidates(_SAMPLE_ASSETS, "cp312")
        self.assertEqual(len(result), 3)

    def test_excludes_wrong_python_version(self):
        """cp311 wheels excluded when asking for cp312."""
        result = _parse_wheel_candidates(_SAMPLE_ASSETS, "cp312")
        names = [r[2] for r in result]
        self.assertTrue(all("cp312" in n for n in names))

    def test_extracts_cuda_and_torch_numbers(self):
        """Parsed tuples contain correct (cu_num, th_num, name)."""
        result = _parse_wheel_candidates(_SAMPLE_ASSETS, "cp312")
        cu_th_pairs = [(r[0], r[1]) for r in result]
        self.assertIn((124, 260), cu_th_pairs)
        self.assertIn((124, 251), cu_th_pairs)
        self.assertIn((121, 260), cu_th_pairs)

    def test_ignores_non_wheel_files(self):
        """Tarballs and text files are not matched."""
        result = _parse_wheel_candidates(_SAMPLE_ASSETS, "cp312")
        names = [r[2] for r in result]
        self.assertTrue(all(n.endswith(".whl") for n in names))

    def test_empty_assets_returns_empty(self):
        """Empty asset list returns empty candidate list."""
        result = _parse_wheel_candidates([], "cp312")
        self.assertEqual(result, [])

    def test_no_matching_python_returns_empty(self):
        """No cp313 wheels in sample -> empty list."""
        result = _parse_wheel_candidates(_SAMPLE_ASSETS, "cp313")
        self.assertEqual(result, [])


@pytest.mark.unit
class TestSelectBestWheel(unittest.TestCase):

    def test_prefers_highest_compatible_cuda_and_torch(self):
        """With cu_installed=128, th_installed=2100, picks cu124/torch260 first."""
        candidates = [
            (121, 251, "flash_attn-cu121torch251-whl"),
            (124, 260, "flash_attn-cu124torch260-whl"),
            (121, 260, "flash_attn-cu121torch260-whl"),
        ]
        result = _select_best_wheel(candidates, cu_installed=128, th_installed=2100)
        self.assertEqual(result[0], (124, 260, "flash_attn-cu124torch260-whl"))

    def test_filters_out_wheels_above_installed_cuda(self):
        """cu126 wheel excluded when installed CUDA is only 124."""
        candidates = [
            (126, 260, "flash_attn-cu126-whl"),
            (124, 260, "flash_attn-cu124-whl"),
        ]
        result = _select_best_wheel(candidates, cu_installed=124, th_installed=260)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 124)

    def test_filters_out_wheels_above_installed_torch(self):
        """torch260 wheel excluded when installed torch is only 251."""
        candidates = [
            (124, 260, "flash_attn-torch260-whl"),
            (124, 251, "flash_attn-torch251-whl"),
        ]
        result = _select_best_wheel(candidates, cu_installed=124, th_installed=251)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 251)

    def test_falls_back_to_highest_when_none_compatible(self):
        """When ALL candidates are newer than installed, returns highest-first."""
        candidates = [
            (126, 260, "flash_attn-cu126torch260-whl"),
            (124, 251, "flash_attn-cu124torch251-whl"),
        ]
        result = _select_best_wheel(candidates, cu_installed=118, th_installed=240)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 126)

    def test_empty_candidates_returns_empty(self):
        """No candidates -> empty result."""
        result = _select_best_wheel([], cu_installed=128, th_installed=2100)
        self.assertEqual(result, [])

    def test_sort_order_cuda_then_torch(self):
        """Among compatible wheels, sort by (cuda DESC, torch DESC)."""
        candidates = [
            (121, 260, "a"),
            (124, 251, "b"),
            (124, 260, "c"),
        ]
        result = _select_best_wheel(candidates, cu_installed=128, th_installed=2100)
        self.assertEqual(
            [(r[0], r[1]) for r in result],
            [(124, 260), (124, 251), (121, 260)]
        )


if __name__ == "__main__":
    unittest.main()
