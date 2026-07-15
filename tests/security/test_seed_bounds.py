"""Tests that GenerateRequest rejects out-of-range seed values before processing."""
import pytest
from pydantic import ValidationError

from qwen3_tts.server.validation import MAX_SEED, GenerateRequest


class TestSeedBounds:
    def _req(self, **kwargs):
        return GenerateRequest(text="hi", **kwargs)

    def test_01_no_seed_accepted(self):
        req = self._req()
        assert req.seed is None

    def test_02_zero_seed_accepted(self):
        req = self._req(seed=0)
        assert req.seed == 0

    def test_03_positive_seed_accepted(self):
        req = self._req(seed=42)
        assert req.seed == 42

    def test_04_max_seed_accepted(self):
        req = self._req(seed=MAX_SEED)
        assert req.seed == MAX_SEED

    def test_05_negative_seed_rejected(self):
        with pytest.raises(ValidationError):
            self._req(seed=-1)

    def test_06_oversized_seed_rejected(self):
        with pytest.raises(ValidationError):
            self._req(seed=MAX_SEED + 1)

    def test_07_astronomically_large_seed_rejected(self):
        with pytest.raises(ValidationError):
            self._req(seed=2**200)

    def test_08_max_seed_constant_is_signed_int32(self):
        assert MAX_SEED == 2**31 - 1
