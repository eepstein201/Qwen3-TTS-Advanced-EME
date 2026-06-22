import pytest

from qwen3_tts.core.config import validate_voice_name


class TestValidateVoiceName:
    def test_01_plain_ascii_accepted(self):
        assert validate_voice_name("alice") == "alice"

    def test_02_dotdot_rejected(self):
        with pytest.raises(ValueError):
            validate_voice_name("../evil")

    def test_03_forward_slash_rejected(self):
        with pytest.raises(ValueError):
            validate_voice_name("dir/file")

    def test_04_backslash_rejected(self):
        with pytest.raises(ValueError):
            validate_voice_name("dir\\file")

    def test_05_absolute_path_rejected(self):
        with pytest.raises(ValueError):
            validate_voice_name("/etc/passwd")

    def test_06_empty_string_rejected(self):
        with pytest.raises(ValueError):
            validate_voice_name("")

    def test_07_too_long_rejected(self):
        with pytest.raises(ValueError):
            validate_voice_name("a" * 129)

    def test_08_nul_byte_rejected(self):
        with pytest.raises(ValueError):
            validate_voice_name("evil\x00name")

    def test_09_whitespace_only_rejected(self):
        with pytest.raises(ValueError):
            validate_voice_name("   ")

    def test_10_unicode_accepted(self):
        assert validate_voice_name("céline") == "céline"
