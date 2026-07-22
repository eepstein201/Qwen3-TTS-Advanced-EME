"""Tests for server validation module."""
import pytest
from fastapi import HTTPException


class TestValidateGenerationRequest:
    """Tests for _validate_generation_request function."""

    @pytest.fixture
    def security_config(self):
        """Standard security config."""
        return {"max_text_length": 10000, "max_batch_size": 20}

    @pytest.fixture
    def valid_request(self):
        """Valid generation request."""
        from qwen3_tts.server.validation import GenerateRequest
        return GenerateRequest(texts=["Hello world"], mode="clone")

    def test_rejects_invalid_mode(self, security_config, valid_request):
        """Validation rejects modes other than clone/design/custom."""
        from qwen3_tts.server.validation import (
            GenerateRequest,
            _validate_generation_request,
        )

        req = GenerateRequest(texts=["test"], mode="invalid")
        with pytest.raises(HTTPException) as exc:
            _validate_generation_request(req, security_config)
        assert exc.value.status_code == 400
        assert "Invalid mode" in exc.value.detail

    def test_rejects_path_traversal_in_prompt_file(self, security_config):
        """Validation rejects path traversal in prompt_file."""
        from qwen3_tts.server.validation import (
            GenerateRequest,
            _validate_generation_request,
        )

        req = GenerateRequest(texts=["test"], mode="clone", prompt_file="../../../etc/passwd")
        with pytest.raises(HTTPException) as exc:
            _validate_generation_request(req, security_config)
        assert exc.value.status_code == 400
        assert "path traversal" in exc.value.detail.lower()

    def test_accepts_subdir_path_in_prompt_file(self, security_config):
        """Validation accepts subdirectory path in prompt_file (pathlib-based check)."""
        from qwen3_tts.server.validation import (
            GenerateRequest,
            _validate_generation_request,
        )

        req = GenerateRequest(texts=["test"], mode="clone", prompt_file="subdir/file.pt")
        # Should NOT raise — pathlib check allows paths that resolve within voice_prompts dir
        _validate_generation_request(req, security_config)

    def test_rejects_absolute_path_in_prompt_file(self, security_config):
        """Validation rejects absolute path in prompt_file."""
        from qwen3_tts.server.validation import (
            GenerateRequest,
            _validate_generation_request,
        )

        req = GenerateRequest(texts=["test"], mode="clone", prompt_file="/etc/passwd")
        with pytest.raises(HTTPException) as exc:
            _validate_generation_request(req, security_config)
        assert exc.value.status_code == 400
        assert "path traversal" in exc.value.detail.lower()

    def test_rejects_invalid_speaker_for_custom_mode(self, security_config):
        """Validation rejects invalid speaker for custom mode."""
        from qwen3_tts.server.validation import (
            GenerateRequest,
            _validate_generation_request,
        )

        req = GenerateRequest(texts=["test"], mode="custom", speaker="nonexistent_speaker")
        with pytest.raises(HTTPException) as exc:
            _validate_generation_request(req, security_config)
        assert exc.value.status_code == 400
        assert "Unknown speaker" in exc.value.detail

    def test_accepts_valid_speaker_lowercase(self, security_config):
        """Validation accepts valid lowercase speaker."""
        from qwen3_tts.server.validation import (
            GenerateRequest,
            _validate_generation_request,
        )

        # Should not raise for valid speaker
        req = GenerateRequest(texts=["test"], mode="custom", speaker="ryan")
        _validate_generation_request(req, security_config)  # No exception

    def test_accepts_valid_clone_mode(self, security_config):
        """Validation accepts clone mode."""
        from qwen3_tts.server.validation import (
            GenerateRequest,
            _validate_generation_request,
        )

        req = GenerateRequest(texts=["test"], mode="clone", prompt_file="my_voice.pt")
        _validate_generation_request(req, security_config)  # No exception

    def test_accepts_valid_design_mode(self, security_config):
        """Validation accepts design mode."""
        from qwen3_tts.server.validation import (
            GenerateRequest,
            _validate_generation_request,
        )

        req = GenerateRequest(texts=["test"], mode="design", voice_description="A calm voice")
        _validate_generation_request(req, security_config)  # No exception


class TestValidatePromptName:
    """Tests for _validate_prompt_name function."""

    def test_rejects_empty_name(self):
        """Validation rejects empty prompt name."""
        from qwen3_tts.server.validation import _validate_prompt_name

        result = _validate_prompt_name("")
        assert result is not None
        assert result[1] == 400  # status code

    def test_rejects_whitespace_only_name(self):
        """Validation rejects whitespace-only name."""
        from qwen3_tts.server.validation import _validate_prompt_name

        result = _validate_prompt_name("   ")
        assert result is not None

    def test_rejects_too_long_name(self):
        """Validation rejects names longer than 255 chars."""
        from qwen3_tts.server.validation import _validate_prompt_name

        long_name = "a" * 300
        result = _validate_prompt_name(long_name)
        assert result is not None
        assert "too long" in result[0]["error"].lower()

    def test_rejects_special_characters(self):
        """Validation rejects names with special characters."""
        from qwen3_tts.server.validation import _validate_prompt_name

        result = _validate_prompt_name("my@voice#prompt")
        assert result is not None
        assert "Invalid" in result[0]["error"]

    def test_rejects_double_dot(self):
        """Validation rejects names with double dot (path traversal)."""
        from qwen3_tts.server.validation import _validate_prompt_name

        result = _validate_prompt_name("..traversal")
        assert result is not None

    def test_accepts_valid_name(self):
        """Validation accepts valid prompt name."""
        from qwen3_tts.server.validation import _validate_prompt_name

        result = _validate_prompt_name("my_voice_prompt")
        assert result is None  # No error

    def test_accepts_name_with_extension(self):
        """Validation accepts name with extension."""
        from qwen3_tts.server.validation import _validate_prompt_name

        result = _validate_prompt_name("my_voice.pt")
        assert result is None

    def test_accepts_name_with_dash_and_underscore(self):
        """Validation accepts name with dash and underscore."""
        from qwen3_tts.server.validation import _validate_prompt_name

        result = _validate_prompt_name("my-voice_prompt-123")
        assert result is None


class TestStripExtension:
    """Tests for _strip_extension function."""

    def test_strips_pt_extension(self):
        """Strips .pt extension."""
        from qwen3_tts.server.validation import _strip_extension

        assert _strip_extension("voice.pt") == "voice"

    def test_strips_wav_extension(self):
        """Strips .wav extension."""
        from qwen3_tts.server.validation import _strip_extension

        assert _strip_extension("voice.wav") == "voice"

    def test_strips_txt_extension(self):
        """Strips .txt extension."""
        from qwen3_tts.server.validation import _strip_extension

        assert _strip_extension("voice.txt") == "voice"

    def test_returns_unchanged_if_no_extension(self):
        """Returns unchanged if no matching extension."""
        from qwen3_tts.server.validation import _strip_extension

        assert _strip_extension("voice") == "voice"

    def test_handles_multiple_dots(self):
        """Handles names with multiple dots."""
        from qwen3_tts.server.validation import _strip_extension

        assert _strip_extension("my.voice.file.pt") == "my.voice.file"


class TestGenCacheKey:
    """Tests for _gen_cache_key function."""

    def test_produces_consistent_hash(self):
        """Same inputs produce same hash."""
        from qwen3_tts.server.validation import _gen_cache_key

        key1 = _gen_cache_key("hello", "clone", {"temp": 0.7})
        key2 = _gen_cache_key("hello", "clone", {"temp": 0.7})
        assert key1 == key2

    def test_different_text_produces_different_hash(self):
        """Different text produces different hash."""
        from qwen3_tts.server.validation import _gen_cache_key

        key1 = _gen_cache_key("hello", "clone", {"temp": 0.7})
        key2 = _gen_cache_key("world", "clone", {"temp": 0.7})
        assert key1 != key2

    def test_different_mode_produces_different_hash(self):
        """Different mode produces different hash."""
        from qwen3_tts.server.validation import _gen_cache_key

        key1 = _gen_cache_key("hello", "clone", {"temp": 0.7})
        key2 = _gen_cache_key("hello", "design", {"temp": 0.7})
        assert key1 != key2

    def test_includes_prompt_file_in_hash(self):
        """Prompt file is included in hash."""
        from qwen3_tts.server.validation import _gen_cache_key

        key1 = _gen_cache_key("hello", "clone", {"temp": 0.7}, prompt_file="voice.pt")
        key2 = _gen_cache_key("hello", "clone", {"temp": 0.7})
        assert key1 != key2

    def test_includes_speaker_in_hash(self):
        """Speaker is included in hash."""
        from qwen3_tts.server.validation import _gen_cache_key

        key1 = _gen_cache_key("hello", "custom", {"temp": 0.7}, speaker="ryan")
        key2 = _gen_cache_key("hello", "custom", {"temp": 0.7})
        assert key1 != key2

    def test_hash_is_hex_string(self):
        """Hash is a hex string."""
        from qwen3_tts.server.validation import _gen_cache_key

        key = _gen_cache_key("hello", "clone", {"temp": 0.7})
        assert all(c in "0123456789abcdef" for c in key)

    def test_different_language_produces_different_hash(self):
        """language changes pronunciation and must distinguish cache entries.

        Pre-fix, language was omitted, so an English request could be served a
        Spanish (etc.) cache entry for the same text/voice/params.
        """
        from qwen3_tts.server.validation import _gen_cache_key

        base = ("hello", "clone", {"temp": 0.7})
        key_en = _gen_cache_key(*base, language="English")
        key_es = _gen_cache_key(*base, language="Spanish")
        assert key_en != key_es

    def test_x_vector_only_mode_distinguishes_hash(self):
        """x_vector_only_mode switches the clone feature path."""
        from qwen3_tts.server.validation import _gen_cache_key

        base = ("hello", "clone", {"temp": 0.7})
        key_off = _gen_cache_key(*base, x_vector_only_mode=False)
        key_on = _gen_cache_key(*base, x_vector_only_mode=True)
        assert key_off != key_on

    def test_max_chunk_chars_distinguishes_hash(self):
        """max_chunk_chars moves chunk boundaries and must not collide."""
        from qwen3_tts.server.validation import _gen_cache_key

        base = ("hello", "clone", {"temp": 0.7})
        key_default = _gen_cache_key(*base)
        key_small = _gen_cache_key(*base, max_chunk_chars=200)
        assert key_default != key_small

    def test_seed_lock_chunks_distinguishes_hash(self):
        """seed_lock_chunks changes the per-chunk seed strategy (not value)."""
        from qwen3_tts.server.validation import _gen_cache_key

        base = ("hello", "clone", {"temp": 0.7})
        key_off = _gen_cache_key(*base, seed_lock_chunks=False)
        key_on = _gen_cache_key(*base, seed_lock_chunks=True)
        assert key_off != key_on

    def test_unset_behavior_toggles_stable_across_calls(self):
        """Default toggles must be deterministic so base keys are stable."""
        from qwen3_tts.server.validation import _gen_cache_key

        base = ("hello", "clone", {"temp": 0.7})
        a = _gen_cache_key(*base)
        b = _gen_cache_key(*base, language=None, x_vector_only_mode=False,
                           seed_lock_chunks=False)
        assert a == b


class TestTranscribeRequestValidation:
    """Tests for TranscribeRequest field validation (R-29, R-30)."""

    @pytest.mark.parametrize("lang", ["en", "zh", "eng", "en-US", "zh-Hans"], ids=[
        "two-letter", "two-letter-zh", "three-letter", "with-region", "with-script",
    ])
    def test_language_accepts_valid(self, lang):
        """TranscribeRequest accepts valid BCP-47 language codes."""
        from qwen3_tts.server.validation import TranscribeRequest
        req = TranscribeRequest(audio_base64="abc", language=lang)
        assert req.language == lang

    @pytest.mark.parametrize("lang", [
        "not_a_lang_code!!",
        "toolongcode",
        "EN",
        "e",
        "123",
    ], ids=["special-chars", "too-long", "uppercase", "too-short", "digits"])
    def test_language_rejects_invalid(self, lang):
        """TranscribeRequest rejects non-BCP-47 language codes."""
        from pydantic import ValidationError

        from qwen3_tts.server.validation import TranscribeRequest
        with pytest.raises(ValidationError):
            TranscribeRequest(audio_base64="abc", language=lang)

    def test_audio_base64_rejects_oversized_payload(self):
        """TranscribeRequest rejects base64 strings over 50MB."""
        from pydantic import ValidationError

        from qwen3_tts.server.validation import TranscribeRequest
        oversized = "A" * (51 * 1024 * 1024)
        with pytest.raises(ValidationError):
            TranscribeRequest(audio_base64=oversized, language="en")

    def test_audio_base64_accepts_normal_payload(self):
        """TranscribeRequest accepts normally-sized base64 audio."""
        from qwen3_tts.server.validation import TranscribeRequest
        # ~1KB base64 — well within limit
        req = TranscribeRequest(audio_base64="A" * 1024, language="en")
        assert len(req.audio_base64) == 1024


class TestCreateVoicePromptRequestValidation:
    """Tests for CreateVoicePromptRequest field validation (R-30)."""

    def test_audio_base64_rejects_oversized_payload(self):
        """CreateVoicePromptRequest rejects base64 strings over 50MB."""
        from pydantic import ValidationError

        from qwen3_tts.server.validation import CreateVoicePromptRequest
        oversized = "A" * (51 * 1024 * 1024)
        with pytest.raises(ValidationError):
            CreateVoicePromptRequest(audio_base64=oversized, name="test")

    def test_audio_base64_accepts_normal_payload(self):
        """CreateVoicePromptRequest accepts normally-sized base64 audio."""
        from qwen3_tts.server.validation import CreateVoicePromptRequest
        req = CreateVoicePromptRequest(audio_base64="A" * 1024, name="test_voice")
        assert req.name == "test_voice"


class TestErrorResponse:
    """Tests for _error_response helper."""

    def test_raises_http_exception(self):
        """_error_response raises HTTPException."""
        from qwen3_tts.server.validation import _error_response

        with pytest.raises(HTTPException) as exc:
            _error_response(400, "Test error", "Test detail", "retry")
        assert exc.value.status_code == 400

    def test_includes_structured_detail(self):
        """_error_response includes structured detail dict."""
        from qwen3_tts.server.validation import _error_response

        with pytest.raises(HTTPException) as exc:
            _error_response(500, "ServerError", "Something went wrong", "restart")
        detail = exc.value.detail
        assert detail["error"] == "ServerError"
        assert detail["detail"] == "Something went wrong"
        assert detail["recovery"] == "restart"
