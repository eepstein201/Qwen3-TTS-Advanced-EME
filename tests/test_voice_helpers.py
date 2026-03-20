"""Tests for voice_helpers module."""


class TestGetProsodyChoices:
    """Tests for get_prosody_choices function."""

    def test_returns_list_with_none_first(self):
        """Returns list with (none) as first option."""
        from qwen3_tts.interface.voice_helpers import get_prosody_choices

        choices = get_prosody_choices()
        assert isinstance(choices, list)
        assert choices[0] == "(none)"

    def test_includes_prosody_presets(self):
        """Includes prosody presets from config."""
        from qwen3_tts.interface.voice_helpers import get_prosody_choices

        choices = get_prosody_choices()
        # Should have at least some presets after (none)
        assert len(choices) > 1


class TestApplyProsodyPreset:
    """Tests for apply_prosody_preset function."""

    def test_returns_empty_for_none_choice(self):
        """Returns empty string for (none) choice."""
        from qwen3_tts.interface.voice_helpers import apply_prosody_preset

        result = apply_prosody_preset("(none)")
        assert result == ""

    def test_returns_existing_text_for_none_choice(self):
        """Returns existing text for (none) choice."""
        from qwen3_tts.interface.voice_helpers import apply_prosody_preset

        result = apply_prosody_preset("(none)", "existing text")
        assert result == "existing text"

    def test_returns_prosody_text_for_empty_existing(self):
        """Returns prosody text when no existing text."""
        from qwen3_tts.interface.voice_helpers import apply_prosody_preset

        # Use a known preset - assuming "calm" exists
        result = apply_prosody_preset("calm - Speak in a calm, soothing, relaxed manner")
        assert "calm" in result.lower()

    def test_appends_to_existing_text(self):
        """Appends prosody text to existing text."""
        from qwen3_tts.interface.voice_helpers import apply_prosody_preset

        result = apply_prosody_preset(
            "calm - Speak in a calm, soothing, relaxed manner",
            "A gentle speaker"
        )
        assert "gentle speaker" in result.lower()
        assert "calm" in result.lower()


class TestComposeVoiceDescription:
    """Tests for compose_voice_description function."""

    def test_returns_empty_for_all_none(self):
        """Returns empty string when all selections are none."""
        from qwen3_tts.interface.voice_helpers import compose_voice_description

        result = compose_voice_description("(none)", "(none)", "(none)", "(none)", "(none)", "(none)")
        assert result == ""

    def test_composes_gender_only(self):
        """Composes description with gender only."""
        from qwen3_tts.interface.voice_helpers import compose_voice_description

        result = compose_voice_description("Female", "(none)", "(none)", "(none)", "(none)", "(none)")
        assert "female" in result.lower()
        assert "speaker" in result.lower()

    def test_composes_age_and_gender(self):
        """Composes description with age and gender."""
        from qwen3_tts.interface.voice_helpers import compose_voice_description

        result = compose_voice_description("Male", "Young (20s-30s)", "(none)", "(none)", "(none)", "(none)")
        assert "young" in result.lower()
        assert "male" in result.lower()

    def test_composes_full_description(self):
        """Composes full description with all attributes."""
        from qwen3_tts.interface.voice_helpers import compose_voice_description

        result = compose_voice_description(
            "Female", "Middle-aged (40s-50s)", "Warm", "Smooth", "Moderate", "British"
        )
        assert "female" in result.lower()
        assert "warm" in result.lower()
        assert "smooth" in result.lower()
        assert "moderate" in result.lower()
        assert "british" in result.lower()

    def test_ends_with_period(self):
        """Description ends with period."""
        from qwen3_tts.interface.voice_helpers import compose_voice_description

        result = compose_voice_description("Female", "(none)", "(none)", "(none)", "(none)", "(none)")
        assert result.endswith(".")


class TestStripExtension:
    """Tests for _strip_extension function (re-exported from validation)."""

    def test_strips_pt_extension(self):
        """Strips .pt extension."""
        from qwen3_tts.interface.voice_helpers import strip_extension

        assert strip_extension("voice.pt") == "voice"

    def test_strips_wav_extension(self):
        """Strips .wav extension."""
        from qwen3_tts.interface.voice_helpers import strip_extension

        assert strip_extension("voice.wav") == "voice"

    def test_strips_txt_extension(self):
        """Strips .txt extension."""
        from qwen3_tts.interface.voice_helpers import strip_extension

        assert strip_extension("voice.txt") == "voice"

    def test_returns_unchanged_if_no_extension(self):
        """Returns unchanged if no matching extension."""
        from qwen3_tts.interface.voice_helpers import strip_extension

        assert strip_extension("voice") == "voice"

    def test_handles_multiple_dots(self):
        """Handles names with multiple dots."""
        from qwen3_tts.interface.voice_helpers import strip_extension

        assert strip_extension("my.voice.prompt.pt") == "my.voice.prompt"


class TestValidatePromptName:
    """Tests for validate_prompt_name function."""

    def test_accepts_valid_name(self):
        """Accepts valid alphanumeric name."""
        from qwen3_tts.interface.voice_helpers import validate_prompt_name

        result = validate_prompt_name("my_voice_123")
        assert result is None

    def test_accepts_name_with_dash(self):
        """Accepts name with dash."""
        from qwen3_tts.interface.voice_helpers import validate_prompt_name

        result = validate_prompt_name("my-voice")
        assert result is None

    def test_accepts_name_with_dot(self):
        """Accepts name with dot."""
        from qwen3_tts.interface.voice_helpers import validate_prompt_name

        result = validate_prompt_name("my.voice")
        assert result is None

    def test_rejects_empty_name(self):
        """Rejects empty name."""
        from qwen3_tts.interface.voice_helpers import validate_prompt_name

        result = validate_prompt_name("")
        assert result is not None
        assert result[1] == 400

    def test_rejects_special_characters(self):
        """Rejects special characters."""
        from qwen3_tts.interface.voice_helpers import validate_prompt_name

        result = validate_prompt_name("my@voice")
        assert result is not None

    def test_rejects_double_dot(self):
        """Rejects double dot (path traversal)."""
        from qwen3_tts.interface.voice_helpers import validate_prompt_name

        result = validate_prompt_name("my..voice")
        assert result is not None
