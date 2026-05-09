"""Voice-related helper functions for TTS UI.

This module contains helper functions for:
- Prosody preset management
- Voice description composition
- Prompt name validation

These functions are used by the Gradio UI but don't depend on Gradio directly.
"""

import re

from qwen3_tts.core.config import get_prosody_presets

# =============================================================================
# Prompt Name Validation (duplicated here to avoid circular imports)
# =============================================================================


def validate_prompt_name(name: str) -> tuple[dict, int] | None:
    """Validate prompt name — returns error tuple or None."""
    if not name or not name.strip():
        return {"error": "Missing prompt name", "recovery": "config"}, 400
    name = name.strip()
    if len(name) > 255:
        return {"error": "Prompt name too long", "recovery": "config"}, 400
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", name):
        return {
            "error": "Invalid prompt name: only alphanumeric, dash, underscore, dot allowed",
            "recovery": "config",
        }, 400
    if ".." in name:
        return {"error": "Invalid prompt name", "recovery": "config"}, 400
    return None


def strip_extension(name: str) -> str:
    """Strip .pt, .wav, or .txt extension from name."""
    base = name
    for ext in (".pt", ".wav", ".txt"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    return base


# =============================================================================
# Prosody Preset Helpers
# =============================================================================


def get_prosody_choices() -> list[str]:
    """Return list of prosody preset choices for dropdown, with (none) first.

    Returns:
        List of strings in format "name - description", with "(none)" first
    """
    presets = get_prosody_presets()
    return ["(none)"] + [f"{name} - {text}" for name, text in sorted(presets.items())]


def apply_prosody_preset(choice: str, existing_text: str | None = None) -> str:
    """Apply a prosody preset to text.

    When a prosody preset is selected, append to existing text or fill in.

    Args:
        choice: The selected prosody choice from dropdown
        existing_text: Current text in the field (optional)

    Returns:
        The combined text with prosody applied
    """
    if not choice or choice == "(none)":
        return existing_text or ""

    # Extract preset name from "name - description" format
    name = choice.split(" - ")[0].strip()
    presets = get_prosody_presets()
    prosody_text = presets.get(name, "")

    if not prosody_text:
        return existing_text or ""

    if existing_text and existing_text.strip():
        return f"{existing_text.strip()}. {prosody_text}"

    return prosody_text


# =============================================================================
# Voice Description Builder
# =============================================================================


def compose_voice_description(
    gender: str,
    age: str,
    tone: str,
    texture: str,
    pace: str,
    accent: str,
) -> str:
    """Compose a voice description from dropdown selections.

    Combines voice attributes into a natural language description.

    Args:
        gender: Gender selection (e.g., "Male", "Female", "(none)")
        age: Age range selection (e.g., "Young (20s-30s)", "(none)")
        tone: Voice tone selection (e.g., "Warm", "Bright", "(none)")
        texture: Voice texture selection (e.g., "Smooth", "Rough", "(none)")
        pace: Speaking pace selection (e.g., "Fast", "Slow", "(none)")
        accent: Accent selection (e.g., "British", "American", "(none)")

    Returns:
        Composed voice description string, Empty string if all selections are "(none)".
    """
    parts = []

    # Handle age and gender combination
    if age and age != "(none)":
        # Extract the age range text (remove parenthetical info)
        age_text = age.lower().split(" (")[0] if " (" in age else age.lower()
        if gender and gender != "(none)":
            parts.append(f"A {age_text} {gender.lower()}")
        else:
            parts.append(f"A {age_text} speaker")
    elif gender and gender != "(none)":
        parts.append(f"A {gender.lower()} speaker")

    # Collect voice qualifiers
    qualifiers = []
    if tone and tone != "(none)":
        qualifiers.append(tone.lower())
    if texture and texture != "(none)":
        qualifiers.append(texture.lower())

    if qualifiers:
        parts.append(f"with a {', '.join(qualifiers)} voice")

    # Add pace
    if pace and pace != "(none)":
        parts.append(f"who speaks at a {pace.lower()} pace")

    # Add accent
    if accent and accent != "(none)" and accent != "None/Default":
        parts.append(f"with a {accent} accent")

    if not parts:
        return ""

    desc = " ".join(parts)
    if not desc.endswith("."):
        desc += "."

    return desc
