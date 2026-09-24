"""Unified list-resource table renderers (interface-quality plan, Phase 2 T2.5).

One Column spec and one render function per resource, shared by the Click
`tts list X` commands and the legacy argparse `tts generate --list-X` flags
so both surfaces produce byte-identical output. This is what kills the
historical 12-vs-18 prosody-table width drift and model_cache's duplicated
size column. Import-light: stdlib + cli_output only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from qwen3_tts.cli_output import Column, render_table

SPEAKER_COLUMNS: tuple[Column, ...] = (
    Column("key", "Speaker", 12),
    Column("desc", "Description", 0),
)

PRESET_COLUMNS: tuple[Column, ...] = (
    Column("name", "Preset", 20),
    Column("settings", "Settings", 0),
)

PROSODY_COLUMNS: tuple[Column, ...] = (
    Column("name", "Preset", 18),
    Column("text", "Description", 0),
)

VOICE_PROMPT_COLUMNS: tuple[Column, ...] = (Column("name", "Voice Prompt", 0),)

CACHE_COLUMNS: tuple[Column, ...] = (
    Column("type", "Model Type", 12),
    Column("size", "Size", 8),
    Column("backend", "Backend", 8),
    Column("last_access", "Last Accessed", 20),
)

HISTORY_COLUMNS: tuple[Column, ...] = (
    Column("timestamp", "Time", 19),
    Column("mode", "Mode", 8),
    Column("voice", "Voice", 12),
)


def render_speakers(speakers: Mapping[str, Mapping[str, str]]) -> str:
    """Group premium CustomVoice speakers by language.

    English and Chinese get their own section (the two languages with
    multiple speakers today); everything else is grouped under "Other
    languages" with the language named inline.
    """
    lines = ["Premium CustomVoice speakers (use with -m custom -s SPEAKER):", ""]
    for group in ("English", "Chinese"):
        rows = [
            {"key": key, "desc": info["desc"]}
            for key, info in speakers.items()
            if info["lang"] == group
        ]
        if not rows:
            continue
        lines.append(f"  {group}:")
        lines.append(render_table(SPEAKER_COLUMNS, rows, indent="    ", header=False))
        lines.append("")
    other_rows = [
        {"key": key, "desc": f"{info['desc']} ({info['lang']})"}
        for key, info in speakers.items()
        if info["lang"] not in ("English", "Chinese")
    ]
    if other_rows:
        lines.append("  Other languages:")
        lines.append(
            render_table(SPEAKER_COLUMNS, other_rows, indent="    ", header=False)
        )
        lines.append("")
    lines.append("Example: tts 'Hello world' -m custom -s ryan -o output")
    return "\n".join(lines)


def render_presets(presets: Mapping[str, Mapping[str, object]]) -> str:
    """Render generation presets, or the pinned empty-state message."""
    if not presets:
        return "No presets configured."
    rows = [
        {
            "name": name,
            "settings": ", ".join(f"{k}={v}" for k, v in params.items()),
        }
        for name, params in presets.items()
    ]
    body = render_table(PRESET_COLUMNS, rows, header=False)
    return "\n".join(["Generation presets:", body])


def render_prosody(presets: Mapping[str, str]) -> str:
    """Render prosody presets sorted alphabetically, or the empty message."""
    if not presets:
        return "No prosody presets configured."
    rows = [{"name": name, "text": text} for name, text in sorted(presets.items())]
    body = render_table(PROSODY_COLUMNS, rows, header=False)
    return "\n".join(["Prosody presets:", body])


def render_voice_prompts(prompts: Sequence[str], default: str | None = None) -> str:
    """Render voice prompt names, marking the default, or the empty message."""
    if not prompts:
        return "No voice prompts found."
    rows = [
        {"name": f"{name}{' (default)' if name == default else ''}"} for name in prompts
    ]
    body = render_table(VOICE_PROMPT_COLUMNS, rows, header=False)
    return "\n".join(["Available voice prompts:", body])


def render_cache(models: Sequence[Mapping[str, object]]) -> str:
    """Render cached-model rows with one size column (no duplication)."""
    if not models:
        return "No TTS models found in cache."
    rows = [
        {
            "type": f"{m['model_type'] or 'unknown'} ({m['model_size'] or 'unknown'})",
            "size": m["size_formatted"],
            "backend": m["backend"] or "unknown",
            "last_access": m["last_access_str"],
        }
        for m in models
    ]
    return render_table(CACHE_COLUMNS, rows)


def render_history(entries: Sequence[Mapping[str, str]]) -> str:
    """Render one aligned summary line per history entry."""
    if not entries:
        return "No generation history found."
    return render_table(HISTORY_COLUMNS, entries, header=False)
