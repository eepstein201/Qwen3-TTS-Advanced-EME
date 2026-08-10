#!/usr/bin/env python3
"""Qwen3-TTS generation script with voice cloning and voice design support.

Architecture:
  --_server-mode / server available  ->  HTTP calls (no torch import)
  --local / no server                ->  lazy import qwen3_tts.core.engine (PyTorch)

This module is the CLI orchestrator. Pure helpers, interactive modes, and server
interaction logic live in the sibling modules:
  - generate_helpers.py     (text/audio utilities, SSML/SRT, history, payload)
  - generate_interactive.py (REPL, watch, interactive_mode, progress, voice mgmt)
  - generate_server.py      (HTTP generation, streaming, local gen, server lifecycle)
"""

import argparse
import json
import logging
import os
import sys

logger = logging.getLogger("tts.cli")

from qwen3_tts.core.config import (  # noqa: E402
    CONFIG_PATH,
    CUSTOM_VOICE_SPEAKERS,
    MODEL_INFO,
    VALID_BACKENDS,
    VALID_MODEL_SIZES,
    get_backend,
    get_default_clone_prompt,
    get_mlx_model_name,
    get_mlx_quantization,
    get_torch_dtype_name,
    is_server_running,
    load_config,
    safe_path_join,
    save_config,
)

# ---------------------------------------------------------------------------
# Re-exports from split modules (backward compatibility)
# ---------------------------------------------------------------------------
from qwen3_tts.interface.generate_helpers import (  # noqa: E402, F401
    LAST_TEXT_FILE,
    _build_generation_payload,
    _decode_base64_result,
    _save_base64_result,
    auto_increment_filename,
    get_clipboard_text,
    get_generation_params,
    get_text,
    get_voice_alias,
    list_voice_prompts,
    log_generation,
    open_file,
    parse_srt,
    parse_ssml,
    play_audio,
    process_audio_args,
    process_ssml_text,
    show_history,
    srt_time_to_ms,
    voice_prompt_exists,
)
from qwen3_tts.interface.generate_interactive import (  # noqa: E402, F401
    _ProgressPoller,
    delete_voice_prompt,
    interactive_mode,
    preview_voice_prompt,
    rename_voice_prompt,
    run_repl,
    run_watch_mode,
)
from qwen3_tts.interface.generate_server import (  # noqa: E402, F401
    _run_single_generation,
    _voice_param_for_log,
    build_ui_and_launch,
    ensure_server_running,
    generate_local,
    generate_streaming,
    generate_via_server,
    launch_gradio_ui,
    load_model_on_server,
)

# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def process_batch(texts, args, config, gen_params, use_server):
    """Process multiple texts."""
    import soundfile as sf

    output_dir = os.path.expanduser(
        args.output or config.get("output_directory", "~/Downloads")
    )
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    language = config.get("language", "English")
    mode = args.mode or "clone"
    prompt_file = args.prompt or get_default_clone_prompt(config)
    voice_description = args.description or config.get("default_voice_description", "")

    output_paths = []
    needs_processing = args.trim_silence or args.normalize or args.speed or args.pitch

    if use_server:
        print(f"Using TTS server for batch of {len(texts)} texts...")
        results = generate_via_server(
            texts,
            mode,
            config,
            gen_params,
            prompt_file=prompt_file if mode == "clone" else None,
            voice_description=voice_description if mode == "design" else None,
        )

        for i, result in enumerate(results):
            output_path = safe_path_join(output_dir, f"output_{i + 1}.wav")
            if needs_processing:
                wav, sr = _decode_base64_result(result)
                wav = process_audio_args(wav, sr, args)
                sf.write(output_path, wav, sr)
            else:
                _save_base64_result(result, output_path)
            output_paths.append(output_path)
            print(f"Saved: {output_path}")
    else:
        for i, text in enumerate(texts):
            print(f"\nProcessing {i + 1}/{len(texts)}...")
            wav, sr = generate_local(
                text,
                mode,
                gen_params,
                language,
                prompt_file=prompt_file,
                voice_description=voice_description,
            )
            wav = process_audio_args(wav, sr, args)

            output_path = safe_path_join(output_dir, f"output_{i + 1}.wav")
            sf.write(output_path, wav, sr)
            output_paths.append(output_path)
            print(f"Saved: {output_path}")

    return output_paths


# ---------------------------------------------------------------------------
# Dialogue / SRT delegators
# ---------------------------------------------------------------------------


def process_dialogue(dialogue_path, config, args, gen_params, use_server):
    """Process a dialogue JSON file with multiple speakers.

    Delegates to qwen3_tts.interface.cli.dialogue to avoid duplication.
    """
    from qwen3_tts.interface.cli.dialogue import process_dialogue as _impl

    return _impl(dialogue_path, config, args, gen_params, use_server)


def process_srt_file(srt_path, config, args, gen_params, use_server):
    """Process an SRT file and generate audio for each subtitle.

    Delegates to qwen3_tts.interface.cli.srt to avoid duplication.
    """
    from qwen3_tts.interface.cli.srt import process_srt_file as _impl

    return _impl(srt_path, config, args, gen_params, use_server)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser():
    """Build the argparse parser for generate.py."""
    parser = argparse.ArgumentParser(description="Qwen3-TTS Generator")
    parser.add_argument(
        "text", nargs="*", help="Text(s) to synthesize or path to text file"
    )
    parser.add_argument("-o", "--output", help="Output filename or directory for batch")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["clone", "design", "custom"],
        help="Voice mode (clone, design, or custom)",
    )
    parser.add_argument("-p", "--prompt", help="Voice clone prompt filename")
    parser.add_argument(
        "-d", "--description", help="Voice description (for design mode)"
    )
    parser.add_argument(
        "-s",
        "--speaker",
        help="Premium speaker name for custom mode (ryan, aiden, vivian, etc.)",
    )
    parser.add_argument(
        "-i",
        "--instruct",
        help="Style instruction for custom mode (e.g., 'very happy', 'speak slowly')",
    )
    parser.add_argument(
        "--prosody",
        metavar="PRESET",
        help="Prosody preset for custom/design mode (excited, calm, whisper, etc.)",
    )
    parser.add_argument(
        "--no-transcript",
        action="store_true",
        dest="no_transcript",
        help="Clone using speaker embedding only (no transcript needed)",
    )
    parser.add_argument("--batch", help="JSON file with array of texts")
    parser.add_argument("--preset", help="Use named preset from config")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, dest="top_k", help="Top-k sampling")
    parser.add_argument(
        "--top-p", type=float, dest="top_p", help="Top-p (nucleus) sampling"
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        dest="repetition_penalty",
        help="Repetition penalty",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        dest="max_chunk_chars",
        metavar="N",
        help="Max chars per chunk for long text (default: 500 from config, 0 to disable)",
    )
    parser.add_argument(
        "--backend",
        choices=["torch", "mlx"],
        help="Override backend for this run (default: from config.json)",
    )
    parser.add_argument(
        "--model-size",
        choices=list(VALID_MODEL_SIZES),
        help="Override model size for this run (default: from config.json)",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List available backends and current setting",
    )
    parser.add_argument(
        "--list-prompts", action="store_true", help="List available voice prompts"
    )
    parser.add_argument(
        "--voices",
        action="store_true",
        help="List available voice prompts (alias for --list-prompts)",
    )
    parser.add_argument(
        "--list-presets", action="store_true", help="List available presets"
    )
    parser.add_argument(
        "--list-aliases", action="store_true", help="List available voice aliases"
    )
    parser.add_argument(
        "--list-speakers", action="store_true", help="List premium CustomVoice speakers"
    )
    parser.add_argument(
        "--list-prosody", action="store_true", help="List available prosody presets"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available TTS models and their load status",
    )
    parser.add_argument("--stats", action="store_true", help="Show server statistics")
    parser.add_argument(
        "--edit-config", action="store_true", help="Edit default voice description"
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Don't open the output file"
    )
    parser.add_argument(
        "--local", action="store_true", help="Force local generation (skip server)"
    )
    parser.add_argument(
        "--play", action="store_true", help="Play audio after generation"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream audio playback as it generates (MLX backend)",
    )
    parser.add_argument(
        "--clipboard", action="store_true", help="Read text from clipboard"
    )
    parser.add_argument(
        "--trim-silence", action="store_true", help="Trim leading/trailing silence"
    )
    parser.add_argument(
        "--normalize", action="store_true", help="Normalize audio to -3dB peak"
    )
    parser.add_argument(
        "--speed",
        type=float,
        metavar="FACTOR",
        help="Speed factor (1.2 = 20%% faster, 0.8 = 20%% slower)",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        metavar="SEMITONES",
        help="Pitch shift in semitones (+2 = higher, -2 = lower)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without running",
    )
    parser.add_argument(
        "-v", "--voice", help="Use a voice alias from config (combines prompt + preset)"
    )
    parser.add_argument("--delete-prompt", metavar="NAME", help="Delete a voice prompt")
    parser.add_argument(
        "--rename-prompt", nargs=2, metavar=("OLD", "NEW"), help="Rename a voice prompt"
    )
    parser.add_argument(
        "--preview-prompt", metavar="NAME", help="Preview a voice prompt"
    )
    parser.add_argument(
        "--history",
        nargs="?",
        const=10,
        type=int,
        metavar="N",
        help="Show last N generations (default: 10)",
    )
    parser.add_argument(
        "--repl", action="store_true", help="Start interactive REPL mode"
    )
    parser.add_argument("--watch", metavar="DIR", help="Watch directory for .txt files")
    parser.add_argument("--srt", metavar="FILE", help="Process SRT subtitle file")
    parser.add_argument(
        "--ssml", action="store_true", help="Enable SSML markup parsing"
    )
    parser.add_argument(
        "--dialogue",
        metavar="FILE",
        help="Process dialogue JSON file with multiple speakers",
    )
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Save individual audio files for each dialogue line",
    )
    parser.add_argument(
        "--ui",
        "--gui",
        action="store_true",
        dest="ui",
        help="Launch the Gradio web interface",
    )
    parser.add_argument(
        "--_server-mode",
        dest="server_mode",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--text-override", dest="text_override", help=argparse.SUPPRESS)
    return parser


# ---------------------------------------------------------------------------
# Info command handlers
# ---------------------------------------------------------------------------


def _handle_info_commands(args, config, gen_params):
    """Handle --list-*, --stats, --edit-config, --history, prompt management.

    Returns False if a command was handled, None if no info command matched.
    """
    if args.list_backends:
        current = get_backend()
        override = f" (overridden to '{args.backend}')" if args.backend else ""
        print(f"Available backends: {', '.join(VALID_BACKENDS)}")
        print(f"Current backend:    {current}{override}")
        print()
        if current == "mlx":
            quant = get_mlx_quantization()
            print(f"  MLX quantization: {quant}")
            for mt in ("clone", "design", "custom"):
                print(f"  {mt}: {get_mlx_model_name(mt)}")
        else:
            dtype = get_torch_dtype_name()
            print(f"  PyTorch dtype: {dtype}")
            for mt, info in MODEL_INFO.items():
                print(f"  {mt}: {info['name']}")
        print()
        print(f"To change: edit {CONFIG_PATH} -> advanced.backend")
        print('Or use: tts --backend mlx "text" -o output')
        return False

    if args.list_prompts or args.voices:
        prompts = list_voice_prompts()
        print("Available voice prompts:")
        for p in prompts:
            default_marker = (
                " (default)" if p == config.get("default_clone_prompt") else ""
            )
            print(f"  - {p}{default_marker}")
        return False

    if args.list_presets:
        presets = config.get("presets", {})
        print("Available presets:")
        for name, settings in presets.items():
            print(f"  - {name}: {settings}")
        return False

    if args.list_aliases:
        aliases = config.get("aliases", {})
        if aliases:
            print("Available voice aliases:")
            for name, settings in aliases.items():
                print(f"  - {name}:")
                for k, v in settings.items():
                    print(f"      {k}: {v}")
        else:
            print("No voice aliases configured.")
            print("Add to config.json under 'aliases', e.g.:")
            print('  "aliases": {')
            print('    "narrator": {"prompt": "narrator.pt", "preset": "consistent"}')
            print("  }")
        return False

    if args.list_prosody:
        from qwen3_tts.core.config import get_prosody_presets

        presets = get_prosody_presets(config)
        print("Available prosody presets (use with --prosody PRESET):\n")
        for name, text in sorted(presets.items()):
            print(f"  {name:<18} {text}")
        print()
        print('Example: tts -m custom -s ryan --prosody excited "Hello!" -o output')
        print(f"\nCustomize in {CONFIG_PATH} under 'prosody_presets'.")
        return False

    if args.list_speakers:
        print("Premium CustomVoice speakers (use with -m custom -s SPEAKER):")
        print()
        for group_name, lang_filter in [("English", "English"), ("Chinese", "Chinese")]:
            print(f"  {group_name}:")
            for key, info in CUSTOM_VOICE_SPEAKERS.items():
                if info["lang"] == lang_filter:
                    print(f"    {key:<12} - {info['desc']}")
            print()
        print("  Other languages:")
        for key, info in CUSTOM_VOICE_SPEAKERS.items():
            if info["lang"] not in ("English", "Chinese"):
                print(f"    {key:<12} - {info['desc']} ({info['lang']})")
        print()
        print("Example: tts 'Hello world' -m custom -s ryan -o output")
        return False

    if args.list_models:
        return _handle_list_models(args, config)

    if args.stats:
        return _handle_stats(config)

    if args.edit_config:
        print(
            f"Current voice description: {config.get('default_voice_description', '')}"
        )
        new_desc = input("Enter new description (or press Enter to keep): ").strip()
        if new_desc:
            config["default_voice_description"] = new_desc
            save_config(config)
        return False

    if args.history is not None:
        show_history(args.history)
        return False

    if args.delete_prompt:
        delete_voice_prompt(args.delete_prompt)
        return False

    if args.rename_prompt:
        rename_voice_prompt(args.rename_prompt[0], args.rename_prompt[1])
        return False

    if args.preview_prompt:
        preview_voice_prompt(args.preview_prompt, config)
        return False

    return None  # No info command matched


def _handle_list_models(args, config):
    """Display available TTS models and their load status."""

    models_config = config.get("models", {})
    print("\nAvailable TTS Models:")
    print("=" * 60)
    print()

    model_display = {
        "clone": {
            "name": "Base (Clone)",
            "usage": "-m clone -p voice.pt",
            "memory": "~3.5GB",
        },
        "design": {
            "name": "VoiceDesign",
            "usage": "-m design -d 'warm female voice'",
            "memory": "~3.5GB",
        },
        "custom": {
            "name": "CustomVoice",
            "usage": "-m custom -s ryan",
            "memory": "~3.5GB",
        },
    }

    server_status = {}
    if is_server_running(config):
        try:
            from qwen3_tts.core.http_client import server_request

            resp = server_request("GET", "/models", timeout=5)
            if resp.status_code == 200:
                server_status = resp.json().get("models", {})
        except Exception:  # nosec B110
            logger.debug("Could not fetch server model status")

    for model_type, info in MODEL_INFO.items():
        display = model_display.get(model_type, {})
        cfg = models_config.get(model_type, {})
        load_at_startup = cfg.get("load_at_startup", False)

        if server_status:
            loaded = server_status.get(model_type, {}).get("loaded", False)
            status = "LOADED" if loaded else "not loaded"
        else:
            status = "server not running"

        startup_str = "YES" if load_at_startup else "no"

        print(f"  {display.get('name', model_type):<16} [{status}]")
        print(f"    Model:       {info['name']}")
        print(f"    Purpose:     {info['description']}")
        print(f"    Usage:       {display.get('usage', '')}")
        print(f"    Memory:      {display.get('memory', '?')}")
        print(
            f"    Auto-load:   {startup_str} (config.json: models.{model_type}.load_at_startup)"
        )
        print()

    print("To change which models load at startup, edit config.json:")
    print('  "models": { "clone": { "load_at_startup": true }, ... }')
    print()
    print(
        "Models can also be loaded on-demand when you use a feature that requires them."
    )
    return False


def _handle_stats(config):
    """Display server statistics."""
    if is_server_running(config):
        from qwen3_tts.core.http_client import server_request

        resp = server_request("GET", "/stats", timeout=5)
        if resp.status_code == 200:
            stats = resp.json()
            print("TTS Server Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
        else:
            print("Error: Failed to get stats")
    else:
        print("Server not running. Start with 'tts server start'.")
    return False


# ---------------------------------------------------------------------------
# Generation dispatch
# ---------------------------------------------------------------------------


def _resolve_prompt_file(alias_prompt, args, config):
    """Resolve the clone prompt, verifying an explicitly-named one exists.

    An *implicit* default may fall back: with no prompt named anywhere,
    get_default_clone_prompt() scans for the first backend-appropriate prompt
    on disk. An *explicit* request must not — silently generating in a
    different voice than the one asked for is worse than failing, because
    nothing in the output tells the caller it happened.

    This line used to read ``alias_prompt or get_default_clone_prompt(config)``.
    A truthy-but-missing alias prompt short-circuited the fallback and the
    missing file surfaced as an unhandled FileNotFoundError from the engine —
    the odd one out among this module's input handlers, which print
    ``Error: <thing> not found: <path>`` and exit (see the SRT, dialogue and
    batch paths). repo-audit-2026-07-31 P1-3.
    """
    if not alias_prompt:
        return get_default_clone_prompt(config)

    from qwen3_tts.core.config import prompt_file_exists

    if prompt_file_exists(alias_prompt):
        return alias_prompt

    # Name the actual source so the user knows where to look. --prompt wins
    # over an alias's prompt upstream, so it takes precedence here too.
    if getattr(args, "prompt", None):
        source = "--prompt"
    elif getattr(args, "voice", None):
        source = f"voice alias '{args.voice}'"
    else:
        source = "configured prompt"
    print(f"Error: {source} points at a missing voice prompt: {alias_prompt}")
    print("Use 'tts voice list' to see available prompts.")
    sys.exit(1)


def _handle_generation(args, config, gen_params, use_server, max_chunk_chars):
    """Handle all generation: special modes, text resolution, and single-text output."""
    # Special modes
    if args.repl:
        run_repl(config, use_server, gen_params)
        return use_server
    if args.watch:
        run_watch_mode(args.watch, config, args, gen_params, use_server)
        return use_server
    if args.srt:
        srt_path = os.path.expanduser(args.srt)
        if not os.path.isfile(srt_path):
            print(f"Error: SRT file not found: {srt_path}")
            sys.exit(1)
        process_srt_file(srt_path, config, args, gen_params, use_server)
        return use_server
    if args.dialogue:
        dialogue_path = os.path.expanduser(args.dialogue)
        if not os.path.isfile(dialogue_path):
            print(f"Error: Dialogue file not found: {dialogue_path}")
            sys.exit(1)
        process_dialogue(dialogue_path, config, args, gen_params, use_server)
        return use_server

    # Voice alias — resolve into local overrides (do not mutate args)
    alias_prompt = args.prompt
    alias_mode = args.mode
    alias_description = args.description
    if args.voice:
        alias = get_voice_alias(args.voice, config)
        if alias is None:
            print(f"Error: Unknown voice alias '{args.voice}'")
            print("Available aliases:")
            aliases = config.get("aliases", {})
            for name, settings in aliases.items():
                print(f"  - {name}: {settings}")
            if not aliases:
                print("  (none configured - add to config.json under 'aliases')")
            sys.exit(1)
        if "prompt" in alias and not alias_prompt:
            alias_prompt = alias["prompt"]
        if "preset" in alias and not args.preset:
            gen_params = get_generation_params(args, config)
        if "mode" in alias and not alias_mode:
            alias_mode = alias["mode"]
        if "description" in alias and not alias_description:
            alias_description = alias["description"]
        print(f"Using voice alias '{args.voice}'")

    # Clipboard input — use local text_args (do not mutate args.text)
    text_args = list(args.text) if args.text else []
    if args.clipboard:
        clipboard_text = get_clipboard_text()
        text_args = [clipboard_text]
        print(f"Read from clipboard ({len(clipboard_text)} chars)")

    # Dry-run mode
    if args.dry_run:
        return _handle_dry_run(args, config, gen_params, use_server, max_chunk_chars)

    # Batch mode from file
    if args.batch:
        batch_path = os.path.expanduser(args.batch)
        if not os.path.isfile(batch_path):
            print(f"Error: Batch file not found: {batch_path}")
            sys.exit(1)
        with open(batch_path) as f:
            texts = json.load(f)
        if not isinstance(texts, list):
            print("Error: Batch file must contain a JSON array of texts")
            sys.exit(1)
        process_batch(texts, args, config, gen_params, use_server)
        return use_server

    # Multiple texts as batch
    if len(text_args) > 1:
        texts = [get_text(t) for t in text_args]
        process_batch(texts, args, config, gen_params, use_server)
        return use_server

    # Read from stdin if piped
    if not text_args and not sys.stdin.isatty():
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            text_args = [stdin_text]

    # Interactive mode if no text
    if not text_args:
        result = interactive_mode(use_server, config, gen_params)
        if result is None:
            return False
        return use_server

    # Resolve text
    text = args.text_override if args.text_override else get_text(text_args[0])
    if args.ssml:
        original_text = text
        text = process_ssml_text(text, args)
        if text != original_text:
            print(f"SSML processed: {len(original_text)} -> {len(text)} chars")

    with open(LAST_TEXT_FILE, "w") as f:
        f.write(text)

    output_name = args.output or "tts_output.wav"
    if ".." in output_name or output_name.startswith("/"):
        print(f"Error: Invalid output path: {output_name}")
        return use_server
    if not output_name.endswith(".wav"):
        output_name += ".wav"
    output_dir = os.path.expanduser(config.get("output_directory", "~/Downloads"))
    output_path = auto_increment_filename(safe_path_join(output_dir, output_name))

    language = config.get("language", "English")
    mode = alias_mode or "clone"
    prompt_file = _resolve_prompt_file(alias_prompt, args, config)
    voice_description = alias_description or config.get("default_voice_description", "")

    # Resolve prosody preset into local instruct (do not mutate args)
    instruct = args.instruct or ""
    if args.prosody and not instruct:
        from qwen3_tts.core.config import get_prosody_presets

        prosody_presets = get_prosody_presets(config)
        if args.prosody in prosody_presets:
            instruct = prosody_presets[args.prosody]
            print(f"Using prosody preset '{args.prosody}': {instruct}")
        else:
            available = ", ".join(sorted(prosody_presets.keys()))
            print(f"Error: Unknown prosody preset '{args.prosody}'")
            print(f"Available: {available}")
            sys.exit(1)

    # Determine mode and speaker
    speaker_name = None
    if alias_mode == "design" or alias_description:
        mode = "design"
    if alias_mode == "custom" or args.speaker:
        mode = "custom"
        speaker_key = (args.speaker or "ryan").lower()
        if speaker_key not in CUSTOM_VOICE_SPEAKERS:
            print(f"Error: Unknown speaker '{args.speaker}'")
            print("Use --list-speakers to see available options.")
            sys.exit(1)
        speaker_name = CUSTOM_VOICE_SPEAKERS[speaker_key]["name"]
        speaker_lang = CUSTOM_VOICE_SPEAKERS[speaker_key]["lang"]
        if not alias_description:
            language = speaker_lang

    return _run_single_generation(
        text,
        args,
        config,
        gen_params,
        use_server,
        max_chunk_chars,
        output_path,
        mode,
        language,
        prompt_file,
        voice_description,
        speaker_name,
        instruct=instruct,
    )


def _handle_dry_run(args, config, gen_params, use_server, max_chunk_chars):
    """Display dry-run summary of what would be generated."""
    mode = args.mode or "clone"
    prompt_file = args.prompt or get_default_clone_prompt(config)
    voice_description = args.description or config.get("default_voice_description", "")
    output_dir = os.path.expanduser(
        args.output or config.get("output_directory", "~/Downloads")
    )

    texts = []
    if args.batch:
        from qwen3_tts.core.config import safe_path_join

        # Security: validate batch_path against traversal
        batch_raw = args.batch
        batch_expanded = os.path.expanduser(batch_raw)
        if os.path.isabs(batch_expanded):
            if ".." in batch_expanded:
                raise ValueError(f"Path traversal detected in batch path: {batch_raw}")
            batch_path = batch_expanded
        else:
            batch_path = safe_path_join(os.getcwd(), batch_expanded)

        with open(batch_path) as f:
            texts = json.load(f)
    elif args.text:
        texts = [get_text(t) for t in args.text]

    print("\n=== DRY RUN ===")
    print(f"Mode: {mode}")
    if mode == "clone":
        print(f"Voice prompt: {prompt_file}")
    elif mode == "custom":
        speaker_key = (args.speaker or "ryan").lower()
        if speaker_key in CUSTOM_VOICE_SPEAKERS:
            speaker_info = CUSTOM_VOICE_SPEAKERS[speaker_key]
            print(f"Speaker: {speaker_info['name']} ({speaker_info['desc']})")
        else:
            print(f"Speaker: {args.speaker} (unknown)")
        if args.instruct:
            print(f"Instruction: {args.instruct}")
    else:
        print(f"Voice description: {voice_description}")
    print(f"Output directory: {output_dir}")
    print(f"Server mode: {'yes' if use_server else 'no (local)'}")
    print("\nAudio processing:")
    print(f"  Trim silence: {'yes' if args.trim_silence else 'no'}")
    print(f"  Normalize: {'yes (-3dB peak)' if args.normalize else 'no'}")
    print(f"  Speed: {args.speed if args.speed else '1.0 (unchanged)'}")
    print(f"  Pitch: {args.pitch if args.pitch else '0 (unchanged)'} semitones")
    print(f"  SSML: {'enabled' if args.ssml else 'disabled'}")
    chunk_cfg = (
        max_chunk_chars
        if max_chunk_chars is not None
        else config.get("generation", {}).get("max_chunk_chars", 500)
    )
    print(
        f"  Text chunking: {'disabled' if chunk_cfg == 0 else f'max {chunk_cfg} chars/chunk'}"
    )
    print("\nGeneration parameters:")
    for k, v in gen_params.items():
        print(f"  {k}: {v}")
    print(f"\nTexts to generate ({len(texts)}):")
    for i, t in enumerate(texts[:5], 1):
        preview = t[:80] + "..." if len(t) > 80 else t
        print(f"  {i}. {preview}")
    if len(texts) > 5:
        print(f"  ... and {len(texts) - 5} more")
    print("\n=== END DRY RUN ===")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _build_parser().parse_args()
    config = load_config()
    gen_params = get_generation_params(args, config)
    max_chunk_chars = getattr(args, "max_chunk_chars", None)

    if args.backend:
        os.environ["TTS_BACKEND"] = args.backend
    if args.model_size:
        os.environ["TTS_MODEL_SIZE"] = args.model_size

    if args.ui:
        launch_gradio_ui(config)
        return False

    result = _handle_info_commands(args, config, gen_params)
    if result is not None:
        return result

    use_server = args.server_mode and not args.local
    return _handle_generation(args, config, gen_params, use_server, max_chunk_chars)


if __name__ == "__main__":
    used_server = main()
    sys.exit(0 if not used_server else 2)
