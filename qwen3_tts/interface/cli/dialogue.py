#!/usr/bin/env python3
"""Dialogue processing for Qwen3-TTS CLI.

This module handles multi-speaker dialogue generation from JSON files.
"""

import json
import logging
import os

from qwen3_tts.core.config import (
    CUSTOM_VOICE_SPEAKERS,
    get_default_clone_prompt,
    safe_path_join,
)
from qwen3_tts.interface.generate import (
    _decode_base64_result,
    generate_local,
    generate_via_server,
    log_generation,
    open_file,
    play_audio,
    process_audio_args,
)

logger = logging.getLogger("tts.cli.dialogue")


def process_dialogue(dialogue_path, config, args, gen_params, use_server):
    """Process a dialogue JSON file with multiple speakers.

    Args:
        dialogue_path: Path to dialogue JSON file
        config: Configuration dictionary
        args: Parsed command line arguments
        gen_params: Generation parameters
        use_server: Whether to use server mode

    The dialogue JSON format supports:
    - Simple array of line objects with text and optional speaker/prompt/mode
    - Object with 'speakers' config and 'lines' array
    - Optional 'pause_ms' to control pause between lines (default 500ms)

    Example:
        {
            "speakers": {
                "narrator": {"prompt": "narrator.pt", "mode": "clone"},
                "alice": {"mode": "custom", "speaker": "vivian"},
                "bob": {"mode": "design", "description": "deep male voice"}
            },
            "pause_ms": 300,
            "lines": [
                {"speaker": "narrator", "text": "Once upon a time..."},
                {"speaker": "alice", "text": "Hello!"},
                {"speaker": "bob", "text": "Hi there!"}
            ]
        }
    """
    import numpy as np  # lazy — heavy import
    import soundfile as sf  # lazy — heavy import
    with open(dialogue_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        lines = data
        speakers = {}
        pause_ms = 500
    else:
        speakers = data.get("speakers", {})
        lines = data.get("lines", data.get("dialogue", []))
        pause_ms = data.get("pause_ms", 500)
        # Validate: clamp to [0, 10000ms] to prevent negative or excessive allocation
        try:
            pause_ms = max(0, min(10000, int(pause_ms)))
        except (TypeError, ValueError):
            pause_ms = 500

    if not lines:
        print("Error: No dialogue lines found in file")
        return

    output_dir = os.path.expanduser(args.output or config.get("output_directory", "~/Downloads"))
    os.makedirs(output_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(dialogue_path))[0]
    language = config.get("language", "English")

    print(f"\nProcessing dialogue: {dialogue_path}")
    print(f"Found {len(lines)} lines, pause between lines: {pause_ms}ms")

    all_audio = []
    sample_rate = None

    for idx, line in enumerate(lines, 1):
        text = line.get("text", "")
        if not text:
            continue

        # Resolve speaker config
        if "speaker" in line and line["speaker"] in speakers:
            speaker_config = speakers[line["speaker"]].copy()
            speaker_name = line["speaker"]
        else:
            speaker_config = line.copy()
            speaker_name = line.get("speaker", line.get("prompt", "unknown"))

        mode = speaker_config.get("mode", "clone")

        prompt_file = speaker_config.get("prompt", get_default_clone_prompt(config))
        voice_description = speaker_config.get("description", config.get("default_voice_description", ""))
        custom_speaker = speaker_config.get("speaker", "ryan")
        instruct = speaker_config.get("instruct", line.get("instruct", ""))

        # Resolve custom speaker name
        if mode == "custom":
            speaker_key = custom_speaker.lower()
            if speaker_key in CUSTOM_VOICE_SPEAKERS:
                resolved_speaker = CUSTOM_VOICE_SPEAKERS[speaker_key]["name"]
            else:
                resolved_speaker = custom_speaker
        else:
            resolved_speaker = None

        preview = text[:40] + "..." if len(text) > 40 else text
        print(f"  [{idx}/{len(lines)}] {speaker_name} ({mode}): \"{preview}\"")

        try:
            if use_server:
                results = generate_via_server(
                    [text], mode, config, gen_params,
                    prompt_file=prompt_file if mode == "clone" else None,
                    voice_description=voice_description if mode == "design" else None,
                    speaker=resolved_speaker if mode == "custom" else None,
                    instruct=instruct if mode == "custom" else None,
                )
                wav, sr = _decode_base64_result(results[0])
            else:
                wav, sr = generate_local(
                    text, mode, gen_params, language,
                    prompt_file=prompt_file,
                    voice_description=voice_description,
                    speaker=resolved_speaker,
                    instruct=instruct,
                )

            wav = process_audio_args(wav, sr, args)

            if sample_rate is None:
                sample_rate = sr

            all_audio.append(wav)

            if args.save_individual:
                individual_path = safe_path_join(output_dir, f"{basename}_{idx:03d}.wav")
                sf.write(individual_path, wav, sr)

        except Exception as e:
            logger.error("Error generating line %d: %s", idx, e)
            print(f"    Error generating line {idx}: {e}")
            continue

    if not all_audio:
        print("Error: No audio generated")
        return

    # Combine with pauses
    print("\nCombining audio...")
    silence_samples = int(sample_rate * pause_ms / 1000)
    combined = []

    for i, wav in enumerate(all_audio):
        combined.extend(wav)
        if i < len(all_audio) - 1:
            combined.extend(np.zeros(silence_samples))

    combined_path = safe_path_join(output_dir, f"{basename}.wav")
    sf.write(combined_path, np.array(combined), sample_rate)

    duration_sec = len(combined) / sample_rate
    print(f"\nSaved: {combined_path}")
    print(f"Duration: {duration_sec:.1f}s ({len(all_audio)} segments)")

    log_generation(
        f"[Dialogue: {len(lines)} lines]",
        "dialogue", basename, combined_path,
        gen_params, duration_sec,
    )

    if args.play:
        play_audio(combined_path)
    elif not args.no_open:
        open_file(combined_path)
