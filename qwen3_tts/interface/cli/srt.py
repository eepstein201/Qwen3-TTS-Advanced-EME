#!/usr/bin/env python3
"""SRT subtitle processing for Qwen3-TTS CLI.

This module handles parsing and processing of SRT subtitle files.
"""

import os

from qwen3_tts.core.config import get_default_clone_prompt, safe_path_join
from qwen3_tts.interface.generate import (
    _decode_base64_result,
    generate_local,
    generate_via_server,
    parse_srt,
    play_audio,
    process_audio_args,
)

# ---------------------------------------------------------------------------
# SRT processing
# ---------------------------------------------------------------------------


def process_srt_file(srt_path, config, args, gen_params, use_server):
    """Process an SRT file and generate audio for each subtitle.

    Args:
        srt_path: Path to .srt file
        config: Configuration dict
        args: Parsed command line arguments
        gen_params: Generation parameters dict
        use_server: Whether to use server for generation
    """
    import numpy as np  # lazy — heavy import
    import soundfile as sf  # lazy — heavy import

    entries = parse_srt(srt_path)
    if not entries:
        print(f"Error: No subtitles found in {srt_path}")
        return

    # Security: validate output directory against path traversal
    base_raw = config.get("output_directory", "~/Downloads")
    base_expanded = os.path.expanduser(base_raw)
    if os.path.isabs(base_expanded):
        if ".." in base_expanded:
            raise ValueError(
                f"Path traversal detected in output_directory config: {base_raw}"
            )
        base_dir = base_expanded
    else:
        base_dir = safe_path_join(os.getcwd(), base_expanded)

    # Security: validate args.output against path traversal
    if args.output:
        output_raw = args.output
        output_expanded = os.path.expanduser(output_raw)
        if os.path.isabs(output_expanded):
            if ".." in output_expanded:
                raise ValueError(
                    f"Path traversal detected in output path: {output_raw}"
                )
            output_dir = output_expanded
        else:
            output_dir = safe_path_join(os.getcwd(), output_expanded)

        # Verify output_dir is under home directory (user data should be in home)
        home = os.path.realpath(os.path.expanduser("~"))
        resolved = os.path.realpath(output_dir)
        if not (resolved == home or resolved.startswith(home + os.sep)):
            raise ValueError(f"output path must be under home directory: {output_raw}")
    else:
        output_dir = base_dir

    os.makedirs(output_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(srt_path))[0]
    mode = args.mode or "clone"
    prompt_file = args.prompt or get_default_clone_prompt(config)
    voice_description = args.description or config.get("default_voice_description", "")

    print(f"\nProcessing SRT: {srt_path}")
    print(f"Found {len(entries)} subtitles")

    all_audio = []
    sample_rate = None
    success_count = 0

    for idx, start_ms, end_ms, text in entries:
        print(f"  [{idx}/{len(entries)}] {text[:50]}{'...' if len(text) > 50 else ''}")

        try:
            if use_server:
                results = generate_via_server(
                    [text],
                    mode,
                    config,
                    gen_params,
                    prompt_file=prompt_file if mode == "clone" else None,
                    voice_description=voice_description if mode == "design" else None,
                )
                wav, sr = _decode_base64_result(results[0])
            else:
                wav, sr = generate_local(
                    text,
                    mode,
                    gen_params,
                    config.get("language", "auto"),
                    prompt_file=prompt_file,
                    voice_description=voice_description,
                )

            wav = process_audio_args(wav, sr, args)

            if sample_rate is None:
                sample_rate = sr

            all_audio.append(wav)
            success_count += 1

            individual_path = safe_path_join(output_dir, f"{basename}_{idx:03d}.wav")
            sf.write(individual_path, wav, sr)
        except Exception as e:
            # One bad entry must not abort the whole file or discard the
            # combined output. Log, skip, and continue (mirrors dialogue.py).
            print(f"  [{idx}/{len(entries)}] FAILED, skipping: {e}")
            continue

    if not all_audio:
        print("\nNo subtitles succeeded; nothing to combine.")
        return

    # Combined file
    print(f"\nCreating combined audio from {success_count}/{len(entries)} subtitles...")
    combined = []
    silence_samples = int(sample_rate * 0.5)

    for i, wav in enumerate(all_audio):
        combined.extend(wav)
        if i < len(all_audio) - 1:
            combined.extend(np.zeros(silence_samples))

    combined_path = safe_path_join(output_dir, f"{basename}_combined.wav")
    sf.write(combined_path, np.array(combined), sample_rate)

    print(f"\nSaved {success_count}/{len(entries)} individual files to: {output_dir}")
    print(f"Combined audio: {combined_path}")

    if args.play:
        play_audio(combined_path)
