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

    output_dir = os.path.expanduser(args.output or config.get("output_directory", "~/Downloads"))
    os.makedirs(output_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(srt_path))[0]
    mode = args.mode or "clone"
    prompt_file = args.prompt or get_default_clone_prompt(config)
    voice_description = args.description or config.get("default_voice_description", "")

    print(f"\nProcessing SRT: {srt_path}")
    print(f"Found {len(entries)} subtitles")

    all_audio = []
    sample_rate = None

    for idx, start_ms, end_ms, text in entries:
        print(f"  [{idx}/{len(entries)}] {text[:50]}{'...' if len(text) > 50 else ''}")

        if use_server:
            results = generate_via_server(
                [text], mode, config, gen_params,
                prompt_file=prompt_file if mode == "clone" else None,
                voice_description=voice_description if mode == "design" else None,
            )
            wav, sr = _decode_base64_result(results[0])
        else:
            wav, sr = generate_local(
                text, mode, gen_params,
                config.get("language", "English"),
                prompt_file=prompt_file,
                voice_description=voice_description,
            )

        wav = process_audio_args(wav, sr, args)

        if sample_rate is None:
            sample_rate = sr

        all_audio.append(wav)

        individual_path = safe_path_join(output_dir, f"{basename}_{idx:03d}.wav")
        sf.write(individual_path, wav, sr)

    # Combined file
    print("\nCreating combined audio...")
    combined = []
    silence_samples = int(sample_rate * 0.5)

    for i, wav in enumerate(all_audio):
        combined.extend(wav)
        if i < len(all_audio) - 1:
            combined.extend(np.zeros(silence_samples))

    combined_path = safe_path_join(output_dir, f"{basename}_combined.wav")
    sf.write(combined_path, np.array(combined), sample_rate)

    print(f"\nSaved {len(entries)} individual files to: {output_dir}")
    print(f"Combined audio: {combined_path}")

    if args.play:
        play_audio(combined_path)


