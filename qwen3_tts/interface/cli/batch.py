#!/usr/bin/env python3
"""Batch processing for Qwen3-TTS CLI.

This module handles batch processing of multiple texts from JSON files.
"""

import json
import logging
import os

from qwen3_tts.core.config import (
    get_default_clone_prompt,
)
from qwen3_tts.interface.generate import (
    _decode_base64_result,
    _save_base64_result,
    generate_local,
    generate_via_server,
    process_audio_args,
)

logger = logging.getLogger("tts.cli.batch")


def process_batch(texts, args, config, gen_params, use_server):
    """Process multiple texts.

    Args:
        texts: List of text strings to synthesize
        args: Parsed command line arguments
        config: Configuration dictionary
        gen_params: Generation parameters
        use_server: Whether to use server mode

    Returns:
        List of output file paths
    """
    import soundfile as sf  # lazy — heavy import
    output_dir = os.path.expanduser(args.output or config.get("output_directory", "~/Downloads"))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    language = config.get("language", "English")
    mode = args.mode or "clone"
    prompt_file = args.prompt or get_default_clone_prompt(config)
    voice_description = args.description or config.get("default_voice_description", "")

    output_paths = []
    needs_processing = args.trim_silence or args.normalize or args.speed or args.pitch

    if use_server:
        logger.info(f"Using TTS server for batch of {len(texts)} texts...")
        print(f"Using TTS server for batch of {len(texts)} texts...")
        results = generate_via_server(
            texts, mode, config, gen_params,
            prompt_file=prompt_file if mode == "clone" else None,
            voice_description=voice_description if mode == "design" else None,
        )

        for i, result in enumerate(results):
            output_path = os.path.join(output_dir, f"output_{i+1}.wav")
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
            print(f"\nProcessing {i+1}/{len(texts)}...")
            wav, sr = generate_local(
                text, mode, gen_params, language,
                prompt_file=prompt_file,
                voice_description=voice_description,
            )
            wav = process_audio_args(wav, sr, args)

            output_path = os.path.join(output_dir, f"output_{i+1}.wav")
            sf.write(output_path, wav, sr)
            output_paths.append(output_path)
            print(f"Saved: {output_path}")

    return output_paths


def load_batch_file(batch_path):
    """Load texts from a batch JSON file.

    Args:
        batch_path: Path to JSON file containing array of texts

    Returns:
        List of text strings

    Raises:
        ValueError: If file is not a valid JSON array
    """
    batch_path = os.path.expanduser(batch_path)
    if not os.path.isfile(batch_path):
        raise FileNotFoundError(f"Batch file not found: {batch_path}")

    with open(batch_path, "r") as f:
        texts = json.load(f)

    if not isinstance(texts, list):
        raise ValueError("Batch file must contain a JSON array of texts")

    return texts
