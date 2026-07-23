#!/usr/bin/env python3
"""Batch processing for Qwen3-TTS CLI.

This module handles batch processing of multiple texts from JSON files.

``process_batch`` is a thin delegator to the canonical, path-traversal-hardened
implementation in :mod:`qwen3_tts.interface.generate`. It is kept here so the
``qwen3_tts.interface.cli.batch`` import path continues to work; the lazy
in-function import mirrors the delegator pattern used by ``process_dialogue``
and ``process_srt_file`` and keeps the heavy import out of module top level.
"""

import json
import os


def process_batch(texts, args, config, gen_params, use_server):
    """Process multiple texts.

    Delegates to :func:`qwen3_tts.interface.generate.process_batch`, which owns
    the single canonical implementation (hardened with :func:`safe_path_join`).
    This wrapper exists only for import-path compatibility — the previous
    duplicate copy had drifted to plain ``os.path.join`` (a path-traversal
    regression) and is now removed in favor of the shared implementation.

    Args:
        texts: List of text strings to synthesize
        args: Parsed command line arguments
        config: Configuration dictionary
        gen_params: Generation parameters
        use_server: Whether to use server mode

    Returns:
        List of output file paths
    """
    from qwen3_tts.interface.generate import process_batch as _impl

    return _impl(texts, args, config, gen_params, use_server)


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

    with open(batch_path) as f:
        texts = json.load(f)

    if not isinstance(texts, list):
        raise ValueError("Batch file must contain a JSON array of texts")

    return texts
