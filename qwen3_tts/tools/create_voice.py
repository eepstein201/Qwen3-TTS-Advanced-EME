#!/usr/bin/env python3
"""Create a custom voice clone prompt from reference audio.

Saves voice prompts in dual format by default:
  - .pt  (PyTorch tensor — used by torch backend)
  - .wav (reference audio — used by MLX backend)
  - .txt (transcript — used by MLX backend)

Use --mlx-only to skip .pt creation (no torch required, works from any env).
"""

import argparse
import os
import shutil
import subprocess  # nosec B404
import sys
import tempfile

from qwen3_tts.core.config import (
    USER_FILES_DIR,
    VOICE_PROMPTS_DIR,
    get_backend,
    safe_path_join,
    validate_voice_name,
)


def create_and_save_voice_prompt(
    audio_path,
    transcript,
    prompt_name,
    test_generation=True,
    mlx_only=False,
    x_vector_only_mode=False,
):
    """Create a voice clone prompt from audio and transcript.

    Args:
        audio_path: Path to reference audio file.
        transcript: Text transcript of the audio.
        prompt_name: Name for the prompt (with or without .pt extension).
        test_generation: Run a test generation after creating the prompt.
        mlx_only: If True, only save .wav + .txt (no .pt, no torch needed).
        x_vector_only_mode: If True, create a transcript-free
            (speaker-embedding-only) prompt. The .txt is written empty and the
            torch .pt stores the flag so generation runs in x-vector-only mode.
    """
    import soundfile as sf  # lazy — not needed at module import time
    from pydub import AudioSegment  # lazy — only used for non-wav format fallback

    # Load audio — try soundfile first (fast, supports wav/flac/ogg),
    # fall back to pydub for other formats (m4a, mp3, etc.)
    ext = os.path.splitext(audio_path)[1].lower().lstrip(".")
    wav_path = None
    try:
        ref_audio, ref_sr = sf.read(audio_path)
        print(f"Audio loaded: {len(ref_audio) / ref_sr:.1f} seconds at {ref_sr}Hz")
    except (sf.SoundFileError, RuntimeError):
        # Format not supported by soundfile — try pydub conversion.
        # PermissionError, FileNotFoundError, MemoryError etc. propagate normally.
        if ext == "m4a":
            ext = "mp4"
        print(f"Converting {audio_path} to wav format...")
        audio = AudioSegment.from_file(audio_path, format=ext)
        # Unique temp name: the old fixed "temp_reference.wav" raced two
        # concurrent creates (B's export overwrote A's input before A read
        # it). Cleanup is guaranteed by the try/finally below, which wraps
        # EVERYTHING after staging — any failure (validation, the writer,
        # torch save) removes the temp instead of orphaning it.
        fd, wav_path = tempfile.mkstemp(suffix=".wav", dir=USER_FILES_DIR)
        os.close(fd)
        audio.export(wav_path, format="wav")
        ref_audio, ref_sr = sf.read(wav_path)
        print(f"Audio loaded: {len(ref_audio) / ref_sr:.1f} seconds at {ref_sr}Hz")

    # Reference audio below the model's native rate makes MLX clone
    # generation fail to emit EOS (measured 2026-08-16 — an 8 kHz prompt ran
    # to the token cap 3/3). The MLX path hands this file's path straight to
    # mlx-audio, so it must be corrected here, at write time.
    from qwen3_tts.core.engine.audio_processing import ensure_min_sample_rate

    # was_modified covers a resample, a stereo downmix, or both — the flag
    # gates whether the .wav below is written from this array or byte-copied
    # from the original, so it must not be read as "was resampled". This call
    # also feeds the printouts + the torch path; the mlx_only delegation
    # re-validates inside the engine writer (idempotent — do not consolidate
    # away the call here without moving the prints).
    ref_audio, new_sr, was_modified = ensure_min_sample_rate(ref_audio, ref_sr)
    if was_modified:
        if new_sr != ref_sr:
            print(
                f"Reference audio upsampled {ref_sr}Hz -> {new_sr}Hz "
                f"(below {new_sr}Hz causes runaway generation)"
            )
        else:
            print("Reference audio downmixed to mono (model expects 1 channel)")
        ref_sr = new_sr

    # Normalize prompt name
    if not prompt_name.endswith(".pt"):
        prompt_name += ".pt"
    base_name = prompt_name[:-3]
    validate_voice_name(base_name)

    try:
            if mlx_only:
                # #236: MLX prompts are a .wav+.txt pair stored at write time —
                # no model, no .pt, no torch. The ENGINE writer owns the
                # validation + storage policy (one ensure_min_sample_rate
                # implementation for the server endpoint and the CLI alike);
                # the pydub conversion above is the CLI-only pre-step this layer
                # keeps. Transcript note: the writer stores it STRIPPED (the
                # old inline write stored it raw; every loader strips on read,
                # so generation behavior is unchanged).
                from qwen3_tts.core.engine import save_voice_prompt_mlx

                mlx_wav_path = save_voice_prompt_mlx(
                    base_name, wav_path or audio_path, transcript
                )
                print(f"MLX files saved: {mlx_wav_path}")
                print(
                    f'\nDone (MLX-only mode)! Use with: tts -p {prompt_name} "Your text here"'
                )
                return mlx_wav_path

            # --- Save MLX-compatible files (.wav + .txt) --- (torch: MLX interop)
            mlx_wav_path = safe_path_join(VOICE_PROMPTS_DIR, f"{base_name}.wav")
            mlx_txt_path = safe_path_join(VOICE_PROMPTS_DIR, f"{base_name}.txt")

            # Save .wav — copy from temp or write from loaded audio. A resampled
            # clip must be written from the array; copying the temp file would
            # restore the original low rate and undo the fix above.
            if wav_path and not was_modified:
                shutil.copy2(wav_path, mlx_wav_path)
            else:
                sf.write(mlx_wav_path, ref_audio, ref_sr)
            with open(mlx_txt_path, "w") as f:
                f.write(transcript)

            print(f"MLX files saved: {mlx_wav_path}")
            print(f"                 {mlx_txt_path}")

            # --- Save PyTorch .pt file (requires torch + qwen-tts) ---
            import torch

            from qwen3_tts.core.engine import (
                create_voice_prompt,
                load_model,
                run_inference,
            )

            print("Loading Qwen3-TTS Base model...")
            model = load_model("clone")

            print("\nCreating voice clone prompt...")
            voice_prompt = create_voice_prompt(
                model, ref_audio, ref_sr, transcript, x_vector_only_mode=x_vector_only_mode
            )

            output_path = safe_path_join(VOICE_PROMPTS_DIR, prompt_name)
            torch.save(voice_prompt, output_path)
            print(f"Torch file saved: {output_path}")

            # Optional test generation
            if test_generation:
                print("\nGenerating test audio with cloned voice...")
                test_text = "This is a test of the cloned voice. How does it sound?"

                wav, sr = run_inference(
                    model=model,
                    text=test_text,
                    mode="clone",
                    gen_params={
                        "temperature": 0.7,
                        "top_k": 50,
                        "top_p": 0.95,
                        "repetition_penalty": 1.05,
                    },
                    language="English",
                    voice_prompt=voice_prompt,
                    x_vector_only_mode=x_vector_only_mode,
                )

                test_output = safe_path_join(USER_FILES_DIR, f"test_{base_name}.wav")
                sf.write(test_output, wav, sr)
                print(f"Test audio saved to: {test_output}")
                from qwen3_tts.core.config import IS_LINUX, IS_MACOS

                if IS_MACOS:
                    subprocess.run(["open", test_output], timeout=10)  # nosec B603 B607
                elif IS_LINUX:
                    subprocess.run(
                        ["xdg-open", test_output], stderr=subprocess.DEVNULL, timeout=10
                    )  # nosec B603 B607
    finally:
        # Guaranteed temp cleanup: any failure after staging (validation,
        # the rate check, the writer, torch save) removes the mkstemp file
        # instead of orphaning it in USER_FILES_DIR.
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
    print(f'\nDone! Use with: tts -p {prompt_name} "Your text here"')
    print("  (Works with both torch and MLX backends)")
    return output_path


def _resolve_audio_path(args) -> str:
    """Resolve the reference audio path from args or interactive input."""
    from qwen3_tts.core.config import safe_path_join

    if not args.audio:
        print("\n=== Create Custom Voice Clone ===\n")
        audio_path = input("Path to reference audio file: ").strip()
        if not audio_path:
            print("Error: Audio path required")
            sys.exit(1)
    else:
        audio_path = args.audio

    # Security: validate audio_path against traversal
    expanded = os.path.expanduser(audio_path)
    if os.path.isabs(expanded):
        # Absolute paths: reject if they contain traversal sequences
        if ".." in expanded:
            raise ValueError(f"Path traversal detected in audio path: {audio_path}")
        safe_path = expanded
    else:
        # Relative paths: validate against current directory
        safe_path = safe_path_join(os.getcwd(), expanded)

    if not os.path.isfile(safe_path):
        # Security: validate downloads fallback path against traversal
        downloads_dir = os.path.expanduser("~/Downloads")
        if ".." in audio_path:
            raise ValueError(f"Path traversal detected in audio filename: {audio_path}")
        downloads_path = safe_path_join(downloads_dir, os.path.basename(audio_path))
        if os.path.isfile(downloads_path):
            return downloads_path
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    return safe_path


def _transcribe_with_asr(audio_path: str) -> str | None:
    """Auto-transcribe audio using MLX ASR. Returns transcript or None on failure/abort."""
    from qwen3_tts.core.engine import transcribe_audio

    print("\nAuto-transcribing reference audio...")
    try:
        transcript = transcribe_audio(audio_path)
        print(f'\nTranscript ({len(transcript)} chars):\n  "{transcript}"')
        confirm = input("\nUse this transcript? [Y/n]: ").strip().lower()
        if confirm and confirm not in ("y", "yes"):
            return None
        return transcript
    except Exception as e:
        print(f"Auto-transcription failed: {e}")
        return None


def _resolve_transcript(args, audio_path: str) -> str:
    """Resolve transcript from flags, ASR, or interactive input."""
    from qwen3_tts.core.engine import is_asr_available

    if args.no_transcript:
        print("Using x-vector only mode (no transcript)")
        return ""

    if args.transcript:
        transcript_path = os.path.expanduser(args.transcript)
        if os.path.isfile(transcript_path):
            with open(transcript_path) as f:
                transcript = f.read().strip()
            print(f"Loaded transcript from file ({len(transcript)} chars)")
            return transcript
        return args.transcript

    if args.auto_transcribe:
        if not is_asr_available():
            print(
                "Error: Auto-transcription requires MLX backend with mlx-audio STT support."
            )
            print("  Switch to MLX: set 'backend': 'mlx' in config.json")
            sys.exit(1)
        result = _transcribe_with_asr(audio_path)
        if result is None:
            print("Aborted. Provide transcript manually with --transcript.")
            sys.exit(1)
        return result

    # Interactive mode: offer ASR if available
    transcript = None
    if is_asr_available():
        print("\nNo transcript provided. Options:")
        print("  1. Auto-transcribe with MLX ASR")
        print("  2. Enter transcript manually")
        if input("Choose [1/2]: ").strip() == "1":
            transcript = _transcribe_with_asr(audio_path)

    if transcript is None:
        print("\nEnter the transcript of what is said in the audio.")
        print("(This helps the model understand the voice characteristics)")
        print("(You can paste the text directly or provide a path to a .txt file)")
        transcript = input("Transcript: ").strip()
        transcript_path = os.path.expanduser(transcript)

        # Security: validate transcript_path against traversal before file operations
        if os.path.isabs(transcript_path):
            if ".." in transcript_path:
                raise ValueError(
                    f"Path traversal detected in transcript path: {transcript}"
                )
            safe_transcript_path = transcript_path
        else:
            from qwen3_tts.core.config import safe_path_join

            safe_transcript_path = safe_path_join(os.getcwd(), transcript_path)

        if os.path.isfile(safe_transcript_path):
            with open(safe_transcript_path) as f:
                transcript = f.read().strip()
            print(f"Loaded transcript from file ({len(transcript)} chars)")

    if transcript is None:
        print("Error: Transcript required (use --no-transcript for x-vector only mode)")
        sys.exit(1)
    return transcript


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create a custom voice clone prompt from reference audio",
        epilog="""Examples:
  tts voice create audio.wav -t "Hello world" -n my_voice
  tts voice create audio.m4a -t transcript.txt -n my_voice --mlx-only
  tts voice create audio.wav -n my_voice --auto-transcribe  # MLX ASR
  tts voice create                                          # Interactive mode
""",
    )
    parser.add_argument("audio", nargs="?", help="Path to reference audio file")
    parser.add_argument(
        "-t", "--transcript", help="Transcript of the audio (text or file path)"
    )
    parser.add_argument(
        "-n", "--name", help="Name for the voice prompt (without .pt extension)"
    )
    parser.add_argument(
        "--no-test", action="store_true", help="Skip test audio generation"
    )
    parser.add_argument(
        "--mlx-only",
        action="store_true",
        help="Save only MLX files (.wav + .txt), skip .pt creation (no torch needed)",
    )
    parser.add_argument(
        "--force-torch",
        action="store_true",
        help="Force .pt creation even when MLX backend is active",
    )
    parser.add_argument(
        "--auto-transcribe",
        action="store_true",
        help="Auto-transcribe reference audio using MLX ASR (MLX backend only)",
    )
    parser.add_argument(
        "--no-transcript",
        action="store_true",
        help="Create voice with empty transcript (x-vector only mode, lower fidelity)",
    )
    args = parser.parse_args(argv)

    try:
        audio_path = _resolve_audio_path(args)
        transcript = _resolve_transcript(args, audio_path)

        if args.name:
            prompt_name = args.name
        else:
            prompt_name = (
                input("\nName for this voice (e.g., 'john_doe'): ").strip()
                or "custom_voice"
            )

        use_mlx_only = args.mlx_only
        if not use_mlx_only and not args.force_torch and get_backend() == "mlx":
            use_mlx_only = True
            print("Note: MLX backend active - using MLX-only mode (skip .pt creation)")
            print("      Use --force-torch to create .pt files for torch compatibility")

        create_and_save_voice_prompt(
            audio_path,
            transcript,
            prompt_name,
            test_generation=not args.no_test and not use_mlx_only,
            mlx_only=use_mlx_only,
            x_vector_only_mode=args.no_transcript,
        )
        return 0
    except SystemExit as e:
        return e.code if e.code is not None else 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
