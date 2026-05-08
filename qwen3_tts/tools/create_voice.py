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

from qwen3_tts.core.config import VOICE_PROMPTS_DIR, USER_FILES_DIR, get_backend, safe_path_join, validate_voice_name


def create_and_save_voice_prompt(audio_path, transcript, prompt_name,
                                  test_generation=True, mlx_only=False):
    """Create a voice clone prompt from audio and transcript.

    Args:
        audio_path: Path to reference audio file.
        transcript: Text transcript of the audio.
        prompt_name: Name for the prompt (with or without .pt extension).
        test_generation: Run a test generation after creating the prompt.
        mlx_only: If True, only save .wav + .txt (no .pt, no torch needed).
    """
    import soundfile as sf  # lazy — not needed at module import time
    from pydub import AudioSegment  # lazy — only used for non-wav format fallback

    # Load audio — try soundfile first (fast, supports wav/flac/ogg),
    # fall back to pydub for other formats (m4a, mp3, etc.)
    ext = os.path.splitext(audio_path)[1].lower().lstrip('.')
    wav_path = None
    try:
        ref_audio, ref_sr = sf.read(audio_path)
        print(f"Audio loaded: {len(ref_audio)/ref_sr:.1f} seconds at {ref_sr}Hz")
    except (sf.SoundFileError, RuntimeError):
        # Format not supported by soundfile — try pydub conversion.
        # PermissionError, FileNotFoundError, MemoryError etc. propagate normally.
        if ext == 'm4a':
            ext = 'mp4'
        print(f"Converting {audio_path} to wav format...")
        audio = AudioSegment.from_file(audio_path, format=ext)
        wav_path = os.path.join(USER_FILES_DIR, "temp_reference.wav")
        audio.export(wav_path, format="wav")
        ref_audio, ref_sr = sf.read(wav_path)
        print(f"Audio loaded: {len(ref_audio)/ref_sr:.1f} seconds at {ref_sr}Hz")

    # Normalize prompt name
    if not prompt_name.endswith('.pt'):
        prompt_name += '.pt'
    base_name = prompt_name[:-3]
    validate_voice_name(base_name)

    # --- Save MLX-compatible files (.wav + .txt) ---
    mlx_wav_path = safe_path_join(VOICE_PROMPTS_DIR, f"{base_name}.wav")
    mlx_txt_path = safe_path_join(VOICE_PROMPTS_DIR, f"{base_name}.txt")

    # Save .wav — copy from temp or write from loaded audio
    if wav_path:
        shutil.copy2(wav_path, mlx_wav_path)
    else:
        sf.write(mlx_wav_path, ref_audio, ref_sr)
    with open(mlx_txt_path, "w") as f:
        f.write(transcript)

    print(f"MLX files saved: {mlx_wav_path}")
    print(f"                 {mlx_txt_path}")

    if mlx_only:
        # Clean up temp and exit early — no torch needed
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
        print(f"\nDone (MLX-only mode)! Use with: tts -p {prompt_name} \"Your text here\"")
        return mlx_wav_path

    # --- Save PyTorch .pt file (requires torch + qwen-tts) ---
    import torch
    from qwen3_tts.core.engine import load_model, create_voice_prompt, run_inference

    print("Loading Qwen3-TTS Base model...")
    model = load_model("clone")

    print("\nCreating voice clone prompt...")
    voice_prompt = create_voice_prompt(model, ref_audio, ref_sr, transcript)

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
            gen_params={"temperature": 0.7, "top_k": 50, "top_p": 0.95, "repetition_penalty": 1.05},
            language="English",
            voice_prompt=voice_prompt,
        )

        test_output = safe_path_join(USER_FILES_DIR, f"test_{base_name}.wav")
        sf.write(test_output, wav, sr)
        print(f"Test audio saved to: {test_output}")
        from qwen3_tts.core.config import IS_MACOS, IS_LINUX
        if IS_MACOS:
            subprocess.run(["open", test_output], timeout=10)  # nosec B603 B607
        elif IS_LINUX:
            subprocess.run(["xdg-open", test_output], stderr=subprocess.DEVNULL, timeout=10)  # nosec B603 B607

    # Cleanup temp file
    if wav_path and os.path.exists(wav_path):
        os.remove(wav_path)

    print(f"\nDone! Use with: tts -p {prompt_name} \"Your text here\"")
    print("  (Works with both torch and MLX backends)")
    return output_path


def _resolve_audio_path(args) -> str:
    """Resolve the reference audio path from args or interactive input."""
    if not args.audio:
        print("\n=== Create Custom Voice Clone ===\n")
        audio_path = input("Path to reference audio file: ").strip()
        if not audio_path:
            print("Error: Audio path required")
            sys.exit(1)
    else:
        audio_path = args.audio

    audio_path = os.path.expanduser(audio_path)
    if not os.path.isfile(audio_path):
        downloads_path = os.path.expanduser(f"~/Downloads/{audio_path}")
        if os.path.isfile(downloads_path):
            return downloads_path
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    return audio_path


def _transcribe_with_asr(audio_path: str) -> str | None:
    """Auto-transcribe audio using MLX ASR. Returns transcript or None on failure/abort."""
    from qwen3_tts.core.engine import transcribe_audio
    print("\nAuto-transcribing reference audio...")
    try:
        transcript = transcribe_audio(audio_path)
        print(f"\nTranscript ({len(transcript)} chars):\n  \"{transcript}\"")
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
            with open(transcript_path, "r") as f:
                transcript = f.read().strip()
            print(f"Loaded transcript from file ({len(transcript)} chars)")
            return transcript
        return args.transcript

    if args.auto_transcribe:
        if not is_asr_available():
            print("Error: Auto-transcription requires MLX backend with mlx-audio STT support.")
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
        if os.path.isfile(transcript_path):
            with open(transcript_path, "r") as f:
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
    parser.add_argument("-t", "--transcript", help="Transcript of the audio (text or file path)")
    parser.add_argument("-n", "--name", help="Name for the voice prompt (without .pt extension)")
    parser.add_argument("--no-test", action="store_true", help="Skip test audio generation")
    parser.add_argument("--mlx-only", action="store_true",
                        help="Save only MLX files (.wav + .txt), skip .pt creation (no torch needed)")
    parser.add_argument("--force-torch", action="store_true",
                        help="Force .pt creation even when MLX backend is active")
    parser.add_argument("--auto-transcribe", action="store_true",
                        help="Auto-transcribe reference audio using MLX ASR (MLX backend only)")
    parser.add_argument("--no-transcript", action="store_true",
                        help="Create voice with empty transcript (x-vector only mode, lower fidelity)")
    args = parser.parse_args(argv)

    try:
        audio_path = _resolve_audio_path(args)
        transcript = _resolve_transcript(args, audio_path)

        if args.name:
            prompt_name = args.name
        else:
            prompt_name = input("\nName for this voice (e.g., 'john_doe'): ").strip() or "custom_voice"

        use_mlx_only = args.mlx_only
        if not use_mlx_only and not args.force_torch and get_backend() == "mlx":
            use_mlx_only = True
            print("Note: MLX backend active - using MLX-only mode (skip .pt creation)")
            print("      Use --force-torch to create .pt files for torch compatibility")

        create_and_save_voice_prompt(
            audio_path, transcript, prompt_name,
            test_generation=not args.no_test and not use_mlx_only,
            mlx_only=use_mlx_only,
        )
        return 0
    except SystemExit as e:
        return e.code if e.code is not None else 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
