#!/usr/bin/env python3
"""Create MLX-compatible voice prompt files (.wav + .txt) from reference audio.

Voice prompts created before the MLX backend was added only have .pt files
(PyTorch tensor format). The MLX backend needs .wav (reference audio) and
.txt (transcript) files instead.

This script does NOT convert .pt tensors — that's not possible because .pt
files store processed embeddings, not raw audio. Instead, you provide the
original reference audio and transcript that were used to create the .pt
file, and this script creates the .wav/.txt companion files alongside it.

Usage:
    # Interactive mode — walks through all .pt-only prompts
    python convert_prompts_mlx.py

    # Convert a single prompt with known audio + transcript
    python convert_prompts_mlx.py --name lsmith --audio ~/Downloads/lsmith_sound.m4a --transcript "Hello, this is..."

    # Provide transcript from a file
    python convert_prompts_mlx.py --name lsmith --audio ~/Downloads/lsmith_sound.m4a --transcript ltref.txt

    # Dry run — show what would be done
    python convert_prompts_mlx.py --dry-run
"""

import argparse
import os
import shutil
import sys

from pydub import AudioSegment

from tts_config import VOICE_PROMPTS_DIR, USER_FILES_DIR


def find_pt_only_prompts():
    """Find .pt files that lack .wav/.txt companions."""
    pt_only = []
    if not os.path.isdir(VOICE_PROMPTS_DIR):
        return pt_only
    for f in sorted(os.listdir(VOICE_PROMPTS_DIR)):
        if not f.endswith(".pt"):
            continue
        base = f[:-3]
        wav = os.path.join(VOICE_PROMPTS_DIR, f"{base}.wav")
        txt = os.path.join(VOICE_PROMPTS_DIR, f"{base}.txt")
        if not os.path.exists(wav) or not os.path.exists(txt):
            pt_only.append(base)
    return pt_only


def convert_audio_to_wav(audio_path, output_wav_path):
    """Convert any audio format to WAV using pydub."""
    ext = os.path.splitext(audio_path)[1].lower().lstrip(".")
    if ext == "m4a":
        ext = "mp4"
    audio = AudioSegment.from_file(audio_path, format=ext)
    audio.export(output_wav_path, format="wav")
    duration_s = len(audio) / 1000
    return duration_s


def create_mlx_files(name, audio_path, transcript, dry_run=False):
    """Create .wav and .txt files for a voice prompt.

    Args:
        name: Base name (without extension), e.g. "lsmith".
        audio_path: Path to reference audio file.
        transcript: Transcript string of what's said in the audio.
        dry_run: If True, print what would be done without writing.

    Returns:
        (wav_path, txt_path) tuple, or None if dry_run.
    """
    wav_path = os.path.join(VOICE_PROMPTS_DIR, f"{name}.wav")
    txt_path = os.path.join(VOICE_PROMPTS_DIR, f"{name}.txt")

    if dry_run:
        print(f"  Would create: {wav_path}")
        print(f"  Would create: {txt_path}")
        print(f"  From audio:   {audio_path}")
        print(f"  Transcript:   {transcript[:80]}{'...' if len(transcript) > 80 else ''}")
        return None

    # Convert audio to wav
    audio_path = os.path.expanduser(audio_path)
    if not os.path.isfile(audio_path):
        print(f"  Error: Audio file not found: {audio_path}")
        return None

    if audio_path.lower().endswith(".wav"):
        shutil.copy2(audio_path, wav_path)
        print(f"  Copied: {wav_path}")
    else:
        duration = convert_audio_to_wav(audio_path, wav_path)
        print(f"  Converted to wav: {wav_path} ({duration:.1f}s)")

    # Write transcript
    with open(txt_path, "w") as f:
        f.write(transcript)
    print(f"  Saved transcript: {txt_path} ({len(transcript)} chars)")

    return wav_path, txt_path


def resolve_transcript(transcript_arg):
    """If transcript_arg is a file path, read it; otherwise return as-is."""
    expanded = os.path.expanduser(transcript_arg)
    if os.path.isfile(expanded):
        with open(expanded, "r") as f:
            text = f.read().strip()
        print(f"  (Loaded transcript from {expanded}: {len(text)} chars)")
        return text
    return transcript_arg


def interactive_mode(dry_run=False):
    """Walk through all .pt-only prompts interactively."""
    pt_only = find_pt_only_prompts()

    if not pt_only:
        print("All voice prompts already have MLX-compatible files (.wav + .txt).")
        print("Nothing to convert.")
        return

    print(f"\nFound {len(pt_only)} voice prompt(s) needing MLX conversion:\n")
    for name in pt_only:
        print(f"  - {name}.pt")

    print(f"\nFor each prompt, provide the original reference audio and transcript.")
    print(f"Press Enter to skip a prompt, or Ctrl+C to quit.\n")

    converted = 0
    skipped = 0

    for name in pt_only:
        print(f"--- {name} ---")

        audio_path = input(f"  Reference audio path for '{name}' (or Enter to skip): ").strip()
        if not audio_path:
            print(f"  Skipping {name}")
            skipped += 1
            continue

        audio_path = os.path.expanduser(audio_path)
        if not os.path.isfile(audio_path):
            # Check Downloads
            dl_path = os.path.expanduser(f"~/Downloads/{audio_path}")
            if os.path.isfile(dl_path):
                audio_path = dl_path
            else:
                print(f"  Error: File not found: {audio_path}")
                skipped += 1
                continue

        transcript = input(f"  Transcript (text or path to .txt file): ").strip()
        if not transcript:
            print(f"  Error: Transcript required. Skipping {name}")
            skipped += 1
            continue

        transcript = resolve_transcript(transcript)

        result = create_mlx_files(name, audio_path, transcript, dry_run=dry_run)
        if result:
            converted += 1
            print(f"  Done: {name} is now MLX-compatible\n")
        else:
            if not dry_run:
                skipped += 1
            print()

    print(f"\nSummary: {converted} converted, {skipped} skipped")
    if converted > 0:
        print("Voice prompts are now usable with both torch and MLX backends.")


def main():
    parser = argparse.ArgumentParser(
        description="Create MLX-compatible files (.wav + .txt) for voice prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python convert_prompts_mlx.py                          # Interactive mode
  python convert_prompts_mlx.py --dry-run                # Preview what needs conversion
  python convert_prompts_mlx.py --name lsmith \\
    --audio ~/Downloads/lsmith_sound.m4a \\
    --transcript "Hello, this is a test"
  python convert_prompts_mlx.py --name lsmith \\
    --audio ~/Downloads/lsmith_sound.m4a \\
    --transcript ltref.txt                               # Transcript from file
""",
    )
    parser.add_argument("--name", help="Voice prompt base name (e.g. 'lsmith')")
    parser.add_argument("--audio", help="Path to reference audio file")
    parser.add_argument("--transcript", help="Transcript text or path to .txt file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--list", action="store_true", help="List prompts needing conversion")

    args = parser.parse_args()

    if args.list:
        pt_only = find_pt_only_prompts()
        if not pt_only:
            print("All voice prompts have MLX-compatible files.")
        else:
            print(f"{len(pt_only)} prompt(s) need MLX conversion:")
            for name in pt_only:
                print(f"  {name}.pt  (missing .wav + .txt)")
        return

    if args.name:
        # Single prompt mode
        if not args.audio or not args.transcript:
            print("Error: --audio and --transcript required with --name")
            sys.exit(1)

        transcript = resolve_transcript(args.transcript)

        print(f"Converting '{args.name}'...")
        result = create_mlx_files(args.name, args.audio, transcript, dry_run=args.dry_run)
        if result and not args.dry_run:
            print(f"\nDone. '{args.name}' is now MLX-compatible.")
        elif args.dry_run:
            print("\n(Dry run — no files written)")
    else:
        # Interactive mode
        interactive_mode(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
