#!/usr/bin/env python3
"""Create a custom voice clone prompt from reference audio."""

import argparse
import os
import sys
import numpy as np
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from pydub import AudioSegment

VOICE_PROMPTS_DIR = os.path.expanduser("~/Qwen3-TTS_UserFiles/voice_prompts")
USER_FILES_DIR = os.path.expanduser("~/Qwen3-TTS_UserFiles")


def create_voice_prompt(audio_path, transcript, prompt_name, test_generation=True):
    """Create a voice clone prompt from audio and transcript."""

    # Determine audio format from extension
    ext = os.path.splitext(audio_path)[1].lower().lstrip('.')
    if ext == 'm4a':
        ext = 'mp4'  # pydub uses mp4 for m4a

    # Convert to wav
    print(f"Converting {audio_path} to wav format...")
    audio = AudioSegment.from_file(audio_path, format=ext)
    wav_path = os.path.join(USER_FILES_DIR, "temp_reference.wav")
    audio.export(wav_path, format="wav")

    # Load the converted audio
    ref_audio, ref_sr = sf.read(wav_path)

    # Convert to mono if stereo (library bug: can't handle multi-channel in-place)
    if ref_audio.ndim > 1:
        ref_audio = np.mean(ref_audio, axis=-1).astype(np.float32)

    print(f"Audio loaded: {len(ref_audio)/ref_sr:.1f} seconds at {ref_sr}Hz")

    # Load the Base model (required for voice cloning from audio)
    print("Loading Qwen3-TTS Base model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="mps",
        dtype=torch.float16,
    )

    # Create reusable voice clone prompt
    print("\nCreating voice clone prompt...")
    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=(ref_audio, ref_sr),
        ref_text=transcript,
    )

    # Save the voice prompt
    if not prompt_name.endswith('.pt'):
        prompt_name += '.pt'
    output_path = os.path.join(VOICE_PROMPTS_DIR, prompt_name)
    torch.save(voice_prompt, output_path)
    print(f"Voice prompt saved to: {output_path}")

    # Optional test generation
    if test_generation:
        print("\nGenerating test audio with cloned voice...")
        test_text = "This is a test of the cloned voice. How does it sound?"

        wavs, sr = model.generate_voice_clone(
            text=test_text,
            language="English",
            voice_clone_prompt=voice_prompt,
        )

        test_output = os.path.join(USER_FILES_DIR, f"test_{prompt_name.replace('.pt', '.wav')}")
        sf.write(test_output, wavs[0], sr)
        print(f"Test audio saved to: {test_output}")
        os.system(f'open "{test_output}"')

    # Cleanup temp file
    if os.path.exists(wav_path):
        os.remove(wav_path)

    print(f"\nDone! Use with: changeVoice -p {prompt_name} \"Your text here\"")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create a custom voice clone prompt")
    parser.add_argument("audio", nargs="?", help="Path to reference audio file")
    parser.add_argument("-t", "--transcript", help="Transcript of the audio (text or file path)")
    parser.add_argument("-n", "--name", help="Name for the voice prompt (without .pt extension)")
    parser.add_argument("--no-test", action="store_true", help="Skip test audio generation")

    args = parser.parse_args()

    # Interactive mode if no arguments
    if not args.audio:
        print("\n=== Create Custom Voice Clone ===\n")
        audio_path = input("Path to reference audio file: ").strip()
        if not audio_path:
            print("Error: Audio path required")
            sys.exit(1)
    else:
        audio_path = args.audio

    # Expand and validate audio path
    audio_path = os.path.expanduser(audio_path)
    if not os.path.isfile(audio_path):
        # Check Downloads
        downloads_path = os.path.expanduser(f"~/Downloads/{audio_path}")
        if os.path.isfile(downloads_path):
            audio_path = downloads_path
        else:
            print(f"Error: Audio file not found: {audio_path}")
            sys.exit(1)

    # Get transcript
    if args.transcript:
        transcript = args.transcript
    else:
        print("\nEnter the transcript of what is said in the audio.")
        print("(This helps the model understand the voice characteristics)")
        print("(You can paste the text directly or provide a path to a .txt file)")
        transcript = input("Transcript: ").strip()
        if not transcript:
            print("Error: Transcript required")
            sys.exit(1)

    # Check if transcript is a file path
    transcript_path = os.path.expanduser(transcript)
    if os.path.isfile(transcript_path):
        with open(transcript_path, "r") as f:
            transcript = f.read().strip()
        print(f"Loaded transcript from file ({len(transcript)} chars)")

    # Get prompt name
    if args.name:
        prompt_name = args.name
    else:
        prompt_name = input("\nName for this voice (e.g., 'john_doe'): ").strip()
        if not prompt_name:
            prompt_name = "custom_voice"

    create_voice_prompt(audio_path, transcript, prompt_name, test_generation=not args.no_test)


if __name__ == "__main__":
    main()
