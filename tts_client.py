#!/usr/bin/env python3
"""
TTS Client Library - HTTP-only Python API for Qwen3-TTS generation.

This module NEVER imports torch or tts_engine — it communicates
exclusively over HTTP to the TTS server.

Usage:
    from tts_client import TTSClient

    client = TTSClient()

    # Generate with default voice
    audio_path = client.generate("Hello world", output="greeting.wav")

    # Generate with specific voice prompt
    audio_path = client.generate("Hello", prompt="narrator.pt")

    # Generate with voice alias
    audio_path = client.generate("Hello", voice="narrator")

    # Generate with premium speaker (custom mode)
    audio_path = client.generate(
        "Hello",
        mode="custom",
        speaker="Ryan",
        instruct="Speak with enthusiasm"
    )

    # Check server status
    if client.is_server_running():
        stats = client.get_stats()
        print(f"Memory: {stats['mps_memory_allocated_mb']}MB")
"""

import json
import os
import shutil

import requests
import soundfile as sf

from tts_config import (
    CONFIG_PATH,
    VOICE_PROMPTS_DIR,
    get_default_clone_prompt,
    get_server_url,
    is_server_running,
    auth_headers,
)


class TTSClient:
    """HTTP-only client for Qwen3-TTS generation."""

    def __init__(self, config_path=None):
        """Initialize the TTS client.

        Args:
            config_path: Path to config.json. Defaults to ~/Qwen3-TTS_UserFiles/config.json
        """
        self.config_path = config_path or CONFIG_PATH
        self.voice_prompts_dir = VOICE_PROMPTS_DIR
        self._config = None

    @property
    def config(self):
        """Load and cache configuration."""
        if self._config is None:
            with open(self.config_path, "r") as f:
                self._config = json.load(f)
        return self._config

    def reload_config(self):
        """Force reload of configuration."""
        self._config = None
        return self.config

    @property
    def server_url(self):
        """Get the server URL from config."""
        return get_server_url(self.config)

    def is_server_running(self):
        """Check if the TTS server is running."""
        return is_server_running(self.config)

    def get_stats(self):
        """Get server statistics."""
        if not self.is_server_running():
            raise ConnectionError("TTS server is not running")
        resp = requests.get(f"{self.server_url}/stats", timeout=5, headers=auth_headers())
        return resp.json()

    def list_prompts(self):
        """List available voice prompts."""
        prompts = [f for f in os.listdir(self.voice_prompts_dir) if f.endswith('.pt')]
        return sorted(prompts)

    def list_presets(self):
        """List available presets."""
        return self.config.get("presets", {})

    def list_aliases(self):
        """List available voice aliases."""
        return self.config.get("aliases", {})

    def resolve_alias(self, alias_name):
        """Resolve a voice alias to its settings."""
        aliases = self.config.get("aliases", {})
        return aliases.get(alias_name)

    def generate(
        self,
        text,
        output=None,
        mode="clone",
        prompt=None,
        description=None,
        speaker=None,
        instruct=None,
        voice=None,
        preset=None,
        temperature=None,
        top_k=None,
        top_p=None,
        seed=None,
        repetition_penalty=None,
        speed=None,
        pitch=None,
        normalize=False,
        trim_silence=False,
    ):
        """Generate speech from text via the server.

        Args:
            text: Text to synthesize
            output: Output file path (default: auto-generated in output_directory)
            mode: "clone", "design", or "custom"
            prompt: Voice prompt filename (for clone mode)
            description: Voice description (for design mode)
            speaker: Speaker name (for custom mode) - Ryan, Aiden, Vivian, etc.
            instruct: Instruction for speech style (for custom mode)
            voice: Voice alias name (overrides prompt/preset)
            preset: Preset name to use
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            seed: Random seed
            repetition_penalty: Repetition penalty
            speed: Speed factor (1.2 = 20% faster)
            pitch: Pitch shift in semitones
            normalize: Normalize audio to -3dB peak
            trim_silence: Trim leading/trailing silence

        Returns:
            Path to the generated audio file
        """
        # Resolve voice alias
        if voice:
            alias = self.resolve_alias(voice)
            if alias:
                if "prompt" in alias and prompt is None:
                    prompt = alias["prompt"]
                if "preset" in alias and preset is None:
                    preset = alias["preset"]
                if "mode" in alias:
                    mode = alias["mode"]
                if "description" in alias and description is None:
                    description = alias["description"]
                if "speaker" in alias and speaker is None:
                    speaker = alias["speaker"]
                if "instruct" in alias and instruct is None:
                    instruct = alias["instruct"]
            else:
                raise ValueError(f"Unknown voice alias: {voice}")

        # Get generation parameters
        gen_config = self.config.get("generation", {})
        gen_params = {
            "temperature": temperature if temperature is not None else gen_config.get("temperature", 0.7),
            "top_k": top_k if top_k is not None else gen_config.get("top_k", 50),
            "top_p": top_p if top_p is not None else gen_config.get("top_p", 0.95),
            "repetition_penalty": repetition_penalty if repetition_penalty is not None else gen_config.get("repetition_penalty", 1.05),
        }

        if seed is not None:
            gen_params["seed"] = seed
        elif gen_config.get("seed"):
            gen_params["seed"] = gen_config["seed"]

        # Apply preset
        if preset:
            presets = self.config.get("presets", {})
            if preset in presets:
                gen_params.update(presets[preset])

        # Default prompt/description/speaker
        if mode == "clone":
            prompt = prompt or get_default_clone_prompt(self.config)
        elif mode == "custom":
            speaker = speaker or self.config.get("default_speaker", "Ryan")
            instruct = instruct or ""
        else:
            description = description or self.config.get("default_voice_description", "")

        # Determine output path
        if output is None:
            output_dir = os.path.expanduser(self.config.get("output_directory", "~/Downloads"))
            output = os.path.join(output_dir, "tts_output.wav")
        else:
            output = os.path.expanduser(output)
            if not output.endswith('.wav'):
                output += '.wav'

        # Generate audio via server
        wav, sr = self._generate_via_server(text, mode, prompt, description, speaker, instruct, gen_params)

        # Apply audio processing (lazy import — only if needed)
        needs_processing = trim_silence or normalize or (speed and speed != 1.0) or (pitch and pitch != 0)
        if needs_processing:
            from tts_engine import process_audio
            wav = process_audio(wav, sr, trim=trim_silence, normalize=normalize,
                                speed=speed, pitch=pitch)

        # Save output
        sf.write(output, wav, sr)
        return output

    def _generate_via_server(self, text, mode, prompt, description, speaker, instruct, gen_params):
        """Generate audio via the TTS server."""
        payload = {
            "texts": [text],
            "mode": mode,
            "language": self.config.get("language", "English"),
            **gen_params,
        }

        if mode == "clone":
            payload["prompt_file"] = prompt
        elif mode == "custom":
            payload["speaker"] = speaker
            payload["instruct"] = instruct or ""
        else:
            payload["voice_description"] = description

        resp = requests.post(f"{self.server_url}/generate", json=payload, timeout=300, headers=auth_headers())
        if resp.status_code != 200:
            try:
                error_msg = resp.json().get("error", "Unknown error")
            except (ValueError, requests.exceptions.JSONDecodeError):
                error_msg = f"Server returned HTTP {resp.status_code} (non-JSON response)"
            raise Exception(f"Server error: {error_msg}")

        result = resp.json()["results"][0]
        wav, sr = sf.read(result["file"])
        os.remove(result["file"])
        return wav, sr

    def generate_dialogue(
        self,
        lines,
        output=None,
        speakers=None,
        pause_ms=500,
        preset=None,
        temperature=None,
        top_k=None,
        top_p=None,
        seed=None,
        repetition_penalty=None,
        speed=None,
        pitch=None,
        normalize=False,
        trim_silence=False,
    ):
        """Generate multi-speaker dialogue audio.

        Args:
            lines: List of dialogue lines. Each line is a dict with:
                - text: The text to speak
                - speaker: Speaker name (if using speakers dict) OR inline config
                - mode: "clone", "design", or "custom" (if inline)
                - prompt: Voice prompt file (for clone mode, if inline)
                - description: Voice description (for design mode, if inline)
                - speaker: Premium speaker name (for custom mode, if inline)
                - instruct: Style instruction (for custom mode, if inline)
            output: Output file path
            speakers: Optional dict mapping speaker names to their config
            pause_ms: Milliseconds of silence between lines (default: 500)
            preset: Preset name to use
            temperature/top_k/top_p/seed/repetition_penalty: Generation params
            speed/pitch/normalize/trim_silence: Audio processing options

        Returns:
            Path to the generated audio file
        """
        import numpy as np

        if not self.is_server_running():
            raise ConnectionError("TTS server must be running for dialogue generation")

        speakers = speakers or {}

        # Get generation parameters
        gen_config = self.config.get("generation", {})
        gen_params = {
            "temperature": temperature if temperature is not None else gen_config.get("temperature", 0.7),
            "top_k": top_k if top_k is not None else gen_config.get("top_k", 50),
            "top_p": top_p if top_p is not None else gen_config.get("top_p", 0.95),
            "repetition_penalty": repetition_penalty if repetition_penalty is not None else gen_config.get("repetition_penalty", 1.05),
        }

        if seed is not None:
            gen_params["seed"] = seed

        # Apply preset
        if preset:
            presets = self.config.get("presets", {})
            if preset in presets:
                gen_params.update(presets[preset])

        all_audio = []
        sample_rate = None

        for line in lines:
            text = line.get("text", "")
            if not text:
                continue

            # Resolve speaker config
            if "speaker" in line and line["speaker"] in speakers:
                speaker_config = speakers[line["speaker"]].copy()
            else:
                speaker_config = line.copy()

            mode = speaker_config.get("mode", "clone")
            prompt = speaker_config.get("prompt", get_default_clone_prompt(self.config))
            description = speaker_config.get("description", self.config.get("default_voice_description", ""))
            custom_speaker = speaker_config.get("speaker", "ryan")
            instruct = speaker_config.get("instruct", line.get("instruct", ""))

            # Generate this line
            payload = {
                "texts": [text],
                "mode": mode,
                "language": self.config.get("language", "English"),
                **gen_params,
            }

            if mode == "clone":
                payload["prompt_file"] = prompt
            elif mode == "design":
                payload["voice_description"] = description
            else:  # custom
                payload["speaker"] = custom_speaker
                payload["instruct"] = instruct

            resp = requests.post(f"{self.server_url}/generate", json=payload, timeout=300, headers=auth_headers())
            if resp.status_code != 200:
                try:
                    error_msg = resp.json().get("error", "Unknown error")
                except (ValueError, requests.exceptions.JSONDecodeError):
                    error_msg = f"Server returned HTTP {resp.status_code} (non-JSON response)"
                raise Exception(f"Server error: {error_msg}")

            result = resp.json()["results"][0]
            wav, sr = sf.read(result["file"])
            os.remove(result["file"])

            # Apply audio processing if needed
            needs_processing = trim_silence or normalize or (speed and speed != 1.0) or (pitch and pitch != 0)
            if needs_processing:
                from tts_engine import process_audio
                wav = process_audio(wav, sr, trim=trim_silence, normalize=normalize,
                                    speed=speed, pitch=pitch)

            if sample_rate is None:
                sample_rate = sr

            all_audio.append(wav)

        if not all_audio:
            raise ValueError("No audio generated from dialogue")

        # Combine with pauses
        silence_samples = int(sample_rate * pause_ms / 1000)
        combined = []

        for i, wav in enumerate(all_audio):
            combined.extend(wav)
            if i < len(all_audio) - 1:
                combined.extend(np.zeros(silence_samples))

        combined = np.array(combined)

        # Determine output path
        if output is None:
            output_dir = os.path.expanduser(self.config.get("output_directory", "~/Downloads"))
            output = os.path.join(output_dir, "dialogue_output.wav")
        else:
            output = os.path.expanduser(output)
            if not output.endswith('.wav'):
                output += '.wav'

        sf.write(output, combined, sample_rate)
        return output


# Convenience function for simple usage
def generate(text, **kwargs):
    """Generate speech from text using default settings.

    This is a convenience function that creates a TTSClient and calls generate().
    For repeated use, create a TTSClient instance instead.

    Args:
        text: Text to synthesize
        **kwargs: Additional arguments passed to TTSClient.generate()

    Returns:
        Path to the generated audio file
    """
    client = TTSClient()
    return client.generate(text, **kwargs)


if __name__ == "__main__":
    # Example usage
    client = TTSClient()

    print("TTS Client Library")
    print(f"Server running: {client.is_server_running()}")
    print(f"Available prompts: {client.list_prompts()}")
    print(f"Available presets: {list(client.list_presets().keys())}")
    print(f"Available aliases: {list(client.list_aliases().keys())}")
