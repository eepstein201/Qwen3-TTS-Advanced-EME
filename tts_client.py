#!/usr/bin/env python3
"""
TTS Client Library - Python API for Qwen3-TTS generation.

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

    # Generate with audio processing
    audio_path = client.generate(
        "Hello",
        speed=1.1,
        pitch=-2,
        normalize=True,
        trim_silence=True
    )

    # Check server status
    if client.is_server_running():
        stats = client.get_stats()
        print(f"Memory: {stats['mps_memory_allocated_mb']}MB")
"""

import json
import os
import shutil
import tempfile
import requests
import soundfile as sf
import numpy as np
import librosa


class TTSClient:
    """Client for Qwen3-TTS generation."""

    def __init__(self, config_path=None):
        """Initialize the TTS client.

        Args:
            config_path: Path to config.json. Defaults to ~/Qwen3-TTS_UserFiles/config.json
        """
        self.config_path = config_path or os.path.expanduser("~/Qwen3-TTS_UserFiles/config.json")
        self.voice_prompts_dir = os.path.expanduser("~/Qwen3-TTS_UserFiles/voice_prompts")
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
        server = self.config.get("server", {})
        host = server.get("host", "127.0.0.1")
        port = server.get("port", 5123)
        return f"http://{host}:{port}"

    def is_server_running(self):
        """Check if the TTS server is running."""
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=2)
            return resp.status_code == 200
        except:
            return False

    def get_stats(self):
        """Get server statistics."""
        if not self.is_server_running():
            raise ConnectionError("TTS server is not running")
        resp = requests.get(f"{self.server_url}/stats", timeout=5)
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
        use_server=True,
    ):
        """Generate speech from text.

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
            use_server: Use server if running (default: True)

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
            prompt = prompt or self.config.get("default_clone_prompt", "default_clone.pt")
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

        # Generate audio
        if use_server and self.is_server_running():
            wav, sr = self._generate_via_server(text, mode, prompt, description, speaker, instruct, gen_params)
        else:
            wav, sr = self._generate_local(text, mode, prompt, description, speaker, instruct, gen_params)

        # Apply audio processing
        wav = self._process_audio(wav, sr, speed, pitch, normalize, trim_silence)

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

        resp = requests.post(f"{self.server_url}/generate", json=payload, timeout=300)
        if resp.status_code != 200:
            raise Exception(f"Server error: {resp.json().get('error', 'Unknown error')}")

        result = resp.json()["results"][0]
        wav, sr = sf.read(result["file"])
        os.remove(result["file"])
        return wav, sr

    def _generate_local(self, text, mode, prompt, description, speaker, instruct, gen_params):
        """Generate audio locally."""
        import torch
        from qwen_tts import Qwen3TTSModel

        if mode == "clone":
            model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                attn_implementation="sdpa",
                device_map="mps",
                dtype=torch.float16,
            )
            prompt_path = os.path.join(self.voice_prompts_dir, prompt)
            voice_prompt = torch.load(prompt_path, weights_only=False)

            if gen_params.get("seed") is not None:
                torch.manual_seed(gen_params["seed"])

            with torch.inference_mode():
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=self.config.get("language", "English"),
                    voice_clone_prompt=voice_prompt,
                    temperature=gen_params.get("temperature", 0.7),
                    top_k=gen_params.get("top_k", 50),
                    top_p=gen_params.get("top_p", 0.95),
                    repetition_penalty=gen_params.get("repetition_penalty", 1.05),
                )
        elif mode == "custom":
            model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                attn_implementation="sdpa",
                device_map="mps",
                dtype=torch.float16,
            )

            if gen_params.get("seed") is not None:
                torch.manual_seed(gen_params["seed"])

            with torch.inference_mode():
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    speaker=speaker,
                    instruct=instruct or "",
                    language=self.config.get("language", "English"),
                    temperature=gen_params.get("temperature", 0.7),
                    top_k=gen_params.get("top_k", 50),
                    top_p=gen_params.get("top_p", 0.95),
                    repetition_penalty=gen_params.get("repetition_penalty", 1.05),
                )
        else:
            model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                attn_implementation="sdpa",
                device_map="mps",
                dtype=torch.float16,
            )

            if gen_params.get("seed") is not None:
                torch.manual_seed(gen_params["seed"])

            with torch.inference_mode():
                wavs, sr = model.generate_voice_design(
                    text=text,
                    instruct=description,
                    language=self.config.get("language", "English"),
                    temperature=gen_params.get("temperature", 0.7),
                    top_k=gen_params.get("top_k", 50),
                    top_p=gen_params.get("top_p", 0.95),
                    repetition_penalty=gen_params.get("repetition_penalty", 1.05),
                )

        return wavs[0], sr

    def _process_audio(self, audio, sample_rate, speed, pitch, normalize, trim_silence):
        """Apply audio processing."""
        # Trim silence
        if trim_silence:
            audio = self._trim_silence(audio, sample_rate)

        # Speed adjustment
        if speed and speed != 1.0:
            audio = librosa.effects.time_stretch(audio, rate=speed)

        # Pitch adjustment
        if pitch and pitch != 0:
            audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch)

        # Normalize
        if normalize:
            peak = np.max(np.abs(audio))
            if peak > 0:
                target_peak = 10 ** (-3 / 20)  # -3dB
                audio = audio * (target_peak / peak)

        return audio

    def _trim_silence(self, audio, sample_rate, threshold_db=-40, min_silence_ms=100):
        """Trim leading and trailing silence."""
        threshold = 10 ** (threshold_db / 20)
        min_samples = int(sample_rate * min_silence_ms / 1000)

        abs_audio = np.abs(audio)
        non_silent = abs_audio > threshold

        if not np.any(non_silent):
            return audio

        start_idx = np.argmax(non_silent)
        start_idx = max(0, start_idx - min_samples)

        end_idx = len(audio) - np.argmax(non_silent[::-1])
        end_idx = min(len(audio), end_idx + min_samples)

        return audio[start_idx:end_idx]


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
