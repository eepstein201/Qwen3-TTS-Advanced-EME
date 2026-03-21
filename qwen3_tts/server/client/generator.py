"""
GeneratorMixin — audio generation methods for TTSClient.

Handles generate(), _generate_via_server(), generate_streaming(),
generate_dialogue(), and cancel_generation().

This module NEVER imports torch or qwen3_tts.core.engine at module scope.
All heavy imports (struct, numpy, soundfile, io, base64) remain lazy inside
method bodies.
"""

import os

import requests

from qwen3_tts.server.client._base import (
    _require_server,
    _resolve_voice_alias,
    _build_gen_params,
    _normalize_speaker_name,
    _extract_error_message,
    MAX_BUFFER_SIZE,
)
from qwen3_tts.core.config import (
    get_default_clone_prompt,
    auth_headers,
    GenerationError,
)


class GeneratorMixin:
    """Mixin providing audio generation capabilities."""

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
        prosody=None,
        temperature=None,
        top_k=None,
        top_p=None,
        seed=None,
        repetition_penalty=None,
        speed=None,
        pitch=None,
        normalize=False,
        trim_silence=False,
        x_vector_only_mode=False,
        max_new_tokens=None,
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
            prosody: Prosody preset name (resolves to instruct text for custom/design)
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
        # Resolve voice alias using helper
        if voice:
            alias = self.resolve_alias(voice)
            if alias:
                resolved = _resolve_voice_alias(
                    alias, prompt, mode, description, speaker, instruct, preset
                )
                prompt = resolved["prompt"]
                mode = resolved["mode"]
                description = resolved["description"]
                speaker = resolved["speaker"]
                instruct = resolved["instruct"]
                preset = resolved["preset"]
            else:
                raise ValueError(f"Unknown voice alias: {voice}")

        # Resolve prosody preset into instruct text
        if prosody and not instruct:
            from qwen3_tts.core.config import get_prosody_presets
            prosody_presets = get_prosody_presets(self.config)
            if prosody in prosody_presets:
                instruct = prosody_presets[prosody]
            else:
                raise ValueError(
                    f"Unknown prosody preset: {prosody}. "
                    f"Available: {', '.join(sorted(prosody_presets.keys()))}"
                )

        # Build generation parameters using helper
        gen_params = _build_gen_params(
            self.config, temperature, top_k, top_p, repetition_penalty, max_new_tokens, seed
        )

        # Apply preset
        if preset:
            presets = self.config.get("presets", {})
            if preset in presets:
                gen_params.update(presets[preset])

        # Default prompt/description/speaker
        if mode == "clone":
            prompt = prompt or get_default_clone_prompt(self.config)
        elif mode == "custom":
            speaker = speaker or self.config.get("default_speaker", "ryan")
            speaker = _normalize_speaker_name(speaker)
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
        wav, sr = self._generate_via_server(text, mode, prompt, description, speaker, instruct, gen_params,
                                            x_vector_only_mode=x_vector_only_mode)

        # Apply audio processing (lazy import — only if needed)
        needs_processing = trim_silence or normalize or (speed and speed != 1.0) or (pitch and pitch != 0)
        if needs_processing:
            from qwen3_tts.core.engine import process_audio
            wav = process_audio(wav, sr, trim=trim_silence, normalize=normalize,
                                speed=speed, pitch=pitch)

        # Save output
        import soundfile as sf  # lazy — not needed at module import time
        sf.write(output, wav, sr)
        return output

    def _generate_via_server(self, text, mode, prompt, description, speaker, instruct, gen_params,
                             x_vector_only_mode=False):
        """Generate audio via the TTS server."""
        payload = self._add_mode_params(
            {
                "texts": [text],
                "mode": mode,
                "language": self.config.get("language", "English"),
                **gen_params,
            },
            mode, prompt=prompt, description=description,
            speaker=speaker, instruct=instruct, x_vector_only_mode=x_vector_only_mode,
        )

        resp = self._session.post(f"{self.server_url}/generate", json=payload, timeout=600, headers=auth_headers())
        if resp.status_code != 200:
            raise GenerationError(_extract_error_message(resp))

        import io
        import base64
        import soundfile as sf
        result = resp.json()["results"][0]
        audio_bytes = base64.b64decode(result["audio_base64"])
        wav, sr = sf.read(io.BytesIO(audio_bytes))
        return wav, sr

    def generate_streaming(
        self,
        text,
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
        x_vector_only_mode=False,
        max_new_tokens=None,
    ):
        """Generate speech with streaming, yielding audio chunks as they're produced.

        This method connects to /generate-stream and yields (wav_chunk, sample_rate)
        tuples as audio is generated. Ideal for real-time playback.

        Args:
            text: Text to synthesize
            mode: "clone", "design", or "custom"
            prompt: Voice prompt filename (for clone mode)
            description: Voice description (for design mode)
            speaker: Speaker name (for custom mode)
            instruct: Instruction for speech style (for custom mode)
            voice: Voice alias name (overrides prompt/preset)
            preset: Preset name to use
            temperature/top_k/top_p/seed/repetition_penalty: Generation params

        Yields:
            (wav_chunk, sample_rate) tuples where wav_chunk is a numpy float32 array
        """
        import struct
        import numpy as np

        # Resolve voice alias using helper
        if voice:
            alias = self.resolve_alias(voice)
            if alias:
                resolved = _resolve_voice_alias(
                    alias, prompt, mode, description, speaker, instruct, preset
                )
                prompt = resolved["prompt"]
                mode = resolved["mode"]
                description = resolved["description"]
                speaker = resolved["speaker"]
                instruct = resolved["instruct"]
                preset = resolved["preset"]
            else:
                raise ValueError(f"Unknown voice alias: {voice}")

        # Build generation parameters using helper
        gen_params = _build_gen_params(
            self.config, temperature, top_k, top_p, repetition_penalty, max_new_tokens, seed
        )

        # Apply preset
        if preset:
            presets = self.config.get("presets", {})
            if preset in presets:
                gen_params.update(presets[preset])

        # Default prompt/description/speaker
        if mode == "clone":
            prompt = prompt or get_default_clone_prompt(self.config)
        elif mode == "custom":
            speaker = speaker or self.config.get("default_speaker", "ryan")
            speaker = _normalize_speaker_name(speaker)
            instruct = instruct or ""
        else:
            description = description or self.config.get("default_voice_description", "")

        # Build payload
        payload = self._add_mode_params(
            {
                "text": text,
                "mode": mode,
                "language": self.config.get("language", "English"),
                **gen_params,
            },
            mode, prompt=prompt, description=description,
            speaker=speaker, instruct=instruct, x_vector_only_mode=x_vector_only_mode,
        )

        # Stream from server
        with self._session.post(
            f"{self.server_url}/generate-stream",
            json=payload,
            headers=auth_headers(),
            stream=True,
            timeout=600,
        ) as resp:
            if resp.status_code != 200:
                error_msg = _extract_error_message(resp)
                # Include model_type prefix if present in JSON response
                try:
                    error_data = resp.json()
                    if "model_type" in error_data:
                        error_msg = f"{error_data['model_type']} model: {error_msg}"
                except (ValueError, requests.exceptions.JSONDecodeError):
                    pass
                raise GenerationError(error_msg)

            buffer = b""
            header_size = 8  # 4 bytes sample_rate + 4 bytes audio_length

            for chunk in resp.iter_content(chunk_size=65536):
                buffer += chunk

                # Protection against unbounded buffer growth from malformed data
                if len(buffer) > MAX_BUFFER_SIZE:
                    raise RuntimeError(
                        f"Streaming buffer exceeded maximum size ({MAX_BUFFER_SIZE} bytes). "
                        "Possible malformed response from server."
                    )

                # Parse complete chunks from buffer
                while len(buffer) >= header_size:
                    sr, audio_len = struct.unpack("<II", buffer[:header_size])
                    total_chunk_size = header_size + audio_len

                    # Protection against malformed headers claiming huge chunks
                    if total_chunk_size > MAX_BUFFER_SIZE:
                        raise RuntimeError(
                            f"Streaming chunk size ({total_chunk_size} bytes) exceeds maximum buffer size "
                            f"({MAX_BUFFER_SIZE} bytes). Possible malformed response from server."
                        )

                    if len(buffer) < total_chunk_size:
                        break  # Wait for more data

                    audio_bytes = buffer[header_size:total_chunk_size]
                    wav_chunk = np.frombuffer(audio_bytes, dtype="<f4")
                    buffer = buffer[total_chunk_size:]

                    yield wav_chunk, sr

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
        max_new_tokens=None,
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
        gen_params = _build_gen_params(
            self.config, temperature, top_k, top_p, repetition_penalty, max_new_tokens, seed
        )

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
            custom_speaker = _normalize_speaker_name(speaker_config.get("speaker", "ryan"))
            instruct = speaker_config.get("instruct", line.get("instruct", ""))

            # Generate this line
            payload = self._add_mode_params(
                {
                    "texts": [text],
                    "mode": mode,
                    "language": self.config.get("language", "English"),
                    **gen_params,
                },
                mode, prompt=prompt, description=description,
                speaker=custom_speaker, instruct=instruct,
            )

            resp = self._session.post(f"{self.server_url}/generate", json=payload, timeout=600, headers=auth_headers())
            if resp.status_code != 200:
                raise GenerationError(_extract_error_message(resp))

            import io
            import base64
            import soundfile as sf
            result = resp.json()["results"][0]
            audio_bytes = base64.b64decode(result["audio_base64"])
            wav, sr = sf.read(io.BytesIO(audio_bytes))

            # Apply audio processing if needed
            needs_processing = trim_silence or normalize or (speed and speed != 1.0) or (pitch and pitch != 0)
            if needs_processing:
                from qwen3_tts.core.engine import process_audio
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

        import soundfile as sf  # lazy — not needed at module import time
        sf.write(output, combined, sample_rate)
        return output

    @_require_server
    def cancel_generation(self):
        """Cancel the current streaming generation.

        Returns:
            Response dict with "status" key ("cancellation_requested" or "no_active_generation").
        """
        resp = self._session.post(
            f"{self.server_url}/cancel-generation",
            timeout=5,
            headers=auth_headers(),
        )
        return resp.json()
