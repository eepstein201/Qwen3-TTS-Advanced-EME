#!/usr/bin/env python3
"""Pure helper functions for Qwen3-TTS generation.

Contains text/audio utilities, SSML/SRT parsing, history, voice alias resolution,
and generation parameter handling. No interactive I/O or server communication.
"""

import json
import logging
import os
import re
import subprocess  # nosec B404
import sys

logger = logging.getLogger("tts.cli")

from qwen3_tts.core.config import (  # noqa: E402
    VOICE_PROMPTS_DIR,
    HISTORY_FILE,
    get_backend,
    safe_path_join,
    sanitize_log,
)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def voice_prompt_exists(prompt_file):
    """Check if a voice prompt exists for the current backend.

    - torch: checks for .pt file
    - mlx: checks for .wav + .txt file pair
    """
    backend = get_backend()
    if backend == "mlx":
        base = prompt_file
        if base.endswith(".pt"):
            base = base[:-3]
        wav = safe_path_join(VOICE_PROMPTS_DIR, f"{base}.wav")
        txt = safe_path_join(VOICE_PROMPTS_DIR, f"{base}.txt")
        return os.path.exists(wav) and os.path.exists(txt)
    else:
        pt_path = safe_path_join(VOICE_PROMPTS_DIR, prompt_file)
        return os.path.exists(pt_path)


def list_voice_prompts():
    """List available voice clone prompts (.pt for torch, .wav+.txt for MLX)."""
    try:
        files = os.listdir(VOICE_PROMPTS_DIR)
    except OSError:
        return []
    pt_prompts = {f for f in files if f.endswith('.pt')}
    # Include MLX prompts: .wav files that have a matching .txt
    txt_bases = {f[:-4] for f in files if f.endswith('.txt')}
    mlx_prompts = {f for f in files if f.endswith('.wav') and f[:-4] in txt_bases}
    return sorted(pt_prompts | mlx_prompts)


def get_text(text_or_file):
    """Get text from argument - either direct text or file path."""
    expanded = os.path.expanduser(text_or_file)
    if os.path.isfile(expanded):
        with open(expanded, "r") as f:
            return f.read().strip()
    # Path traversal guard: only look up bare filenames in ~/Downloads
    if ".." not in text_or_file and "/" not in text_or_file:
        downloads_path = os.path.expanduser(f"~/Downloads/{text_or_file}")
        if os.path.isfile(downloads_path):
            with open(downloads_path, "r") as f:
                return f.read().strip()
    return text_or_file


def get_clipboard_text():
    """Get text from system clipboard (platform-aware)."""
    from qwen3_tts.core.config import IS_MACOS, IS_LINUX, IN_COLAB
    if IN_COLAB:
        print("Error: Clipboard not available in Colab environment")
        sys.exit(1)
    if IS_MACOS:
        cmd = ["pbpaste"]
    elif IS_LINUX:
        cmd = ["xclip", "-selection", "clipboard", "-o"]
    else:
        print("Error: Clipboard not supported on this platform")
        sys.exit(1)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # nosec B603
        text = result.stdout.strip()
        if not text:
            print("Error: Clipboard is empty")
            sys.exit(1)
        return text
    except subprocess.CalledProcessError:
        print("Error: Failed to read clipboard")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: {cmd[0]} not found")
        sys.exit(1)


LAST_TEXT_FILE = os.path.expanduser("~/.voice_last_text")


def auto_increment_filename(path):
    """Auto-increment filename if it already exists.

    output.wav -> output_2.wav -> output_3.wav
    """
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    # Check if base already ends with _N
    match = re.match(r'^(.+)_(\d+)$', base)
    if match:
        base_stem = match.group(1)
        n = int(match.group(2))
    else:
        base_stem = base
        n = 1

    while True:
        n += 1
        candidate = f"{base_stem}_{n}{ext}"
        if not os.path.exists(candidate):
            return candidate


def play_audio(file_path):
    """Play audio file using system player (platform-aware)."""
    from qwen3_tts.core.config import IS_MACOS, IS_LINUX, IN_COLAB
    if IN_COLAB:
        logger.info("Audio generated (playback skipped in headless mode)")
        return
    # Validate file exists before playback
    if not os.path.isfile(file_path):
        logger.warning("Audio file not found for playback")
        return
    if IS_MACOS:
        cmd = ["afplay", file_path]
    elif IS_LINUX:
        cmd = ["ffplay", "-nodisp", "-autoexit", file_path]
    else:
        logger.warning("Audio playback not supported on this platform")
        return
    try:
        subprocess.run(cmd, check=True)  # nosec B603  # CodeQL: cmd is a validated hardcoded list [py/command-line-injection]
    except subprocess.CalledProcessError:
        logger.warning("Failed to play audio")
    except FileNotFoundError:
        logger.warning("%s not found — audio playback unavailable", sanitize_log(cmd[0]))


def open_file(path):
    """Open a file with the system default handler (platform-aware)."""
    from qwen3_tts.core.config import IS_MACOS, IS_LINUX, IN_COLAB
    if IN_COLAB:
        print(f"File saved: {path}")
        return
    if IS_MACOS:
        subprocess.run(["open", path])  # nosec B603 B607
    elif IS_LINUX:
        try:
            subprocess.run(["xdg-open", path])  # nosec B603 B607
        except FileNotFoundError:
            logger.warning("xdg-open not found — cannot open file automatically")


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def log_generation(text, mode, voice_param, output_path, gen_params, duration_sec=None):
    """Log generation to history file."""
    import datetime
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "text": text[:200] + "..." if len(text) > 200 else text,
        "text_length": len(text),
        "mode": mode,
        "voice": voice_param,
        "output": output_path,
        "params": gen_params,
    }
    if duration_sec is not None:
        entry["duration_sec"] = round(duration_sec, 2)

    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def show_history(count=10):
    """Show recent generation history."""
    if not os.path.exists(HISTORY_FILE):
        print("No generation history found.")
        return

    with open(HISTORY_FILE, "r") as f:
        lines = f.readlines()

    if not lines:
        print("No generation history found.")
        return

    recent = lines[-count:]
    print(f"\nRecent generations (last {min(count, len(recent))}):\n")

    for line in reversed(recent):
        entry = json.loads(line)
        ts = entry.get("timestamp", "")[:19].replace("T", " ")
        text_preview = entry.get("text", "")[:50]
        if len(entry.get("text", "")) > 50:
            text_preview += "..."
        mode = entry.get("mode", "?")
        voice = entry.get("voice", "?")
        output = os.path.basename(entry.get("output", "?"))

        print(f"  {ts}  [{mode}] {voice}")
        print(f"    \"{text_preview}\"")
        print(f"    -> {output}")
        print()


# ---------------------------------------------------------------------------
# Voice alias
# ---------------------------------------------------------------------------

def get_voice_alias(alias_name, config):
    """Resolve a voice alias from config."""
    aliases = config.get("aliases", {})
    if alias_name in aliases:
        return aliases[alias_name]
    return None


# ---------------------------------------------------------------------------
# SSML parsing
# ---------------------------------------------------------------------------

def parse_ssml(text):
    """Parse SSML markup and return processed text with metadata.

    Supported tags:
    - <break time="500ms"/> or <break time="1s"/> - Insert pause
    - <emphasis>text</emphasis> - Emphasis
    - <sub alias="replacement">original</sub> - Substitution
    - <say-as interpret-as="characters">ABC</say-as> - Spell out
    - <prosody rate="slow|fast" pitch="low|high">text</prosody> - Prosody hints
    """
    metadata = {"has_ssml": False, "breaks": [], "prosody": None}

    # Guard against ReDoS on extremely long inputs
    if len(text) > 50000:
        return text, metadata

    if not re.search(r'<[a-z][a-z0-9-]*(?:\s[^>]{0,500})?>', text, re.IGNORECASE):
        return text, metadata

    metadata["has_ssml"] = True
    processed = text

    # <break> tags
    def replace_break(match):
        time_str = match.group(1)
        if 'ms' in time_str:
            ms = int(time_str.replace('ms', ''))
            if ms >= 1000:
                return '... '
            elif ms >= 500:
                return '.. '
            else:
                return '. '
        elif 's' in time_str:
            seconds = float(time_str.replace('s', ''))
            if seconds >= 2:
                return '.... '
            elif seconds >= 1:
                return '... '
            else:
                return '.. '
        return '. '

    processed = re.sub(r'<break\s+time=["\']([^"\']+)["\']\s*/>', replace_break, processed, flags=re.IGNORECASE)

    # <sub> tags
    processed = re.sub(r'<sub\s+alias=["\']([^"\']+)["\']>([^<]*)</sub>', r'\1', processed, flags=re.IGNORECASE)

    # <say-as interpret-as="characters">
    def spell_out(match):
        chars = match.group(1)
        return ' '.join(chars.upper())

    processed = re.sub(r'<say-as\s+interpret-as=["\']characters["\']>([^<]*)</say-as>', spell_out, processed, flags=re.IGNORECASE)

    # <emphasis>
    processed = re.sub(r'<emphasis(?:\s+level=["\'][^"\']+["\'])?>([^<]*)</emphasis>', r'\1', processed, flags=re.IGNORECASE)

    # <prosody>
    prosody_match = re.search(r'<prosody\s+([^>]{1,200})>([^<]{0,5000})</prosody>', processed, flags=re.IGNORECASE)
    if prosody_match:
        attrs = prosody_match.group(1)
        rate_match = re.search(r'rate=["\']([^"\']+)["\']', attrs)
        pitch_match = re.search(r'pitch=["\']([^"\']+)["\']', attrs)

        if rate_match:
            rate = rate_match.group(1).lower()
            if rate in ('slow', 'x-slow'):
                metadata["prosody"] = metadata.get("prosody") or {}
                metadata["prosody"]["speed"] = 0.8
            elif rate in ('fast', 'x-fast'):
                metadata["prosody"] = metadata.get("prosody") or {}
                metadata["prosody"]["speed"] = 1.2

        if pitch_match:
            pitch = pitch_match.group(1).lower()
            if pitch in ('low', 'x-low'):
                metadata["prosody"] = metadata.get("prosody") or {}
                metadata["prosody"]["pitch"] = -2
            elif pitch in ('high', 'x-high'):
                metadata["prosody"] = metadata.get("prosody") or {}
                metadata["prosody"]["pitch"] = 2

    processed = re.sub(r'<prosody\s+[^>]{1,200}>([^<]{0,5000})</prosody>', r'\1', processed, flags=re.IGNORECASE)

    # Remove remaining XML tags
    processed = re.sub(r'<[^>]{1,500}>', '', processed)
    processed = re.sub(r'\s+', ' ', processed).strip()

    return processed, metadata


def process_ssml_text(text, args):
    """Process SSML in text and update args with prosody hints."""
    processed, metadata = parse_ssml(text)

    if metadata.get("prosody"):
        if metadata["prosody"].get("speed") and not args.speed:
            args.speed = metadata["prosody"]["speed"]
        if metadata["prosody"].get("pitch") and not args.pitch:
            args.pitch = metadata["prosody"]["pitch"]

    return processed


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

def parse_srt(srt_path):
    """Parse an SRT subtitle file."""
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = r"(\d+)\s*\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n(.*?)(?=\n\n|\Z)"
    matches = re.findall(pattern, content, re.DOTALL)

    entries = []
    for match in matches:
        index = int(match[0])
        start_time = srt_time_to_ms(match[1])
        end_time = srt_time_to_ms(match[2])
        text = match[3].strip().replace("\n", " ")
        entries.append((index, start_time, end_time, text))

    return entries


def srt_time_to_ms(time_str):
    """Convert SRT timestamp to milliseconds."""
    parts = time_str.replace(",", ":").split(":")
    hours, minutes, seconds, ms = map(int, parts)
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + ms


# ---------------------------------------------------------------------------
# Audio processing helper (delegates to engine for heavy ops)
# ---------------------------------------------------------------------------

def process_audio_args(audio, sample_rate, args):
    """Apply audio processing based on argparse args.

    Lazily imports qwen3_tts.core.engine only when processing is actually needed.
    """
    needs = args.trim_silence or args.normalize or (args.speed and args.speed != 1.0) or (args.pitch and args.pitch != 0)
    if not needs:
        return audio

    from qwen3_tts.core.engine import process_audio
    return process_audio(
        audio, sample_rate,
        trim=args.trim_silence,
        normalize=args.normalize,
        speed=args.speed if args.speed and args.speed != 1.0 else None,
        pitch=args.pitch if args.pitch and args.pitch != 0 else None,
    )


# ---------------------------------------------------------------------------
# Generation payload helpers
# ---------------------------------------------------------------------------

def _build_generation_payload(mode, config, gen_params, prompt_file=None,
                              voice_description=None, speaker=None,
                              instruct=None, x_vector_only_mode=False,
                              max_chunk_chars=None):
    """Build request payload for /generate or /generate-stream."""
    payload = {
        "mode": mode,
        "language": config.get("language", "English"),
        **gen_params,
    }
    if max_chunk_chars is not None:
        payload["max_chunk_chars"] = max_chunk_chars
    if mode == "clone":
        payload["prompt_file"] = prompt_file
        if x_vector_only_mode:
            payload["x_vector_only_mode"] = True
    elif mode == "design":
        payload["voice_description"] = voice_description
    elif mode == "custom":
        payload["speaker"] = speaker
        payload["instruct"] = instruct or ""
    return payload


def _decode_base64_result(result):
    """Decode base64 audio from server response to numpy array + sample rate."""
    import base64
    import io
    import soundfile as sf
    audio_bytes = base64.b64decode(result["audio_base64"])
    wav, sr = sf.read(io.BytesIO(audio_bytes))
    return wav, sr


def _save_base64_result(result, output_path):
    """Save base64 audio from server response directly to a WAV file."""
    import base64
    audio_bytes = base64.b64decode(result["audio_base64"])
    with open(output_path, "wb") as f:
        f.write(audio_bytes)


# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------

def get_generation_params(args, config):
    """Get generation parameters from args, preset, or config defaults."""
    gen_config = config.get("generation", {})
    presets = config.get("presets", {})

    params = {
        "temperature": gen_config.get("temperature", 0.7),
        "top_k": gen_config.get("top_k", 50),
        "top_p": gen_config.get("top_p", 0.95),
        "repetition_penalty": gen_config.get("repetition_penalty", 1.05),
        "seed": gen_config.get("seed"),
    }

    if args.preset and args.preset in presets:
        params.update(presets[args.preset])

    if args.temperature is not None:
        params["temperature"] = args.temperature
    if args.top_k is not None:
        params["top_k"] = args.top_k
    if args.top_p is not None:
        params["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        params["repetition_penalty"] = args.repetition_penalty
    if args.seed is not None:
        params["seed"] = args.seed

    return params
