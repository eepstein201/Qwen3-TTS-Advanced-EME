#!/usr/bin/env python3
"""Qwen3-TTS generation script with voice cloning and voice design support.

Architecture:
  --_server-mode / server available  ->  HTTP calls (no torch import)
  --local / no server                ->  lazy import qwen3_tts.core.engine (PyTorch)
"""

import argparse
import json
import logging
import os
import re
import subprocess  # nosec B404
import sys
import threading
import time

logger = logging.getLogger("tts.cli")

from qwen3_tts.core.config import (  # noqa: E402
    CONFIG_PATH,
    VOICE_PROMPTS_DIR,
    HISTORY_FILE,
    MODEL_INFO,
    CUSTOM_VOICE_SPEAKERS,
    VALID_BACKENDS,
    load_config,
    save_config,
    get_server_url,
    is_server_running,
    auth_headers,
    get_backend,
    get_torch_dtype_name,
    get_mlx_quantization,
    get_mlx_model_name,
    get_default_clone_prompt,
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
        wav = os.path.join(VOICE_PROMPTS_DIR, f"{base}.wav")
        txt = os.path.join(VOICE_PROMPTS_DIR, f"{base}.txt")
        return os.path.exists(wav) and os.path.exists(txt)
    else:
        pt_path = os.path.join(VOICE_PROMPTS_DIR, prompt_file)
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
    if IS_MACOS:
        cmd = ["afplay", file_path]
    elif IS_LINUX:
        cmd = ["ffplay", "-nodisp", "-autoexit", file_path]
    else:
        logger.warning("Audio playback not supported on this platform")
        return
    try:
        subprocess.run(cmd, check=True)  # nosec B603
    except subprocess.CalledProcessError:
        logger.warning("Failed to play audio")
    except FileNotFoundError:
        logger.warning("%s not found — audio playback unavailable", cmd[0])


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
# Voice prompt management
# ---------------------------------------------------------------------------

def delete_voice_prompt(prompt_name):
    """Delete a voice prompt file (supports .pt, .wav+.txt formats)."""
    if ".." in prompt_name or "/" in prompt_name or "\\" in prompt_name:
        print(f"Error: Invalid prompt name: {prompt_name!r}")
        return False

    # Strip known extensions to get the base name
    base = prompt_name
    for ext in ('.pt', '.wav', '.txt'):
        if base.endswith(ext):
            base = base[:-len(ext)]
            break

    # Find all format files that exist
    pt_path = os.path.join(VOICE_PROMPTS_DIR, f"{base}.pt")
    wav_path = os.path.join(VOICE_PROMPTS_DIR, f"{base}.wav")
    txt_path = os.path.join(VOICE_PROMPTS_DIR, f"{base}.txt")
    to_delete = [p for p in (pt_path, wav_path, txt_path) if os.path.exists(p)]

    if not to_delete:
        print(f"Error: Voice prompt not found: {prompt_name}")
        return False

    filenames = ", ".join(os.path.basename(p) for p in to_delete)
    confirm = input(f"Delete '{base}' ({filenames})? This cannot be undone. [y/N]: ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return False

    for p in to_delete:
        os.remove(p)
    print(f"Deleted: {filenames}")
    return True


def rename_voice_prompt(old_name, new_name):
    """Rename a voice prompt file (supports .pt, .wav+.txt formats)."""
    for _name in (old_name, new_name):
        if ".." in _name or "/" in _name or "\\" in _name:
            print(f"Error: Invalid prompt name: {_name!r}")
            return False

    # Strip known extensions to get base names
    old_base = old_name
    new_base = new_name
    for ext in ('.pt', '.wav', '.txt'):
        if old_base.endswith(ext):
            old_base = old_base[:-len(ext)]
        if new_base.endswith(ext):
            new_base = new_base[:-len(ext)]

    # Find all format files that exist for old name
    rename_pairs = []
    for ext in ('.pt', '.wav', '.txt'):
        old_path = os.path.join(VOICE_PROMPTS_DIR, f"{old_base}{ext}")
        new_path = os.path.join(VOICE_PROMPTS_DIR, f"{new_base}{ext}")
        if os.path.exists(old_path):
            if os.path.exists(new_path):
                print(f"Error: Voice prompt already exists: {new_base}{ext}")
                return False
            rename_pairs.append((old_path, new_path))

    if not rename_pairs:
        print(f"Error: Voice prompt not found: {old_name}")
        return False

    completed = []
    try:
        for old_path, new_path in rename_pairs:
            os.rename(old_path, new_path)
            completed.append((old_path, new_path))
    except OSError as e:
        # Rollback completed renames on failure
        for done_old, done_new in completed:
            try:
                os.rename(done_new, done_old)
            except OSError:
                pass
        print(f"Error: Rename failed: {e}")
        return False

    print(f"Renamed: {old_base} -> {new_base}")
    return True


def preview_voice_prompt(prompt_name, config):
    """Preview a voice prompt by generating a short sample."""
    import requests  # lazy
    if not prompt_name.endswith('.pt'):
        prompt_name += '.pt'

    if not voice_prompt_exists(prompt_name):
        print(f"Error: Voice prompt not found: {prompt_name}")
        return False

    if is_server_running(config):
        url = get_server_url(config)
        payload = {
            "texts": ["This is a preview of the voice prompt."],
            "mode": "clone",
            "prompt_file": prompt_name,
            "language": config.get("language", "English"),
            "temperature": 0.7,
        }
        print(f"Generating preview for '{prompt_name}'...")
        resp = requests.post(f"{url}/generate", json=payload, timeout=60, headers=auth_headers())
        if resp.status_code == 200:
            result = resp.json()["results"][0]
            print("Playing preview...")
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            _save_base64_result(result, tmp.name)
            play_audio(tmp.name)
            os.remove(tmp.name)
            return True
        else:
            try:
                error_msg = resp.json().get('error', 'Unknown error')
            except (ValueError, requests.exceptions.JSONDecodeError):
                error_msg = f"Server returned HTTP {resp.status_code}"
            print(f"Error: {error_msg}")
            return False
    else:
        print("Error: TTS server must be running for preview. Start with 'tts server start'.")
        return False


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

    if not re.search(r'<[a-z]+[^>]*>', text, re.IGNORECASE):
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
    prosody_match = re.search(r'<prosody\s+([^>]+)>([^<]*)</prosody>', processed, flags=re.IGNORECASE)
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

    processed = re.sub(r'<prosody\s+[^>]+>([^<]*)</prosody>', r'\1', processed, flags=re.IGNORECASE)

    # Remove remaining XML tags
    processed = re.sub(r'<[^>]+>', '', processed)
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
# Server helpers
# ---------------------------------------------------------------------------

def ensure_server_running(config):
    """Ensure the TTS server is running, starting it if necessary."""
    if is_server_running(config):
        return True

    print("TTS Server is not running.")
    print("Starting server (this may take 30-180 seconds on first run to download models)...")

    import shutil
    tts_cmd = shutil.which("tts")
    if tts_cmd:
        result = subprocess.run([tts_cmd, "server", "start"], capture_output=False)  # nosec B603
        return result.returncode == 0

    # Fallback: start server directly as a module
    from qwen3_tts.core.config import LOG_FILE
    with open(LOG_FILE, "w") as log:
        subprocess.Popen(  # nosec B603
            [sys.executable, "-m", "qwen3_tts.server.app"],
            stdout=log, stderr=log,
            start_new_session=True,
        )

    print("Waiting for server to be ready...")
    for i in range(300):
        if is_server_running(config):
            print("TTS Server is ready!")
            return True
        time.sleep(1)
        if (i + 1) % 10 == 0:
            print(f"  Still loading models... ({i + 1} seconds)")

    print("Error: Server failed to start. Check log:", LOG_FILE)
    return False


def build_ui_and_launch(config):
    """Build and launch the Gradio UI in-process."""
    from qwen3_tts.interface.ui import build_ui, _find_available_port
    from qwen3_tts.core.config import IN_COLAB
    preferred = config.get("ui", {}).get("port", 7860)
    port = _find_available_port(preferred)
    if port is None:
        print(f"Error: No available port found near {preferred}.")
        return
    share = bool(os.environ.get("TTS_UI_SHARE")) or IN_COLAB
    inbrowser = not bool(os.environ.get("TTS_UI_NO_BROWSER")) and not IN_COLAB
    demo = build_ui()
    demo.launch(server_port=port, share=share, inbrowser=inbrowser)


def launch_gradio_ui(config):
    """Launch the Gradio web interface, starting server if needed."""
    if not ensure_server_running(config):
        print("\nCannot launch web interface without the server.")
        print("Please check the server logs and try again.")
        return

    print("\nLaunching Gradio web interface...")
    build_ui_and_launch(config)


def load_model_on_server(config, model_type):
    """Request the server to load a model on demand."""
    import requests  # lazy
    url = get_server_url(config)
    print(f"Loading {model_type} model on server (this may take 30-60 seconds)...")
    resp = requests.post(f"{url}/load-model", json={"model_type": model_type}, timeout=120, headers=auth_headers())
    if resp.status_code == 200:
        result = resp.json()
        if result.get("status") == "loaded":
            print(f"Model '{model_type}' loaded successfully!")
            return True
        elif result.get("status") == "already_loaded":
            print(f"Model '{model_type}' was already loaded.")
            return True
    print(f"Failed to load model: {resp.json().get('error', 'Unknown error')}")
    return False


class _ProgressPoller:
    """Background thread that polls /generation-status and displays progress.

    Uses Rich for pretty progress bars if available, falls back to print-based progress.
    """

    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    # Try to import Rich
    try:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
        from rich.console import Console
        HAS_RICH = True
    except ImportError:
        HAS_RICH = False

    def __init__(self, server_url, batch_total=1):
        self.server_url = server_url
        self.batch_total = batch_total
        self._stop = threading.Event()
        self._thread = None
        self._rich_progress = None
        self._rich_task_id = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        # Clean up Rich display if used
        if self.HAS_RICH and self._rich_progress:
            self._rich_progress.stop()
        else:
            # Clear the progress line (fallback)
            sys.stderr.write("\r" + " " * 80 + "\r")
            sys.stderr.flush()

    def _run(self):
        # Use Rich if available
        if self.HAS_RICH:
            self._run_rich()
        else:
            self._run_fallback()

    def _run_rich(self):
        """Run with Rich progress bar."""
        import requests  # lazy
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
        from rich.console import Console

        console = Console(stderr=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),  # noqa: F821
            console=console,
            transient=True,
        ) as progress:
            task_id = progress.add_task(
                "Generating audio...",
                total=100 if self.batch_total > 1 else None,
            )

            while not self._stop.is_set():
                try:
                    resp = requests.get(f"{self.server_url}/generation-status", timeout=2)
                    if resp.status_code == 200:
                        state = resp.json()
                        if state.get("active"):
                            elapsed = state.get("elapsed_sec", 0)
                            eta = state.get("eta_sec")

                            # Chunk progress suffix
                            chunk_total = state.get("chunk_total", 0)
                            chunk_suffix = ""
                            if chunk_total > 1:
                                chunk_idx = state.get("chunk_index", 0) + 1
                                chunk_suffix = f" [chunk {chunk_idx}/{chunk_total}]"
                                progress.update(task_id, description=f"Generating...{chunk_suffix}")

                            if self.batch_total > 1:
                                idx = state.get("batch_index", 0) + 1
                                progress.update(task_id, description=f"[{idx}/{self.batch_total}] Generating...{chunk_suffix}")
                                if eta is not None:
                                    total_est = elapsed + eta
                                    pct = min(95, int(elapsed / total_est * 100)) if total_est > 0 else 0
                                    progress.update(task_id, completed=pct)
                except Exception as e:
                    logger.debug("Progress poller (_run_rich) error: %s", e)

                self._stop.wait(1.0)

    def _run_fallback(self):
        """Run with print-based progress (original implementation)."""
        import requests  # lazy
        tick = 0
        while not self._stop.is_set():
            try:
                resp = requests.get(f"{self.server_url}/generation-status", timeout=2)
                if resp.status_code == 200:
                    state = resp.json()
                    if state.get("active"):
                        elapsed = state.get("elapsed_sec", 0)
                        eta = state.get("eta_sec")
                        spinner = self.SPINNER[tick % len(self.SPINNER)]

                        # Chunk progress suffix
                        chunk_total = state.get("chunk_total", 0)
                        chunk_suffix = ""
                        if chunk_total > 1:
                            chunk_idx = state.get("chunk_index", 0) + 1
                            chunk_suffix = f" [chunk {chunk_idx}/{chunk_total}]"

                        if self.batch_total > 1:
                            idx = state.get("batch_index", 0) + 1
                            if eta is not None:
                                total_est = elapsed + eta
                                pct = min(95, int(elapsed / total_est * 100)) if total_est > 0 else 0
                                bar_filled = pct // 5
                                bar = "=" * bar_filled + ">" + " " * (19 - bar_filled)
                                line = f"\r{spinner} [{idx}/{self.batch_total}] Generating... {elapsed:.0f}s / ~{elapsed + eta:.0f}s [{bar}] {pct}%{chunk_suffix}"
                            else:
                                line = f"\r{spinner} [{idx}/{self.batch_total}] Generating... {elapsed:.0f}s elapsed{chunk_suffix}"
                        else:
                            if eta is not None:
                                line = f"\r{spinner} Generating... {elapsed:.0f}s elapsed (ETA ~{eta:.0f}s){chunk_suffix}"
                            else:
                                line = f"\r{spinner} Generating... {elapsed:.0f}s elapsed{chunk_suffix}"

                        sys.stderr.write(line)
                        sys.stderr.flush()
            except Exception as e:  # nosec B110
                logger.debug("Progress poller (_run_fallback) error: %s", e)

            tick += 1
            self._stop.wait(1.0)


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


def generate_via_server(texts, mode, config, gen_params,
                        prompt_file=None, voice_description=None,
                        speaker=None, instruct=None, auto_load_model=True,
                        max_chunk_chars=None, x_vector_only_mode=False):
    """Generate audio via the TTS server."""
    import requests  # lazy
    url = get_server_url(config)

    payload = _build_generation_payload(
        mode, config, gen_params,
        prompt_file=prompt_file, voice_description=voice_description,
        speaker=speaker, instruct=instruct,
        x_vector_only_mode=x_vector_only_mode,
        max_chunk_chars=max_chunk_chars,
    )
    payload["texts"] = texts

    # Start progress polling
    progress = _ProgressPoller(url, batch_total=len(texts))
    progress.start()

    try:
        resp = requests.post(f"{url}/generate", json=payload, timeout=600, headers=auth_headers())
    finally:
        progress.stop()

    # Handle model not loaded
    if resp.status_code == 503:
        try:
            error_data = resp.json()
        except (ValueError, requests.exceptions.JSONDecodeError):
            raise Exception("Server returned HTTP 503 (non-JSON response)")

        if error_data.get("error") == "model_not_loaded":
            model_type = error_data.get("model_type")
            description = error_data.get("description")
            print(f"\nThe '{model_type}' model is not loaded.")
            print(f"  Purpose: {description}")
            print()

            if auto_load_model:
                choice = input(f"Would you like to load the '{model_type}' model now? [Y/n]: ").strip().lower()
                if choice != 'n':
                    if load_model_on_server(config, model_type):
                        progress = _ProgressPoller(url, batch_total=len(texts))
                        progress.start()
                        try:
                            resp = requests.post(f"{url}/generate", json=payload, timeout=600, headers=auth_headers())
                        finally:
                            progress.stop()
                    else:
                        raise Exception(f"Failed to load {model_type} model")
                else:
                    raise Exception(f"Model '{model_type}' not loaded. Enable in config.json or load with server.")

    if resp.status_code != 200:
        try:
            error_data = resp.json()
            error_msg = error_data.get('error', 'Unknown error')
            detail = error_data.get('detail', '')
            recovery = error_data.get('recovery', '')
        except (ValueError, requests.exceptions.JSONDecodeError):
            error_msg = f"Server returned HTTP {resp.status_code} (non-JSON response)"
            detail = ""
            recovery = ""

        msg = f"Server error: {error_msg}"
        if detail:
            msg += f" [{detail}]"
        if recovery == "restart":
            msg += "\n  Suggestion: Try restarting the server with 'tts server start'."
        elif recovery == "config":
            msg += f"\n  Suggestion: Check your configuration in {CONFIG_PATH}."
        elif recovery == "retry":
            msg += "\n  Suggestion: Try again; the issue may be transient."
        raise Exception(msg)

    return resp.json()["results"]


def generate_streaming(text, mode, config, gen_params, output_path,
                       prompt_file=None, voice_description=None,
                       speaker=None, instruct=None, x_vector_only_mode=False):
    """Generate and stream audio playback in real-time (MLX backend).

    Streams from server and plays audio chunks as they arrive.
    Also saves the complete audio to output_path.
    """
    import requests  # lazy
    import struct
    import tempfile
    import numpy as np
    import soundfile as sf

    url = get_server_url(config)

    payload = _build_generation_payload(
        mode, config, gen_params,
        prompt_file=prompt_file, voice_description=voice_description,
        speaker=speaker, instruct=instruct,
        x_vector_only_mode=x_vector_only_mode,
    )
    payload["text"] = text

    print("Streaming generation...")

    try:
        resp = requests.post(
            f"{url}/generate-stream",
            json=payload,
            headers=auth_headers(),
            stream=True,
            timeout=600,
        )

        if resp.status_code != 200:
            error_data = resp.json()
            raise Exception(f"Server error: {error_data.get('error', 'Unknown')}")

        # Collect all chunks for saving
        all_chunks = []
        sample_rate = None
        chunk_count = 0

        # Read streamed chunks with length-prefixed format:
        # Each chunk: [sample_rate:4][length:4][audio:length]
        buffer = b""
        header_size = 8  # 4 bytes sample_rate + 4 bytes length

        for data in resp.iter_content(chunk_size=4096):
            buffer += data

            # Process complete chunks in buffer
            while len(buffer) >= header_size:
                # Read header
                sr, audio_len = struct.unpack("<II", buffer[:header_size])

                # Check if we have the full audio chunk
                total_chunk_size = header_size + audio_len
                if len(buffer) < total_chunk_size:
                    break  # Need more data

                if sample_rate is None:
                    sample_rate = sr

                # Extract audio
                audio_bytes = buffer[header_size:total_chunk_size]
                chunk = np.frombuffer(audio_bytes, dtype="<f4")
                all_chunks.append(chunk)
                chunk_count += 1

                # Play chunk using platform-aware player
                temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(temp.name, chunk, sr)
                try:
                    play_audio(temp.name)
                except Exception:  # nosec B110
                    pass
                finally:
                    os.unlink(temp.name)

                # Remove processed chunk from buffer
                buffer = buffer[total_chunk_size:]

        print(f"Streaming complete: {chunk_count} chunks received")

        # Save combined audio
        if all_chunks and sample_rate:
            combined = np.concatenate(all_chunks)
            sf.write(output_path, combined, sample_rate)
            print(f"Saved: {output_path}")
            return output_path

        return None

    except requests.exceptions.RequestException as e:
        raise Exception(f"Streaming request failed: {e}")


# ---------------------------------------------------------------------------
# Local generation (lazy import of qwen3_tts.core.engine)
# ---------------------------------------------------------------------------

def generate_local(text, mode, gen_params, language="English",
                   prompt_file=None, voice_description=None,
                   speaker=None, instruct=None, max_chunk_chars=None):
    """Generate speech locally using qwen3_tts.core.engine (imports torch on first call)."""
    from qwen3_tts.core.engine import load_model, run_inference, load_voice_prompt

    print(f"Loading {mode} model locally...")
    model = load_model(mode)

    voice_prompt = None
    if mode == "clone":
        if not prompt_file:
            print("Error: Voice prompt required for clone mode")
            sys.exit(1)
        if not voice_prompt_exists(prompt_file):
            backend = get_backend()
            if backend == "mlx":
                base = prompt_file[:-3] if prompt_file.endswith(".pt") else prompt_file
                print(f"Error: MLX voice prompt not found for '{base}'.")
                print(f"  Need: voice_prompts/{base}.wav + voice_prompts/{base}.txt")
                print(f"  Create with: tts voice create <audio> -t <transcript> -n {base} --mlx-only")
            else:
                print(f"Error: Voice prompt not found: {os.path.join(VOICE_PROMPTS_DIR, prompt_file)}")
            sys.exit(1)
        print(f"Loading voice prompt: {prompt_file}")
        voice_prompt = load_voice_prompt(prompt_file)
    elif mode == "custom":
        if not speaker:
            speaker = "Ryan"
        print(f"Using speaker: {speaker}")
        if instruct:
            print(f"With instruction: {instruct}")
    elif mode == "design":
        print(f"Using voice description: {voice_description}")

    print("Generating audio...")
    wav, sr = run_inference(
        model=model,
        text=text,
        mode=mode,
        gen_params=gen_params,
        language=language,
        voice_prompt=voice_prompt,
        voice_description=voice_description,
        speaker=speaker,
        instruct=instruct,
        max_chunk_chars=max_chunk_chars,
    )
    return wav, sr


# ---------------------------------------------------------------------------
# REPL mode
# ---------------------------------------------------------------------------

def run_repl(config, use_server):
    """Run interactive REPL mode for rapid TTS iteration."""
    import soundfile as sf
    print("\n=== TTS REPL Mode ===")
    print("Commands:")
    print("  Type text to generate speech")
    print("  /voice NAME    - Switch voice alias")
    print("  /preset NAME   - Switch preset")
    print("  /prompt NAME   - Switch voice prompt")
    print("  /play on|off   - Toggle auto-play")
    print("  /speed FACTOR  - Set speed (e.g., 1.2)")
    print("  /pitch SEMI    - Set pitch shift")
    print("  /status        - Show current settings")
    print("  /quit or /q    - Exit REPL")
    print()

    state = {
        "mode": "clone",
        "prompt": get_default_clone_prompt(config),
        "preset": None,
        "auto_play": True,
        "speed": None,
        "pitch": None,
        "counter": 1,
    }

    gen_params = config.get("generation", {}).copy()

    while True:
        try:
            user_input = input("tts> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting REPL.")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None

            if cmd in ("/quit", "/q"):
                print("Exiting REPL.")
                break
            elif cmd == "/voice":
                if arg:
                    alias = get_voice_alias(arg, config)
                    if alias:
                        if "prompt" in alias:
                            state["prompt"] = alias["prompt"]
                        if "preset" in alias:
                            state["preset"] = alias["preset"]
                            preset_params = config.get("presets", {}).get(alias["preset"], {})
                            gen_params.update(preset_params)
                        print(f"Switched to voice alias: {arg}")
                    else:
                        print(f"Unknown alias: {arg}")
                else:
                    print("Usage: /voice NAME")
            elif cmd == "/preset":
                if arg:
                    presets = config.get("presets", {})
                    if arg in presets:
                        state["preset"] = arg
                        gen_params.update(presets[arg])
                        print(f"Switched to preset: {arg}")
                    else:
                        print(f"Unknown preset: {arg}")
                else:
                    print("Usage: /preset NAME")
            elif cmd == "/prompt":
                if arg:
                    prompt_name = arg if arg.endswith(".pt") else arg + ".pt"
                    prompt_path = os.path.join(VOICE_PROMPTS_DIR, prompt_name)
                    if os.path.exists(prompt_path):
                        state["prompt"] = prompt_name
                        print(f"Switched to prompt: {prompt_name}")
                    else:
                        print(f"Prompt not found: {prompt_name}")
                else:
                    print("Usage: /prompt NAME")
            elif cmd == "/play":
                if arg and arg.lower() in ("on", "off"):
                    state["auto_play"] = arg.lower() == "on"
                    print(f"Auto-play: {'on' if state['auto_play'] else 'off'}")
                else:
                    print("Usage: /play on|off")
            elif cmd == "/speed":
                if arg:
                    try:
                        state["speed"] = float(arg)
                        print(f"Speed: {state['speed']}")
                    except ValueError:
                        print("Invalid speed value")
                else:
                    state["speed"] = None
                    print("Speed: reset to default")
            elif cmd == "/pitch":
                if arg:
                    try:
                        state["pitch"] = float(arg)
                        print(f"Pitch: {state['pitch']} semitones")
                    except ValueError:
                        print("Invalid pitch value")
                else:
                    state["pitch"] = None
                    print("Pitch: reset to default")
            elif cmd == "/status":
                print(f"  Mode: {state['mode']}")
                print(f"  Prompt: {state['prompt']}")
                print(f"  Preset: {state['preset'] or 'default'}")
                print(f"  Auto-play: {'on' if state['auto_play'] else 'off'}")
                print(f"  Speed: {state['speed'] or '1.0'}")
                print(f"  Pitch: {state['pitch'] or '0'}")
                print(f"  Server: {'connected' if use_server else 'local'}")
            else:
                print(f"Unknown command: {cmd}")
            continue

        # Generate speech
        text = user_input
        output_dir = os.path.expanduser(config.get("output_directory", "~/Downloads"))
        output_path = os.path.join(output_dir, f"repl_{state['counter']}.wav")

        try:
            if use_server:
                results = generate_via_server(
                    [text], state["mode"], config, gen_params,
                    prompt_file=state["prompt"]
                )
                wav, sr = _decode_base64_result(results[0])
            else:
                wav, sr = generate_local(
                    text, state["mode"], gen_params,
                    config.get("language", "English"),
                    prompt_file=state["prompt"],
                )

            # Apply audio processing
            if state["speed"] and state["speed"] != 1.0:
                from qwen3_tts.core.engine import adjust_speed
                wav = adjust_speed(wav, sr, state["speed"])
            if state["pitch"] and state["pitch"] != 0:
                from qwen3_tts.core.engine import adjust_pitch
                wav = adjust_pitch(wav, sr, state["pitch"])

            sf.write(output_path, wav, sr)
            print(f"  -> {output_path}")

            if state["auto_play"]:
                play_audio(output_path)

            state["counter"] += 1

        except Exception as e:
            print(f"Error: {e}")


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------

def run_watch_mode(watch_dir, config, args, gen_params, use_server):
    """Watch a directory for new .txt files and generate TTS."""
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    watch_dir = os.path.expanduser(watch_dir)
    if not os.path.isdir(watch_dir):
        print(f"Error: Directory not found: {watch_dir}")
        return

    output_dir = os.path.expanduser(args.output or config.get("output_directory", "~/Downloads"))
    os.makedirs(output_dir, exist_ok=True)

    mode = args.mode or "clone"
    prompt_file = args.prompt or get_default_clone_prompt(config)
    voice_description = args.description or config.get("default_voice_description", "")

    processed_files = set()

    class TTSHandler(FileSystemEventHandler):
        def on_created(self, event):
            import soundfile as sf
            if event.is_directory or not event.src_path.endswith(".txt"):
                return
            if event.src_path in processed_files:
                return

            time.sleep(0.5)

            try:
                with open(event.src_path, "r") as f:
                    text = f.read().strip()

                if not text:
                    return

                processed_files.add(event.src_path)
                basename = os.path.splitext(os.path.basename(event.src_path))[0]
                output_path = os.path.join(output_dir, f"{basename}.wav")

                print(f"\nProcessing: {event.src_path}")

                if use_server:
                    results = generate_via_server(
                        [text], mode, config, gen_params,
                        prompt_file=prompt_file if mode == "clone" else None,
                        voice_description=voice_description if mode == "design" else None,
                    )
                    wav, sr = _decode_base64_result(results[0])
                else:
                    wav, sr = generate_local(
                        text, mode, gen_params,
                        config.get("language", "English"),
                        prompt_file=prompt_file,
                        voice_description=voice_description,
                    )

                wav = process_audio_args(wav, sr, args)
                sf.write(output_path, wav, sr)

                print(f"  -> {output_path}")

                if args.play:
                    play_audio(output_path)

            except Exception as e:
                print(f"Error processing {event.src_path}: {e}")

    print("\n=== Watch Mode ===")
    print(f"Watching: {watch_dir}")
    print(f"Output to: {output_dir}")
    print("Drop .txt files into the watch directory to generate TTS.")
    print("Press Ctrl+C to stop.\n")

    event_handler = TTSHandler()
    observer = Observer()
    observer.schedule(event_handler, watch_dir, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nWatch mode stopped.")

    observer.join()


# ---------------------------------------------------------------------------
# Dialogue processing
# ---------------------------------------------------------------------------

def process_dialogue(dialogue_path, config, args, gen_params, use_server):
    """Process a dialogue JSON file with multiple speakers.

    Delegates to qwen3_tts.interface.cli.dialogue to avoid duplication.
    """
    from qwen3_tts.interface.cli.dialogue import process_dialogue as _impl
    return _impl(dialogue_path, config, args, gen_params, use_server)


# ---------------------------------------------------------------------------
# SRT processing
# ---------------------------------------------------------------------------

def process_srt_file(srt_path, config, args, gen_params, use_server):
    """Process an SRT file and generate audio for each subtitle.

    Delegates to qwen3_tts.interface.cli.srt to avoid duplication.
    """
    from qwen3_tts.interface.cli.srt import process_srt_file as _impl
    return _impl(srt_path, config, args, gen_params, use_server)


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


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive_mode(use_server, config, gen_params):
    """Run in interactive mode with prompts."""
    import soundfile as sf
    print("\n=== Qwen3-TTS Generator ===\n")

    print("How would you like to generate speech?")
    print("  1. Command Line (enter text here)")
    print("  2. Web Interface (Gradio UI in browser)")
    print()
    mode_choice = input("Select [1/2]: ").strip()

    if mode_choice == "2":
        launch_gradio_ui(config)
        return None

    text_input = input("\nEnter text or file path: ").strip()
    if not text_input:
        print("Error: No text provided")
        sys.exit(1)
    text = get_text(text_input)
    print(f"\nText to synthesize ({len(text)} chars):\n{text[:200]}{'...' if len(text) > 200 else ''}\n")

    print("Voice mode:")
    print("  1. Default (VoiceDesign with description)")
    print("  2. Custom (Voice clone prompt)")
    mode_choice = input("Select [1/2]: ").strip()

    if mode_choice == "2":
        prompts = list_voice_prompts()
        if not prompts:
            print("Error: No voice prompts found in", VOICE_PROMPTS_DIR)
            sys.exit(1)

        print("\nAvailable voice prompts:")
        for i, p in enumerate(prompts, 1):
            default_marker = " (default)" if p == config.get("default_clone_prompt") else ""
            print(f"  {i}. {p}{default_marker}")

        choice = input(f"Select [1-{len(prompts)}]: ").strip()
        try:
            prompt_file = prompts[int(choice) - 1]
        except (ValueError, IndexError):
            print("Invalid selection, using default")
            prompt_file = config.get("default_clone_prompt", prompts[0])

        mode = "clone"
        voice_param = prompt_file
    else:
        mode = "design"
        voice_param = config.get("default_voice_description", "")
        print(f"\nCurrent voice description: {voice_param}")
        custom = input("Use this description? [Y/n]: ").strip().lower()
        if custom == 'n':
            voice_param = input("Enter new description: ").strip()

    output_name = input("\nOutput filename (saved to ~/Downloads/): ").strip()
    if not output_name:
        output_name = "tts_output.wav"
    if not output_name.endswith('.wav'):
        output_name += '.wav'

    output_dir = os.path.expanduser(config.get("output_directory", "~/Downloads"))
    output_path = os.path.join(output_dir, output_name)

    print()
    language = config.get("language", "English")

    if use_server:
        print("Using TTS server...")
        if mode == "clone":
            results = generate_via_server([text], mode, config, gen_params, prompt_file=voice_param)
        else:
            results = generate_via_server([text], mode, config, gen_params, voice_description=voice_param)
        _save_base64_result(results[0], output_path)
    else:
        wav, sr = generate_local(
            text, mode, gen_params, language,
            prompt_file=voice_param if mode == "clone" else None,
            voice_description=voice_param if mode == "design" else None,
        )
        sf.write(output_path, wav, sr)

    print(f"Saved to: {output_path}")
    open_file(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_batch(texts, args, config, gen_params, use_server):
    """Process multiple texts."""
    import soundfile as sf
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_parser():
    """Build the argparse parser for generate.py."""
    parser = argparse.ArgumentParser(description="Qwen3-TTS Generator")
    parser.add_argument("text", nargs="*", help="Text(s) to synthesize or path to text file")
    parser.add_argument("-o", "--output", help="Output filename or directory for batch")
    parser.add_argument("-m", "--mode", choices=["clone", "design", "custom"], help="Voice mode (clone, design, or custom)")
    parser.add_argument("-p", "--prompt", help="Voice clone prompt filename")
    parser.add_argument("-d", "--description", help="Voice description (for design mode)")
    parser.add_argument("-s", "--speaker", help="Premium speaker name for custom mode (ryan, aiden, vivian, etc.)")
    parser.add_argument("-i", "--instruct", help="Style instruction for custom mode (e.g., 'very happy', 'speak slowly')")
    parser.add_argument("--prosody", metavar="PRESET", help="Prosody preset for custom/design mode (excited, calm, whisper, etc.)")
    parser.add_argument("--no-transcript", action="store_true", dest="no_transcript",
                        help="Clone using speaker embedding only (no transcript needed)")
    parser.add_argument("--batch", help="JSON file with array of texts")
    parser.add_argument("--preset", help="Use named preset from config")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, dest="top_k", help="Top-k sampling")
    parser.add_argument("--top-p", type=float, dest="top_p", help="Top-p (nucleus) sampling")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--repetition-penalty", type=float, dest="repetition_penalty", help="Repetition penalty")
    parser.add_argument("--max-chunk-chars", type=int, dest="max_chunk_chars", metavar="N",
                        help="Max chars per chunk for long text (default: 500 from config, 0 to disable)")
    parser.add_argument("--backend", choices=["torch", "mlx"], help="Override backend for this run (default: from config.json)")
    parser.add_argument("--model-size", choices=["1.7B", "0.6B"], help="Override model size for this run (default: from config.json)")
    parser.add_argument("--list-backends", action="store_true", help="List available backends and current setting")
    parser.add_argument("--list-prompts", action="store_true", help="List available voice prompts")
    parser.add_argument("--voices", action="store_true", help="List available voice prompts (alias for --list-prompts)")
    parser.add_argument("--list-presets", action="store_true", help="List available presets")
    parser.add_argument("--list-aliases", action="store_true", help="List available voice aliases")
    parser.add_argument("--list-speakers", action="store_true", help="List premium CustomVoice speakers")
    parser.add_argument("--list-prosody", action="store_true", help="List available prosody presets")
    parser.add_argument("--list-models", action="store_true", help="List available TTS models and their load status")
    parser.add_argument("--stats", action="store_true", help="Show server statistics")
    parser.add_argument("--edit-config", action="store_true", help="Edit default voice description")
    parser.add_argument("--no-open", action="store_true", help="Don't open the output file")
    parser.add_argument("--local", action="store_true", help="Force local generation (skip server)")
    parser.add_argument("--play", action="store_true", help="Play audio after generation")
    parser.add_argument("--stream", action="store_true", help="Stream audio playback as it generates (MLX backend)")
    parser.add_argument("--clipboard", action="store_true", help="Read text from clipboard")
    parser.add_argument("--trim-silence", action="store_true", help="Trim leading/trailing silence")
    parser.add_argument("--normalize", action="store_true", help="Normalize audio to -3dB peak")
    parser.add_argument("--speed", type=float, metavar="FACTOR", help="Speed factor (1.2 = 20%% faster, 0.8 = 20%% slower)")
    parser.add_argument("--pitch", type=float, metavar="SEMITONES", help="Pitch shift in semitones (+2 = higher, -2 = lower)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated without running")
    parser.add_argument("-v", "--voice", help="Use a voice alias from config (combines prompt + preset)")
    parser.add_argument("--delete-prompt", metavar="NAME", help="Delete a voice prompt")
    parser.add_argument("--rename-prompt", nargs=2, metavar=("OLD", "NEW"), help="Rename a voice prompt")
    parser.add_argument("--preview-prompt", metavar="NAME", help="Preview a voice prompt")
    parser.add_argument("--history", nargs="?", const=10, type=int, metavar="N", help="Show last N generations (default: 10)")
    parser.add_argument("--repl", action="store_true", help="Start interactive REPL mode")
    parser.add_argument("--watch", metavar="DIR", help="Watch directory for .txt files")
    parser.add_argument("--srt", metavar="FILE", help="Process SRT subtitle file")
    parser.add_argument("--ssml", action="store_true", help="Enable SSML markup parsing")
    parser.add_argument("--dialogue", metavar="FILE", help="Process dialogue JSON file with multiple speakers")
    parser.add_argument("--save-individual", action="store_true", help="Save individual audio files for each dialogue line")
    parser.add_argument("--ui", "--gui", action="store_true", dest="ui", help="Launch the Gradio web interface")
    parser.add_argument("--_server-mode", dest="server_mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--text-override", dest="text_override", help=argparse.SUPPRESS)
    return parser


def _handle_info_commands(args, config, gen_params):
    """Handle --list-*, --stats, --edit-config, --history, prompt management.

    Returns False if a command was handled, None if no info command matched.
    """
    if args.list_backends:
        current = get_backend()
        override = f" (overridden to '{args.backend}')" if args.backend else ""
        print(f"Available backends: {', '.join(VALID_BACKENDS)}")
        print(f"Current backend:    {current}{override}")
        print()
        if current == "mlx":
            quant = get_mlx_quantization()
            print(f"  MLX quantization: {quant}")
            for mt in ("clone", "design", "custom"):
                print(f"  {mt}: {get_mlx_model_name(mt)}")
        else:
            dtype = get_torch_dtype_name()
            print(f"  PyTorch dtype: {dtype}")
            for mt, info in MODEL_INFO.items():
                print(f"  {mt}: {info['name']}")
        print()
        print(f"To change: edit {CONFIG_PATH} -> advanced.backend")
        print("Or use: tts --backend mlx \"text\" -o output")
        return False

    if args.list_prompts or args.voices:
        prompts = list_voice_prompts()
        print("Available voice prompts:")
        for p in prompts:
            default_marker = " (default)" if p == config.get("default_clone_prompt") else ""
            print(f"  - {p}{default_marker}")
        return False

    if args.list_presets:
        presets = config.get("presets", {})
        print("Available presets:")
        for name, settings in presets.items():
            print(f"  - {name}: {settings}")
        return False

    if args.list_aliases:
        aliases = config.get("aliases", {})
        if aliases:
            print("Available voice aliases:")
            for name, settings in aliases.items():
                print(f"  - {name}:")
                for k, v in settings.items():
                    print(f"      {k}: {v}")
        else:
            print("No voice aliases configured.")
            print("Add to config.json under 'aliases', e.g.:")
            print('  "aliases": {')
            print('    "narrator": {"prompt": "narrator.pt", "preset": "consistent"}')
            print('  }')
        return False

    if args.list_prosody:
        from qwen3_tts.core.config import get_prosody_presets
        presets = get_prosody_presets(config)
        print("Available prosody presets (use with --prosody PRESET):\n")
        for name, text in sorted(presets.items()):
            print(f"  {name:<18} {text}")
        print()
        print("Example: tts -m custom -s ryan --prosody excited \"Hello!\" -o output")
        print(f"\nCustomize in {CONFIG_PATH} under 'prosody_presets'.")
        return False

    if args.list_speakers:
        print("Premium CustomVoice speakers (use with -m custom -s SPEAKER):")
        print()
        for group_name, lang_filter in [("English", "English"), ("Chinese", "Chinese")]:
            print(f"  {group_name}:")
            for key, info in CUSTOM_VOICE_SPEAKERS.items():
                if info["lang"] == lang_filter:
                    print(f"    {key:<12} - {info['desc']}")
            print()
        print("  Other languages:")
        for key, info in CUSTOM_VOICE_SPEAKERS.items():
            if info["lang"] not in ("English", "Chinese"):
                print(f"    {key:<12} - {info['desc']} ({info['lang']})")
        print()
        print("Example: tts 'Hello world' -m custom -s ryan -o output")
        return False

    if args.list_models:
        return _handle_list_models(args, config)

    if args.stats:
        return _handle_stats(config)

    if args.edit_config:
        print(f"Current voice description: {config.get('default_voice_description', '')}")
        new_desc = input("Enter new description (or press Enter to keep): ").strip()
        if new_desc:
            config["default_voice_description"] = new_desc
            save_config(config)
        return False

    if args.history is not None:
        show_history(args.history)
        return False

    if args.delete_prompt:
        delete_voice_prompt(args.delete_prompt)
        return False

    if args.rename_prompt:
        rename_voice_prompt(args.rename_prompt[0], args.rename_prompt[1])
        return False

    if args.preview_prompt:
        preview_voice_prompt(args.preview_prompt, config)
        return False

    return None  # No info command matched


def _handle_list_models(args, config):
    """Display available TTS models and their load status."""
    import requests  # lazy
    models_config = config.get("models", {})
    print("\nAvailable TTS Models:")
    print("=" * 60)
    print()

    model_display = {
        "clone": {"name": "Base (Clone)", "usage": "-m clone -p voice.pt", "memory": "~3.5GB"},
        "design": {"name": "VoiceDesign", "usage": "-m design -d 'warm female voice'", "memory": "~3.5GB"},
        "custom": {"name": "CustomVoice", "usage": "-m custom -s ryan", "memory": "~3.5GB"},
    }

    server_status = {}
    if is_server_running(config):
        try:
            url = get_server_url(config)
            resp = requests.get(f"{url}/models", timeout=5, headers=auth_headers())
            if resp.status_code == 200:
                server_status = resp.json().get("models", {})
        except Exception:  # nosec B110
            pass

    for model_type, info in MODEL_INFO.items():
        display = model_display.get(model_type, {})
        cfg = models_config.get(model_type, {})
        load_at_startup = cfg.get("load_at_startup", False)

        if server_status:
            loaded = server_status.get(model_type, {}).get("loaded", False)
            status = "LOADED" if loaded else "not loaded"
        else:
            status = "server not running"

        startup_str = "YES" if load_at_startup else "no"

        print(f"  {display.get('name', model_type):<16} [{status}]")
        print(f"    Model:       {info['name']}")
        print(f"    Purpose:     {info['description']}")
        print(f"    Usage:       {display.get('usage', '')}")
        print(f"    Memory:      {display.get('memory', '?')}")
        print(f"    Auto-load:   {startup_str} (config.json: models.{model_type}.load_at_startup)")
        print()

    print("To change which models load at startup, edit config.json:")
    print('  "models": { "clone": { "load_at_startup": true }, ... }')
    print()
    print("Models can also be loaded on-demand when you use a feature that requires them.")
    return False


def _handle_stats(config):
    """Display server statistics."""
    import requests  # lazy
    if is_server_running(config):
        url = get_server_url(config)
        resp = requests.get(f"{url}/stats", timeout=5, headers=auth_headers())
        if resp.status_code == 200:
            stats = resp.json()
            print("TTS Server Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
        else:
            print("Error: Failed to get stats")
    else:
        print("Server not running. Start with 'tts server start'.")
    return False


def _handle_generation(args, config, gen_params, use_server, max_chunk_chars):
    """Handle all generation: special modes, text resolution, and single-text output."""
    # Special modes
    if args.repl:
        run_repl(config, use_server)
        return use_server
    if args.watch:
        run_watch_mode(args.watch, config, args, gen_params, use_server)
        return use_server
    if args.srt:
        srt_path = os.path.expanduser(args.srt)
        if not os.path.isfile(srt_path):
            print(f"Error: SRT file not found: {srt_path}")
            sys.exit(1)
        process_srt_file(srt_path, config, args, gen_params, use_server)
        return use_server
    if args.dialogue:
        dialogue_path = os.path.expanduser(args.dialogue)
        if not os.path.isfile(dialogue_path):
            print(f"Error: Dialogue file not found: {dialogue_path}")
            sys.exit(1)
        process_dialogue(dialogue_path, config, args, gen_params, use_server)
        return use_server

    # Voice alias
    if args.voice:
        alias = get_voice_alias(args.voice, config)
        if alias is None:
            print(f"Error: Unknown voice alias '{args.voice}'")
            print("Available aliases:")
            aliases = config.get("aliases", {})
            for name, settings in aliases.items():
                print(f"  - {name}: {settings}")
            if not aliases:
                print("  (none configured - add to config.json under 'aliases')")
            sys.exit(1)
        if "prompt" in alias and not args.prompt:
            args.prompt = alias["prompt"]
        if "preset" in alias and not args.preset:
            args.preset = alias["preset"]
            gen_params = get_generation_params(args, config)
        if "mode" in alias and not args.mode:
            args.mode = alias["mode"]
        if "description" in alias and not args.description:
            args.description = alias["description"]
        print(f"Using voice alias '{args.voice}'")

    # Clipboard input
    if args.clipboard:
        clipboard_text = get_clipboard_text()
        args.text = [clipboard_text]
        print(f"Read from clipboard ({len(clipboard_text)} chars)")

    # Dry-run mode
    if args.dry_run:
        return _handle_dry_run(args, config, gen_params, use_server, max_chunk_chars)

    # Batch mode from file
    if args.batch:
        batch_path = os.path.expanduser(args.batch)
        if not os.path.isfile(batch_path):
            print(f"Error: Batch file not found: {batch_path}")
            sys.exit(1)
        with open(batch_path, "r") as f:
            texts = json.load(f)
        if not isinstance(texts, list):
            print("Error: Batch file must contain a JSON array of texts")
            sys.exit(1)
        process_batch(texts, args, config, gen_params, use_server)
        return use_server

    # Multiple texts as batch
    if len(args.text) > 1:
        texts = [get_text(t) for t in args.text]
        process_batch(texts, args, config, gen_params, use_server)
        return use_server

    # Read from stdin if piped
    if not args.text and not sys.stdin.isatty():
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            args.text = [stdin_text]

    # Interactive mode if no text
    if not args.text:
        result = interactive_mode(use_server, config, gen_params)
        if result is None:
            return False
        return use_server

    # Resolve text
    text = args.text_override if args.text_override else get_text(args.text[0])
    if args.ssml:
        original_text = text
        text = process_ssml_text(text, args)
        if text != original_text:
            print(f"SSML processed: {len(original_text)} -> {len(text)} chars")

    with open(LAST_TEXT_FILE, "w") as f:
        f.write(text)

    output_name = args.output or "tts_output.wav"
    if ".." in output_name or output_name.startswith("/"):
        print(f"Error: Invalid output path: {output_name}")
        return use_server
    if not output_name.endswith('.wav'):
        output_name += '.wav'
    output_dir = os.path.expanduser(config.get("output_directory", "~/Downloads"))
    output_path = auto_increment_filename(os.path.join(output_dir, output_name))

    language = config.get("language", "English")
    mode = args.mode or "clone"
    prompt_file = args.prompt or get_default_clone_prompt(config)
    voice_description = args.description or config.get("default_voice_description", "")

    # Resolve prosody preset into instruct text
    if args.prosody and not args.instruct:
        from qwen3_tts.core.config import get_prosody_presets
        prosody_presets = get_prosody_presets(config)
        if args.prosody in prosody_presets:
            args.instruct = prosody_presets[args.prosody]
            print(f"Using prosody preset '{args.prosody}': {args.instruct}")
        else:
            available = ", ".join(sorted(prosody_presets.keys()))
            print(f"Error: Unknown prosody preset '{args.prosody}'")
            print(f"Available: {available}")
            sys.exit(1)

    # Determine mode and speaker
    speaker_name = None
    if args.mode == "design" or args.description:
        mode = "design"
    if args.mode == "custom" or args.speaker:
        mode = "custom"
        speaker_key = (args.speaker or "ryan").lower()
        if speaker_key not in CUSTOM_VOICE_SPEAKERS:
            print(f"Error: Unknown speaker '{args.speaker}'")
            print("Use --list-speakers to see available options.")
            sys.exit(1)
        speaker_name = CUSTOM_VOICE_SPEAKERS[speaker_key]["name"]
        speaker_lang = CUSTOM_VOICE_SPEAKERS[speaker_key]["lang"]
        if not args.description:
            language = speaker_lang

    return _run_single_generation(
        text, args, config, gen_params, use_server, max_chunk_chars,
        output_path, mode, language, prompt_file, voice_description, speaker_name,
    )


def _handle_dry_run(args, config, gen_params, use_server, max_chunk_chars):
    """Display dry-run summary of what would be generated."""
    mode = args.mode or "clone"
    prompt_file = args.prompt or get_default_clone_prompt(config)
    voice_description = args.description or config.get("default_voice_description", "")
    output_dir = os.path.expanduser(args.output or config.get("output_directory", "~/Downloads"))

    texts = []
    if args.batch:
        batch_path = os.path.expanduser(args.batch)
        with open(batch_path, "r") as f:
            texts = json.load(f)
    elif args.text:
        texts = [get_text(t) for t in args.text]

    print("\n=== DRY RUN ===")
    print(f"Mode: {mode}")
    if mode == "clone":
        print(f"Voice prompt: {prompt_file}")
    elif mode == "custom":
        speaker_key = (args.speaker or "ryan").lower()
        if speaker_key in CUSTOM_VOICE_SPEAKERS:
            speaker_info = CUSTOM_VOICE_SPEAKERS[speaker_key]
            print(f"Speaker: {speaker_info['name']} ({speaker_info['desc']})")
        else:
            print(f"Speaker: {args.speaker} (unknown)")
        if args.instruct:
            print(f"Instruction: {args.instruct}")
    else:
        print(f"Voice description: {voice_description}")
    print(f"Output directory: {output_dir}")
    print(f"Server mode: {'yes' if use_server else 'no (local)'}")
    print("\nAudio processing:")
    print(f"  Trim silence: {'yes' if args.trim_silence else 'no'}")
    print(f"  Normalize: {'yes (-3dB peak)' if args.normalize else 'no'}")
    print(f"  Speed: {args.speed if args.speed else '1.0 (unchanged)'}")
    print(f"  Pitch: {args.pitch if args.pitch else '0 (unchanged)'} semitones")
    print(f"  SSML: {'enabled' if args.ssml else 'disabled'}")
    chunk_cfg = max_chunk_chars if max_chunk_chars is not None else config.get("generation", {}).get("max_chunk_chars", 500)
    print(f"  Text chunking: {'disabled' if chunk_cfg == 0 else f'max {chunk_cfg} chars/chunk'}")
    print("\nGeneration parameters:")
    for k, v in gen_params.items():
        print(f"  {k}: {v}")
    print(f"\nTexts to generate ({len(texts)}):")
    for i, t in enumerate(texts[:5], 1):
        preview = t[:80] + "..." if len(t) > 80 else t
        print(f"  {i}. {preview}")
    if len(texts) > 5:
        print(f"  ... and {len(texts) - 5} more")
    print("\n=== END DRY RUN ===")
    return False


def _voice_param_for_log(mode, prompt_file, voice_description, speaker_name, instruct):
    """Build the voice_param string for log_generation()."""
    if mode == "clone":
        return prompt_file
    elif mode == "design":
        return voice_description
    return f"{speaker_name}" + (f" ({instruct})" if instruct else "")


def _run_single_generation(text, args, config, gen_params, use_server, max_chunk_chars,
                           output_path, mode, language, prompt_file, voice_description,
                           speaker_name):
    """Execute a single text generation (streaming, server, or local) and save output."""
    import soundfile as sf  # lazy
    gen_start = time.time()
    instruct = args.instruct or ""

    if getattr(args, "stream", False) and use_server:
        print("Using TTS server (streaming mode)...")
        if mode == "clone":
            generate_streaming(text, mode, config, gen_params, output_path,
                               prompt_file=prompt_file,
                               x_vector_only_mode=getattr(args, 'no_transcript', False))
        elif mode == "design":
            generate_streaming(text, mode, config, gen_params, output_path,
                               voice_description=voice_description)
        else:
            generate_streaming(text, mode, config, gen_params, output_path,
                               speaker=speaker_name, instruct=instruct)
        gen_duration = time.time() - gen_start
        print(f"Streaming complete ({gen_duration:.1f}s)")
        voice_param = _voice_param_for_log(mode, prompt_file, voice_description, speaker_name, instruct)
        log_generation(text, mode, voice_param, output_path, gen_params, duration_sec=gen_duration)
        return use_server

    if use_server:
        print("Using TTS server...")
        if mode == "clone":
            results = generate_via_server([text], mode, config, gen_params, prompt_file=prompt_file,
                                          max_chunk_chars=max_chunk_chars,
                                          x_vector_only_mode=getattr(args, 'no_transcript', False))
        elif mode == "design":
            results = generate_via_server([text], mode, config, gen_params, voice_description=voice_description,
                                          max_chunk_chars=max_chunk_chars)
        else:
            results = generate_via_server([text], mode, config, gen_params,
                                          speaker=speaker_name, instruct=instruct,
                                          max_chunk_chars=max_chunk_chars)
        needs_processing = args.trim_silence or args.normalize or args.speed or args.pitch
        if needs_processing:
            wav, sr = _decode_base64_result(results[0])
            wav = process_audio_args(wav, sr, args)
            sf.write(output_path, wav, sr)
        else:
            _save_base64_result(results[0], output_path)
    else:
        if mode == "custom":
            wav, sr = generate_local(text, mode, gen_params, language,
                                     speaker=speaker_name, instruct=instruct,
                                     max_chunk_chars=max_chunk_chars)
        elif mode == "design":
            wav, sr = generate_local(text, mode, gen_params, language,
                                     voice_description=voice_description,
                                     max_chunk_chars=max_chunk_chars)
        else:
            wav, sr = generate_local(text, mode, gen_params, language,
                                     prompt_file=prompt_file,
                                     max_chunk_chars=max_chunk_chars)
        wav = process_audio_args(wav, sr, args)
        sf.write(output_path, wav, sr)

    gen_duration = time.time() - gen_start
    print(f"Saved to: {output_path} ({gen_duration:.1f}s)")
    voice_param = _voice_param_for_log(mode, prompt_file, voice_description, speaker_name, instruct)
    log_generation(text, mode, voice_param, output_path, gen_params, duration_sec=gen_duration)

    if args.play:
        play_audio(output_path)
    elif not args.no_open:
        open_file(output_path)

    return use_server


def main():
    args = _build_parser().parse_args()
    config = load_config()
    gen_params = get_generation_params(args, config)
    max_chunk_chars = getattr(args, "max_chunk_chars", None)

    if args.backend:
        os.environ["TTS_BACKEND"] = args.backend
    if args.model_size:
        os.environ["TTS_MODEL_SIZE"] = args.model_size

    if args.ui:
        launch_gradio_ui(config)
        return False

    result = _handle_info_commands(args, config, gen_params)
    if result is not None:
        return result

    use_server = args.server_mode and not args.local
    return _handle_generation(args, config, gen_params, use_server, max_chunk_chars)


if __name__ == "__main__":
    used_server = main()
    sys.exit(0 if not used_server else 2)
