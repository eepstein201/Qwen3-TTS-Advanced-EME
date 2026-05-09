#!/usr/bin/env python3
"""vLLM-Omni adapter for high-performance TTS inference.

This module provides VLLMAdapter, which manages a vLLM-Omni subprocess
for accelerated TTS generation. It handles:
- Dynamic port allocation
- Subprocess spawning with POSIX process groups
- Health checks via /v1/models endpoint
- Async generation and streaming proxies
- Cancellation callbacks for graceful shutdown

vLLM-Omni is an external server process that exposes OpenAI-compatible
endpoints for audio generation. This adapter manages the subprocess lifecycle
and provides a clean Python interface.
"""

import asyncio
import base64
import io
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("tts.engine.vllm")

# Named constants for timeouts and limits (M25)
_STARTUP_TIMEOUT_SECS = 300.0
_HEALTH_CHECK_TIMEOUT_SECS = 5.0
_HEALTH_CHECK_POLL_SECS = 2.0
_GRACEFUL_STOP_TIMEOUT_SECS = 10.0
_GENERATION_TIMEOUT_SECS = 600.0
_GENERATION_CONNECT_TIMEOUT_SECS = 60.0


class VLLMAdapter:
    """Adapter for vLLM-Omni TTS server subprocess.

    Manages the vLLM-Omni server lifecycle:
    - Spawns the subprocess in a POSIX process group (prevents zombies)
    - Routes stdout/stderr to log file (prevents buffer freeze)
    - Polls /v1/models until ready
    - Proxies generation requests to OpenAI-compatible endpoints
    - Supports cancellation via callback registration
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        gpu_memory_utilization: float = 0.7,
        port: int | None = None,
        log_file: str | None = None,
        max_model_len: int = 4096,
        dtype: str = "bfloat16",
    ):
        """Initialize the vLLM adapter.

        Args:
            model_name: HuggingFace model ID for vLLM to load
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            port: Port for vLLM server (None = auto-find open port)
            log_file: Path to log file for vLLM output (None = use default)
            max_model_len: Maximum model context length
            dtype: Model dtype (bfloat16 recommended for modern GPUs)
        """
        self.model_name = model_name
        self.gpu_memory_utilization = gpu_memory_utilization
        self.port = port
        self.log_file = log_file or self._get_default_log_file()
        self.max_model_len = max_model_len
        self.dtype = dtype

        self._process: subprocess.Popen | None = None
        self._client: httpx.AsyncClient | None = None
        self._log_fh = None  # File handle for vLLM subprocess stdout/stderr
        self._ready_event = asyncio.Event()
        self._cancellation_callback: Callable[[], bool] | None = None
        self._auto_port = port is None

        logger.debug(
            "VLLMAdapter initialized: model=%s, gpu_memory=%.2f, port=%s",
            model_name,
            gpu_memory_utilization,
            port or "auto",
        )

    def _get_default_log_file(self) -> str:
        """Return default vLLM log file path."""
        from qwen3_tts.core.config import USER_FILES_DIR

        log_dir = Path(USER_FILES_DIR) / ".vllm_logs"
        log_dir.mkdir(exist_ok=True)
        return str(log_dir / "vllm_server.log")

    def _find_open_port(self, start_port: int = 8100) -> int:
        """Find an available port starting from start_port.

        Tries ports sequentially until one is available.

        Args:
            start_port: First port to try

        Returns:
            An available port number
        """
        for port in range(start_port, start_port + 100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    s.listen(1)
                    return port
                except OSError:
                    continue
        raise RuntimeError(
            f"Could not find open port in range {start_port}-{start_port + 99}"
        )

    def _start(self) -> None:
        """Start the vLLM-Omni subprocess.

        The subprocess is launched with:
        - POSIX process group (preexec_fn=os.setsid) to prevent zombies
        - stdout/stderr redirected to log file (prevents buffer freeze)
        - Model name and GPU memory utilization as arguments
        """
        if self._process is not None:
            raise RuntimeError("vLLM process is already running")

        # Find open port if not specified
        if self.port is None:
            self.port = self._find_open_port()
            logger.info("Auto-assigned vLLM port: %d", self.port)

        # Prepare log file
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Build command line
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_name,
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--port",
            str(self.port),
            "--disable-log-requests",
            "--limit-mm-per-prompt",
            "audio=1",
            "--enable-chunked-prefill",
            "--dtype",
            self.dtype,
            "--max-model-len",
            str(self.max_model_len),
        ]

        logger.info("Starting vLLM subprocess: %s", " ".join(cmd))

        # Open log file for subprocess output (stored so stop() can close it)
        self._log_fh = open(log_path, "w")  # noqa: PTH123

        # Start subprocess with POSIX process group
        self._process = subprocess.Popen(
            cmd,
            stdout=self._log_fh,
            stderr=self._log_fh,
            preexec_fn=os.setsid,  # Create new process group
            start_new_session=True,  # Windows equivalent (ignored on POSIX)
        )

        logger.info(
            "vLLM subprocess started with PID %d (port %d, log: %s)",
            self._process.pid,
            self.port,
            self.log_file,
        )

    async def _wait_until_ready(self, timeout: float = _STARTUP_TIMEOUT_SECS) -> None:
        """Wait until vLLM server is ready by polling /v1/models.

        Args:
            timeout: Maximum seconds to wait

        Raises:
            TimeoutError: If server does not become ready within timeout
            RuntimeError: If server exits unexpectedly
        """
        base_url = f"http://127.0.0.1:{self.port}"
        start_time = asyncio.get_running_loop().time()

        logger.info("Waiting for vLLM server to be ready at %s", base_url)

        while True:
            # Check if process exited
            if self._process and self._process.poll() is not None:
                returncode = self._process.returncode
                raise RuntimeError(
                    f"vLLM process exited with code {returncode}. "
                    f"Check log file: {self.log_file}"
                )

            # Try health check
            try:
                async with httpx.AsyncClient(
                    timeout=_HEALTH_CHECK_TIMEOUT_SECS
                ) as client:
                    response = await client.get(f"{base_url}/v1/models")
                    if response.status_code == 200:
                        logger.info("vLLM server is ready")
                        self._ready_event.set()
                        return
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                logger.debug("vLLM not ready yet: %s", e)

            # Check timeout
            elapsed = asyncio.get_running_loop().time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f"vLLM server did not become ready within {timeout}s. "
                    f"Check log file: {self.log_file}"
                )

            # Wait before retry
            await asyncio.sleep(_HEALTH_CHECK_POLL_SECS)

    async def start(self) -> None:
        """Start vLLM server and wait until ready.

        Retries up to 3 times on failure. On each failure, stops the zombie
        process before retrying. If using auto-port, picks a new port on retry.
        """
        max_attempts = 3
        for attempt in range(max_attempts):
            if attempt > 0 and self._auto_port:
                self.port = None  # Force _start to find a new port

            self._start()

            try:
                await self._wait_until_ready()
                return  # Success
            except Exception as e:
                logger.warning(
                    "vLLM start failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_attempts,
                    e,
                )
                self.stop()  # Kill zombie process

                if attempt < max_attempts - 1:
                    continue
                raise  # Exhausted retries

    def stop(self) -> None:
        """Stop the vLLM subprocess and close HTTP client.

        Sends SIGTERM to the process group, waits for graceful shutdown,
        then forces SIGKILL if needed. Also closes the httpx.AsyncClient.
        """
        if self._process is None:
            logger.warning("stop() called but no vLLM process running")
            return

        pid = self._process.pid
        logger.info("Stopping vLLM subprocess (PID %d)", pid)

        try:
            # Send SIGTERM to process group (negative PID)
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except ProcessLookupError:
                logger.warning("Process group not found for PID %d", pid)

            # Wait for graceful shutdown (max 10 seconds)
            try:
                self._process.wait(timeout=_GRACEFUL_STOP_TIMEOUT_SECS)
                logger.info("vLLM subprocess shut down gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("vLLM did not shut down gracefully, forcing SIGKILL")
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._process.wait()
                logger.info("vLLM subprocess force-killed")

        finally:
            self._process = None
            self._ready_event.clear()

        # Close log file handle
        if self._log_fh:
            try:
                self._log_fh.close()
            except Exception as e:
                logger.warning("Error closing vLLM log file handle: %s", e)
            self._log_fh = None

        # Close HTTP client if exists
        if self._client:
            try:
                asyncio.get_running_loop().create_task(self._client.aclose())
            except Exception as e:
                logger.warning("Error closing HTTP client: %s", e)
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for vLLM requests.

        Returns:
            httpx.AsyncClient configured for vLLM server
        """
        if self._client is None:
            base_url = f"http://127.0.0.1:{self.port}"
            timeout = httpx.Timeout(
                _GENERATION_TIMEOUT_SECS, connect=_GENERATION_CONNECT_TIMEOUT_SECS
            )
            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
            self._client = httpx.AsyncClient(
                base_url=base_url, timeout=timeout, limits=limits
            )
        return self._client

    def _build_request(
        self,
        text: str,
        mode: str,
        prompt_audio: bytes | None,
        voice_description: str | None,
        speaker: str | None,
        **kwargs,
    ) -> dict:
        """Build the vLLM request dict for clone/design/custom modes.

        Args:
            text: Input text to synthesize
            mode: Generation mode ("clone", "design", "custom")
            prompt_audio: Reference audio bytes (clone mode only)
            voice_description: Voice description (design mode only)
            speaker: Speaker name (custom mode only)
            **kwargs: Additional generation parameters merged into input

        Returns:
            Request dict ready for /v1/audio/generations

        Raises:
            ValueError: If required mode parameter is missing or mode is invalid
        """
        if mode == "clone":
            if prompt_audio is None:
                raise ValueError("prompt_audio required for clone mode")
            # Write to a temp file; caller is responsible for cleanup via the
            # returned path stored in request["input"]["prompt_audio"].
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            try:
                tmp.write(prompt_audio)
                prompt_path = tmp.name
            except Exception:
                tmp.close()
                os.unlink(tmp.name)
                raise
            finally:
                tmp.close()
            request = {
                "model": self.model_name,
                "input": {"text": text, "mode": "clone", "prompt_audio": prompt_path},
            }

        elif mode == "design":
            if voice_description is None:
                raise ValueError("voice_description required for design mode")
            request = {
                "model": self.model_name,
                "input": {
                    "text": text,
                    "mode": "design",
                    "voice_description": voice_description,
                },
            }

        elif mode == "custom":
            if speaker is None:
                raise ValueError("speaker required for custom mode")
            request = {
                "model": self.model_name,
                "input": {"text": text, "mode": "custom", "speaker": speaker},
            }

        else:
            raise ValueError(f"Invalid mode: {mode}")

        request["input"].update(kwargs)
        return request

    def set_cancellation_callback(self, callback: Callable[[], bool]) -> None:
        """Register a callback to check for cancellation requests.

        The callback should return True if generation should be cancelled.
        It is called periodically during streaming generation.

        Args:
            callback: Function that returns bool (True = cancel requested)
        """
        self._cancellation_callback = callback
        logger.debug("Cancellation callback registered")

    async def generate(
        self,
        text: str,
        mode: str = "clone",
        prompt_audio: bytes | None = None,
        voice_description: str | None = None,
        speaker: str | None = None,
        **kwargs,
    ) -> tuple[int, Any]:
        """Generate audio from text using vLLM-Omni.

        Args:
            text: Input text to synthesize
            mode: Generation mode ("clone", "design", "custom")
            prompt_audio: Reference audio bytes (clone mode only)
            voice_description: Voice description (design mode only)
            speaker: Speaker name (custom mode only)
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (sample_rate, audio_array) where audio_array is float32 numpy array
        """
        if not self._ready_event.is_set():
            raise RuntimeError("vLLM server is not ready")

        client = self._get_client()
        request = self._build_request(
            text, mode, prompt_audio, voice_description, speaker, **kwargs
        )

        try:
            response = await client.post("/v1/audio/generations", json=request)
            response.raise_for_status()

            data = response.json()
            # vLLM-Omni returns audio in OpenAI-compatible format
            audio_base64 = data.get("data", [{}])[0].get("audio")

            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_base64)

            # Load audio bytes into numpy array
            import soundfile as sf  # lazy — heavy import

            audio, sr = sf.read(io.BytesIO(audio_bytes))
            return sr, audio

        except httpx.HTTPStatusError as e:
            logger.error("vLLM generation failed: %s", e.response.text)
            raise RuntimeError(f"vLLM generation failed: {e}") from e
        except Exception as e:
            logger.error("vLLM generation error: %s", e)
            raise

    async def generate_stream(
        self,
        text: str,
        mode: str = "clone",
        prompt_audio: bytes | None = None,
        voice_description: str | None = None,
        speaker: str | None = None,
        **kwargs,
    ) -> AsyncIterator[tuple[int, Any]]:
        """Generate audio from text with streaming using vLLM-Omni.

        Yields audio chunks as they are generated, enabling real-time playback.

        Args:
            text: Input text to synthesize
            mode: Generation mode ("clone", "design", "custom")
            prompt_audio: Reference audio bytes (clone mode only)
            voice_description: Voice description (design mode only)
            speaker: Speaker name (custom mode only)
            **kwargs: Additional generation parameters

        Yields:
            Tuple of (sample_rate, audio_chunk) where audio_chunk is float32 numpy array
        """
        if not self._ready_event.is_set():
            raise RuntimeError("vLLM server is not ready")

        client = self._get_client()
        request = self._build_request(
            text, mode, prompt_audio, voice_description, speaker, **kwargs
        )
        request["stream"] = True

        try:
            async with client.stream(
                "POST", "/v1/audio/generations", json=request
            ) as response:
                response.raise_for_status()

                # Process streaming response
                async for line in response.aiter_lines():
                    if not line:
                        continue

                    # Check for cancellation
                    if self._cancellation_callback and self._cancellation_callback():
                        logger.info("Generation cancelled by callback")
                        response.close()
                        raise RuntimeError("Generation cancelled")

                    # Parse SSE format
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            chunk_data = json.loads(data_str)
                            audio_base64 = chunk_data.get("audio")
                            if audio_base64:
                                audio_bytes = base64.b64decode(audio_base64)
                                import soundfile as sf  # lazy — heavy import

                                audio, sr = sf.read(io.BytesIO(audio_bytes))
                                yield sr, audio

                        except json.JSONDecodeError:
                            logger.warning(
                                "Failed to parse streaming response: %s", data_str
                            )

        except httpx.HTTPStatusError as e:
            logger.error("vLLM streaming failed: %s", e.response.text)
            raise RuntimeError(f"vLLM streaming failed: {e}") from e
        except Exception as e:
            logger.error("vLLM streaming error: %s", e)
            raise

    def is_ready(self) -> bool:
        """Check if vLLM server is ready."""
        return self._ready_event.is_set()

    @property
    def base_url(self) -> str:
        """Return the base URL for vLLM server."""
        if self.port is None:
            raise RuntimeError("vLLM server not started")
        return f"http://127.0.0.1:{self.port}"
