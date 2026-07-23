#!/usr/bin/env python3
"""Async HTTPX client for vLLM server with circuit breaker pattern.

This module provides AsyncVLLMClient, which:
- Uses httpx.AsyncClient for non-blocking HTTP calls
- Implements circuit breaker pattern for vLLM failures
- Provides retry logic with exponential backoff
- Supports async generation and streaming
- Decouples FastAPI from direct VLLMAdapter imports

The circuit breaker prevents cascade failures by automatically opening
after repeated failures and closing after a cooldown period.
"""

import asyncio
import base64
import io
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger("tts.server.vllm_client")


# Circuit breaker state constants
_CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3
_CIRCUIT_BREAKER_COOLDOWN_SECS = 60
_CIRCUIT_BREAKER_HALF_OPEN_SECS = 30


class CircuitBreaker:
    """Circuit breaker for vLLM server to prevent cascade failures.

    The circuit breaker has three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Circuit is tripped, requests fail immediately
    - HALF_OPEN: Testing if service has recovered
    """

    def __init__(
        self,
        failure_threshold: int = _CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        cooldown_secs: float = _CIRCUIT_BREAKER_COOLDOWN_SECS,
        half_open_secs: float = _CIRCUIT_BREAKER_HALF_OPEN_SECS,
    ):
        """Initialize the circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures to trip circuit
            cooldown_secs: Seconds to wait before attempting recovery
            half_open_secs: Seconds to wait in HALF_OPEN state before closing
        """
        self.failure_threshold = failure_threshold
        self.cooldown_secs = cooldown_secs
        self.half_open_secs = half_open_secs

        self._failure_count = 0
        self._last_failure_time = 0.0
        self._state = "CLOSED"  # CLOSED, OPEN, or HALF_OPEN
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        """Enter circuit breaker context.

        Returns:
            self

        Raises:
            RuntimeError: If circuit is OPEN or HALF_OPEN
        """
        await self._check_state()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit circuit breaker context.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised
        """
        if exc_type is not None:
            await self._on_failure()
        else:
            await self._on_success()

    async def _check_state(self) -> None:
        """Check circuit state and raise if OPEN.

        Raises:
            RuntimeError: If circuit is OPEN
        """
        async with self._lock:
            if self._state == "OPEN":
                # Check if cooldown period has elapsed
                if time.time() - self._last_failure_time >= self.cooldown_secs:
                    logger.info("Circuit breaker: entering HALF_OPEN state")
                    self._state = "HALF_OPEN"
                else:
                    raise RuntimeError(
                        f"Circuit breaker is OPEN (cooldown: {self.cooldown_secs}s)"
                    )
            elif self._state == "HALF_OPEN":
                # Allow one request through to test service
                logger.debug(
                    "Circuit breaker: allowing test request in HALF_OPEN state"
                )

    async def _on_success(self) -> None:
        """Handle successful request.

        Resets failure count and closes circuit if HALF_OPEN.
        """
        async with self._lock:
            self._failure_count = 0
            if self._state == "HALF_OPEN":
                logger.info("Circuit breaker: closing after successful test request")
                self._state = "CLOSED"

    async def _on_failure(self) -> None:
        """Handle failed request.

        Increments failure count and trips circuit if threshold exceeded.
        """
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                logger.warning(
                    f"Circuit breaker: tripped after {self._failure_count} failures"
                )
                self._state = "OPEN"

    @property
    def state(self) -> str:
        """Get current circuit state.

        Returns:
            State string: CLOSED, OPEN, or HALF_OPEN
        """
        return self._state


class AsyncVLLMClient:
    """Async HTTP client for vLLM server with circuit breaker pattern.

    This client provides:
    - Non-blocking HTTP calls via httpx.AsyncClient
    - Automatic retry with exponential backoff
    - Circuit breaker to prevent cascade failures
    - Async generation and streaming support
    - Decoupling from VLLMAdapter subprocess management
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 300.0,
        circuit_breaker_failure_threshold: int = _CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    ):
        """Initialize the async vLLM client.

        Args:
            base_url: Base URL for vLLM server (e.g., "http://127.0.0.1:5123")
            timeout: Request timeout in seconds
            circuit_breaker_failure_threshold: Failures before tripping circuit
        """
        self.base_url = base_url
        self.timeout = timeout
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_failure_threshold
        )

        # Initialize HTTP client (will be created on first use)
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client.

        Returns:
            httpx.AsyncClient configured for vLLM server
        """
        if self._client is None:
            timeout = httpx.Timeout(self.timeout, connect=60.0)
            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=timeout, limits=limits
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _decode_audio(audio_base64: str) -> tuple[int, Any]:
        """Decode base64 audio and read it using soundfile."""
        import soundfile as sf

        audio_bytes = base64.b64decode(audio_base64)
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        return sr, audio

    async def generate(
        self,
        text: str,
        mode: str = "clone",
        prompt_audio: bytes | None = None,
        voice_description: str | None = None,
        speaker: str | None = None,
        **kwargs,
    ) -> tuple[int, Any]:
        """Generate audio from text using vLLM server.

        Args:
            text: Text to synthesize
            mode: Generation mode (clone/design/custom)
            prompt_audio: Reference audio bytes (clone mode only)
            voice_description: Voice description (design mode only)
            speaker: Speaker name (custom mode only)
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (sample_rate, audio_array) where audio_array is float32 numpy array

        Raises:
            RuntimeError: If circuit breaker is OPEN or generation fails
        """
        async with self.circuit_breaker:
            client = self._get_client()

            # Build request payload
            request = {
                "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                "input": {"text": text, "mode": mode},
            }

            # Track any temp file so it can be cleaned up in the finally block.
            tmp_path: str | None = None

            if mode == "clone" and prompt_audio:
                import tempfile

                # Write audio to temp file for vLLM
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(prompt_audio)
                    tmp_path = tmp.name
                request["input"]["prompt_audio"] = tmp_path

            elif mode == "design" and voice_description:
                request["input"]["voice_description"] = voice_description

            elif mode == "custom" and speaker:
                request["input"]["speaker"] = speaker

            request["input"].update(kwargs)

            try:
                # Implement retry with exponential backoff
                for attempt in range(3):  # Max 3 retries
                    try:
                        response = await client.post(
                            "/v1/audio/generations", json=request
                        )
                        response.raise_for_status()

                        data = response.json()
                        audio_base64 = data.get("data", [{}])[0].get("audio")

                        # Decode base64 audio in a threadpool to avoid
                        # blocking the event loop on CPU-bound decode + I/O.
                        sr, audio = await asyncio.to_thread(
                            self._decode_audio, audio_base64
                        )
                        return sr, audio

                    except httpx.HTTPStatusError as e:
                        if attempt < 2 and e.response.status_code >= 500:
                            # Server error - retry with backoff
                            wait_time = 2**attempt  # 1s, 2s, 4s
                            logger.warning(
                                f"vLLM request failed (attempt {attempt + 1}), retrying in {wait_time}s"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            # Client error or final retry - raise
                            logger.error("vLLM generation failed: %s", e.response.text)
                            raise RuntimeError(f"vLLM generation failed: {e}") from e

            except Exception as e:
                logger.error("vLLM generation error: %s", e)
                raise
            finally:
                # Always remove the temp prompt-audio file to avoid leaks.
                if tmp_path is not None and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError as unlink_err:
                        logger.warning(
                            "Failed to remove temp prompt-audio file %s: %s",
                            tmp_path,
                            unlink_err,
                        )

    async def health_check(self) -> bool:
        """Check if vLLM server is healthy.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            client = self._get_client()
            response = await client.get("/v1/models", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.debug("vLLM health check failed: %s", e)
            return False

    @property
    def circuit_state(self) -> str:
        """Get current circuit breaker state.

        Returns:
            Circuit state: CLOSED, OPEN, or HALF_OPEN
        """
        return self.circuit_breaker.state
