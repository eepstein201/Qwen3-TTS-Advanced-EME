"""Audio generation endpoint handlers — /generate and /generate-stream.

Extracted from app.py to keep each module under 800 lines.
These are plain async functions called by thin endpoint wrappers in app.py.
"""

import asyncio
import base64
import io
import json
import logging
import os
import secrets
import struct
import tempfile
import threading
import time
import uuid

from fastapi import HTTPException, Response
from fastapi.responses import StreamingResponse

from qwen3_tts.core.config import get_generation_cache_max
from qwen3_tts.server.app_lifespan import _check_memory_available
from qwen3_tts.server.validation import (
    MAX_SEED,
    _error_response,
    _gen_cache_key,
    _validate_generation_request,
)

logger = logging.getLogger("tts")


def _resolve_generation_seed(req_seed: int | None) -> int:
    """Return the seed to use for generation.

    When the caller supplies a seed (including 0), it is used verbatim so the
    request stays reproducible. When none is given, a random seed is generated
    server-side so the value can be recorded in history and reused later — the
    UI otherwise has no way to know which seed produced a "random" generation.
    """
    if req_seed is not None:
        return req_seed
    return secrets.randbelow(MAX_SEED + 1)


def _should_stop_streaming(stop_event, generation_state) -> bool:
    """Return True if streaming generation should stop.

    Stops when either the client disconnected (``stop_event`` set by the
    generator's finally) or the user cancelled via /cancel-generation
    (``generation_state['cancelled']``), matching the batch path's cancel check.
    """
    return stop_event.is_set() or bool(generation_state.get("cancelled"))


# Floor for the streaming inference thread join. The wait must cover ONE chunk's
# generation; on slow macOS CI a 500-char chunk takes ~30-60 s, so 90 s is the
# margin for default-sized chunks.
_STREAM_THREAD_JOIN_FLOOR_SEC: float = 90.0

# Conservative per-character generation cost, matching TTSClient's
# _generation_timeout (server/client/generator.py). Chunks generate sequentially
# at ~40-70 s each on MLX/M2 Pro for ~500 chars.
_STREAM_SECONDS_PER_CHAR: float = 0.25


def _stream_thread_join_timeout(
    text_len: int, max_chunk_chars: int | None = None
) -> float:
    """Seconds to wait for the streaming inference thread to finish its chunk.

    Must scale with ``max_chunk_chars``: a fixed timeout sized for the 500-char
    default expires mid-generation once the limit is raised, and the caller then
    releases ``inference_lock`` while the model is still running on the GPU —
    the exact race this join exists to prevent. Never reintroduce a constant
    here (same failure mode as the old hardcoded ``timeout=600`` in
    generate_via_server).

    ``max_chunk_chars`` of 0/None disables chunking, so the whole text is
    generated in one call and the bound is the full text length.
    """
    effective_chars = (
        min(max_chunk_chars, text_len) if max_chunk_chars else text_len
    )
    return max(_STREAM_THREAD_JOIN_FLOOR_SEC, effective_chars * _STREAM_SECONDS_PER_CHAR)


# Back-compat alias: existing callers/tests reference the old constant name.
_STREAM_THREAD_JOIN_TIMEOUT_SEC: float = _STREAM_THREAD_JOIN_FLOOR_SEC

# Terminal error frame for the length-prefixed streaming wire format (WS2 2.5).
# Frames are [sample_rate:4][length:4][payload:length]; a real chunk always
# carries a non-zero sample rate, so 0 is a free sentinel. The payload is JSON
# {"error": str, "code": str}. The client mirror of this constant lives in
# qwen3_tts/interface/generate_server.py and is kept in lockstep by
# tests/test_stream_error_frame.py.
STREAM_ERROR_SENTINEL_SR: int = 0
STREAM_ERROR_CODE_INFERENCE_FAILED = "inference_failed"


def encode_stream_error_frame(message: str, code: str = STREAM_ERROR_CODE_INFERENCE_FAILED) -> bytes:
    """Build a terminal error frame for the streaming wire format."""
    payload = json.dumps({"error": message, "code": code}).encode("utf-8")
    return struct.pack("<II", STREAM_ERROR_SENTINEL_SR, len(payload)) + payload


async def _await_inference_thread_done(
    done_event: threading.Event,
    timeout: float = _STREAM_THREAD_JOIN_TIMEOUT_SEC,
) -> bool:
    """Block (threadpool worker) until the streaming inference thread sets
    done_event or timeout elapses. Call ONLY in a streaming consumer's finally,
    BEFORE the `async with inference_lock` block exits, so an in-flight
    model.generate() cannot race the next request's GPU access. Returns True if
    the thread signaled done; False on timeout (lock released anyway -> the
    daemon thread finishes its chunk and self-terminates)."""
    return await asyncio.to_thread(done_event.wait, timeout)


async def handle_generate(request, state, req, security, config_provider):
    """Core logic for the /generate endpoint.

    Args:
        request: FastAPI Request object
        state: app.state
        req: GenerateRequest
        security: security config dict
        config_provider: optional ConfigLoader for DI
    """
    # Validate and normalize request
    max_text_length = security.get("max_text_length", 50000)
    max_batch_size = security.get("max_batch_size", 20)

    # Support both text and texts
    if req.text:
        texts = [req.text]
    elif req.texts:
        texts = req.texts
    else:
        raise HTTPException(status_code=400, detail="No text provided")

    if isinstance(texts, str):
        texts = [texts]

    if len(texts) > max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(texts)} exceeds limit of {max_batch_size}",
        )

    for i, t in enumerate(texts):
        if not isinstance(t, str) or not t.strip():
            raise HTTPException(
                status_code=400, detail=f"Text at index {i} is empty or invalid"
            )
        if len(t) > max_text_length:
            raise HTTPException(
                status_code=400,
                detail=f"Text at index {i} exceeds {max_text_length} character limit ({len(t)} chars)",
            )

    _validate_generation_request(req, security)

    # Memory guard — prevent OOM crash. Runs after validation so a malformed
    # request is always a 400 regardless of host memory pressure.
    mem_ok, available_mb = _check_memory_available()
    if not mem_ok:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "insufficient_memory",
                "detail": f"Only {available_mb}MB available. Unload unused models to free memory.",
                "recovery": "unload",
            },
        )

    mode = req.mode
    prompt_file = req.prompt_file
    speaker = req.speaker

    # Check if required model is loaded
    model = state.models.get(mode)
    if model is None:
        from qwen3_tts.core.config import get_model_info

        info = get_model_info(mode)
        detail = info.get("description", "")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_loaded",
                "detail": f"The '{mode}' model is not loaded. {detail}",
                "recovery": "restart",
                "model_type": mode,
            },
        )

    # Generation parameters
    gen_params = {
        "temperature": req.temperature,
        "top_k": req.top_k,
        "top_p": req.top_p,
        "repetition_penalty": req.repetition_penalty,
        "max_new_tokens": req.max_new_tokens,
    }
    if req.seed is not None:
        gen_params["seed"] = req.seed

    # Resolve the actual seed to apply and report. When the caller supplies no
    # seed we generate one so it can be shown in history and reused. It is kept
    # OUT of gen_params (and thus the cache key) so repeated blank-seed requests
    # still cache-hit; the seed that produced the cached audio is stored on the
    # cache entry and echoed back on hits.
    used_seed = _resolve_generation_seed(req.seed)

    voice_description = req.voice_description
    language = req.language
    instruct = req.instruct
    x_vector_only_mode = req.x_vector_only_mode
    max_chunk_chars = req.max_chunk_chars

    # Track this request in queue (thread-safe, R-19)
    request_id = id(request)
    with state.request_queue_lock:
        state.request_queue.add(request_id)

    # Stamp a per-batch ownership id so the finally can check whether this
    # batch still owns generation_state before resetting it.  Without this,
    # an all-cache-hit batch (which never sets active=True) or a batch whose
    # state was since overwritten by a concurrent stream would clobber the
    # stream's generation_state.  Mirrors the streaming guard at :729.
    batch_gen_id = str(uuid.uuid4())[:8]

    try:
        import soundfile as sf

        from qwen3_tts.core.engine import load_voice_prompt, run_inference

        # Clear any stale cancellation flag from a prior request. Without this,
        # a cancel from a previous request would immediately abort this new one
        # on the first loop iteration, returning an empty results array.
        state.generation_state["cancelled"] = False

        def _read_cache_file_b64(filepath: str) -> str:
            with open(filepath, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        # Pre-lock cache check
        pre_lock_cache_keys = {}
        pre_lock_results = {}
        for i, text in enumerate(texts):
            cache_key = _gen_cache_key(
                text,
                mode,
                gen_params,
                prompt_file=prompt_file,
                voice_description=voice_description,
                speaker=speaker,
                instruct=instruct,
                language=language,
                x_vector_only_mode=x_vector_only_mode,
                max_chunk_chars=max_chunk_chars,
                seed_lock_chunks=req.seed_lock_chunks,
            )
            pre_lock_cache_keys[i] = cache_key
            with state.gen_cache_lock:
                entry = state.gen_cache.get(cache_key)
            if entry:
                cache_file = entry.get("main_file") or entry.get("file")
                if cache_file and os.path.exists(cache_file):
                    b64_audio = await asyncio.to_thread(_read_cache_file_b64, cache_file)
                    pre_lock_results[i] = {
                        "index": i,
                        "audio_base64": b64_audio,
                        "sample_rate": entry["sample_rate"],
                        "chunks": entry.get("chunks", 0),
                        "seed": entry.get("seed"),
                    }
                    logger.info(
                        "Generation cache hit (pre-lock) for text %d/%d",
                        i + 1,
                        len(texts),
                    )

        # If ALL texts hit cache, skip the lock entirely
        if len(pre_lock_results) == len(texts):
            results = [pre_lock_results[i] for i in range(len(texts))]
            with state.request_queue_lock:
                state.request_queue.discard(request_id)
            return {"results": results}

        results = []

        for i, text in enumerate(texts):
            # Check for cancellation before each batch item (R-44)
            if state.generation_state.get("cancelled"):
                logger.info(
                    "Batch generation cancelled at item %d/%d", i + 1, len(texts)
                )
                break

            # Use pre-lock cache hit if available
            if i in pre_lock_results:
                results.append(pre_lock_results[i])
                continue

            # Post-lock cache check (does not need inference_lock — pure dict
            # lookup under gen_cache_lock)
            cache_key = pre_lock_cache_keys[i]
            with state.gen_cache_lock:
                entry = state.gen_cache.get(cache_key)
            if entry:
                cache_file = entry.get("main_file") or entry.get("file")
                if cache_file and os.path.exists(cache_file):
                    b64_audio = await asyncio.to_thread(_read_cache_file_b64, cache_file)
                    results.append(
                        {
                            "index": i,
                            "audio_base64": b64_audio,
                            "sample_rate": entry["sample_rate"],
                            "chunks": entry.get("chunks", 0),
                            "seed": entry.get("seed"),
                        }
                    )
                    logger.info(
                        "Generation cache hit (post-lock) for text %d/%d",
                        i + 1,
                        len(texts),
                    )
                continue

            # Load the voice prompt BEFORE acquiring inference_lock — disk I/O +
            # tensor deserialize is not GPU work, so holding the GPU-serialization
            # lock during it needlessly blocks every other generation. Mirrors
            # the streaming path (:611 before lock :644) and the WS path
            # (websocket.py:356 before :433).
            voice_prompt = None
            if mode == "clone":
                if not prompt_file:
                    raise HTTPException(
                        status_code=400,
                        detail="prompt_file required for clone mode",
                    )
                voice_prompt = await asyncio.to_thread(load_voice_prompt, prompt_file)
                if voice_prompt is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Voice prompt not found: {prompt_file}",
                    )

            # Acquire inference_lock ONLY for GPU-bound work: the state update
            # and the inference call itself. Everything after chunk_count
            # capture is CPU-only on the local wav array and must run with the
            # lock released so other requests can inference in parallel with our
            # encode/peaks. (generation_lock is a separate short-lived lock used
            # only for state updates.)
            async with state.inference_lock:
                # Brief lock to set generation state
                async with state.generation_lock:
                    state.generation_state.update(
                        {
                            "active": True,
                            "start_time": time.time(),
                            "text_length": len(text),
                            "mode": mode,
                            "batch_index": i,
                            "batch_total": len(texts),
                            "generation_id": batch_gen_id,
                            # Re-clear per item so a stale flag left by a
                            # concurrent request's cancel cannot truncate this
                            # batch. Mirrors the streaming-path clear at :658.
                            "cancelled": False,
                        }
                    )

                def _chunk_progress(chunk_idx, chunk_total):
                    state.generation_state.update(
                        {
                            "chunk_index": chunk_idx,
                            "chunk_total": chunk_total,
                        }
                    )

                # Apply the resolved seed for this generation. gen_params itself
                # stays seed-free for blank-seed requests so the cache key is
                # stable; the actual seed is injected only for inference.
                seeded_params = {**gen_params, "seed": used_seed}

                # Run inference: vLLM adapter takes priority when available
                vllm_adapter = getattr(request.app.state, "vllm_adapter", None)
                vllm_client = getattr(request.app.state, "vllm_client", None)

                # Check if vLLM should be used based on config and circuit state
                config = request.app.state.server_config
                vllm_enabled = config.get("vllm", {}).get("enabled", False)
                vllm_fallback_enabled = config.get("vllm", {}).get(
                    "fallback_to_torch", True
                )

                use_vllm = False
                if (
                    vllm_enabled
                    and vllm_adapter is not None
                    and vllm_client is not None
                ):
                    # Check circuit breaker state
                    circuit_state = vllm_client.circuit_state
                    if circuit_state == "CLOSED":
                        use_vllm = True
                        logger.debug(
                            "Using vLLM for generation (circuit state: CLOSED)"
                        )
                    else:
                        logger.warning(
                            f"vLLM circuit breaker state: {circuit_state}, falling back to torch/MLX"
                        )
                        if not vllm_fallback_enabled:
                            raise RuntimeError(
                                f"vLLM circuit breaker is {circuit_state} and fallback is disabled"
                            )
                elif vllm_enabled:
                    logger.info(
                        "vLLM is enabled but not available (adapter/client not initialized)"
                    )

                if use_vllm:
                    try:
                        wav, sr = await vllm_adapter.generate(
                            text=text,
                            mode=mode,
                            prompt_audio=voice_prompt,
                            voice_description=voice_description,
                            speaker=speaker,
                            **seeded_params,
                        )
                        logger.info("vLLM generation completed successfully")
                    except Exception as e:
                        logger.error(f"vLLM generation failed: {e}")
                        if vllm_fallback_enabled:
                            logger.info("Falling back to torch/MLX due to vLLM failure")
                            use_vllm = False
                        else:
                            raise RuntimeError(
                                f"vLLM generation failed and fallback is disabled: {e}"
                            ) from e

                if not use_vllm:
                    logger.debug("Using torch/MLX backend for generation")
                    wav, sr = await asyncio.to_thread(
                        run_inference,
                        model=model,
                        text=text,
                        mode=mode,
                        gen_params=seeded_params,
                        language=language,
                        voice_prompt=voice_prompt,
                        voice_description=voice_description,
                        speaker=speaker,
                        instruct=instruct,
                        max_chunk_chars=max_chunk_chars,
                        progress_callback=_chunk_progress,
                        x_vector_only_mode=x_vector_only_mode,
                        config_provider=config_provider,
                        seed_lock_chunks=req.seed_lock_chunks,
                    )

                # Capture chunk count IMMEDIATELY, while still serialized under
                # inference_lock. A later request's inference overwrites
                # generation_state["chunk_total"], so reading it after the lock
                # releases would surface another generation's chunk count. This
                # MUST stay inside the lock block.
                chunk_count = state.generation_state.get("chunk_total", 0)

            # inference_lock is now RELEASED. Everything below is CPU-only and
            # operates on the local (wav, sr) arrays returned by inference, so
            # it does not need GPU serialization and can run concurrently with
            # another request's inference.

            # Encode audio to base64 WAV in memory (off the event loop)
            buf = io.BytesIO()
            await asyncio.to_thread(sf.write, buf, wav, sr, format="WAV")
            b64_audio = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Store persistent cache file for future hits
            cache_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            cache_file.close()  # Close handle before sf.write to avoid leak
            os.chmod(cache_file.name, 0o600)
            await asyncio.to_thread(sf.write, cache_file.name, wav, sr)

            with state.gen_cache_lock:
                if len(state.gen_cache) >= get_generation_cache_max():
                    oldest_key = min(
                        state.gen_cache,
                        key=lambda k: state.gen_cache[k]["timestamp"],
                    )
                    old_entry = state.gen_cache.pop(oldest_key)
                    old_main = old_entry.get("main_file")
                    if old_main and os.path.exists(old_main):
                        try:
                            os.remove(old_main)
                        except OSError:
                            pass
                state.gen_cache[cache_key] = {
                    "main_file": cache_file.name,
                    "sample_rate": sr,
                    "timestamp": time.time(),
                    "chunks": chunk_count,
                    "seed": used_seed,
                }

            from qwen3_tts.core.engine.audio_processing import (
                calculate_waveform_peaks,
            )

            peaks = await asyncio.to_thread(
                calculate_waveform_peaks, wav, num_peaks=500
            )
            results.append(
                {
                    "index": i,
                    "audio_base64": b64_audio,
                    "sample_rate": sr,
                    "peaks": peaks,
                    "chunks": chunk_count,
                    "seed": used_seed,
                }
            )

        # Content negotiation: return binary WAV if Accept header contains audio/wav
        accept = request.headers.get("accept", "application/json")
        if "audio/wav" in accept and len(results) == 1:
            # Single text generation with audio/wav Accept: return binary WAV directly
            result = results[0]
            if result.get("audio_base64"):
                audio_bytes = base64.b64decode(result["audio_base64"])
                return Response(
                    content=audio_bytes,
                    media_type="audio/wav",
                    headers={"X-Sample-Rate": str(result["sample_rate"])},
                )

        return {"results": results}

    except HTTPException:
        raise
    except (
        RuntimeError,
        OSError,
        ValueError,
        MemoryError,
        TypeError,
        ImportError,
    ) as e:
        logger.error("Generation failed: %s", e, exc_info=True)
        _error_response(
            500,
            "Audio generation failed",
            "An internal error occurred. Check server logs for details.",
            "retry",
        )
    finally:
        # Reset generation_state ONLY if this batch still owns it.  A
        # concurrent stream may have overwritten generation_id mid-batch;
        # an all-cache-hit batch never stamped one at all.  In either case
        # resetting would clobber the other request's state.  Mirrors the
        # streaming-path guard at :729.
        if state.generation_state.get("generation_id") == batch_gen_id:
            state.generation_state.update(
                {
                    "active": False,
                    "start_time": 0.0,
                    "text_length": 0,
                    "mode": "",
                    "batch_index": 0,
                    "batch_total": 0,
                    "chunk_index": 0,
                    "chunk_total": 0,
                    "generation_id": None,
                    # A cancelled batch must not leave the shared flag dirty
                    # for the next request's cancel check (:249). Mirrors
                    # the streaming finally reset at :764.
                    "cancelled": False,
                }
            )
        with state.request_queue_lock:
            state.request_queue.discard(request_id)


async def handle_generate_stream(request, state, req, security, config_provider):
    """Core logic for the /generate-stream endpoint.

    Returns a StreamingResponse with chunked audio.

    Wire format per chunk (little-endian):
        [sample_rate: 4 bytes uint32][audio_len: 4 bytes uint32][audio: audio_len bytes float32]
    """
    # Validate request
    max_text_length = security.get("max_text_length", 50000)

    if not req.text:
        raise HTTPException(status_code=400, detail="No text provided")

    text = req.text
    if len(text) > max_text_length:
        raise HTTPException(
            status_code=400,
            detail=f"Text exceeds {max_text_length} character limit ({len(text)} chars)",
        )

    # Shared validation (path traversal, speaker, mode)
    _validate_generation_request(req, security)

    # Memory guard — prevent OOM crash. Runs after validation so a malformed
    # request is always a 400 regardless of host memory pressure.
    mem_ok, available_mb = _check_memory_available()
    if not mem_ok:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "insufficient_memory",
                "detail": f"Only {available_mb}MB available. Unload unused models to free memory.",
                "recovery": "unload",
            },
        )

    mode = req.mode

    # Check if required model is loaded
    model = state.models.get(mode)
    if model is None:
        error_msg = state.model_load_errors.get(mode, "Model not loaded")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_loaded",
                "message": error_msg,
                "model_type": mode,
            },
        )

    # Generation parameters
    gen_params = {
        "temperature": req.temperature,
        "top_k": req.top_k,
        "top_p": req.top_p,
        "repetition_penalty": req.repetition_penalty,
        "max_new_tokens": req.max_new_tokens,
    }
    if req.seed is not None:
        gen_params["seed"] = req.seed

    # Resolve the actual seed and apply it to inference so streaming generations
    # are reproducible and can be reported to the client (via the X-Seed header
    # below), matching the /generate batch path.
    used_seed = _resolve_generation_seed(req.seed)
    seeded_params = {**gen_params, "seed": used_seed}

    # Prepare mode-specific params
    from qwen3_tts.core.engine import load_voice_prompt

    voice_prompt = None
    if mode == "clone":
        prompt_file = req.prompt_file
        if not prompt_file:
            raise HTTPException(
                status_code=400, detail="prompt_file required for clone mode"
            )
        voice_prompt = await asyncio.to_thread(load_voice_prompt, prompt_file)
        if voice_prompt is None:
            raise HTTPException(
                status_code=404, detail=f"Voice prompt not found: {prompt_file}"
            )

    voice_description = req.voice_description
    language = req.language
    speaker = req.speaker
    instruct = req.instruct
    x_vector_only_mode = req.x_vector_only_mode

    # Create queue for streaming chunks
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    stop_event = threading.Event()
    inference_lock = state.inference_lock

    # Register in pending queue
    queue_entry = {
        "id": str(uuid.uuid4())[:8],
        "text_preview": text[:40],
        "mode": mode,
        "queued_at": time.time(),
    }

    async def audio_stream_generator():
        """Async generator that yields audio chunks."""
        # Track in pending queue while waiting for inference lock
        async with state.pending_lock:
            state.pending_requests.append(queue_entry)

        # Acquire inference_lock to serialize GPU access
        async with inference_lock:
            # Remove from pending queue once we have the lock
            async with state.pending_lock:
                if queue_entry in state.pending_requests:
                    state.pending_requests.remove(queue_entry)

            gen_id = str(uuid.uuid4())[:8]
            state.generation_state.update(
                {
                    "active": True,
                    "start_time": time.time(),
                    "text_length": len(text),
                    "mode": mode,
                    "generation_id": gen_id,
                    "cancelled": False,
                }
            )

            def _chunk_progress(chunk_idx, chunk_total):
                """Update generation_state with chunk progress from streaming callback."""
                state.generation_state.update(
                    {
                        "chunk_index": chunk_idx,
                        "chunk_total": chunk_total,
                    }
                )

            def inference_thread():
                """Run inference in a thread and push chunks to queue."""
                try:
                    from qwen3_tts.core.engine import run_inference_streaming

                    for wav_chunk, sr in run_inference_streaming(
                        model=model,
                        text=text,
                        mode=mode,
                        gen_params=seeded_params,
                        language=language,
                        voice_prompt=voice_prompt,
                        voice_description=voice_description,
                        speaker=speaker,
                        instruct=instruct,
                        x_vector_only_mode=x_vector_only_mode,
                        max_chunk_chars=req.max_chunk_chars,
                        config_provider=config_provider,
                        progress_callback=_chunk_progress,
                    ):
                        if _should_stop_streaming(
                            stop_event, state.generation_state
                        ):
                            logger.info("Generation cancelled by user")
                            break

                        # Length-prefixed format: [sample_rate:4][length:4][audio:length]
                        audio_bytes = wav_chunk.astype("<f4").tobytes()
                        header = struct.pack("<II", sr, len(audio_bytes))

                        # Use call_soon_threadsafe to safely put from thread to async queue
                        loop.call_soon_threadsafe(
                            queue.put_nowait, header + audio_bytes
                        )

                except Exception as e:
                    # Broad catch: any exception outside the old narrow tuple
                    # (e.g. AttributeError) would otherwise kill the thread
                    # without the None sentinel, deadlocking the consumer.
                    thread_error[0] = str(e)
                    logger.error("Streaming inference failed: %s", e, exc_info=True)
                finally:
                    # Signal the consumer that the thread has fully stopped.
                    # Separate from the queue-None sentinel (which means "no
                    # more chunks"); done means "thread finished" and is awaited
                    # by the consumer's finally BEFORE releasing inference_lock.
                    done.set()
                    # Always send the completion sentinel so the consumer never
                    # blocks forever on queue.get() — even on unexpected errors
                    # (deadlock fix, H3).
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            # thread_error holds the stringified failure if the thread caught one
            # (closure-captured by the thread and read by the consumer after it
            # awaits done). chunk_count tracks delivered chunks so a pre-chunk
            # error can be surfaced instead of a silent empty 200.
            thread_error: list[str | None] = [None]
            chunk_count = 0
            # Event signals the inference thread has fully stopped; the consumer
            # awaits it in its finally BEFORE releasing inference_lock so an
            # in-flight model.generate() cannot race the next request.
            done = threading.Event()
            # Start inference thread
            thread = threading.Thread(target=inference_thread, daemon=True)
            thread.start()

            try:
                # Yield chunks as they arrive
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    yield chunk
                    chunk_count += 1
            finally:
                stop_event.set()
                join_timeout = _stream_thread_join_timeout(
                    len(text), req.max_chunk_chars
                )
                await _await_inference_thread_done(done, timeout=join_timeout)
                if not done.is_set():
                    logger.error(
                        "streaming inference thread did not stop within %ss; "
                        "releasing inference_lock",
                        join_timeout,
                    )
                # Reset generation state if still our generation
                if state.generation_state.get("generation_id") == gen_id:
                    state.generation_state.update(
                        {
                            "active": False,
                            "start_time": 0.0,
                            "text_length": 0,
                            "mode": "",
                            "chunk_index": 0,
                            "chunk_total": 0,
                            "generation_id": None,
                            "cancelled": False,
                        }
                    )
            # Terminal error frame (WS2 Task 2.5). Starlette commits the 200
            # headers before the body is iterated, so once streaming starts we
            # cannot signal failure with a status code. Raising here would just
            # truncate the connection, which the client cannot distinguish from
            # a network drop and which carries no error context. Instead emit an
            # in-band terminal frame — sample_rate 0 is never valid for real
            # audio — and let the stream end cleanly.
            #
            # This fires whether or not chunks were already delivered: a
            # mid-stream failure used to be dropped entirely, so the client
            # accepted truncated audio as a complete generation.
            # On client disconnect the finally returns early via
            # GeneratorExit/aclose and this code is skipped.
            if thread_error[0] is not None:
                yield encode_stream_error_frame(thread_error[0])

    return StreamingResponse(
        audio_stream_generator(),
        media_type="application/octet-stream",
        headers={
            "X-Content-Type": "audio/raw-float32",
            # Actual seed used for this generation (server-generated when the
            # caller supplied none), so the client can record/reuse it.
            "X-Seed": str(used_seed),
            # Approximate: read without lock since response is already committed.
            # Exact position available via /queue-status endpoint.
            "X-Queue-Position": str(len(state.pending_requests)),
        },
    )
