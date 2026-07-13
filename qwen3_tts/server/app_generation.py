"""Audio generation endpoint handlers — /generate and /generate-stream.

Extracted from app.py to keep each module under 800 lines.
These are plain async functions called by thin endpoint wrappers in app.py.
"""

import asyncio
import base64
import io
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
    _error_response,
    _gen_cache_key,
    _validate_generation_request,
)

logger = logging.getLogger("tts")

# Upper bound for a server-generated seed. Kept within signed int32 so it is
# safe across torch/MLX seeding APIs and JSON-serializes cleanly.
_MAX_SEED = 2**31 - 1


def _resolve_generation_seed(req_seed: int | None) -> int:
    """Return the seed to use for generation.

    When the caller supplies a seed (including 0), it is used verbatim so the
    request stays reproducible. When none is given, a random seed is generated
    server-side so the value can be recorded in history and reused later — the
    UI otherwise has no way to know which seed produced a "random" generation.
    """
    if req_seed is not None:
        return req_seed
    return secrets.randbelow(_MAX_SEED + 1)


def _should_stop_streaming(stop_event, generation_state) -> bool:
    """Return True if streaming generation should stop.

    Stops when either the client disconnected (``stop_event`` set by the
    generator's finally) or the user cancelled via /cancel-generation
    (``generation_state['cancelled']``), matching the batch path's cancel check.
    """
    return stop_event.is_set() or bool(generation_state.get("cancelled"))


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
    max_text_length = security.get("max_text_length", 10000)
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

    try:
        import soundfile as sf

        from qwen3_tts.core.engine import load_voice_prompt, run_inference

        # Clear any stale cancellation flag from a prior request. Without this,
        # a cancel from a previous request would immediately abort this new one
        # on the first loop iteration, returning an empty results array.
        state.generation_state["cancelled"] = False

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
            )
            pre_lock_cache_keys[i] = cache_key
            with state.gen_cache_lock:
                entry = state.gen_cache.get(cache_key)
            if entry:
                cache_file = entry.get("main_file") or entry.get("file")
                if cache_file and os.path.exists(cache_file):
                    with open(cache_file, "rb") as f:
                        b64_audio = base64.b64encode(f.read()).decode("utf-8")
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

        # Acquire inference_lock for GPU serialization (generation_lock used only for state updates)
        async with state.inference_lock:
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

                # Post-lock cache check
                cache_key = pre_lock_cache_keys[i]
                with state.gen_cache_lock:
                    entry = state.gen_cache.get(cache_key)
                if entry:
                    cache_file = entry.get("main_file") or entry.get("file")
                    if cache_file and os.path.exists(cache_file):
                        with open(cache_file, "rb") as f:
                            b64_audio = base64.b64encode(f.read()).decode("utf-8")
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
                        }
                    )

                # Prepare mode-specific params
                voice_prompt = None
                if mode == "clone":
                    if not prompt_file:
                        raise HTTPException(
                            status_code=400,
                            detail="prompt_file required for clone mode",
                        )
                    voice_prompt = await asyncio.to_thread(
                        load_voice_prompt, prompt_file
                    )
                    if voice_prompt is None:
                        raise HTTPException(
                            status_code=404,
                            detail=f"Voice prompt not found: {prompt_file}",
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

                # Capture chunk count now (still serialized under inference_lock)
                # so the value reported and cached is this generation's, not a
                # later request's, from the shared generation_state.
                chunk_count = state.generation_state.get("chunk_total", 0)

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
        # Clear generation state
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
    max_text_length = security.get("max_text_length", 10000)

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

                except (RuntimeError, OSError, ValueError, MemoryError, TypeError) as e:
                    logger.error("Streaming inference failed: %s", e, exc_info=True)
                    loop.call_soon_threadsafe(queue.put_nowait, None)
                else:
                    # Signal completion
                    loop.call_soon_threadsafe(queue.put_nowait, None)

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
            finally:
                stop_event.set()
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
