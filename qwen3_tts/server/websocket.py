"""WebSocket endpoint for bidirectional audio streaming.

Provides a WebSocket handler that accepts text input and streams
TTS audio chunks back in real-time, enabling low-latency bidirectional
communication between client and server.

Wire format (server → client):
    Binary frames: [sample_rate:4 bytes LE uint32][length:4 bytes LE uint32][audio:length bytes float32 LE]

Wire format (client → server):
    JSON text frames: {"text": "...", "mode": "clone|design|custom", ...}
    Control frames: {"action": "cancel"} to abort current generation
"""

import asyncio
import contextlib
import json
import logging
import struct
import threading
import time
import uuid

from fastapi import WebSocket, WebSocketDisconnect

from qwen3_tts.core.config import sanitize_log
from qwen3_tts.server.app_generation import (
    _await_inference_thread_done,
    _resolve_generation_seed,
    _stream_thread_join_timeout,
)

logger = logging.getLogger("tts.server.websocket")

# Concurrent WebSocket connection caps. accept() precedes auth, so an
# unauthenticated client could otherwise pin file descriptors by flooding /ws.
_WS_MAX_PER_IP = 5
_WS_MAX_TOTAL = 50
_ws_conn_lock = threading.Lock()


def _ws_try_acquire(app_state, client_ip: str) -> bool:
    """Reserve a WebSocket slot for ``client_ip``; False if over per-IP/global cap."""
    with _ws_conn_lock:
        conns = getattr(app_state, "_ws_connections", None)
        if conns is None:
            conns = {}
            app_state._ws_connections = conns
        if sum(conns.values()) >= _WS_MAX_TOTAL:
            return False
        if conns.get(client_ip, 0) >= _WS_MAX_PER_IP:
            return False
        conns[client_ip] = conns.get(client_ip, 0) + 1
        return True


def _ws_release(app_state, client_ip: str) -> None:
    """Release a previously reserved WebSocket slot for ``client_ip``."""
    with _ws_conn_lock:
        conns = getattr(app_state, "_ws_connections", None)
        if not conns or client_ip not in conns:
            return
        conns[client_ip] -= 1
        if conns[client_ip] <= 0:
            del conns[client_ip]


async def websocket_tts_handler(
    websocket: WebSocket,
    app_state,
    verify_token_fn,
    config_provider=None,
) -> None:
    """Handle a WebSocket connection for bidirectional TTS streaming.

    Protocol:
    1. Client connects and sends auth token as first text message
    2. Client sends JSON generation requests
    3. Server streams binary audio chunks back
    4. Client can send {"action": "cancel"} to abort

    Args:
        websocket: The FastAPI WebSocket connection.
        app_state: The FastAPI app.state with models, config, etc.
        verify_token_fn: Function to validate auth tokens.
    """
    # Cap concurrent connections BEFORE accepting, so a flood can't pin FDs
    # through the pre-auth window.
    client_ip = websocket.client.host if websocket.client else "unknown"
    if not _ws_try_acquire(app_state, client_ip):
        logger.warning("WebSocket connection rejected (limit reached) from %s", client_ip)
        await websocket.close(code=1013, reason="Too many connections")
        return

    await websocket.accept()

    # Step 1: Authenticate
    try:
        auth_msg = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        auth_data = json.loads(auth_msg)
        # A valid-JSON non-object payload (e.g. "42", "[1]") must be rejected
        # explicitly. Without this guard, auth_data.get() below raised
        # AttributeError, which escaped the (previously narrow) except clause
        # and skipped _ws_release — leaking a connection slot. Repeating 50x
        # exhausted _WS_MAX_TOTAL (unauthenticated slot-exhaustion DoS).
        if not isinstance(auth_data, dict):
            logger.warning(
                "WebSocket auth failed from %s: first message not a JSON object",
                sanitize_log(client_ip),
            )
            await websocket.close(code=4001, reason="Authentication failed")
            _ws_release(app_state, client_ip)
            return
        token = auth_data.get("token", "")
        if not verify_token_fn(token):
            logger.warning(
                "WebSocket auth failed from %s: invalid token",
                sanitize_log(client_ip),
            )
            await websocket.send_json({"error": "Authentication failed"})
            await websocket.close(code=4001, reason="Authentication failed")
            _ws_release(app_state, client_ip)
            return
        await websocket.send_json({"status": "authenticated"})
    except Exception as e:
        # Any auth-path failure (timeout, malformed JSON, disconnect, or an
        # unexpected error) must release the reserved slot. Broad catch
        # guarantees no leak regardless of payload shape.
        logger.warning(
            "WebSocket auth failed from %s: %s",
            sanitize_log(client_ip),
            sanitize_log(e),
            exc_info=True,
        )
        await websocket.close(code=4001, reason="Authentication failed")
        _ws_release(app_state, client_ip)
        return

    # Step 2: Message loop
    stop_event = threading.Event()
    # Distinct from stop_event: set ONLY when the concurrent cancel-watcher
    # observes a WebSocketDisconnect, so the consumer can tell a real cancel
    # (stop_event alone) from a client disconnect (stop_event + disconnect_event)
    # and avoid emitting a terminal "cancelled" frame on the dead socket.
    disconnect_event = threading.Event()

    try:
        while True:
            try:
                message = await websocket.receive_text()
            except WebSocketDisconnect:
                break

            # Reject oversized messages (64KB limit)
            if len(message) > 65536:
                await websocket.send_json({"error": "Message too large (max 64KB)"})
                continue

            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            # Handle control messages
            action = data.get("action")
            if action == "cancel":
                # Set the flag and leave it set. Clearing it here (as this once
                # did) raced the inference thread's stop_event.is_set() check:
                # the flag could be cleared microseconds after being set,
                # before inference observed it — losing the cancel. The flag is
                # cleared at the start of the next generation instead.
                stop_event.set()
                await websocket.send_json({"status": "cancelled"})
                continue

            # Handle generation request
            text = data.get("text", "")
            if not text:
                await websocket.send_json({"error": "No text provided"})
                continue

            # Enforce max text length (matches HTTP endpoint validation)
            security = (
                app_state.server_config.get("security", {})
                if hasattr(app_state, "server_config")
                else {}
            )
            max_text_length = security.get("max_text_length", 50000)
            if len(text) > max_text_length:
                await websocket.send_json(
                    {"error": f"Text exceeds {max_text_length} character limit"}
                )
                continue

            mode = data.get("mode", "clone")
            stop_event.clear()

            # Concurrent cancel-watcher: the main loop is blocked inside
            # _stream_generation for the duration of generation, so without a
            # watcher a {"action":"cancel"} frame sent mid-generation is never
            # read until generation finishes.  The watcher is the SOLE
            # receive_text reader during generation (the main loop is blocked
            # in the await below), so there is no concurrent-reader race.  On
            # normal completion the finally cancels and reaps the watcher.
            async def _cancel_watcher() -> None:
                try:
                    while True:
                        msg = await websocket.receive_text()
                        try:
                            frame = json.loads(msg)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(frame, dict) and frame.get("action") == "cancel":
                            stop_event.set()
                            return
                        # Other frames mid-generation: ignore (client protocol
                        # violation) — the main loop will resume after gen.
                except WebSocketDisconnect:
                    # The client went away. Signal disconnect BEFORE stop_event
                    # so the consumer can distinguish a real cancel (stop_event
                    # alone) from a disconnect and skip the terminal frame it
                    # would otherwise send on the dead socket.
                    disconnect_event.set()
                    stop_event.set()
                    return
                except Exception:
                    return  # a watcher error must never abort generation

            watcher = asyncio.create_task(_cancel_watcher())
            try:
                await _stream_generation(
                    websocket=websocket,
                    app_state=app_state,
                    text=text,
                    mode=mode,
                    data=data,
                    stop_event=stop_event,
                    disconnect_event=disconnect_event,
                    config_provider=config_provider,
                )
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                # Client validation errors - bad request data
                logger.error("WebSocket generation request error: %s", e, exc_info=True)
                await websocket.send_json({"error": f"Invalid request: {str(e)}"})
            except (ConnectionError, OSError) as e:
                # Network/connection issues
                logger.error("WebSocket connection error: %s", e, exc_info=True)
                await websocket.send_json({"error": "Connection error"})
            except Exception as e:
                # Unexpected errors during generation
                logger.error("WebSocket generation error: %s", e, exc_info=True)
                from qwen3_tts.server.app_lifespan import _sanitize_error

                await websocket.send_json({"error": _sanitize_error(str(e))})
                # RFC 6455 §7.4.1: 1011 means the server hit an unexpected
                # condition. Without an explicit close code the socket ends as a
                # normal 1000, so a client that missed the error message reads a
                # server-side failure as a clean finish (WS2 2.5, the WebSocket
                # counterpart of the HTTP terminal error frame).
                with contextlib.suppress(RuntimeError):
                    await websocket.close(
                        code=1011, reason="Generation failed"
                    )
            finally:
                watcher.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await watcher

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except (asyncio.TimeoutError, RuntimeError) as e:
        # Async/await and runtime errors in WebSocket lifecycle
        logger.error("WebSocket lifecycle error: %s", e, exc_info=True)
    except Exception as e:
        # Unexpected errors in WebSocket handler
        logger.error("WebSocket handler error: %s", e, exc_info=True)
    finally:
        stop_event.set()
        _ws_release(app_state, client_ip)


async def _stream_generation(
    websocket: WebSocket,
    app_state,
    text: str,
    mode: str,
    data: dict,
    stop_event: threading.Event,
    disconnect_event: threading.Event,
    config_provider=None,
) -> None:
    """Run TTS generation and stream audio chunks over WebSocket.

    Args:
        websocket: The WebSocket connection.
        app_state: FastAPI app state.
        text: Input text to synthesize.
        mode: Generation mode (clone/design/custom).
        data: Full request data dict.
        stop_event: Event to signal cancellation.
        disconnect_event: Event set by the concurrent cancel-watcher when the
            client disconnects, so the outcome is reported as a disconnect
            rather than misread as a cancel.
    """
    model = app_state.models.get(mode)
    if model is None:
        await websocket.send_json({"error": f"Model '{mode}' not loaded"})
        return

    # Validate request (path traversal, speaker, mode) — same checks as HTTP endpoints
    try:
        from fastapi import HTTPException
        from pydantic import ValidationError

        from qwen3_tts.server.validation import (
            GenerateRequest,
            _validate_generation_request,
        )

        # Construct the FULL request so Pydantic validates every generation
        # parameter (types + ranges). Pre-fix, only 6 fields were validated and
        # temperature/top_k/top_p/repetition_penalty/max_new_tokens/seed flowed
        # from the raw `data` dict unvalidated — letting a client send
        # max_new_tokens=2_147_483_647 and monopolize the inference_lock.
        try:
            req = GenerateRequest(
                text=text,
                mode=mode,
                prompt_file=data.get("prompt_file"),
                speaker=data.get("speaker"),
                voice_description=data.get("voice_description", ""),
                instruct=data.get("instruct", ""),
                temperature=data.get("temperature", 0.7),
                top_k=data.get("top_k", 50),
                top_p=data.get("top_p", 0.95),
                repetition_penalty=data.get("repetition_penalty", 1.05),
                max_new_tokens=data.get("max_new_tokens", 2048),
                seed=data.get("seed"),
                language=data.get("language", "auto"),
                max_chunk_chars=data.get("max_chunk_chars"),
                x_vector_only_mode=data.get("x_vector_only_mode", False),
                seed_lock_chunks=data.get("seed_lock_chunks", False),
            )
        except ValidationError as e:
            # Keep the details in locals rather than falling back to a bare {},
            # which mypy rejects as an incomplete pydantic ``ErrorDetails``.
            errors = e.errors()
            field = "parameter"
            reason = "validation failed"
            if errors:
                first = errors[0]
                field = ".".join(str(p) for p in first.get("loc", ())) or field
                reason = first.get("msg", reason)
            await websocket.send_json({"error": f"Invalid {field}: {reason}"})
            return
        security = (
            app_state.server_config.get("security", {})
            if hasattr(app_state, "server_config")
            else {}
        )
        _validate_generation_request(req, security)
    except HTTPException as e:
        detail = (
            e.detail
            if isinstance(e.detail, str)
            else e.detail.get("detail", str(e.detail))
        )
        await websocket.send_json({"error": detail})
        return

    # Clone-mode prompt validation — mirror HTTP 400/404 (app_generation.py).
    # MUST precede the "generating" frame so a missing prompt never looks like
    # generation started (false success). Loads the prompt here so the inference
    # thread can consume it via closure without a redundant load.
    voice_prompt = None
    if mode == "clone":
        from qwen3_tts.server.prompt_loading import load_voice_prompt_serialized

        if not req.prompt_file:
            await websocket.send_json(
                {"error": "prompt_file required for clone mode"}
            )
            return
        try:
            voice_prompt = await load_voice_prompt_serialized(
                app_state, req.prompt_file
            )
        except FileNotFoundError as e:
            # MLX loader raises (torch returns None) — report like the HTTP 404.
            await websocket.send_json({"error": str(e)})
            return
        if voice_prompt is None:
            await websocket.send_json(
                {"error": f"Voice prompt not found: {req.prompt_file}"}
            )
            return

    # Resolve the actual seed (server-generated when the client sends none) so it
    # can be applied to inference and reported in the completion message,
    # matching the /generate and /generate-stream paths.
    used_seed = _resolve_generation_seed(req.seed)

    from qwen3_tts.server.app_lifespan import _check_memory_available
    mem_ok, available_mb = _check_memory_available()
    if not mem_ok:
        await websocket.send_json({"status": "error",
            "detail": f"Insufficient memory: only {available_mb}MB available. Unload unused models to free memory."})
        return

    await websocket.send_json({"status": "generating", "text_length": len(text)})

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def inference_thread():
        try:
            from qwen3_tts.core.engine import run_inference_streaming

            # Derive gen_params from the validated request, never from raw data.
            gen_params = {
                "temperature": req.temperature,
                "top_k": req.top_k,
                "top_p": req.top_p,
                "repetition_penalty": req.repetition_penalty,
                "max_new_tokens": req.max_new_tokens,
                "seed": used_seed,
            }

            # voice_prompt is loaded and validated before the "generating"
            # frame (above); the thread consumes it via closure.
            for wav_chunk, sr in run_inference_streaming(
                model=model,
                text=text,
                mode=mode,
                gen_params=gen_params,
                language=req.language,
                voice_prompt=voice_prompt,
                voice_description=req.voice_description,
                speaker=req.speaker,
                instruct=req.instruct,
                x_vector_only_mode=req.x_vector_only_mode,
                max_chunk_chars=req.max_chunk_chars,
                config_provider=config_provider,
            ):
                if stop_event.is_set():
                    break

                audio_bytes = wav_chunk.astype("<f4").tobytes()
                header = struct.pack("<II", sr, len(audio_bytes))
                loop.call_soon_threadsafe(queue.put_nowait, header + audio_bytes)

        except (RuntimeError, ValueError, AttributeError, OSError) as e:
            # Model inference, audio processing, or file operation errors
            logger.error("WebSocket inference thread error: %s", e, exc_info=True)
            thread_error[0] = str(e)
        except Exception as e:
            # Unexpected errors in inference thread
            logger.error(
                "WebSocket inference thread unexpected error: %s", e, exc_info=True
            )
            thread_error[0] = str(e)
        finally:
            # Signal the consumer that the thread has fully stopped BEFORE the
            # queue-None sentinel. done is separate from None (None = "no more
            # chunks"; done = "thread finished"); the consumer awaits done in
            # its finally to hold inference_lock until the thread stops.
            done.set()
            loop.call_soon_threadsafe(queue.put_nowait, None)

    ws_gen_id = str(uuid.uuid4())[:8]

    async with app_state.inference_lock:
        # T5: re-read the slot under the lock — /ws captures the model into
        # a local long before this acquire, so an unload landing in between
        # must surface as a retryable 503-shaped error frame, never as an
        # orphan generation.
        from qwen3_tts.server.app_generation import _require_model_under_lock

        _require_model_under_lock(app_state, mode)

        # Mark this generation active in the shared generation_state so the
        # public /generation-status, /cancel-generation, and
        # detect_degraded_generation() see WebSocket work — without this the WS
        # path is invisible to the HTTP control plane (an HTTP /generate would
        # queue behind an unseen job, and the degraded-generation watchdog could
        # not see a runaway WS generation). generation_lock is nested inside
        # inference_lock (same order as app_generation.py:291-304) and held only
        # for the dict mutation.
        async with app_state.generation_lock:
            app_state.generation_state.update(
                {
                    "active": True,
                    "start_time": time.time(),
                    "text_length": len(text),
                    "mode": mode,
                    "generation_id": ws_gen_id,
                    "cancelled": False,
                }
            )
        # Event signals the inference thread has fully stopped; the consumer
        # awaits it in its finally BEFORE releasing inference_lock so an
        # in-flight model.generate() cannot race the next request.
        done = threading.Event()
        # Holder for an exception string captured by the inference thread's
        # except clauses. None = no error; non-None = terminal frame reports
        # status=="error" instead of false success.
        thread_error: list[str | None] = [None]
        thread = threading.Thread(target=inference_thread, daemon=True)
        thread.start()

        chunk_count = 0
        was_cancelled = False
        disconnected = False
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                await websocket.send_bytes(chunk)
                chunk_count += 1
        except WebSocketDisconnect:
            disconnected = True
        finally:
            # A disconnect may be observed two ways: this consumer's own
            # send_bytes raising WebSocketDisconnect (local ``disconnected``),
            # or the concurrent cancel-watcher, which signals via
            # disconnect_event. Either way a disconnect must NOT be reported as
            # a cancel — checked before the unconditional set below so a normal
            # completion isn't misread either.
            if disconnect_event.is_set():
                disconnected = True
            if stop_event.is_set() and not disconnected:
                was_cancelled = True
            # stop_event must be set BEFORE awaiting done so the thread breaks
            # out of its generate loop; awaiting done first risks deadlock if
            # the thread is mid-chunk and waiting for the stop signal.
            stop_event.set()
            # The join must cover ONE chunk's generation, so it scales with the
            # text length and the configured chunk size — same rule as the HTTP
            # streaming path (app_generation.py). A flat timeout sized for the
            # 500-char default expires mid-generation once max_chunk_chars is
            # raised, releasing inference_lock while model.generate() is still
            # on the GPU. Never reintroduce a constant here.
            join_timeout = _stream_thread_join_timeout(
                len(text), req.max_chunk_chars
            )
            finished = await _await_inference_thread_done(done, timeout=join_timeout)
            if not finished:
                logger.error(
                    "WebSocket inference thread did not stop within %ss; "
                    "releasing inference_lock",
                    join_timeout,
                )
            # Release the shared generation_state slot, but only if this
            # generation still owns it (a concurrent request may have
            # overwritten generation_id). Mirrors app_generation.py:514/754.
            async with app_state.generation_lock:
                if app_state.generation_state.get("generation_id") == ws_gen_id:
                    app_state.generation_state.update(
                        {
                            "active": False,
                            "start_time": 0.0,
                            "text_length": 0,
                            "mode": "",
                            "chunk_index": 0,
                            "chunk_total": 0,
                            "cancelled": False,
                            "generation_id": None,
                        }
                    )

    # Terminal frame — branch on outcome (outside inference_lock so the lock
    # isn't held while sending the final JSON). Pre-fix this was an
    # unconditional {"status": "complete"} that reported false success even
    # when the inference thread excepted or the prompt was missing.
    if disconnected:
        return
    if was_cancelled:
        await websocket.send_json(
            {"status": "cancelled", "chunks": chunk_count, "seed": used_seed}
        )
    elif thread_error[0] is not None:
        from qwen3_tts.server.app_lifespan import _sanitize_error

        await websocket.send_json(
            {
                "status": "error",
                "detail": _sanitize_error(thread_error[0]),
                "chunks": chunk_count,
                "seed": used_seed,
            }
        )
    else:
        await websocket.send_json(
            {
                "status": "complete",
                "chunks": chunk_count,
                "seed": used_seed,
            }
        )
