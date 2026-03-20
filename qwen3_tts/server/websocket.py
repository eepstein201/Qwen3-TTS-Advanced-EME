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
import json
import logging
import struct
import threading

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("tts.server.websocket")


async def websocket_tts_handler(
    websocket: WebSocket,
    app_state,
    verify_token_fn,
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
    await websocket.accept()

    # Step 1: Authenticate
    try:
        auth_msg = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        auth_data = json.loads(auth_msg)
        token = auth_data.get("token", "")
        if not verify_token_fn(token):
            await websocket.send_json({"error": "Authentication failed"})
            await websocket.close(code=4001, reason="Authentication failed")
            return
        await websocket.send_json({"status": "authenticated"})
    except (asyncio.TimeoutError, json.JSONDecodeError, WebSocketDisconnect):
        await websocket.close(code=4001, reason="Authentication failed")
        return

    # Step 2: Message loop
    stop_event = threading.Event()

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
                stop_event.set()
                await websocket.send_json({"status": "cancelled"})
                stop_event.clear()
                continue

            # Handle generation request
            text = data.get("text", "")
            if not text:
                await websocket.send_json({"error": "No text provided"})
                continue

            mode = data.get("mode", "clone")
            stop_event.clear()

            try:
                await _stream_generation(
                    websocket=websocket,
                    app_state=app_state,
                    text=text,
                    mode=mode,
                    data=data,
                    stop_event=stop_event,
                )
            except Exception as e:
                logger.error("WebSocket generation error: %s", e, exc_info=True)
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket handler error: %s", e, exc_info=True)
    finally:
        stop_event.set()


async def _stream_generation(
    websocket: WebSocket,
    app_state,
    text: str,
    mode: str,
    data: dict,
    stop_event: threading.Event,
) -> None:
    """Run TTS generation and stream audio chunks over WebSocket.

    Args:
        websocket: The WebSocket connection.
        app_state: FastAPI app state.
        text: Input text to synthesize.
        mode: Generation mode (clone/design/custom).
        data: Full request data dict.
        stop_event: Event to signal cancellation.
    """
    model = app_state.models.get(mode)
    if model is None:
        await websocket.send_json({"error": f"Model '{mode}' not loaded"})
        return

    # Validate request (path traversal, speaker, mode) — same checks as HTTP endpoints
    try:
        from fastapi import HTTPException
        from qwen3_tts.server.validation import GenerateRequest, _validate_generation_request
        req = GenerateRequest(
            text=text,
            mode=mode,
            prompt_file=data.get("prompt_file"),
            speaker=data.get("speaker"),
            voice_description=data.get("voice_description", ""),
            instruct=data.get("instruct", ""),
        )
        security = app_state.server_config.get("security", {}) if hasattr(app_state, "server_config") else {}
        _validate_generation_request(req, security)
    except HTTPException as e:
        detail = e.detail if isinstance(e.detail, str) else e.detail.get("detail", str(e.detail))
        await websocket.send_json({"error": detail})
        return

    await websocket.send_json({"status": "generating", "text_length": len(text)})

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def inference_thread():
        try:
            from qwen3_tts.core.engine import run_inference_streaming

            gen_params = {
                "temperature": data.get("temperature", 0.7),
                "top_k": data.get("top_k", 50),
                "top_p": data.get("top_p", 0.95),
                "repetition_penalty": data.get("repetition_penalty", 1.05),
                "max_new_tokens": data.get("max_new_tokens", 2048),
            }

            voice_prompt = None
            if mode == "clone":
                from qwen3_tts.core.engine import load_voice_prompt
                prompt_file = data.get("prompt_file")
                if prompt_file:
                    voice_prompt = load_voice_prompt(prompt_file)

            for wav_chunk, sr in run_inference_streaming(
                model=model,
                text=text,
                mode=mode,
                gen_params=gen_params,
                language=data.get("language", "English"),
                voice_prompt=voice_prompt,
                voice_description=data.get("voice_description"),
                speaker=data.get("speaker"),
                instruct=data.get("instruct"),
                x_vector_only_mode=data.get("x_vector_only_mode", False),
            ):
                if stop_event.is_set():
                    break

                audio_bytes = wav_chunk.astype("<f4").tobytes()
                header = struct.pack("<II", sr, len(audio_bytes))
                loop.call_soon_threadsafe(queue.put_nowait, header + audio_bytes)

        except Exception as e:
            logger.error("WebSocket inference thread error: %s", e, exc_info=True)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    thread = threading.Thread(target=inference_thread, daemon=True)
    thread.start()

    chunk_count = 0
    try:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            await websocket.send_bytes(chunk)
            chunk_count += 1
    except WebSocketDisconnect:
        stop_event.set()
        return

    await websocket.send_json({
        "status": "complete",
        "chunks": chunk_count,
    })
