#!/usr/bin/env python3
"""Serialize the torch auto-create-from-.wav inference path (#214 item 1).

Module scope imports asyncio ONLY -- no torch/mlx/transformers (lazy-import
project rule). All engine imports are function-local (also preserves
patchability at the call sites this module's callers use).
"""

import asyncio


async def load_voice_prompt_serialized(state, prompt_file: str):
    """Load a voice prompt; serialize the torch auto-create inference (#214 item 1).

    Fast path is a plain unlocked disk load. Only when the torch backend must BUILD the
    prompt -- real create inference -- do we re-enter under inference_lock as a leaf
    acquisition.

    load_model() is NOT memoized (model_loader.py:552) -- every call is a full multi-minute
    weight construction. So we reuse the already-loaded clone model when there is one,
    otherwise build it OUTSIDE the lock, and forward it via clone_model= so the locked
    section runs create inference ONLY. Dropping that forwarding puts weight construction
    back under the lock -- the exact starvation the #212 split exists to prevent.

    On MLX this is a no-op: load_voice_prompt_mlx never creates.
    """
    from qwen3_tts.core.engine import VoicePromptCreateRequired, load_voice_prompt

    try:
        return await asyncio.to_thread(load_voice_prompt, prompt_file, allow_create=False)
    except VoicePromptCreateRequired:
        pass

    model = state.models.get("clone")
    came_from_slot = model is not None
    if model is None:
        # Defensive only: all three server call sites 503 earlier on a None clone
        # model, so this build runs just if the model was unloaded mid-flight.
        # Not the hot path -- never memoized (see model_loader.load_model).
        from qwen3_tts.core.engine.model_loader import load_model

        model = await asyncio.to_thread(load_model, "clone", warmup=False)

    async with state.inference_lock:  # leaf acquisition
        # T5 sibling: the clone slot above was captured BEFORE this acquire, and
        # the wait here is a whole queued generation -- the widest
        # capture->acquire window in the server. The locked call below is REAL
        # create inference, so re-read the slot here and rebind: an
        # unload->RELOAD in the window leaves the slot non-None, so a null check
        # alone would still build the prompt on the ORPHANED pre-unload object
        # (unload_model_cleanup is gc.collect + a cache flush -- it never
        # destroys weights, so the orphan silently keeps working).
        current = state.models.get("clone")  # .get(): partial-dict states exist
        if came_from_slot:
            # The model we hold IS the slot's -- an empty slot now means a real
            # unload landed in the window. Same classified, retryable 503 the
            # generation paths raise. Function-local import: same precedent as
            # websocket.py, and it keeps this module free of a module-level
            # dependency on a sibling handler.
            from qwen3_tts.server.app_generation import _require_model_under_lock

            model = _require_model_under_lock(state, "clone")
        elif current is not None:
            # Fallback branch: load_model() returns the model and never writes
            # state.models (core/engine/model_loader.py), so an empty slot here
            # is the NORMAL case and must not 503 -- keep the locally built
            # model. But a /load-model that finished while we waited published
            # the live model into the slot; prefer that over our private copy.
            model = current

        return await asyncio.to_thread(
            load_voice_prompt, prompt_file, allow_create=True, clone_model=model
        )
