# TDD Evidence — Phase 1e: one streaming frame parser (U3 + DRY-U7)

**Source plan:** `~/.claude/plans/review-entire-repo-for-ancient-possum.md`, Phase 1e.
**Branch:** `fix/phase1-stream-error-sentinel` · **Base:** `main` @ `0ddaad8`

## The defect

The length-prefixed streaming wire format was parsed by **two independent
implementations**:

| | `interface/generate_server.py` (CLI) | `server/client/generator.py` (`TTSClient`) |
|---|---|---|
| terminal error sentinel (`sr == 0`) | checked | **not checked** |
| max chunk / buffer cap | 200 MB | 100 MB |

Starlette commits the 200 headers before the body is iterated, so a mid-stream
failure cannot use a status code — the server emits a final frame with
`sample_rate == 0` whose payload is JSON. The CLI has handled that since
WS2 2.5. The **public** client (README:303) did not: it ran the JSON bytes
through `np.frombuffer` and yielded them as audio.

## RED

```
$ pytest tests/test_client_generator.py::TestGenerateStreaming -v

FAILED test_terminal_error_frame_raises_instead_of_yielding_garbage
E   ValueError: buffer size must be a multiple of element size
    qwen3_tts/server/client/generator.py:433

FAILED test_partial_audio_before_error_frame_is_not_a_success
E   AssertionError: GenerationError not raised

2 failed, 4 passed
```

Both failure modes, from the same defect:

- when the JSON payload length is **not** a multiple of 4 → a raw, unhelpful
  `ValueError` escapes to the caller;
- when it **is** a multiple of 4 → no error at all. The generator yields
  garbage samples with `sr=0` and the stream ends cleanly, so a failed
  generation is indistinguishable from a successful one.

The second is the dangerous one and is exactly what the second test caught.

## GREEN

New `qwen3_tts/core/stream_protocol.py` — no FastAPI, torch or mlx imports
(which is why the sentinel constant used to be duplicated rather than shared),
numpy imported lazily inside the parser. It owns:

- `STREAM_ERROR_SENTINEL_SR`, `STREAM_ERROR_CODE_INFERENCE_FAILED`,
  `STREAM_HEADER_SIZE`, `MAX_STREAM_CHUNK_BYTES`
- `encode_stream_error_frame()` / `decode_stream_error_payload()`
- `iter_stream_chunks(byte_iter)` — the single parser

Consumers now re-raise in their own idiom:

```python
except StreamProtocolError as e:
    raise TTSGenericError(f"Server error during streaming: {e}") from e   # CLI
    raise GenerationError(f"Server error during streaming: {e}") from e   # TTSClient
```

`app_generation.py`, `generate_server.py` and `client/_base.py` re-export the
constants so every existing import path keeps working;
`_base.MAX_BUFFER_SIZE` is now an **alias** of the shared cap rather than an
independent literal.

```
$ pytest tests/test_stream_protocol.py -q          →  16 passed
$ pytest tests/test_generate_server.py tests/test_stream_error_frame.py \
         tests/test_client_generator.py tests/test_client.py \
         tests/test_streaming_thread_lifecycle.py -q
124 passed
```

## Cap alignment: 200 MB → 100 MB

The stricter of the two divergent values wins. 100 MB of float32 at 24 kHz is
~17 minutes of audio **in a single frame**, against a real chunk of ~2 s — so
this is far above any legitimate frame while bounding what a corrupt or hostile
length prefix (an attacker-influenceable uint32, up to ~4 GB) can make the
reader buffer.

The unified error message deliberately contains **both** "exceeds" and
"buffer", because the two pre-existing tests key on different words:

- `test_generate_server.py` asserts `"exceeds"` + the byte count
- `test_client.py` asserts `"buffer"` + `"exceed"`

Neither test had to be modified — which is itself evidence the merge preserved
both consumers' contracts rather than bending the tests to fit.

## Anti-re-fork guards

`test_stream_protocol.py::TestSingleImplementation` asserts the *singularity*,
not just the behavior — a future re-fork has to defeat these:

- all three modules report the same sentinel value
- CLI cap and client cap are the same object as the shared cap
- both consumer modules reference `iter_stream_chunks` and contain no local
  `np.frombuffer(audio_bytes, dtype="<f4")`

Plus parser-level cases the old copies handled inconsistently: frames split
across block boundaries (**including mid-header**), multiple frames per block,
trailing partial frame dropped rather than emitted as short audio, corrupt JSON
payload still raising, buffer-growth cap, and non-multiple-of-4 payloads
raising with the `ValueError` chained.

## Gates

```
$ pytest tests/ -m "not e2e" -q --ignore=tests/evaluations/test_speaker_similarity.py
2986 passed, 11 skipped, 88 deselected

$ python tests/run_batches.py --batch 2   →  1/1 batches passed
$ python tests/run_batches.py --batch 3   →  1/1 batches passed
$ ruff check qwen3_tts tests              →  All checks passed
$ mypy qwen3_tts/{core,server,interface}  →  Success: no issues found in 54 source files
$ bandit -r qwen3_tts -c pyproject.toml   →  0 findings (7 pre-existing stale-nosec warnings)
$ wc -l CLAUDE.md                         →  298 (unchanged; the edit rewrote an existing line)
```

`tests.test_stream_protocol` is registered in `BATCHES` (batch 3), so the
Phase 0b guard passes.

`--ignore=tests/evaluations/test_speaker_similarity.py` is the plan's interim
workaround for pre-existing **P1**. Local-env-only; CI unaffected; owned by
Phase 0, not fixed here.

## Not covered

No e2e exercises a live sentinel frame end-to-end — the plan's Phase 2d owns
the e2e tier. The unit coverage drives the exact bytes the server's
`encode_stream_error_frame()` produces, so the two sides are pinned to the same
encoder, but the wire has not been exercised against a running server here.
