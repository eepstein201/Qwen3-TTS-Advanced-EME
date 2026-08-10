# Python Code Review — Qwen3-TTS (whole repo)
**Date:** 2026-08-10 · **Reviewer:** python-reviewer (6 parallel passes) + automated baselines + live profiling/E2E
**Scope:** entire `qwen3_tts/` tree (~24.6k LOC, 80 files) across best practices, reliability, error handling, performance, quality, security.
**Verdict: WARNING** — 0 CRITICAL, **9 HIGH**, 33 MEDIUM, 16 LOW. No blockers, but 4 HIGHs cause silent data loss / wrong output and should ship first.

---

## 1. Methodology (what actually ran)

| Check | Tool | Result |
|---|---|---|
| Type check | `mypy qwen3_tts/{core,server,interface}` | **PASS** — 0 errors / 0 `# type: ignore` (53 files; untyped bodies not checked, so missing annotations remain fair game) |
| Lint | `ruff check qwen3_tts tests` | **PASS** — all checks |
| Security | `bandit -r qwen3_tts` | **PASS** — 0 issues (44 justified `# nosec`, mostly B104 bind on 127.0.0.1) |
| Complexity | `ruff --select C901` | **45 functions > cc 10** — worst `handle_generate` = **41** |
| Coverage | `pytest -m "not e2e" --cov` | **86%** (9906 stmt / 1371 miss) · 2692 passed / 5 skipped / 49s |
| Dep audit | `pip-audit` both envs | mlx: 8 pkgs w/ CVEs · torch: 61/13 (gradio 6.8.0 — known un-upgradable knot) |
| Live perf | server gen ×3 modes | clone ≈9.5 c/s · design ≈14 c/s · custom ≈19 c/s (M2 Pro / MLX 8bit 1.7B) |
| E2E | `test_e2e_security_auth` | **11/11 passed** |
| Browser | Chrome DevTools MCP drive | Gradio UI renders, live clone generation produced audio + history row |

**Hygiene confirmed clean:** 0 bare `except:`, 0 mutable default args, 0 `# type: ignore`, 5 TODO/FIXME total (low latent debt). Lazy-import discipline holds everywhere (no torch/mlx at module scope). SSRF protection on `http_client` is sound; atomic config save is correct.

---

## 2. CRITICAL — none

---

## 3. HIGH (9) — silent data loss / wrong output / crashes

### H1. Stale `cancelled` flag silently truncates an in-flight batch
**`server/app_generation.py:249, 294`** · reliability · `read`
The batch path clears `generation_state["cancelled"]` once before the loop (l.195) and checks it per item (l.249) but never re-clears it per item. When request A's batch releases `inference_lock` between items and request B is cancelled (`/cancel-generation` sets `cancelled=True`), request A's next iteration sees the stale `True` and **breaks early** — the caller gets `{"results":[…]}` with fewer items than submitted and **no error**. The streaming path does this correctly (l.658).
**Fix:** add `"cancelled": False` to the per-item `generation_state.update(...)` at l.294 (inside `inference_lock`), mirroring l.658.

### H2. Streaming path silently skips post-processing → different audio than `/generate`
**`core/engine/inference.py:1376–1476`** · quality/reliability · `read`
`run_inference` (batch) applies `_trim_icl_echo → _maybe_apply_speed → _maybe_apply_lufs`. `run_inference_streaming` applies **none**. A user with `clone_speed: 1.5` + `lufs_normalize: true` gets correctly processed audio from `/generate` but **raw unprocessed audio** from `/generate-stream` and `/ws`. MLX streaming also skips `_validate_audio`, so NaN/clipping goes uncaught.
**Fix:** wrap each yielded chunk through `_maybe_apply_speed` + `_validate_audio` (per-chunk feasible); LUFS is whole-signal — document the streaming limitation in the docstring.

### H3. torch backend crashes with `IndexError` on empty audio list
**`core/engine/inference.py:255`** · reliability · `read`
`_run_inference_torch` indexes `wavs[0]` unconditionally; if the model returns `([], sr)` (empty text surviving normalization, or an internal failure) it raises an opaque `IndexError`. The MLX path (l.417) guards this explicitly — asymmetric.
**Fix:** `if not wavs: raise RuntimeError("torch generation returned no audio segments")` before l.255.

### H4. Watch mode marks files processed *before* generation succeeds
**`interface/generate_interactive.py:713`** · reliability · `read`
`processed_files.add(path)` runs at l.713, before the generation call (l.719–739). If generation raises (server down, network, model error) the `except` prints the error but the file stays in `processed_files` — **never retried**. User must rename/re-touch it.
**Fix:** move `processed_files.add(path)` to after successful generation (after the `sf.write` at l.742).

### H5. REPL silently ignores all CLI sampling flags
**`interface/generate_interactive.py:471` + `interface/generate.py:638`** · quality · `read`
`run_repl(config, use_server)` is called **without** the `gen_params` that `main()` already computed. Inside the REPL it rebuilds params from raw `config["generation"]`, dropping all CLI overrides. `tts --repl --seed 42 --temperature 0.5` silently ignores both flags. `run_watch_mode` and `interactive_mode` correctly receive `gen_params`.
**Fix:** add `gen_params` param to `run_repl`, pass it from generate.py:638; remove the internal reconstruction at l.500–501.

### H6. SRT processing has no per-entry recovery — one error aborts the whole file
**`interface/cli/srt.py:88–119`** · reliability · `read`
The entry loop has no try/except. A transient 503 or model error on any entry propagates and aborts; individual files before the failure are saved but **all remaining entries + the combined output are lost**. `dialogue.py:162–203` does this correctly.
**Fix:** wrap each entry's generation in try/except, log + continue; build the combined file from whatever succeeded. (Same gap exists in `process_batch` local path, generate.py:131–146.)

### H7. Streaming client trusts the wire format; corrupt stream → uncaught traceback
**`interface/generate_server.py:315–344`** · error-handling · `read`
`struct.unpack` (l.317) and `np.frombuffer` (l.329) raise `struct.error`/`ValueError` on a corrupt/truncated stream (server crash mid-stream, proxy). These are **not** `RequestException` subclasses, so the `except` at l.357 doesn't catch them → raw traceback. There's also **no upper bound on `audio_len`** (l.320) — a corrupt header claiming multi-GB blocks the parser until the 600s timeout.
**Fix:** wrap the parse loop in `try/except (struct.error, ValueError)`; `if audio_len > MAX_CHUNK_SIZE: raise IOError(...)`.

### H8. vLLM `stop()` leaks the AsyncClient connection pool outside an event loop
**`core/engine_vllm.py:415–420`** · reliability · `read`
Sync `stop()` calls `asyncio.get_running_loop().create_task(self._client.aclose())`. Outside a loop, `get_running_loop()` raises `RuntimeError`, swallowed by the broad `except` — `_client` is set to `None` without closing, **leaking up to 10 keepalive connections + transport**. Even with a loop (the `start()` retry at l.363) the close task is never awaited.
**Fix:** make `stop()` async (`await self._client.aclose()`) with a sync fallback for non-async callers.

### H9. `_load_initial_history` wipes a fresh in-memory entry when disk scan is empty
**`interface/ui/_facade.py:163–169`** · reliability · `read`
The timestamp guard requires `and disk_history` to be truthy. When `disk_history == []` (fresh install / all cleared) the guard never fires even if `current_history` holds a brand-new entry from a generation that completed during the scan — so it returns `[]`, **clobbering the fresh entry**. Violates the function's own docstring.
**Fix:** change the condition to `(not disk_history or current_history[0]["timestamp"] > disk_history[0]["timestamp"])`.

---

## 4. MEDIUM (33) — themed

### Theme A — Streaming path divergence (the #1 structural issue)
H2 + H7 above, plus:
- **`server/app_generation.py:676` & `server/websocket.py:396`** — both streaming call sites omit `max_chunk_chars` and (WS) `config_provider`. A client requesting `max_chunk_chars=200` via stream gets the config default (500) → different chunk boundaries than `/generate`. *(read)*
- **`server/websocket.py:433–472`** — WS generation **never sets `generation_state`** → invisible to `/generation-status`, `/queue-status`, and `detect_degraded_generation` (the detector built to catch the 7314s/22-char incident). A wedged WS generation is undetectable. *(read)*
- **`server/app_generation.py:773`** — HTTP streaming swallows a mid-stream inference error once ≥1 chunk was delivered → client gets a truncated stream with no error signal (WS handles this correctly with a terminal error frame). *(read)*

> **Recommendation:** treat batch + streaming as one pipeline. Extract a shared `_postprocess_chunk()` + shared `generation_state` lifecycle and call both from all three entry points (`/generate`, `/generate-stream`, `/ws`). This single refactor retires H2 and 3 MEDIUMs at once.

### Theme B — Concurrency / races
- **`core/engine/asr.py:234–252`** — `unload_asr_model` mutates ASR globals **without `_asr_lock`**, racing the locked load paths. *(read)*
- **`server/app_generation.py:219–224`** — cache TOCTOU: entry looked up under lock, file read after lock release; a concurrent model-unload can `os.remove` it → 500 instead of cache-miss-and-regenerate. *(read)*
- **`server/app_models.py:378–380`** — `handle_update_model_config` nulls all models with **no active-generation guard**, unlike `handle_unload_model` (which 409s). *(read)*
- **`core/engine/inference.py:1191, 207`** — `seed_lock_chunks` reseeds the global backend and `_apply_mps_float32_guard` mutates the shared model dtype in place → race if two generations share a model object (severity depends on whether generation is fully serialized; if so, add a documenting comment). *(inferred)*

### Theme C — Silent error-swallow / opaque crashes
- **`interface/generate_server.py:189,217,219,243,299,358`** — six generic `raise Exception(...)` force every caller into `except Exception` (can't distinguish transient vs permanent). Define `TTSGenericError(RuntimeError)` / use `ConnectionError`. *(read)*
- **`interface/generate_helpers.py:338`** — `parse_ssml` crashes on `<break time="5seconds"/>` (`"s" in time_str` substring match → `float("5econd")`). Use `endswith`. *(read)*
- **`interface/generate_helpers.py:271`** — `show_history` crashes on one corrupt JSONL line; append-only files can have partial last lines. Wrap per-line. *(read)*
- **`interface/generate_server.py:245`** — `resp.json()["results"]` unguarded → `KeyError` on an unexpected 200 shape. *(read)*
- **`interface/generate_interactive.py:439`** — `interactive_mode` is the one interactive path with **no** error recovery (REPL and watch both have it); a transient failure exits with a traceback. *(read)*
- **`server/websocket.py:209`** — cancel-watcher `except Exception: return` with **zero logging** → mid-generation cancel silently dies, undiagnosable. *(read)*
- **`server/app_generation.py:448`, `app_models.py:320,389,550`** — `except OSError: pass` on cache removal hides persistent FS failures; the LRU eviction pops the dict entry but the file persists → **disk grows unbounded while the cache appears to evict**. At least `logger.debug` the failure. *(read)*
- **`core/config/io.py:92`** — `validate_config` does `dict(config)` → opaque `TypeError` if config.json is valid JSON but not an object (`42`, `null`). Add a type guard. *(inferred)*

### Theme D — Resource leaks / non-atomic writes
- **`core/engine_vllm.py:469,548,604`** — clone-mode temp `.wav` (`delete=False`) never unlinked in `generate()`/`generate_stream()` → one leaked file per clone generation. *(read)*
- **`tools/create_voice.py:64–67`** — pydub temp `temp_reference.wav` orphaned if `sf.read` raises; fixed filename clobbers on concurrent use. *(read)*
- **`tools/uninstall.py:193`** — `uninstall_config` does a raw `open("w")` write, the **only** config writer bypassing the atomic `save_config` (temp+fsync+`os.replace`). A crash mid-write corrupts config.json. *(read)*

### Theme E — Async/event-loop blocking
- **`server/app.py:498` (`/stats`)** — calls `torch.mps.current_allocated_memory()` / `mx.get_active_memory()` (GIL-holding C extensions) directly on the event loop; delays chunk delivery to concurrent streaming clients. Every sibling endpoint uses `asyncio.to_thread` — this one doesn't. *(read)*
- **`interface/ui/shared.py:197–223`** — `apply_model_settings` blocks up to **90 s** (30 s × 3 models) synchronously with no progress indicator; UI frozen. *(read)*

### Theme F — Correctness edge cases
- **`core/engine/text_processing.py:105`** — `_TIME_RE` matches `25:99`, `12:70` → nonsensical speech / wrong Chinese `分` forms. Validate ranges after capture. *(read)*
- **`core/engine/inference.py:1042–1073`** — `max_chunk_chars=0` ("0 disables chunking" per docs) does **not** disable the token-based split path. Live corroboration: a 765-char generation ran as **1 chunk** in 38.5s during profiling. *(read)*
- **`core/engine/inference.py:1244`** — multi-chunk `_trim_icl_echo` probes 4 s of *combined* audio (can span chunk boundaries); the 50% cap is calibrated for single-chunk output and could cut legitimate content across a 12-chunk generation. *(read)*
- **`core/engine/inference.py:1308`** — Metal retry trigger `"kernel"` matches generic errors ("kernel size mismatch") → **expensive false-positive recursive retries** (2–4 min wasted per false positive). Tighten to Metal-specific phrases. *(read)*

### Theme G — i18n / quality
- **`core/engine/voice_prompt.py:66,245,316`** — `open(txt_path)` with no `encoding="utf-8"` → silent mojibake on Chinese transcripts under a C/POSIX locale (Docker). *(read)*
- **`interface/ui/shared.py:528–709`** — the expanduser→isabs→`..`→`startswith(home+sep)` path-containment check is **duplicated 4×** in security-sensitive code, pushing shared.py to 803 lines (past the 800 ceiling). Extract one `_ensure_path_under_home()`. *(read)*
- **`interface/generate.py:741`** — output-path validation uses `return` instead of `sys.exit(1)` → misleading success exit code (every sibling validation `sys.exit(1)`s). *(read)*
- **`interface/ui/history_panel.py:161`** — delete banner says "file was already gone" when the file **exists** but deletion was refused (outside output dir) → falsehood. *(read)*
- **`interface/ui/model_management.py:123,182`** — `ProgressIndicator` constructed but never rendered → misleading dead code. *(read)*
- **`core/config/auth.py:24`** — no token-file permission check (defense-in-depth; a 0644 token file is silently accepted). *(read)*

---

## 5. LOW (16) — compact

- `core/engine/inference.py:1234 vs 1354` — `silence_gap_seconds` default inconsistent (0.0 vs 0.1) → audible gap only in retried chunks. *(read)*
- `core/engine/inference.py:698` — `_crossfade_chunks` returns original array by ref for single-chunk case (latent mutation hazard). *(read)*
- `core/engine_vllm.py:317` — fresh `httpx.AsyncClient` per poll (~150× over 300s startup). *(read)*
- `core/engine_vllm.py:274` — log opened `"w"` truncates prior-retry diagnostics. *(read)*
- `server/app_generation.py:429` — `base64.b64encode` of full audio on event loop. *(read)*
- `server/app_generation.py:446` — `os.remove` inside `gen_cache_lock`. *(read)*
- `interface/generate_server.py:312` — streaming buffer uses immutable `bytes +=` → O(n²) copies (use `bytearray`). *(read)*
- `interface/generate.py:700` — batch-file elements not validated as strings. *(inferred)*
- `interface/generate_helpers.py:430` — `process_ssml_text` mutates `args` in place (immutability rule). *(read)*
- `interface/ui/generation.py:204` — dead `prosody_preset` param/branch. *(read)*
- `interface/ui/_facade.py:92` — comment says timer tick costs 2 requests, actually 3. *(read)*
- `tools/healthcheck.py:181` — `UnboundLocalError` if cache > 1 TiB (reimplemented size loop missing TB). *(read)*
- `core/config/runtime.py:308` — `is_server_running` uses raw `requests.get`, bypassing the `http_client` choke-point (CodeQL-visible). *(read)*
- `core/config/pid.py:29` — fixed `.pid.tmp` filename → clobber race on concurrent start. *(read)*
- `core/config/auth.py:24` — TOCTOU between `exists` check and `open()`. *(read)*
- `core/protocols.py` — 5 `@runtime_checkable` Protocols are decorative (no `isinstance`, no consuming signatures) — document as non-binding or actually enforce. *(inferred)*

---

## 6. Dependency CVEs (pip-audit)

**mlx env (8 packages):**
- **`python-multipart` 0.0.22 → 0.0.31** — **5 DoS CVEs (PYSEC-2026-3036–3040)** in FastAPI's multipart/form parser, reached by `/create-voice-prompt` file uploads. **Most actionable server-facing finding.**
- `pillow` 12.1.1 → 12.3.0 (many; gradio dep) · `click` 8.3.1 → 8.3.3 · `idna` 3.11 → 3.15 · `msgpack` 1.1.2 → 1.2.1 · `pip` 26.0.1 → 26.1.2 · `pygments` 2.19.2 → 2.20.0 · `pytest` 9.0.2 → 9.0.3 (dev only)

**torch env (61 vulns / 13 packages):**
- **`gradio` 6.8.0** — PYSEC-2026-211, 2178 (fixed 6.15), 2179 (fixed 6.16). **Known-unupgradable**: the documented `transformers<5` / `huggingface-hub<1.0` ↔ gradio-floor conflict blocks any gradio ≥ 6.15 in this env. Structural debt, not a quick fix — tracked separately.

---

## 7. Recommended fix order

1. **Silent data loss / wrong output (ship first):** H1 (stale cancelled), H5 (REPL flags), H2 (streaming post-processing), H4 (watch drop), H6 (SRT recovery).
2. **Crashes on edge cases:** H3 (torch empty audio), H7 (wire format), then MEDIUMs `validate_config` non-dict, `parse_ssml`, `show_history` JSONL.
3. **Concurrency:** ASR-unload lock, cache TOCTOU, `update_model_config` active-gen guard.
4. **Resource leaks:** vLLM temp `.wav`, `create_voice` temp, `uninstall_config` atomicity.
5. **Async blocking:** `/stats` → `asyncio.to_thread`, `apply_model_settings` streaming progress.
6. **Structural refactor (high-leverage):** unify batch + streaming pipeline (Theme A) — retires H2 + 3 MEDIUMs.
7. **Deps:** bump `python-multipart` (server-facing DoS). Gradio/torch-env knot stays tracked.
8. **Quality/complexity:** decompose `handle_generate` (cc 41) / `_handle_generation` (35) / `run_repl` (31); dedup shared.py path validation; kill dead `ProgressIndicator`/`prosody_preset`.

---

## 8. What's healthy (don't break these)

- Static gates are genuinely green; mypy/ruff/bandit all PASS with **zero** suppressions-that-hide-real-issues.
- Lazy-import discipline is perfect across all 80 files.
- Auth surface is solid — 11/11 E2E auth tests pass (401s, no injection, token never leaked).
- Atomic config save is correct (temp + fsync + `os.replace` + `BaseException` cleanup).
- SSRF protection on `http_client` is sound (host allowlist, path/method validation).
- Coverage 86% is above target; only `_shared.py` (36%) and `auth.py` (49%) are thin.
- Live system is healthy: all 3 modes generate real audio in-browser; model loads are fast (1.6–2s).
- The recent PRF audio work (phase-align crossfade, clone-speed, echo-trim) is correct in the batch path — the gap is purely that the streaming path doesn't share it.

**Evidence artifacts:** live UI screenshot at `.claude/reviews/ui_generation_live.png`; profiling log lines in `.voice_server.log` (2026-08-10 09:38–09:39).
