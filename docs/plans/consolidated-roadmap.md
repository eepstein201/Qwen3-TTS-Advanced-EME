# Qwen3-TTS Consolidated Development Roadmap

> **Status:** All P0/P1/P2 items complete. Open work starts at Priority 2 (Performance).
> **Last Updated:** 2026-04-26

---

## ✅ Completed

| ID | Task | Completed |
|----|------|-----------|
| P0-1 | Remove duplicate Pydantic models (`validation.py`) | ✅ |
| P0-2 | Consolidate print helpers (`tools/_shared.py`) | ✅ |
| CRITICAL-1 | Deconstruct `test_voice.py` (31k tokens → 8 domain files) | ✅ |
| CRITICAL-2 | Fix Docker IPC (`--ipc=host`, HF cache volume) | ✅ |
| CRITICAL-3 | Add `_require_server` decorator (client package) | ✅ |
| P1-1 | `_get_config_value` helper in `config.py` | ✅ |
| P1-2 | `_prepare_text_chunks` helper in `inference.py` | ✅ |
| P1-3 | Audio processing constants (`LUFS_TARGET`, `DEFAULT_SAMPLE_RATE`) | ✅ |
| P1-4 | `_extract_error_message` helper in `server/client/_base.py` | ✅ |
| P2-1 | CLAUDE.md trimmed to <300 lines (currently 293) | ✅ |
| P2-2 | Split `ui.py` into `interface/ui/` package | ✅ |
| P2-3 | Split `generate.py` into `interface/cli/` package | ✅ |
| P2-4 | Split `TTSClient` into `server/client/` package | ✅ |
| HIGH-3 | WebSocket bidirectional audio streaming (`websocket.py`) | ✅ |
| R-13 | Rate limiting (15 tests passing) | ✅ |
| R-50 | Thread-safe concurrent generation (`_history_lock`) | ✅ |
| R-51 | History panel chunk count always showed 0 — `/generate` now returns `"chunks"` per result; client stores `last_chunk_count`; UI passes actual count to `add_to_history` and metadata JSON | ✅ |
| QC-1 | Type hints added to `core/config.py`, `server/app.py`, `core/engine/inference.py` | ✅ |
| QC-2 | `print()` → `click.echo()` in `tools/model_cache.py` | ✅ |
| QC-3 | Specific exception types in `server/websocket.py` (replaces broad `except Exception`) | ✅ |
| QC-4 | `_build_torch_params()` helper extracted in `inference.py`; `model_size` added to `/stats` response | ✅ |
| QC-5 | Remove duplicate module-level definitions in `inference.py` (logger, strategies defined twice) | ✅ |
| QC-6 | `GenerateResult` Pydantic model includes `chunks` field — FastAPI no longer strips it from response | ✅ |
| QC-7 | Token-ownership guard in `cleanup_pid()` / lifespan / shutdown — only delete token file if on-disk matches `app_state.auth_token` | ✅ |
| QC-8 | `TTS_LOG_LEVEL` env var replaces hardcoded `logging.DEBUG` in `run_server()` | ✅ |
| QC-9 | 5bit/6bit added to `VALID_MLX_QUANTIZATIONS`; `transformers` pin relaxed to `>=4.57.3`; `gradio` minimum bumped to `>=5.0.0` | ✅ |
| QC-10 | xdist autouse fixture skips re-init when `app.state.auth_token` is already set, preventing it from clobbering unittest setUpClass state | ✅ |

---

## Priority 1: Performance & Infrastructure (Open)

| ID | Task | Impact | Effort | Files |
|----|------|--------|--------|-------|
| **HIGH-1** | vLLM multimodal params: `--limit-mm-per-prompt audio=1`, `--enable-chunked-prefill`, `bfloat16` | High | Medium | Docker configs, vLLM init |
| **HIGH-2** | Decouple FastAPI from vLLM inference (httpx.AsyncClient) | High | High | `app.py`, `engine_vllm.py` |
| **MED-1** | Pre-calculate wavesurfer.js peaks on backend | Medium | Medium | `audio_processing.py`, `wavesurfer_js.py` |
| **MED-2** | Optimize `engine_vllm.py` parameters | High | Medium | `engine_vllm.py` |

**Why HIGH-1:** vLLM defaults are not tuned for audio multimodal. Missing params cause OOM and slow prefill.
**Why HIGH-2:** Current `app.py` calls vLLM synchronously inside FastAPI request handler — blocks the event loop.
**Why MED-1:** Wavesurfer peaks currently computed client-side; pre-computing saves ~200ms per playback.
**Why MED-2:** Default vLLM params (max_model_len, tensor_parallel_size, etc.) not set for Qwen3-TTS workload.

---

## Priority 2: Enhancements (Open)

| ID | Task | Impact | Effort | Files |
|----|------|--------|--------|-------|
| **R-23** | LUFS normalization option (`pyloudnorm`) — expose as config toggle | Medium | Low | `audio_processing.py`, `config.json` |
| **R-27** | Configurable silence gap (`generation.silence_gap_seconds`) | Low | Low | `inference.py`, `config.json` |
| **R-29** | `TranscribeRequest.language` pattern constraint (validation) | Low | Low | `validation.py` |
| ~~**R-30**~~ ✅ | Unbounded base64 payload size limit — ASGI 413 body-size middleware in `app.py` (2026-07-06) | Low | Low | `app.py` |
| **R-43** | `create_voice.main()` testability (dependency injection) | Low | Low | `tools/create_voice.py` |
| **FOLLOWUP-1** | Decide whether `config.json` defaults should set `design.load_at_startup` and `custom.load_at_startup` to `true` (currently `false` and `{}` in the checked-in config; my local working tree has them on, ~7.5GB at startup vs ~2.5GB) | Low | Low | `config.json` |

---

## Priority 3: Future / Upstream-Dependent

| ID | Task | Blocker | Effort |
|----|------|---------|--------|
| **R-28** | Speculative decoding (1.5-3x speedup) | Upstream library support | High |
| **FUTURE-1** | Entropy-based hallucination monitoring | vLLM forward pass modification | High |
| **FUTURE-2** | GFlowNet distribution alignment | Research integration | High |
| **FUTURE-3** | Adaptive attention head deactivation | Per-model profiling required | High |

---

## Recommended Execution Order

### Sprint 1: Performance Quick Wins
1. HIGH-1: vLLM multimodal params (Docker config change, low risk)
2. MED-2: Optimize `engine_vllm.py` parameters
3. R-29: TranscribeRequest language constraint
4. R-30: Base64 payload size limit
5. R-27: Configurable silence gap

### Sprint 2: Infrastructure
6. HIGH-2: Decouple FastAPI from vLLM (httpx.AsyncClient)
7. MED-1: Pre-calculate wavesurfer peaks
8. R-23: LUFS normalization config toggle
9. R-43: `create_voice.main()` testability

---

## Success Criteria

- [ ] HIGH-1/MED-2 vLLM params validated in Docker environment
- [ ] HIGH-2 FastAPI decoupled — event loop no longer blocked during generation
- [ ] Full test suite passes (2163+ tests)
- [ ] R-29/R-30 validation gaps closed

---

## Notes

- **E2E Testing:** Dedicated plan at `docs/plans/e2e-testing-implementation-plan.md`
  - Phases 1-2 complete (50 tests passing): security, performance E2E tests
  - Phases 3-4 pending: UI tests, cross-browser, accessibility
- **AI Regression Prevention:** All implementations must follow TDD (red-green-refactor)
- **Git Workflow:** Use feature branches, never push directly to main without approval
