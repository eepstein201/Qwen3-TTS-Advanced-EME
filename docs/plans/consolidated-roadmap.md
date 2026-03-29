# Qwen3-TTS Consolidated Development Roadmap

> **Status:** Combined from 5 plan documents into one prioritized roadmap
> **Items:** ~27 open tasks consolidated and prioritized
> **Last Updated:** 2026-03-29

---

## Priority 0: Critical Fixes & Stability (DO FIRST)

| ID | Task | Impact | Effort | Risk | Files |
|----|------|--------|--------|------|-------|
| **P0-1** | Remove duplicate Pydantic models | High | Low | Low | `qwen3_tts/server/validation.py` |
| **P0-2** | Consolidate print helpers | High | Low | Low | `qwen3_tts/tools/_shared.py` |
| **CRITICAL-1** | Deconstruct `test_voice.py` (31k tokens) | High | Medium | Medium | `tests/test_voice.py` → 8 files |
| **CRITICAL-2** | Fix Docker IPC (`--ipc=host`, HF cache mount) | High | Low | Low | `Dockerfile.vllm`, `docker-compose.yml` |
| **CRITICAL-3** | Add `@require_server` decorator (13x repeated code) | High | Low | Low | `qwen3_tts/server/client.py` |

**Why P0-1:** 6 duplicate Pydantic models in `validation.py` (lines 49-81 and 120-152) violate DRY.
**Why P0-2:** Duplicate print functions in `uninstall.py` and `healthcheck.py`.
**Why CRITICAL-1:** 31,688-token monolith causes AI context rot. Decompose into 8 domain files.
**Why CRITICAL-2:** Docker IPC crashes under multi-GPU tensor parallel. HF cache remount on every restart.
**Why CRITICAL-3:** `is_server_running()` check repeated 13+ times in `client.py`.

---

## Priority 1: Code Quality & Modularity

| ID | Task | Impact | Effort | Files |
|----|------|--------|--------|-------|
| **P1-1** | Extract config value helper (`_get_config_value`) | Medium | Low | `qwen3_tts/core/config.py` |
| **P1-2** | Extract text chunking helper (`_prepare_text_chunks`) | Medium | Low | `qwen3_tts/core/engine/inference.py` |
| **P1-3** | Define audio processing constants | Medium | Low | `qwen3_tts/core/engine/audio_processing.py` |
| **P1-4** | Add `_extract_error_message` helper | Medium | Low | `qwen3_tts/server/client.py` |
| **P2-1** | Refactor CLAUDE.md to <300 lines | High | Low | `CLAUDE.md`, `docs/00-Foundations/ARCHITECTURE.md` |
| **P2-2** | Split `ui.py` into modules | Medium | High | `interface/ui/` package |
| **P2-3** | Split `generate.py` into modules | Medium | High | `interface/cli/` package |
| **P2-4** | Split `TTSClient` into focused interfaces | Medium | High | `server/client/` package |

**Why P2-1:** CLAUDE.md is 539 lines (should be <300). Extract deep-dive content to `docs/00-Foundations/ARCHITECTURE.md`.
**Why P2-2:** 2100-line monolithic UI file. Split into `clone_tab.py`, `design_tab.py`, `custom_tab.py`, `voice_management.py`, `shared.py`.
**Why P2-3:** 2400-line CLI file. Split into `parser.py`, `generation.py`, `batch.py`, `srt.py`, `dialogue.py`.
**Why P2-4:** TTSClient has 30+ methods. Split into `generator.py`, `models.py`, `voices.py`, `config.py`.

---

## Priority 2: Performance & Infrastructure

| ID | Task | Impact | Effort | Files |
|----|------|--------|--------|-------|
| **HIGH-1** | vLLM multimodal params: `--limit-mm-per-prompt audio=1`, `--enable-chunked-prefill`, `bfloat16` | High | Medium | Docker configs, vLLM init |
| **HIGH-2** | Decouple FastAPI from vLLM inference (httpx.AsyncClient) | High | High | `app.py`, `engine_vllm.py` |
| **HIGH-3** | Implement WebSocket bidirectional audio streaming | High | Medium | `websocket.py` (new) |
| **MED-1** | Pre-calculate wavesurfer.js peaks on backend | Medium | Medium | `audio_processing.py`, `wavesurfer_js.py` |
| **MED-2** | Optimize `engine_vllm.py` parameters | High | Medium | `engine_vllm.py` |

---

## Priority 3: Enhancements (Low Priority)

| ID | Task | Impact | Effort | Files |
|----|------|--------|--------|-------|
| **R-23** | LUFS Normalization option (`pyloudnorm`) | Medium | Low | `audio_processing.py`, config |
| **R-27** | Configurable silence gap (`generation.silence_gap_seconds`) | Low | Low | `inference.py`, config |
| **R-43** | `create_voice.main()` testability (PARTIAL) | Low | Low | `create_voice.py` |

---

## Priority 4: Future / Upstream-Dependent

| ID | Task | Blocker | Effort |
|----|------|---------|--------|
| **R-28** | Speculative decoding (1.5-3x speedup) | Upstream library support | High |
| **R-29** | TranscribeRequest.language pattern constraint | None | Low |
| **R-30** | Unbounded base64 payload size limit | None | Low |
| **FUTURE-1** | Entropy-based hallucination monitoring | vLLM forward pass modification | High |
| **FUTURE-2** | GFlowNet distribution alignment | Research integration | High |
| **FUTURE-3** | Adaptive attention head deactivation | Per-model profiling required | High |

---

## Execution Order (Recommended)

### Sprint 1: Quick Wins (1-2 days)
1. P0-1: Remove duplicate Pydantic models
2. P0-2: Consolidate print helpers
3. CRITICAL-2: Fix Docker IPC configs
4. P1-3: Define audio constants
5. P1-4: Add `_extract_error_message` helper

### Sprint 2: Code Quality (3-5 days)
6. P0-3: Deconstruct `test_voice.py` (run tests after each extraction)
7. P2-1: Refactor CLAUDE.md to <300 lines
8. P1-1: Extract config value helper
9. P1-2: Extract text chunking helper
10. CRITICAL-3: Add `@require_server` decorator

### Sprint 3: Modularity (1-2 weeks)
11. P2-2: Split `ui.py` into modules
12. P2-3: Split `generate.py` into modules
13. P2-4: Split `TTSClient` into interfaces

### Sprint 4: Performance (1-2 weeks)
14. HIGH-1: vLLM multimodal params optimization
15. HIGH-2: Decouple FastAPI from vLLM
16. MED-1: Pre-calculate wavesurfer peaks

---

## Success Criteria

- [ ] All P0 tasks complete
- [ ] All CRITICAL tasks complete
- [ ] Full test suite passes (1970+ tests)
- [ ] CLAUDE.md <300 lines
- [ ] `test_voice.py` decomposed into 8 files
- [ ] Docker configs validated for production
- [ ] Code quality metrics improved (DRY, SOLID)

---

## Notes

- **E2E Testing:** Has dedicated plan at `docs/plans/e2e-testing-implementation-plan.md`
  - Phases 1-2 complete (50 tests passing): security, performance E2E tests
  - Phases 3-4 pending: UI tests, cross-browser, accessibility (covered by existing `test_e2e_playwright.py`)
- **R-13 Rate Limiting:** COMPLETED (15 tests passing) - E2E verification in separate E2E plan
- **R-50 Concurrent Generation:** FIXED (thread-safe `_history_lock`)
- **AI Regression Prevention:** All implementations must follow TDD (red-green-refactor)
- **Git Workflow:** Use feature branches, never push directly to main without approval
