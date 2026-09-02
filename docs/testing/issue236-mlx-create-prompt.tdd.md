# #236 — MLX-native /create-voice-prompt — TDD evidence

Branch `fix/issue236-mlx-create-voice-prompt`. Plan:
`~/.claude/plans/peppy-moseying-diffie.md` (r3, team-reviewed design + full santa loop).

## Gate A — RED (commits `23a8770`, `6ff5f65`)

`tests/test_issue236_mlx_create_prompt.py`: 18 tests. Honest ledger after
convergence: **12 genuine-RED** (9 `TypeError` on the missing `backend` kwarg =
the missing dispatch; 2 no-kwarg `503 != 400` behavior gaps; 1 route-source AST
RED) + **4 RED-by-artifact** (tests passing `backend="torch"`/pins whose first
failure is the same dispatch TypeError) + **1 GREEN-by-design pin**
(`test_mlx_load_path_stays_create_free`) + **1 librosa-gated skip**
(`test_rewrite_property_sub24k_stereo_lands_24k_mono`). All RED reasons
verified one-by-one under **pytest and bare unittest** (no conftest).

Three adversarial reviews, FAIL→fixed:

- **python-reviewer FAIL (1 CRITICAL):** `_drive_mlx` could never reach GREEN
  (3-tuple vs 4-target unpack; `patch.object` on the function object; the
  writer stub resolving the patched attribute inside the patch scope →
  RecursionError; cross-class helper call). Rewritten module-level; honest
  ledger corrected (the commit's initial "16 genuine-RED" was overstated by
  four test-bug REDs).
- **fastapi-reviewer FAIL:** writer + torch-engine-guard seams moved to the
  engine **facade** (the handler's house-rule function-local facade imports
  make submodule patches inert — verified by GREEN-simulation); the no-kwarg
  rate test was RED-at-GREEN on torch-ambient CI → ambient resolution pinned
  via `TTS_BACKEND` env; the listing test crashed on `query_params=None` and
  didn't prove intersection → fixed + orphan-`.wav` negative added; `Request`
  import path fix.
- **agy round-1 FAIL → round-2 PASS:** cache-clear pinned on BOTH facade and
  submodule seams (exact-once); decode stub returns real bytes (M-b dies on
  the assertion, not a None crash); M-e hardened from substring to AST-arg +
  behavioral route test; invalid_name torch variant; no_transcript asserts
  the `.wav`; response-model round-trip explicitly delegated to
  `test_response_contracts.py`; its "M-a-variant not killed" claim answered
  with evidence (any tools reroute skips the engine writer → `writer_calls`
  assertion fails).

## GREEN (commits `08c4545`, `3be1541`, + fixes)

17 passed + 1 librosa-gated skip, pytest AND bare unittest.

**Mutants killed** (each applied temporarily → owning tests failed → reverted):

| Mutant | Result |
|---|---|
| M-a: MLX branch removed (both occurrences) | **10 failed** (partial single-occurrence mutant: 1 failed) |
| M-b: blank-check truthiness-only (no `.strip()`) | 2 failed |
| M-c: typed exception untranslated (→ 500 `creation_failed`) | 1 failed |
| M-d: cache-clear dropped | 1 failed |
| M-e: route drops `get_backend()` | 1 failed (AST guard; the behavioral route half is killed on torch-ambient legs — locally ambient is already mlx) |
| M-f: MLX branch also writes a `.pt` | 2 failed |

## Existing-test edits + notes

- `test_issue192_create_prompt_serialization.py` (3 direct calls) +
  `test_python_review_fixes.py` (1): explicit `backend="torch"` — the fixtures
  pass no backend today; the ambient fallback would split-brain between the
  mlx dev env and torch-default CI.
- `test_create_voice_endpoint.py`: 503-gate + success tests force the torch
  branch via the patched `qwen3_tts.server.app.get_backend` seam (module-scope
  from-import; facade/definition patches don't reach it).
- `test_response_contracts.py`: +`CreateVoicePromptResponse` round-trip
  (route was the last untyped JSON route).
- `test_e2e_queueing.py` test_02: payload gains a transcript (the 400 policy
  would reject the old blank payload); stale-server skip branch kept, its
  "possibly fixed by the bump" message corrected (disproven by the eval doc).
- `test_create_voice_functions.py`: 8 harness sites now patch the ENGINE
  `VOICE_PROMPTS_DIR` binding too (the delegation writes via
  `voice_prompt`'s module global; tool-binding-only patches leaked two files
  into the real `voice_prompts/` during a RED-state run — moved out, and this
  is the exact dual-binding trap the plan named).

## Pre-existing local-env gap (stash-proven, not a regression)

9 `test_create_voice_functions.py` tests fail identically on pre-change code
in `.venv-310` (16 kHz fixtures + no librosa; CI's `[test]` extra supplies
librosa and passes).

## Live payoff

e2e `test_02` (create-queuing behind a real `/generate`) flips from skip to
expected PASS on the MLX server — see the PR body for the run output.
