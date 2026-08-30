# TDD Evidence — Phase 2a (#214 item 1): the torch auto-create-from-.wav path

## The defect

Torch-backend only. `load_voice_prompt` -> `_load_voice_prompt_torch` ->
`_auto_create_pt_from_wav` (`core/engine/voice_prompt.py`) calls
`load_model("clone")` **and** `create_voice_prompt(...)` — real GPU inference —
as a side effect of "just loading a prompt" whenever the `.pt` is missing or
corrupt and a sibling `.wav` exists. All three server call sites
(`app_generation.py` batch `/generate`, `app_generation.py` streaming
`/generate-stream`, `websocket.py` `/ws`) invoked this via
`asyncio.to_thread(load_voice_prompt, ...)` strictly BEFORE acquiring
`inference_lock` — the same unsynchronized-concurrent-inference shape PRs
#211/#212/#213/#230 already closed everywhere else
(ml-explore/mlx#3078, Blaizzy/mlx-audio#638/#733).

MLX is unaffected: `load_voice_prompt_mlx` never creates — it only builds a
`{"ref_audio", "ref_text"}` dict from disk.

## The trap that makes the naive fix useless

`load_model()` (`core/engine/model_loader.py`) has NO memoization of any
kind — every call runs a full `from_pretrained` / `mlx_load_model` weight
construction (~2.5–3.5 GB, minutes). A naive "pre-load then lock" fix:

```python
await asyncio.to_thread(load_model, "clone")     # result discarded!
async with state.inference_lock:
    await asyncio.to_thread(load_voice_prompt, prompt_file, allow_create=True)
```

discards the pre-built model and lets `_auto_create_pt_from_wav` call
`load_model("clone")` again, internally, now INSIDE the lock — reintroducing
exactly the multi-minute-under-lock starvation the #212 split exists to
prevent. The fix must **forward** the already-built model via `clone_model=`
(mirrors `migrate_orphan_mlx_prompts(clone_model=None)`, already in this
file).

## RED

New module `tests/test_issue214_prompt_create_serialization.py`, written
before any implementation existed. At that point `VoicePromptCreateRequired`
did not exist on the engine facade, `qwen3_tts/server/prompt_loading.py` did
not exist, and neither `load_voice_prompt` nor `_load_voice_prompt_torch`
accepted `allow_create`/`clone_model`:

```
FAILED ...TestLoadVoicePromptSerializedOrdering::test_already_loaded_clone_model_skips_load_model
  - ImportError: cannot import name 'VoicePromptCreateRequired' from 'qwen3_tts.core.engine'
FAILED ...TestLoadVoicePromptSerializedOrdering::test_fast_path_never_touches_lock_or_load_model
  - ModuleNotFoundError: No module named 'qwen3_tts.server.prompt_loading'
FAILED ...TestLoadVoicePromptSerializedOrdering::test_load_model_unlocked_then_create_locked
  - ImportError: cannot import name 'VoicePromptCreateRequired' from 'qwen3_tts.core.engine'
FAILED ...TestLoadVoicePromptSerializedOrdering::test_mlx_is_a_provable_no_op
  - ModuleNotFoundError: No module named 'qwen3_tts.server.prompt_loading'
FAILED ...TestEngineAllowCreateContract::test_allow_create_true_never_raises_create_required
  - TypeError: load_voice_prompt() got an unexpected keyword argument 'allow_create'
FAILED ...TestEngineAllowCreateContract::test_corrupt_pt_no_wav_reraises_original_error
  - TypeError: _load_voice_prompt_torch() got an unexpected keyword argument 'allow_create'
FAILED ...TestEngineAllowCreateContract::test_corrupt_pt_with_wav_raises_create_required
  - ImportError: cannot import name 'VoicePromptCreateRequired' from 'qwen3_tts.core.engine'
FAILED ...TestEngineAllowCreateContract::test_missing_pt_no_wav_returns_none
  - TypeError: _load_voice_prompt_torch() got an unexpected keyword argument 'allow_create'
FAILED ...TestEngineAllowCreateContract::test_missing_pt_with_wav_raises_create_required
  - ImportError: cannot import name 'VoicePromptCreateRequired' from 'qwen3_tts.core.engine'
FAILED ...TestConcurrentCreateConvergence::test_two_racing_callers_create_exactly_once
  - ModuleNotFoundError: No module named 'qwen3_tts.server.prompt_loading'

10 failed
```

This is a genuine RED (real absent behavior — not a manufactured failure):
the collaborators the tests need did not exist yet.

## The fix

1. **`core/engine/voice_prompt.py`**
   - New `VoicePromptCreateRequired(Exception)` — internal control-flow
     signal, deliberately NOT in the `TTSError` hierarchy
     (`core/config/errors.py`): it is raised in a worker thread, caught one
     frame later in `server/prompt_loading.py`, and never reaches a user.
   - `_auto_create_pt_from_wav(..., *, model=None)` — calls
     `load_model("clone")` internally ONLY when `model is None`.
   - `_load_voice_prompt_torch(prompt_file, *, allow_create: bool = True,
     clone_model=None)` — when `allow_create=False` and a create WOULD be
     needed (missing `.pt` + sibling `.wav`, or the corrupt-`.pt` fallback +
     sibling `.wav`), raises `VoicePromptCreateRequired(prompt_file)` instead
     of running the create inline. The `.wav`-exists check is duplicated at
     both decision points (also done again inside `_auto_create_pt_from_wav`)
     because the allow_create decision must happen BEFORE calling it.
     `allow_create=True` (the default) preserves every existing direct
     caller's behavior byte-for-byte, including
     `tests/test_voice_prompts.py::TestCorruptPtFallback`'s positional calls.
   - `load_voice_prompt(prompt_file, *, allow_create=True, clone_model=None)`
     forwards both kwargs to the torch path; MLX ignores them (never raises
     `VoicePromptCreateRequired`).
   - The top-of-function cache re-check inside `_load_voice_prompt_torch` is
     annotated as load-bearing for concurrent convergence: two callers racing
     the same missing prompt each run an unlocked `allow_create=False` probe
     (raises, no create), then serialize on `inference_lock` for the
     `allow_create=True` retry. The FIRST locked caller creates and populates
     the cache; the SECOND locked caller's own top-of-function cache check
     then returns the cached result instead of creating a second time.

2. **New `qwen3_tts/server/prompt_loading.py`** — `load_voice_prompt_serialized(state, prompt_file)`:
   - Module scope imports `asyncio` ONLY; engine imports are function-local.
   - Unlocked probe: `load_voice_prompt(prompt_file, allow_create=False)`. If
     it succeeds (or raises `FileNotFoundError`/returns `None` per the
     existing contract), that result is returned directly — no lock, no
     `load_model` call.
   - On `VoicePromptCreateRequired`: reuse `state.models.get("clone")` if
     already loaded; otherwise build it via `load_model("clone",
     warmup=False)` OUTSIDE the lock.
   - Re-enter `load_voice_prompt(prompt_file, allow_create=True,
     clone_model=model)` under `state.inference_lock` as a leaf acquisition —
     the locked section now runs create inference ONLY, never weight
     construction.

3. **Three call sites swapped** — `await asyncio.to_thread(load_voice_prompt,
   pf)` replaced with `await load_voice_prompt_serialized(state, pf)` (or
   `app_state` in `websocket.py`, matching that module's local parameter
   name) in `app_generation.py` (batch `/generate` and streaming
   `/generate-stream`) and `websocket.py` (`/ws`). Every surrounding
   `try/except FileNotFoundError` → 404/error-frame and `if voice_prompt is
   None` block is unchanged — `load_voice_prompt_serialized` is a drop-in
   replacement with the same contract.

4. **Facade export**: `VoicePromptCreateRequired` added to
   `core/engine/__init__.py`'s imports and `__all__`.

## GREEN

```
tests/test_issue214_prompt_create_serialization.py .......... [10 passed]
tests/test_voice_prompts.py ................................ [46 passed]
```

## An existing test's assertion needed updating — annotating the type made a latent Path/str mismatch visible

`_load_voice_prompt_torch` previously had NO type annotations at all, so
mypy (repo config: neither `disallow_untyped_defs` nor
`check_untyped_defs`) never checked its body. Adding the pinned
`allow_create: bool = True` annotation to its signature made the function
"partially annotated," which mypy DOES check regardless of the
`check_untyped_defs` setting. That surfaced five pre-existing latent
`arg-type` errors: `VOICE_PROMPTS_DIR` is a `pathlib.Path`
(`core/config/paths.py`) passed into `safe_path_join(base_dir: str, ...)`,
which only tolerates it at runtime via an internal `str(base_dir)`. Fixed
locally by wrapping the five call sites in this function with
`str(VOICE_PROMPTS_DIR)` — a real correctness fix (matches the declared
type), not a suppression, and scoped to the one function my change newly
exposed to type checking.

## Two existing tests were updated — justification for each

**`tests/test_voice_prompts.py::TestMLXVoicePrompt::test_load_voice_prompt_dispatch_torch`**
asserted `mock.assert_called_once_with("test.pt")` — i.e. that
`load_voice_prompt` dispatches to `_load_voice_prompt_torch` with exactly
one positional argument. Since `load_voice_prompt` now always forwards
`allow_create`/`clone_model` (with their pre-#214 defaults) to preserve every
existing caller's behavior, the dispatch call legitimately gained two
keyword arguments. Updated to
`mock.assert_called_once_with("test.pt", allow_create=True,
clone_model=None)` — a strictly more precise assertion pinning the new
forwarding contract, not a weakening.

**`tests/test_generation_offload.py::TestGenerationOffload::test_voice_prompt_load_is_offloaded`**
is a source-inspection guard asserting `load_voice_prompt` is dispatched via
a literal `asyncio.to_thread(load_voice_prompt` in `app_generation.py`. That
offload now lives one layer down, inside
`load_voice_prompt_serialized` (`server/prompt_loading.py`), which
`app_generation.py` calls directly (`await
load_voice_prompt_serialized(...)`). The property the guard protects — voice
prompt loading never blocks the event loop — is still true, just relocated.
Updated to assert (a) `app_generation.py` calls
`load_voice_prompt_serialized` and (b) `prompt_loading.py` itself dispatches
`load_voice_prompt` via `asyncio.to_thread`. Verified this is not a
regression: `python -m pytest tests/ -m "not e2e" -q
--continue-on-collection-errors` went from 3027 passed (branch point,
`git stash -u`) to 3037 passed on this change — exactly the 10 new tests in
`test_issue214_prompt_create_serialization.py`, zero elsewhere.

## Mutation evidence

Passing tests prove little on their own; each guard was verified to **fail**
against a targeted mutant, then reverted.

| Mutant | Before | After |
|---|---|---|
| `load_voice_prompt_serialized` forwards `clone_model=None` instead of the pre-built model to the locked call | **KILLED** — `test_load_model_unlocked_then_create_locked`, `test_already_loaded_clone_model_skips_load_model`, `test_two_racing_callers_create_exactly_once` (3 failed) | GREEN after revert |
| The unlocked `allow_create=False` probe is skipped — the function always goes straight to `load_model`+lock+create | **KILLED** — `test_fast_path_never_touches_lock_or_load_model`, `test_load_model_unlocked_then_create_locked`, `test_mlx_is_a_provable_no_op` (3 failed) | GREEN after revert |
| `load_model` construction moved INSIDE `async with state.inference_lock` (the exact naive-fix trap the brief calls out) | **KILLED** — `test_load_model_unlocked_then_create_locked`: `[True, True] != [False, True]` | GREEN after revert |
| `_load_voice_prompt_torch`'s top-of-function cache re-check disabled (`if False and prompt_file in _torch_prompt_cache`) | **KILLED** — `test_two_racing_callers_create_exactly_once`: `2 != 1` | GREEN after revert |

Every mutant was applied to the actual source, run against the real test
file, observed to fail with the assertion shown, then reverted and
re-verified GREEN before moving to the next.

## Gates

- `ruff check qwen3_tts tests` — All checks passed (required `# noqa: N818`
  on `VoicePromptCreateRequired` — the brief pins that exact name, which
  ruff's `N818` rule wants suffixed `Error`; it is deliberately not an
  `Error` subclass per its own docstring)
- `mypy qwen3_tts/{core,server,interface}` — Success, 55 source files
- `bandit -r qwen3_tts -c pyproject.toml -q` — exit 0 (pre-existing B104
  `nosec` warnings, unrelated files, unchanged)
- `pytest tests/test_issue214_prompt_create_serialization.py
  tests/test_voice_prompts.py -v` — 56 passed
- `pytest tests/ -m "not e2e" -q` — one collection error,
  `tests/evaluations/test_speaker_similarity.py`
  (`RuntimeError: Could not load libtorchcodec` — a local ffmpeg/torchcodec
  dylib mismatch on this machine). **Proven pre-existing**: reproduced
  identically with `git stash -u` (this branch's changes fully removed).
  With `--continue-on-collection-errors` (not `--ignore` — nothing is
  skipped, the failure is still reported): 3037 passed, 4 skipped vs. 3027
  passed, 4 skipped on the stashed baseline — exactly +10, matching the new
  test module, zero regressions.
- `python tests/run_batches.py --batch 3` — Ran 675 tests, OK (skipped=3)

## Not covered

No e2e test drives a real `/generate` against a live server with a missing
`.pt` during a real concurrent generation — this is a torch-only defect and
the local development machine runs the MLX backend, so the unit-test mutation
evidence above is the only available proof on this box.

## Deferred (tracked, not fixed here)

- `load_model` deduplication when two racing callers both need to build the
  clone model (both currently call `load_model("clone")` once each,
  unlocked, if neither has a pre-loaded model — wasteful but not unsafe,
  since only the CREATE is serialized). Tracked for a follow-up
  `test_issue214_load_model_dedup` module per the registration ordering note
  in the plan.
- The pre-existing `tests/evaluations/test_speaker_similarity.py` collection
  failure (local ffmpeg/torchcodec dylib mismatch) is environmental, not a
  code defect, and out of scope for this change.
