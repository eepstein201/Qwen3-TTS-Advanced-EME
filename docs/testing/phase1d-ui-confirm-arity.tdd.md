# TDD Evidence — Phase 1d: UI confirm flows (U1 + U2)

**Source plan:** `~/.claude/plans/review-entire-repo-for-ancient-possum.md`, Phase 1d.
**Branch:** `fix/phase1-ui-confirm-arity` · **Base:** `main` @ `0ddaad8`

## Why this survived a test suite with 2,968 tests

Every existing test of these flows either exercised `confirm_step` in isolation
or **grepped the source for a symbol name**:

```python
def test_16_delete_confirm_state_in_manage_voices_tab(self):
    src = self._read_source("qwen3_tts/interface/ui/tabs_management.py")
    self.assertIn("delete_confirm_state", src)
```

A grep for `delete_confirm_state` passes whether the handler is correct or
completely broken. Nothing executed a wired handler, so two different defects
shipped in three documented destructive flows: Stop, Delete Voice, Unload Model.

## The seam used instead

Two real seams, no new production abstractions:

- **`_wire_generation_tab`** is called with `Mock(spec=gr.Button)` components
  (the pattern already in `test_ui_audio_reset.py`), so
  `cancel_btn.click.call_args` yields the actual `fn` **and** its `outputs`
  list. The handler is then executed directly.
- **`gr.Blocks()`** records every registered listener in `demo.fns`, each with
  its resolved `outputs`. Building `_build_manage_voices_tab` /
  `_build_manage_models_tab` inside a Blocks and looking up the handler by
  `__name__` gives the real wiring rather than a source string.

Both let the test assert the thing that actually broke: **returned arity vs.
wired output count**, and **the type stored into `gr.State`**.

## U1 — Stop returned 3 values for 4 wired outputs

`generation.py`'s `on_cancel_click` is wired to
`[cancel_confirm_state, cancel_btn, status, status_html]`. The fast path (:671)
and the confirmed path (:688) returned 4 values. The **arm** path (:673-678) and
the **canceled** path (:685) returned 3 — Gradio's `validate_outputs` raises
`ValueError`, so the documented Stop/confirm flow crashed instead of arming.

Line 673 additionally bound `ConfirmButton.click()`'s whole 4-tuple to
`new_state`, putting a tuple into the `gr.State`.

## U2 — Delete and Unload stored the 4-tuple into `gr.State`

`tabs_management.py:225` and `:392` did the same tuple binding. Here the arity
*matched* (5 values for 5 outputs), so Gradio accepted it silently and the bug
only bit on the **second** click, where `state.get("armed")` hits a tuple:
`AttributeError: 'tuple' object has no attribute 'get'`. Both destructive
confirms could therefore never reach their confirmed branch.

## RED

```
$ pytest tests/test_ui_confirm_patterns.py::TestConfirmHandlerContracts -v

FAILED test_cancel_arm_path_returns_one_value_per_wired_output
E   AssertionError: 3 != 4 : Stop arm path returned 3 values for 4 wired
    outputs; Gradio raises ValueError

FAILED test_cancel_canceled_path_returns_one_value_per_wired_output
E   AssertionError: 3 != 4 : Stop canceled path returned 3 values for 4 wired
    outputs; Gradio raises ValueError

FAILED test_cancel_arm_path_stores_a_state_dict_not_a_tuple
E   AssertionError: ({'armed': True, 'ts': ...}, {'value': 'Stop — click again
    to confirm', ...}, {...}, False) is not an instance of <class 'dict'>

FAILED test_delete_voice_second_click_reaches_the_confirmed_branch
E   AssertionError: ({'armed': True, 'ts': ...}, ...) is not an instance of
    <class 'dict'>

FAILED test_unload_model_second_click_reaches_the_confirmed_branch
E   AssertionError: ({'armed': True, 'ts': ...}, ...) is not an instance of
    <class 'dict'>

5 failed
```

## GREEN

Unpack the 4-tuple and discard what the branch does not forward; return one
value per wired output in every branch.

```
$ pytest tests/test_ui_confirm_patterns.py -q
33 passed
```

The two destructive tests do not stop at "did not raise" — they feed the state
from click 1 straight back into click 2 (exactly as Gradio does) and then
assert the destructive call was actually reached:

```python
delete_voice.assert_called_once_with("my-voice")
toggle_model.assert_called_once_with("clone", "unload")
```

Without that, a handler that silently swallowed the second click would still
pass.

**One test correction during GREEN:** the first drafts patched
`shared.delete_voice_prompt` and `model_management.unload_model`, neither of
which exists. The real confirmed-branch calls are
`voice_management.delete_voice(selected)` and
`model_management.toggle_model(mt, "unload")`. `patch` raised `AttributeError`
on the bad targets rather than silently creating them — the tests were wrong,
not the implementation.

## Stale e2e corrected

`test_e2e_playwright.py::test_09_unload_model` and `test_10_load_unload_cycle`
clicked "Unload" **once** and then waited for the model to report unloaded.
With a two-step confirm the first click only arms the button, so these
assertions could never have passed against the current wiring. Added
`GradioPage.click_confirm_button()`, which clicks, waits for the
`Confirm <label>? (click again)` label to appear, and clicks again — awaiting
the label rather than sleeping, so both clicks land inside the 5 s window.

**Not verified locally:** these are `-m e2e` tests requiring a live server,
loaded models and Chromium. The change is mechanical and the previous form was
provably unreachable, but it has not been executed here.

## Gates

```
$ pytest tests/ -m "not e2e" -q --ignore=tests/evaluations/test_speaker_similarity.py
2973 passed, 11 skipped, 88 deselected

$ python tests/run_batches.py --batch 4   →  1/1 batches passed
$ ruff check qwen3_tts tests              →  All checks passed
$ mypy qwen3_tts/{core,server,interface}  →  Success: no issues found in 53 source files
$ wc -l CLAUDE.md                         →  298 (unchanged)
```

`--ignore=tests/evaluations/test_speaker_similarity.py` is the plan's interim
workaround for pre-existing **P1** (libtorchcodec fails at collection, which
`Interrupt`s the whole run). Local-env-only; CI is unaffected. Still owned by
Phase 0, not fixed here.

## Browser verification

Per the repo's own recorded lesson ("MCP clicks miss a 5 s confirm window;
invoke the handler directly as the definitive check"), the definitive check
here is the direct handler execution above, which drives both clicks with no
timing dependency at all.
