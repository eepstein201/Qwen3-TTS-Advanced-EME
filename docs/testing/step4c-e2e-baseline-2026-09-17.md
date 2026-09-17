# Step 4C — pre-change live E2E baseline (and the Step 4E failures it surfaced)

**Captured:** 2026-09-17 on `e12bf21d` (= `origin/main` at the time), **before** any Step 4C
extraction edits existed on disk. Env: `conda run -n qwen3-tts-mlx`, live server on :5123.

Command (the `-m e2e` flag is mandatory — `pytest.ini` deselects the `e2e` marker, so a bare run
reports "0 selected" and looks like a clean pass while verifying nothing):

```
conda run -n qwen3-tts-mlx python -m pytest tests/test_e2e_playwright.py \
  tests/test_e2e_history_clear_copy.py tests/test_e2e_tab_navigation.py \
  tests/test_e2e_wavesurfer_live.py -m e2e -v --tb=short -p no:cacheprovider
```

**Result: 3 failed, 22 passed, 11 subtests passed in 503.17s.** Collection counts (with `-m e2e`):
playwright 13 · history_clear_copy 3 · tab_navigation 3 · wavesurfer_live 6 = **25**.

Two uses:

1. **Step 4C's exit criterion.** The extraction is behavior-identical only if the post-change run
   reproduces this exactly — same 3 failures, same 22 passes. A post-change *green* means the
   extraction changed behavior and is a FAILURE of 4C, not a bonus fix.
2. **Step 4E's evidence.** The three failures are proven pre-existing on unmodified `main`; they
   went unreported because `make test-e2e`/batch 6 runs only `test_e2e_playwright.py`.

---

## Raw output

```
ERROR conda.cli.main_run:execute(148): `conda run python -m pytest tests/test_e2e_playwright.py tests/test_e2e_history_clear_copy.py tests/test_e2e_tab_navigation.py tests/test_e2e_wavesurfer_live.py -m e2e -v --tb=short -p no:cacheprovider` failed. (See above for error)
============================= test session starts ==============================
platform darwin -- Python 3.11.14, pytest-9.0.2, pluggy-1.6.0 -- /Users/ericepstein/miniforge3/envs/qwen3-tts-mlx/bin/python
rootdir: /Users/ericepstein/Qwen3-TTS_UserFiles/.claude/worktrees/e2e-shared-page-object
configfile: pytest.ini
plugins: anyio-4.12.1, asyncio-1.3.0, cov-7.0.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 25 items

tests/test_e2e_playwright.py::TestE2EPlaywright::test_01_clone_generation PASSED [  4%]
tests/test_e2e_playwright.py::TestE2EPlaywright::test_02_design_generation PASSED [  8%]
tests/test_e2e_playwright.py::TestE2EPlaywright::test_03_custom_generation PASSED [ 12%]
tests/test_e2e_playwright.py::TestE2EPlaywright::test_04_clone_validation_empty_text PASSED [ 16%]
tests/test_e2e_playwright.py::TestE2EPlaywright::test_05_design_validation_empty_description PASSED [ 20%]
tests/test_e2e_playwright.py::TestE2EPlaywright::test_06_cancel_generation PASSED [ 24%]
tests/test_e2e_playwright.py::TestE2EPlaywright::test_07_concurrent_generation PASSED [ 28%]
tests/test_e2e_playwright.py::TestE2EPlaywright::test_08_load_model PASSED [ 32%]
tests/test_e2e_playwright.py::TestE2EPlaywright::test_09_unload_model PASSED [ 36%]
tests/test_e2e_playwright.py::TestE2EPlaywright::test_10_load_unload_cycle PASSED [ 40%]
tests/test_e2e_playwright.py::TestE2EPlaywright::test_11_history_panel_below_tabs PASSED [ 44%]
tests/test_e2e_playwright.py::TestE2EPlaywright::test_12_json_sidecar_and_history_columns PASSED [ 48%]
tests/test_e2e_playwright.py::TestE2EPlaywright::test_13_history_row_populates_seed_in_all_tabs PASSED [ 52%]
tests/test_e2e_history_clear_copy.py::TestE2EHistoryClearCopy::test_01_copy_transcript_to_clipboard_with_visible_status PASSED [ 56%]
tests/test_e2e_history_clear_copy.py::TestE2EHistoryClearCopy::test_02_remove_row_shows_status_and_clears_waveform PASSED [ 60%]
tests/test_e2e_history_clear_copy.py::TestE2EHistoryClearCopy::test_03_clear_all_two_step_with_visible_status FAILED [ 64%]
tests/test_e2e_tab_navigation.py::TestTabNavigationNeverKillsThePage::test_baseline_load_produces_no_page_errors PASSED [ 68%]
tests/test_e2e_tab_navigation.py::TestTabNavigationNeverKillsThePage::test_dataframe_tabs_still_render_after_the_sweep 
tests/test_e2e_tab_navigation.py::TestTabNavigationNeverKillsThePage::test_dataframe_tabs_still_render_after_the_sweep PASSED [ 72%]
tests/test_e2e_tab_navigation.py::TestTabNavigationNeverKillsThePage::test_full_tab_sweep_produces_no_page_errors 
tests/test_e2e_tab_navigation.py::TestTabNavigationNeverKillsThePage::test_full_tab_sweep_produces_no_page_errors PASSED [ 76%]
tests/test_e2e_wavesurfer_live.py::TestWaveSurferLoadsViaProductionPath::test_gradio_still_emits_the_module_script_into_the_dom FAILED [ 80%]
tests/test_e2e_wavesurfer_live.py::TestWaveSurferLoadsViaProductionPath::test_page_loads_without_javascript_errors PASSED [ 84%]
tests/test_e2e_wavesurfer_live.py::TestWaveSurferLoadsViaProductionPath::test_player_controls_render FAILED [ 88%]
tests/test_e2e_wavesurfer_live.py::TestWaveSurferLoadsViaProductionPath::test_player_factory_is_defined_without_test_side_injection PASSED [ 92%]
tests/test_e2e_wavesurfer_live.py::TestWaveSurferLoadsViaProductionPath::test_script_reexecutor_ran_and_injected_blob_module PASSED [ 96%]
tests/test_e2e_wavesurfer_live.py::TestWaveSurferLoadsViaProductionPath::test_streaming_player_module_body_executed PASSED [100%]

=================================== FAILURES ===================================
____ TestE2EHistoryClearCopy.test_03_clear_all_two_step_with_visible_status ____
tests/test_e2e_history_clear_copy.py:383: in test_03_clear_all_two_step_with_visible_status
    self._wait_for_visible_status("cleared", timeout=10_000)
tests/test_e2e_history_clear_copy.py:268: in _wait_for_visible_status
    self.page.wait_for_function(
../../../../miniforge3/envs/qwen3-tts-mlx/lib/python3.11/site-packages/playwright/sync_api/_generated.py:11599: in wait_for_function
    self._sync(
../../../../miniforge3/envs/qwen3-tts-mlx/lib/python3.11/site-packages/playwright/_impl/_page.py:1110: in wait_for_function
    return await self._main_frame.wait_for_function(**locals_to_params(locals()))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../../../miniforge3/envs/qwen3-tts-mlx/lib/python3.11/site-packages/playwright/_impl/_frame.py:878: in wait_for_function
    await self._channel.send("waitForFunction", self._timeout, params)
../../../../miniforge3/envs/qwen3-tts-mlx/lib/python3.11/site-packages/playwright/_impl/_connection.py:69: in send
    return await self._connection.wrap_api_call(
../../../../miniforge3/envs/qwen3-tts-mlx/lib/python3.11/site-packages/playwright/_impl/_connection.py:559: in wrap_api_call
    raise rewrite_error(error, f"{parsed_st['apiName']}: {error}") from None
E   playwright._impl._errors.TimeoutError: Page.wait_for_function: Timeout 10000ms exceeded.
_ TestWaveSurferLoadsViaProductionPath.test_gradio_still_emits_the_module_script_into_the_dom _
tests/test_e2e_wavesurfer_live.py:244: in test_gradio_still_emits_the_module_script_into_the_dom
    self.assertGreater(
E   AssertionError: 0 not greater than 1000 : module script present but suspiciously small: {'module_scripts_in_dom': 1, 'largest_module_script': 0, 'blob_scripts_in_head': 0, 'get_or_create_player': 'undefined', 'streaming_players': 'undefined', 'clone_waveform': False, 'clone_play_btn': False}
_______ TestWaveSurferLoadsViaProductionPath.test_player_controls_render _______
tests/test_e2e_wavesurfer_live.py:252: in test_player_controls_render
    self.assertTrue(probe["clone_waveform"], f"#clone-waveform missing: {probe}")
E   AssertionError: False is not true : #clone-waveform missing: {'module_scripts_in_dom': 1, 'largest_module_script': 0, 'blob_scripts_in_head': 0, 'get_or_create_player': 'undefined', 'streaming_players': 'undefined', 'clone_waveform': False, 'clone_play_btn': False}
=========================== short test summary info ============================
FAILED tests/test_e2e_history_clear_copy.py::TestE2EHistoryClearCopy::test_03_clear_all_two_step_with_visible_status - playwright._impl._errors.TimeoutError: Page.wait_for_function: Timeout 10000ms exceeded.
FAILED tests/test_e2e_wavesurfer_live.py::TestWaveSurferLoadsViaProductionPath::test_gradio_still_emits_the_module_script_into_the_dom - AssertionError: 0 not greater than 1000 : module script present but suspiciously small: {'module_scripts_in_dom': 1, 'largest_module_script': 0, 'blob_scripts_in_head': 0, 'get_or_create_player': 'undefined', 'streaming_players': 'undefined', 'clone_waveform': False, 'clone_play_btn': False}
FAILED tests/test_e2e_wavesurfer_live.py::TestWaveSurferLoadsViaProductionPath::test_player_controls_render - AssertionError: False is not true : #clone-waveform missing: {'module_scripts_in_dom': 1, 'largest_module_script': 0, 'blob_scripts_in_head': 0, 'get_or_create_player': 'undefined', 'streaming_players': 'undefined', 'clone_waveform': False, 'clone_play_btn': False}
========= 3 failed, 22 passed, 11 subtests passed in 503.17s (0:08:23) =========

[exited with code 1]
```
