# Gradio InvalidPathError + History Playback Fix

**Status: Implementation complete. All 6 batches pass (1,591 tests). Awaiting commit.**

## Bugs Fixed
1. `InvalidPathError` when generating audio via `tts ui` — `~/Downloads` not in Gradio `allowed_paths`
2. History row clicks fail to play audio — same root cause

## Root Cause
Config drift between two `demo.launch()` call sites:
- `_facade.py:main()` had `allowed_paths`
- `generate_server.py:build_ui_and_launch()` did NOT

## Changes Made

### Core fix (DRY shared launch kwargs)
| File | Change |
|------|--------|
| `qwen3_tts/interface/ui/shared.py` | Added `_resolve_output_dir()`, `get_gradio_launch_kwargs()` — single source of truth |
| `qwen3_tts/interface/generate_server.py` | Uses `**get_gradio_launch_kwargs(config)`, `logger.error` instead of `print` |
| `qwen3_tts/interface/ui/_facade.py` | Uses shared launch kwargs in `main()` |

### Security hardening
| File | Change |
|------|--------|
| `qwen3_tts/interface/ui/_facade.py` | `on_history_select`: path containment + temp copy; `_sanitize_voice_name`: allowlist regex |
| `qwen3_tts/interface/ui/shared.py` | `format_status_display`: `html.escape()` on server-derived strings |

### Dead code removal
| File | Change |
|------|--------|
| `qwen3_tts/interface/ui/generation.py` | Removed `_save_completed_audio`, `_validate_inputs`, unused `import base64` |
| `qwen3_tts/interface/ui/__init__.py` | Removed dead exports |

### Test fixes
| File | Change |
|------|--------|
| `tests/test_ui_facade.py` | 21 new tests (launch kwargs, output dir, history select, voice name, XSS) |
| `tests/test_fastapi_app_ext2.py` | Fixed `test_auto_shutdown`: mock `os.kill` not `sys.exit` |
| `tests/test_websocket.py` | Added `inference_lock` to `_setup_app_state` |
| `tests/test_generate_server.py` | Fixed `test_build_ui_and_launch_no_port`: check `logger.error` not `print` |
| `tests/test_wavesurfer_js.py` | Removed dead `TestSaveCompletedAudio` tests |
| `tests/test_ui_generation_ext.py` | Removed dead tests for removed functions |
| `tests/run_batches.py` | Added `_ensure_models_loaded` setup for Batch 6 (auto-loads clone/design/custom) |

## Test Results (all passing)
- Batch 1 (Core Utilities): 257 tests
- Batch 2 (Voice & CLI): 479 tests
- Batch 3 (Server Infrastructure): 359 tests
- Batch 4 (Engine & UI): 443 tests
- Batch 5 (Optional Tests): 43 tests
- Batch 6 (E2E Playwright): 10 tests (server running, all models loaded)
- **Total: 1,591 tests**

## Next Steps
- [ ] Commit changes
- [ ] Manual smoke test: `tts ui` → generate → click history row
- [ ] Clean up `test_results.txt` temp file
