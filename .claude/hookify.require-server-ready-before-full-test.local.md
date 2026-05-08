---
name: require-server-ready-before-full-test
enabled: true
event: bash
action: warn
pattern: "python -m pytest(?! tests/test_[a-z_]+\\.py)|python tests/run_full_suite\\.py"
---
[Hook] Running the full test suite requires the server to be running with all models loaded.

Before proceeding:
1. tts server stop && tts server start
2. Load all 3 models (clone, design, custom)
3. Confirm via: tts server status

Cancel this run if server state has not been confirmed.
