---
name: prefer-batch-runner-over-raw-pytest
enabled: true
event: bash
action: warn
pattern: "python -m (pytest|unittest discover)\\s+(tests/[^t]|tests/ |-v\\s+tests/[^t])"
---
[Hook] This project uses a batch runner to prevent hangs and OOM kills (exit code 137).

Use instead:
  python tests/run_batches.py        # all batches
  make test-batch                    # same via Makefile
  python tests/run_batches.py --batch 3  # specific batch

Running 2100+ tests raw is known to hit memory limits. If targeting a single file, use an explicit path:
  python -m pytest tests/test_specific.py
