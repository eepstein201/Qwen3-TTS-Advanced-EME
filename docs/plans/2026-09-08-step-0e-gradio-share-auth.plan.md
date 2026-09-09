# Step 0E execution plan — Gradio share requires auth (amber-drift)

Spec: `docs/plans/2026-09-06-consolidated-backlog-priority.plan.md` Wave 0, Step 0E (reachable, read 2026-09-08).
BASE: `c7483e1` (main = the #273 reconciliation merge). Branch: `fix/gradio-share-requires-auth`.
Worktree: primary (standing 0B–0D precedent).

## Survey (verified against source @ c7483e1, 2026-09-08)

- **Shared choke point:** `get_gradio_launch_kwargs(config)` (`interface/ui/shared.py:872-887`) —
  returns server_name (`"0.0.0.0" if IN_COLAB else "127.0.0.1"`, nosec), `allowed_paths` =
  `{output_dir, realpath(~/Downloads), tempdir}`, theme, css. **No `auth=` anywhere in the
  package** (grep-exhaustive).
- **Two real launch sites** (the step's third cite, `ui/__init__.py:21`, is a docstring example,
  not code):
  1. `_facade.py:571-581` `main()` — `share = args.share or IN_COLAB` (:573); passes
     `share=share, inbrowser=..., **get_gradio_launch_kwargs(config)`.
  2. `generate_server.py:124-133` `build_ui_and_launch(config)` (the `tts ui` path) —
     `share = bool(os.environ.get("TTS_UI_SHARE")) or IN_COLAB` (:124); same shape.
  `tts ui --share` (`cli_config.py:218-226`) sets `TTS_UI_SHARE=1` → `launch_gradio_ui` →
  `ensure_server_running` → `build_ui_and_launch`. `--share` also on the facade argparse
  (`_facade.py:544`). `IN_COLAB` forces share at BOTH sites.
- **Threat model:** the UI process holds the server bearer token (`server_request` reads the token
  file), so a shared `*.gradio.live` URL hands a stranger: generation, prompt delete/rename,
  `/update-model-config`, `/shutdown`, hard-delete of output files, and Gradio's file route over
  `allowed_paths` — which today includes **all of `~/Downloads`**.
- **allowed_paths narrowing:** the app's own files live under
  `resolve_automated_output_dir(config)` / `resolve_manual_downloads_dir(config)`
  (`shared.py:631-663`, subdirs of the configured output dir). The `realpath(~/Downloads)` entry is
  the over-broad one; `tempdir` must stay (history_panel copies to temp for Gradio serving,
  `history_panel.py:94,326`). New allowed set = `{output_dir, tempdir}`.
- **Existing tests:** `test_ui_headless.py:57` builds its OWN launch string in a subprocess (does
  not use the helper) — unaffected. `test_ui_facade.py` (batch 4) patches facade attrs with
  MagicMock; `test_ui_port_flag.py` (batch 4) and `test_colab_paths.py` (batch 4) exercise
  `build_ui_and_launch`; `test_ui_generation_ext.py` shows the `IN_COLAB` module-global patch
  precedent (`_cfg.IN_COLAB = True`, restore via addCleanup at TOP). `secrets` module usage has
  precedent in `server/app_generation.py:12`, `app_lifespan.py:16`, `app.py:20`.
- **Batch ownership:** touched test files sit in batch 4 (UI) and batch 2 (`test_server_helpers`).
- **ARCHITECTURE.md** `## Security` (:62-70) is a server-surface bullet list — the UI surface is
  absent, as the step claims.

## Design rulings (controller, binding)

1. `get_gradio_launch_kwargs(config, *, share: bool)` gains an explicit keyword-only `share`
   parameter. The single enforcement point lives in the helper: **share truthy ⇒ `auth` tuple
   present**; share falsy ⇒ no `auth` key (byte-identical to today). Call sites keep computing and
   passing their own `share=` to `demo.launch` (no duplicate-kwarg conflict; sites pass the same
   value into the helper).
2. Credentials: env `TTS_UI_USERNAME` + `TTS_UI_PASSWORD` (both-or-nothing) override; otherwise
   auto-generate (`secrets.token_urlsafe`, user `tts-` + 6 chars) and `print()` both to console —
   the step's "print it to the console/log". Auto-gen makes creds always exist; the step's
   "refuse to launch" clause is the fail-closed path: if credential material cannot be established
   (secrets raises, or env gives only one of the pair), raise RuntimeError BEFORE returning kwargs
   — never fall through to an unauthenticated shared launch.
3. `allowed_paths` = `{output_dir, tempdir}` (drop `realpath(~/Downloads)`). `output_dir` is the
   parent of both app subfolders and is already home-contained via
   `resolve_history_output_dir`'s traversal guard.

## Global Constraints (bind every task)

- Lazy imports: no torch/mlx/transformers/gradio at module scope in touched files (gradio is
  function-local in shared.py today — keep it that way).
- TDD: RED commit before GREEN; every new test verified to fail on unmodified source, with the
  failure output in the implementer report. Mutation-verify guards (edit file, grep/diff the edit
  landed, re-run) — `conda run` does not forward stdin.
- Env: `conda run -n qwen3-tts-mlx` for python/pytest/ruff; **mypy ENTRY-POINT form only**
  (`conda run -n qwen3-tts-mlx mypy qwen3_tts/core qwen3_tts/server qwen3_tts/interface`);
  torchless RUN-not-skip via `./.venv-310/bin/python`.
- Never stage `config.json`; explicit-pathspec commits; never `--amend`; NO AI attribution; commit
  subjects single-line, NO backticks.
- New test module MUST be registered in `tests/run_batches.py` BATCHES (batch 4, with the other UI
  tests) — `test_batches_coverage.py` enforces.
- Existing UI suites (`test_ui_facade`, `test_ui_port_flag`, `test_colab_paths`,
  `test_ui_headless`, `test_server_helpers`) stay green UNCHANGED.

## Task 1 — RED→GREEN: the auth invariant + narrowed allowed_paths

- **RED first** — new `tests/test_ui_share_auth.py`:
  1. `share=True` ⇒ kwargs contain non-empty `auth` (2-tuple user/password); credentials printed
     (capture via `patch("builtins.print")` or capsys-equivalent); when env creds absent, values
     are generated (user startswith `tts-`).
  2. `share=False` ⇒ NO `auth` key (and no print of credentials).
  3. Env override: `TTS_UI_USERNAME`/`TTS_UI_PASSWORD` both set ⇒ used verbatim; only one set ⇒
     RuntimeError (fail-closed), never partial creds.
  4. Fail-closed: `secrets.token_urlsafe` patched to raise ⇒ RuntimeError propagates from the
     helper (launch never reached).
  5. Per-site behavioral: patch `build_ui` to return a MagicMock demo and capture
     `demo.launch.call_args.kwargs` — facade `main()` under `TTS_UI_SHARE=1` ⇒ launch received
     `share=True` AND the helper's auth is in kwargs; `build_ui_and_launch` same. (Follow
     test_ui_facade's patch-target conventions; patch where defined.)
  6. allowed_paths: output_dir present, tempdir present, `os.path.realpath("~/Downloads")`
     ABSENT (even when output_dir is the default under Downloads).
- **GREEN** — `shared.py`: helper signature + auth block + narrowed allowed set; both call sites
  pass `share=share` into the helper; no other behavior change (`git diff -w` minimal).
- **Verify:** new module green on mlx + torchless (RUN not skip); named suites
  (`test_ui_facade.py test_ui_port_flag.py test_colab_paths.py test_ui_headless.py
  test_server_helpers.py test_ui_generation_ext.py`) green UNCHANGED; batches 2+4 green; ruff;
  mypy (entry-point); stale-pin grep (`grep -rn "get_gradio_launch_kwargs" qwen3_tts tests` —
  every caller updated; update any source-string pin that names changed lines).
- **Commits:** (1) RED tests + batch registration; (2) GREEN implementation. Suggested subjects:
  `test(ui): pin share-requires-auth and the narrowed allowed paths` /
  `fix(ui): Gradio share now requires generated or configured credentials`.

## Task 2 — gates, evidence, notebook mirror, docs (final task)

- Full gates from the FINAL state: full non-e2e
  (`conda run -n qwen3-tts-mlx python -m pytest tests/ -m "not e2e" -q --tb=short --ignore=tests/evaluations`;
  prior 3224/4/0 — expect +new); batch 2 + batch 4 runners; ruff; mypy entry-point; torchless
  RUN-not-skip on the new class; stale-count grep (`git grep -n "3224\|3217" -- CLAUDE.md docs/`
  — sync only files clean in `git status`).
- **Colab mirror:** `colab_notebook.ipynb` cell 6 — direct `demo.launch(share=True, ...)`; add
  auth kwargs consistent with the helper's behavior (generate + print; keep the cell
  self-contained as it is today). Valid notebook JSON after edit.
- **Evidence doc:** `docs/testing/ui-share-auth.tdd.md` (0C/0D style) — threat model, the two
  launch sites + the docstring-only third cite, RED verbatim, design rulings 1–3 with the
  fail-closed rationale, Final-state block with counted totals.
- **Tracked plan copy:** `docs/plans/2026-09-08-step-0e-gradio-share-auth.plan.md` — byte-faithful
  copy of THIS execution plan's spec section (Survey + Design rulings + tasks), 0D precedent.
- **ARCHITECTURE.md:** Security section gains a Gradio-UI-surface entry (launch-auth invariant,
  allowed_paths scope, the bearer-token custody rationale). No fabrication — mirror what shipped.
- **CLAUDE.md:** at most ONE invariant sentence (share ⇒ auth at every launch site; allowed_paths
  narrowed to the app output dir + temp). Check ≤300 lines before AND after (readlines() method).
- **Commits:** single-purpose: (1) notebook mirror; (2) evidence doc + tracked plan copy;
  (3) ARCHITECTURE.md + CLAUDE.md.
