# TDD Evidence: Step 0E — Gradio share requires auth (narrowed `allowed_paths`)

**Source plan:** docs/plans/2026-09-08-step-0e-gradio-share-auth.plan.md
(Step 0E of docs/plans/2026-09-06-consolidated-backlog-priority.plan.md; byte-faithful
copy of the execution plan's spec section)

**Branch:** `fix/gradio-share-requires-auth` (cut from main @ `c7483e1`, the #273
reconciliation merge)

## Problem and threat model

The Gradio UI had **no `auth=` anywhere in the package** (grep-exhaustive at BASE), while
two real launch paths can put it on a public URL: `--share`/`TTS_UI_SHARE` and `IN_COLAB`
(which forces share at BOTH sites). The UI process holds the server **bearer token**
(`server_request` reads the token file), so a shared `*.gradio.live` URL handed a stranger
the full authenticated server surface: generation, prompt delete/rename,
`/update-model-config`, `/shutdown`, and — via the web-UI Remove button — hard-delete of
output files. Gradio's `allowed_paths` made it worse: the shared set-literal was
`{output_dir, realpath(~/Downloads), tempdir}`, i.e. a blanket grant over **all of
`~/Downloads`**, not just the app's folder.

The fix is one enforcement point in the shared launch-kwarg helper:
`get_gradio_launch_kwargs(config, *, share: bool = False)` — share truthy ⇒ an `auth`
tuple is on the returned kwargs (fail-closed to establish), and `allowed_paths` narrows to
`{output_dir, tempdir}`.

## The launch sites

| Site | Share decision | Status |
|---|---|---|
| `ui/_facade.py` `main()` (`:571,580`) | `share = args.share or IN_COLAB` (`:573`) | real — forwards `share=share` into the helper |
| `interface/generate_server.py` `build_ui_and_launch` (`:105,132`; the `tts ui` path) | `share = bool(os.environ.get("TTS_UI_SHARE")) or IN_COLAB` (`:125`) | real — same shape |
| `ui/__init__.py:21` | module docstring usage example — `demo.launch()` with no kwargs | **docstring only**, not code; zero helper hits (Task 1 census). Untouched. |

`tts ui --share` (`cli_config.py:218-226`) sets `TTS_UI_SHARE=1` → `build_ui_and_launch`;
`--share` is also on the facade argparse (`_facade.py:544`).

## Task report

| Task | Scope | RED | GREEN |
|---|---|---|---|
| 1 | helper auth block + narrowed `allowed_paths` + both call sites; `tests/test_ui_share_auth.py` (new, batch 4) | **9 failed / 2 passed** at BASE+tests (the three right-reasons below) | 9 passed + 2 subtests, both interpreters |
| 2 (fold) | headless launch literal modernization; notebook mirror; gates + docs | n/a (docs/input folds; live-launch verified, below) | full non-e2e **3233 passed / 4 skipped / 0 failed** |

## The RED run (Task 1, verbatim)

`conda run -n qwen3-tts-mlx python -m pytest tests/test_ui_share_auth.py -v` against
BASE+tests (pre-fix tree) → **9 failed, 2 passed**. The "2 passed" are intentional
pins that are green by construction: `test_default_call_without_share_kwarg_has_no_auth`
(the share-falsy byte-identical-dict ruling) and the PARENT line of
`test_share_true_single_env_var_fails_closed`, whose 2 subtests both SUBFAILED and ARE
counted in the 9. Every failure was one of exactly three right-reasons:

1. **`TypeError` on `share`** — the 6 helper tests + 2 subtests, because the target API
   did not exist yet:
   ```
   FAILED tests/test_ui_share_auth.py::TestShareRequiresAuthHelper::test_share_true_generates_and_prints_credentials - TypeError: get_gradio_launch_kwargs() got an unexpected keyword argument 'share'
   SUBFAILED(partial={'TTS_UI_PASSWORD': '', 'TTS_UI_USERNAME': 'alice'}) tests/test_ui_share_auth.py::TestShareRequiresAuthHelper::test_share_true_single_env_var_fails_closed - TypeError: get_gradio_launch_kwargs() got an unexpected keyword argument 'share'
   ```
2. **Downloads-present `AssertionError`** — the narrowing test, which uses the bare
   one-arg call deliberately so its RED reason is the narrowing violation, not the
   signature:
   ```
   FAILED tests/test_ui_share_auth.py::TestShareRequiresAuthHelper::test_allowed_paths_narrow_to_output_dir_and_tempdir - AssertionError: '/Users/ericepstein/Downloads' unexpectedly found in ['/Users/ericepstein/Downloads/Qwen3-TTS Output', '/Users/ericepstein/Downloads', '/var/folders/1c/6pc7zwfs2md_h34lpj7wfjz40000gp/T']
   ```
3. **`{} != {'share': True}`** — both per-site tests: the spy on
   `shared.get_gradio_launch_kwargs` captured no `share` kwarg because neither call site
   forwarded its share decision yet.

## GREEN

- `get_gradio_launch_kwargs(config, *, share=False)` (`ui/shared.py:872`) — keyword-only
  `share`, default False, so every existing one-arg caller (six in `test_ui_facade.py`)
  is unchanged and the share-falsy dict is byte-identical to BASE (no `auth` key, no
  credential prints).
- share truthy: BOTH `TTS_UI_USERNAME`+`TTS_UI_PASSWORD` used verbatim; exactly one (or an
  empty value — the `if user or password` gate makes "" behave as unset) ⇒ `RuntimeError`
  naming both vars; neither ⇒ generated (`"tts-" + secrets.token_urlsafe(4)` → exactly 6
  urlsafe chars; `password = secrets.token_urlsafe(16)`), both printed to the console via
  `print()`; `secrets.token_urlsafe` raising ⇒ chained `RuntimeError`. Every
  credential-establishment path either lands the tuple or raises BEFORE kwargs return.
- `allowed_paths = list({history_root, tempfile.gettempdir()})` — only the blanket
  `~/Downloads` entry dropped. *(Corrected in fix round 1: this bullet originally said
  `output_dir` still comes from `_resolve_output_dir(config)` and that `~/Downloads`
  stays served as the output dir only when configured "(non-default)" — inverted, since
  `~/Downloads` IS the `output_directory` default. The resolver is now
  `resolve_history_output_dir(config)`, the `history_output_directory` key with default
  `~/Downloads/Qwen3-TTS Output`; under `_resolve_output_dir` the default-config grant
  `{~/Downloads, tempdir}` equalled the effective BASE grant, so the narrowing was a
  no-op until the swap.)*
- Both call sites: `**get_gradio_launch_kwargs(config, share=share)`; the helper returns no
  `share` key, so there is no duplicate-kwarg conflict with the site's own `share=share`.
- GREEN runs: **9 passed + 2 subtests** on qwen3-tts-mlx (py3.11.14) and on torchless
  `.venv-310` (py3.10.21) — RUN, not skip (1 pre-existing starlette/httpx TestClient
  deprecation warning, the same one 0C/0D recorded).

## Mutation verification (Task 1, each on top of committed `afd5704`)

| Mutation | Caught by |
|---|---|
| auth skipped (`if not share:` → `if True:` early return) | **7 failed / 4 passed** — every share=True test |
| fail-open partial env creds (raise → fallback tuple) | **2 failed / 9 passed** — both fail-closed subtests |
| Downloads re-added to `allowed` | **1 failed / 8 passed** — the narrowing test, same AssertionError as RED |

## Design rulings (controller, binding — and why)

1. **Keyword-only `share` on the helper, enforcement in the helper, sites forward their
   own decision.** One enforcement point cannot drift from two launch sites; the sites
   keep computing share (`--share` argv vs `TTS_UI_SHARE` env) because that is genuinely
   per-site, and pass the SAME value to the helper and to `demo.launch` (no duplicate
   kwarg). Default False keeps the six existing one-arg calls and the share-falsy dict
   byte-identical.
2. **Credentials: env pair both-or-nothing, else generate + print once.** Auto-generation
   makes the invariant enforceable without configuration ("share ⇒ auth" can hold
   unconditionally); printing to the console is the only channel that reaches the human
   who launched the process — under PM2 that is the PM2 log file. The step's "refuse to
   launch" clause is the FAIL-CLOSED arm: partial env creds or a raising `secrets` raise
   RuntimeError BEFORE kwargs exist — there is no code path that returns shared launch
   kwargs without `auth`, so a future refactor cannot reintroduce the unauthenticated
   launch by accident (it would have to delete the raise, which the mutation table shows
   is caught).
3. **`allowed_paths` = `{output_dir, tempdir}`.** Both app folders (Automated Output /
   Manual Downloads) live under the configured output dir, which is home-contained via
   `resolve_history_output_dir`'s traversal guard; `tempdir` must stay (history_panel
   copies files there for Gradio serving, `history_panel.py:94,326`). The blanket
   `~/Downloads` entry granted everything else in the tree to any holder of the shared
   URL.

## The SUBFAIL subtest-counting trap

pytest reports `test_share_true_single_env_var_fails_closed` at PARENT level even when its
subtests fail — a tally grep keyed on `FAILED`/`passed` per parent line reads the RED run
as "2 passed" and misses the failure entirely. The failures appear as `SUBFAILED(...)`
lines. Any future audit of this module must `grep SUBFAILED`, not just parent result
lines. Both RED subtests and both mutation-2 subtests were confirmed by their SUBFAILED
lines (counted: RED 9F includes them; mutation 2 = exactly those 2).

## The argv-vs-env facade-test deviation (disclosed)

The plan's per-site item said "facade `main()` under `TTS_UI_SHARE=1`", but
`TTS_UI_SHARE` is read only by `build_ui_and_launch`; the facade site's real share knobs
are the `--share` argv flag and `IN_COLAB`. The facade test therefore drives
`sys.argv = ["qwen3-tts-ui", "--share"]`; `build_ui_and_launch` is driven by
`TTS_UI_SHARE=1`. The pinned invariant is identical at both sites (launch receives
`share=True` AND the helper's `auth` tuple); only the share-actuation mechanism differs,
and each test uses its site's real one.

## Task 2 folds

- **Headless launch literal** (`tests/test_ui_headless.py:57`): the subprocess launch
  string carried the pre-change `allowed_paths=[os.path.expanduser('~/Downloads'), '/tmp']`.
  Modernized to mirror the production shape —
  `os.path.realpath(os.path.expanduser('~/Downloads/Qwen3-TTS Output'))` + tempdir. The
  literal is INPUT to that launch, not an assertion about the helper. Live verification:
  the UI subprocess launched with the new literal, answered readiness within the window,
  and the gradio_client handshake + `view_api` enumeration succeeded (the launch phase the
  literal feeds). Residual below on the script's generation phase.
- **Colab mirror** (`colab_notebook.ipynb` cell 6): the notebook's direct
  `demo.launch(share=True, ...)` gained an inline, self-contained mirror of the helper's
  auth block (env pair both-or-nothing; else `tts-` + `token_urlsafe` generate + print
  once; partial env ⇒ RuntimeError) and the same narrowed `allowed_paths` (blanket
  `~/Downloads` dropped). IN_COLAB forces share at the package sites, and the notebook
  hard-codes `share=True`, so this was the most public unauthenticated surface left.

## Final state (counted, not assumed)

- `tests/test_ui_share_auth.py` — **9 tests** (8 helper/facade + 1 with 2 subtests);
  `pytest --collect-only`-equivalent: 9 collected on both interpreters.
- Full non-e2e at HEAD: **3233 passed, 4 skipped, 92 deselected, 0 failed** — exactly
  **+9** over the 3224 baseline at `a8ac23b` (the new module; nothing else moved).
- Call-site census: `get_gradio_launch_kwargs` appears at the `:872` definition, the
  `_facade.py:571,580` site, and `generate_server.py:105,132` — no other code hits;
  `ui/__init__.py` remains docstring-only.

## Gates (Step 0E, observed at HEAD, qwen3-tts-mlx unless stated)

| Gate | Command | Result |
|---|---|---|
| Full non-e2e | `conda run -n qwen3-tts-mlx python -m pytest tests/ -m "not e2e" -q --tb=short --ignore=tests/evaluations` | **3233 passed, 4 skipped, 92 deselected, 0 failed** in 46.29 s |
| Batch 2 (Voice & CLI) | `conda run -n qwen3-tts-mlx python tests/run_batches.py --batch 2` | 1/1 batches passed |
| Batch 4 (Engine & UI, incl. the new module) | `conda run -n qwen3-tts-mlx python tests/run_batches.py --batch 4` | 1/1 batches passed |
| ruff | `conda run -n qwen3-tts-mlx ruff check qwen3_tts tests` | All checks passed (exit 0) |
| mypy (entry-point form) | `conda run -n qwen3-tts-mlx mypy qwen3_tts/core qwen3_tts/server qwen3_tts/interface` | Success: no issues found in 58 source files (the two `app_generation.py` `annotation-unchecked` notes are pre-existing) |
| Torchless RUN-not-skip | `./.venv-310/bin/python -m pytest tests/test_ui_share_auth.py -v` | **9 passed, 2 subtests passed** in 1.69 s (ran, did not skip) |
| Stale-count sync grep | `git grep -n "3224\|3217\|3228" -- CLAUDE.md docs/COMMANDS.md docs/CONTRIBUTING.md docs/RUNBOOK.md Makefile` | zero hits (grep exit 1) — no count line to sync |
| Notebook JSON validity | `python3 -c "import json; json.load(open('colab_notebook.ipynb')); print('valid')"` | valid (cell 6 also `compile()`s) |

bandit was not re-run (not on this step's gate list): Task 2 touches only a test-file
launch string, the notebook, and docs.

## Commit list (in order)

- `5238f32` — `test(ui): pin share-requires-auth and the narrowed allowed paths` (Task 1)
- `afd5704` — `fix(ui): Gradio share now requires generated or configured credentials` (Task 1)
- `15ce05a` — `test(ui): mirror the production allowed-paths shape in the headless launch literal`
- `63692f9` — `fix(colab): mirror share-auth and narrowed allowed paths in the notebook`
- `docs(testing): Gradio share-auth TDD evidence` — this file
- `docs: track the Step 0E plan copy and update the security surfaces` — the tracked plan
  copy, ARCHITECTURE.md Security, CLAUDE.md

## Accepted residuals (disclosed)

1. **`tests/test_ui_headless.py`'s generation phase is stale (pre-existing, proven).** Its
   single `predict(api_name="/generate_clone")` runs only the two-step chain's FIRST step
   (`config_handler`, `tabs_generation.py:146`), which returns status "Generating..." —
   the blocking `_generate_server_side` step is a `.then()` link gradio_client never
   invokes. With the live server up (models ready), the script exits 1 at "Generation did
   not succeed. Status: Generating..." — verified IDENTICALLY with and without the fold
   (stash → run → pop, exit 1 both ways), so it is not caused by the allowed-paths
   modernization; with the server DOWN the same step returns "Error: TTS server is not
   running.", so the script cannot complete in either state. The batch runner only imports
   the module (unittest, 0 tests), so batch 4 is unaffected. Fix = drive both chain steps
   (or assert on the launch phase only); owned outside Step 0E — needs a plan line.
2. **Generated share credentials print to stdout, not the log file.** For `tts server`-style
   supervised launches the console IS the PM2 log; for a foreground `tts ui` it is the
   terminal. There is no persistence beyond the one print — re-launching regenerates new
   credentials by design (the env pair is the stable-credential path).

## Fix round 1 (allowed-paths resolver swap)

The whole-branch review's single MEDIUM: the GREEN used `_resolve_output_dir(config)` for
the allowed-paths entry, but that reads the legacy `output_directory` key whose DEFAULT is
`~/Downloads` — so under default config the grant `{~/Downloads, tempdir}` equalled the
effective BASE grant and the narrowing was a no-op. The binding "output_dir is the parent
of both app subfolders" means the HISTORY root: `resolve_history_output_dir(config)`
(the `history_output_directory` key, default `~/Downloads/Qwen3-TTS Output`, same
traversal guard).

- **RED** (`a43afa8`): `test_allowed_paths_narrow_to_output_dir_and_tempdir` re-keyed onto
  `history_output_directory` and renamed to
  `test_allowed_paths_narrow_to_history_root_and_tempdir`; NEW
  `test_default_config_grants_history_root_not_downloads` pins the all-default grant
  (history root PRESENT, realpath(~/Downloads) ABSENT); the legacy
  `tests/test_ui_facade.py` test that PINS Downloads-present under `{}` was re-pinned
  (controller-authorized for exactly this test) to
  `test_includes_history_root_not_downloads_in_allowed_paths`. RED run: **3 failed / 80
  passed** — all three failures show the history root missing while `~/Downloads` is
  still in the granted set (the wrong-resolver reason).
- **GREEN** (`705e3bc`): the one-line resolver swap in `get_gradio_launch_kwargs`
  (`_resolve_output_dir` → `resolve_history_output_dir`, local renamed `output_dir` →
  `history_root`). Mutation-verified: reverting the resolver briefly → exactly the same
  3 narrowing tests RED → restored (grep/diff clean) → green again. New module is now
  **10 tests** on both interpreters; the full non-e2e count moves accordingly.
- **Downstream alignment verified, no edits:** `colab_notebook.ipynb` cell 6 hard-codes
  `os.path.realpath(os.path.expanduser('~/Downloads/Qwen3-TTS Output'))` (ipynb line
  ~175) — the exact default-config grant post-swap; `tests/test_ui_headless.py`'s
  modernized literal mirrors the same shape. CLAUDE.md's `tts ui` entry says
  "narrowed to the app output dir + temp" — names no config key and is true as written
  post-swap, so it was left untouched per the controller ruling.
