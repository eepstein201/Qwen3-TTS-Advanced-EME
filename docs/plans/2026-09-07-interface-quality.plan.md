# Interface Quality Improvement Plan — Web UI · CLI · HTTP API

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all three user-facing surfaces of Qwen3-TTS (Gradio web UI, CLI, HTTP API) coherent and polished: one design language in the UI, one output/error idiom in the CLI, one error/status envelope in the API — plus fixing the verified latent defects found along the way.

**Architecture:** Three independent phases, each landing as its own worktree + branch + PR. Phase 1 (Web UI) owns one shared enabler — auth-gated progress details on `/generation-status` — which Phase 2 (CLI) later consumes. Phase 3 (API) is self-contained. Each phase keeps the app shippable at every task boundary.

**Tech Stack:** Gradio 6 (pinned `>=6.0,!=6.14.*,<7`; 6.20 in mlx env) · FastAPI + Pydantic · Click + argparse (dual-layer CLI) · pytest (2163+ tests, batch-registered).

**Spec:** This plan is its own spec — the current-state audit below is the evidence base. Repo convention: the approved plan doc is moved into the repo as `docs/plans/2026-09-07-interface-quality.plan.md` (first commit of the first phase branch).

## Execution environment (user-directed)

- **Every phase executes in its own git worktree** created from latest `origin/main` (superpowers:using-git-worktrees flow), never in the developer's working copy and never on an existing feature branch.
- The current branch `fix/require-model-under-lock-returns-model` (Step 0B work) is **untouched** by this plan.
- Phase branches: `feat/ui-polish-pass` → `feat/cli-output-consistency` → `feat/api-error-contract` (order below). Each is a normal feature PR; user merges.
- Within a phase branch, tasks commit sequentially in the order given; each task is independently revertible.

## Context

The user asked for a plan to improve the interfaces of the TTS application "across all of its interfaces." A three-agent audit (Web UI, CLI, HTTP API) found the same root condition on every surface: **capability is present but presentation is unstandardized** — every module invented its own idiom, and in several places code consumes data that doesn't exist.

Scope decisions (user-confirmed):
- **Web UI:** polish pass — one custom-CSS design layer + shared status/feedback components. NOT a full custom theme.
- **HTTP API:** breaking response-shape changes ARE acceptable (sole consumers are this repo's CLI and UI).
- **Plan shape:** one document, three phases, each phase an independent worktree + branch + PR.

## Verified current-state findings (evidence base)

### Web UI (`qwen3_tts/interface/ui/`, ~5,133 lines, 10 modules)
- **No design layer.** `gr.Blocks` is bare (`_facade.py:233-234`); the only theme+CSS ride in `get_gradio_launch_kwargs` (`shared.py:872-887` — `gr.themes.Soft()` plus one `.gr-hidden` utility class) and reach only 2 of 3 launch paths; the in-process path (`ui/__init__.py:20-21`) gets stock defaults.
- **Six status surfaces, three renderings:** plaintext Textboxes, `StatusBanner` HTML (`components.py:127-183`), `status_badge` spans (`components.py:105-119`). `loading` and `info` share one color (`components.py:36-45`).
- **No live generation progress** (`gr.Progress`: zero hits). Status Textbox says "Generating..." then silence (`generation.py:266,480`); errors render as `f"Error: {e}"` in the same widget as success — visually identical (`generation.py:395-415`). Three error idioms coexist (`gr.Error` toasts, `gr.Warning`, plain text).
- **Latent defect:** `_prepare_cancel_confirmation` (`generation.py:146-175`) reads `chunk_total`/`eta_sec` from `/generation-status`, which never emits them (`validation.py:196-203`); `chunk_total` defaults to 1 → `progress_pct = chunk_index*100` → "Progress: 300%" mid-generation.
- **`eta_sec` does not exist server-side at all** — read in 3 client sites, written in 0. It must be *computed* (from the generation's own chunk rate), not unstripped.
- **`chunk_index` conventions differ by path:** engine `inference.py:1406` passes 0-based current index, `:1643/:1687` pass 1-based completed count, `:724` passes total `0`=unknown. Any percent math on raw `chunk_index/chunk_total` is wrong on one path or the other — this is the root cause of the "Progress: 300%" class of bug.
- **Zero `tabular-nums` anywhere;** four ETA spellings; memory as `2500MB` / `{:.1f}MB` / `—`; history time column is `%H:%M:%S` only (no date).
- History action cells are bare text glyphs `✕`/`⭳` — no hit area, no hover (`shared.py:109-118`, `_facade.py:375-388`).
- WaveSurfer player: 4 instances each with its own `<style>` (`wavesurfer_js.py:136-168`); hardcoded focus outline `#4a9eff`; small hit areas.
- Dead code: `ProgressIndicator` constructed-then-discarded at 3 sites (`model_management.py:201`, `shared.py:146`, `voice_management.py:247`). Rename voice has no confirm (`tabs_management.py:178-182`); preview voice blocks up to 60 s with no loading indicator (`voice_management.py:343-392`).

### CLI (`cli*.py` + `interface/`)
- **358 `print()` vs 146 `click.echo()`**, split exactly at the argparse/Click seam; no shared output helper; two private near-duplicates (`tools/_shared.py:13-37`, `tools/healthcheck.py:33-57` — the package's ONLY color, raw ANSI, not tty-guarded). `click.secho/style`: zero uses.
- **`TTSGenericError`: 8 raise sites, zero catch sites.** No CLI exception boundary → server failures print raw Python tracebacks. Four error styles; only 3 `err=True` sites (errors go to **stdout**). `tts.cli` logger never configured (`basicConfig`: zero hits outside the server).
- **Exit codes broken:** `cli.py:141` exits 2 when "server was used" (not an error — pinned deliberately by `tests/test_cli_commands.py:136`); several error paths print `Error:` and exit 0 (`generate.py:740-741`, `cli/srt.py:39-40`).
- **Progress:** `_ProgressPoller` (`generate_interactive.py:179-349`) consumes the never-emitted `chunk_total`/`eta_sec`; single-text gets an indeterminate spinner; not started for `--stream`, local, interactive, or REPL; double-renders alongside SRT/dialogue per-item counters.
- **Duplicate renderers** for every `tts list X` (Click) vs legacy `--list-X` (argparse) with mismatched widths/ordering/empty states (prosody 12 vs 18 chars). `tts cache list` has two visible bugs (separator reuses header widths; same value printed in two columns — `model_cache.py:187,192`). `cache prune` epilog documents a `30d` suffix that `type=int` rejects.
- `_FLAG_MAP` (`cli.py:77-117`) has **no** entries for `--list-speakers`/`--list-presets`/`--list-prosody` — those argparse flags are unreachable from Click.

### HTTP API (`qwen3_tts/server/`)
- **No envelope; 20+ ad-hoc flat shapes;** `status` values uncontrolled (15 strings). THREE error families: string `detail`, structured `{"error","detail","recovery"}` via `_error_response` (`validation.py:488-506`), and `/generate-stream`'s `{"error","message","model_type"}` (`app_generation.py:765-772`). No error-code enum (13 bare literals); `recovery` vocabulary drifts from its own docstring (code emits `"unload"`, docs say `restart|config|bug|retry`).
- `ErrorResponse` model (`validation.py:125-130`) is dead on the wire. 500 used for client-recoverable `import_error`. Client treats unload-409 as success (`models.py:57`).
- **Client gaps:** `get_models/get_stats/get_health/shutdown/cancel_generation` never check `status_code` (a 401/500 HTML page becomes a raw `JSONDecodeError`); the stream path can't map `model_not_loaded` (reads `model_type` from top level where the server never puts it). `TTSError.format_cli()/.format_gradio()` exist (`core/config/errors.py:27-69`) with **zero production callers**.
- Naming drift: `ModelOpResponse.model` vs `model_type`; `detail` vs `message`; four time-key conventions (`load_time_sec`/`model_load_times`/`elapsed_sec`/`sec_per_char`).
- OpenAPI: hardcoded `"3.0.0"` duplicated with `pyproject.toml:7`; no tags/summaries/`Field(description=)`; `/health` exposes no version/uptime.

## Global Constraints

- **Lazy imports everywhere** — heavy deps (`torch`, `mlx`, `transformers`) only inside functions, never at module scope. New CLI/theme modules must be import-light (stdlib + click/gradio-only as appropriate).
- **UI collaborators imported module-style** (`shared.get_presets()`, `theme.var(...)`) so `mock.patch` targets definition sites.
- **Gradio bans (each enforced or documented in-repo):** no `select` listener on `gr.Tab`; `gr.Audio` gets `None`, never `""`; no `visible=False` on JS-bridge components (use `elem_classes=["gr-hidden"]`); js-only `.then(fn=None, js=...)` never runs (needs `fn=lambda p: p`); handlers return tuples (generators not iterated); every branch returns exactly one value per wired output.
- **Polling budget:** UI currently issues ~24–36 req/min against the 120/min global pre-auth limit. The one new progress timer must keep total ≤ ~66/min and be `active=False` at build.
- **`/generation-status` privacy:** the public (unauthenticated) payload stays **byte-identical** to today. `batch_total`/`chunk_total`/`progress_pct`/`eta_sec` are emitted **only to authenticated** callers (input-size must not leak; UI Python handlers and the CLI poller both auth via `core/http_client.server_request`).
- **Testing:** `conda run -n qwen3-tts-mlx python -m pytest tests/ -m "not e2e"`; every new test module registered in `tests/run_batches.py` `BATCHES` (enforced by `tests/test_batches_coverage.py`); async tests need `IsolatedAsyncioTestCase` or `pytest.mark.asyncio`; static gates `ruff`, `mypy` (core/server/interface), `bandit` 0 HIGH. Batches: 1=Core Utilities, 2=Voice & CLI, 3=Server Infrastructure, 4=Engine & UI, 5=Optional.
- **Server changes require restart** (`tts server stop && tts server start`; PM2-managed) before manual verification.
- **No new config.json keys, no new CLI flags** — `TTS_LOG_LEVEL` and `NO_COLOR` are env vars (out of `check-config-docs` scope).
- **No AI authorship attribution** anywhere; conventional commits (`feat:`/`fix:`/`refactor:`/`test:`/`docs:`); feature-branch workflow; never amend/force-push.
- **CLAUDE.md updated** with any significant change (primary context file).
- **Test-pinned literals** (`"No presets configured"`, `"(default)"`, `"=== TTS REPL Mode ==="`, `"Exiting REPL."`, exact API `status` strings, `exclude_unset` absence semantics, `sys.exit(2)`) change only deliberately, inside the task that owns them, with the pinning test updated in the same commit.
- **Gate A / Gate B (two-gate adversarial review, user's standing practice):** review the tests alone at RED before implementation exists; review the full diff at GREEN.

## Non-goals (explicit scope cuts)

- **`tts config` bash wizard rewrite** — an installer, not a CLI surface; Python rewrite duplicates `install.sh` for a once-per-machine command.
- **REPL/prompt_toolkit rewrite** — interactive modes keep `input()`; only output/error idioms are unified.
- **WebSocket/stream-frame envelope changes** — `/ws` flat `{"error"}` and the binary terminal-frame protocol stay (single source in `core/stream_protocol.py`; a change forks clients). Documented as a family instead.
- **No response-envelope wrapper** (`{success,data,error}`) — HTTP status already carries success/failure; the plan standardizes error bodies, status vocabulary, and naming.
- **History Duration/Size columns — cut.** Positional row contract pinned by `tests/test_ui_history_helpers.py:108`, column-index click routing (`history_panel.py:26-27`), and a per-file stat call — negative cost/benefit.
- **No `gr.Progress`** — it only works when the generator function *is* the event handler; this repo's flow is a multi-step `.then` chain.
- **No interleaving of srt/dialogue item lines with the progress bar on one terminal line** — two streams (stdout items, stderr bar) is the correct idiom.
- **No `importlib.metadata` for version** — `qwen3_tts.__version__` + hatchling dynamic works uninstalled (Colab/Docker).
- **No renaming `model_load_times`** — a dict of durations, consumed by UI `/health` display and pinned; documented as conforming instead of renamed.

## Phase 1 — Web UI polish (worktree → branch `feat/ui-polish-pass`)

Branch-internal order: **T1.0 → T1.1 → T1.2 → T1.3 → {T1.4, T1.5, T1.8} → T1.6 → T1.7 → T1.9**. Everything after T1.3 parallelizes.

### Task 1.0: Authed `/generation-status` progress details (shared enabler) — M

**Files:**
- Modify: `qwen3_tts/server/app_generation.py` (both `_chunk_progress` closures: ~`:474` non-streaming, ~`:885` streaming)
- Modify: `qwen3_tts/server/app_lifespan.py` (new pure helper)
- Modify: `qwen3_tts/server/app.py` (`generation_status()` ~`:592`; helper beside `verify_auth` ~`:244`)
- Modify: `qwen3_tts/server/validation.py:196-203` (`GenerationStatusResponse`)
- Test: `tests/test_generation_status_authed.py` (new; batch 3 "Server Infrastructure")

**Interfaces (produces — consumed by T1.6 and T2.4):**
- Wire (authed only): `{"batch_total": int, "chunk_total": int, "progress_pct": float|None, "eta_sec": float|None}` — `progress_pct`/`eta_sec` omitted when `None` (`response_model_exclude_unset=True` makes this free). Public payload byte-identical to today.
- `chunk_progress_pct(completed: int, total: int) -> float | None` — `None` when `total <= 0 or total == 1`; **never ≥ 100** (completion is signalled by `active=False`, so a saturated bar can't race the result).
- `estimate_eta_sec(gen_state: dict, now: float | None = None) -> float | None` — `per_chunk = elapsed / completed; eta = per_chunk * (chunk_total - completed)`; `None` when `< 1` chunk completed or `chunk_total` unknown (0 = unknown per engine contract).
- `request_is_authed(request: Request) -> bool` — never raises, never logs; `secrets.compare_digest` against the same token source as `verify_auth` (factor token extraction into `_bearer_token(request) -> str` shared with `verify_auth` so comparison logic can't drift).
- `_authed_generation_details(gen_state: dict) -> dict` (app.py module helper).

**Steps:**
- [ ] **Step 1 (RED):** write `tests/test_generation_status_authed.py`: unauthed body has exactly the 4 public keys (+`elapsed_sec` when active) — byte-identity guard; authed body adds the four fields; `progress_pct` never ≥ 100 and `None` when `chunk_total in (0, 1)`; `estimate_eta_sec` zero-completed → `None`, known rate → expected, `chunk_total=0` → `None`; `chunk_progress_pct` handles `total=0`; **wrong token behaves identically to absent token**. Mirror the harness of `tests/test_response_contracts.py:33` (`_ContractTestBase`). Run: `conda run -n qwen3-tts-mlx python -m pytest tests/test_generation_status_authed.py -v` → FAIL.
- [ ] **Step 2 (GREEN):** `validation.py` optional fields `batch_total/chunk_total: int | None = None`, `progress_pct/eta_sec: float | None = None` + docstring update ("…emitted only when the request carries a valid token"). Then `app.py`: `_bearer_token` + `request_is_authed` + the `if request_is_authed(request): result.update(_authed_generation_details(gen_state))` block. Then `app_lifespan.estimate_eta_sec`, then the two `_chunk_progress` closures write normalized `progress_pct` — each closure comments its own index convention (0-based current ⇒ `completed=chunk_idx`; 1-based completed ⇒ `completed=chunk_idx` as-is) so ambiguity never crosses the module again.
- [ ] **Step 3:** `pytest tests/test_generation_status_authed.py tests/test_response_contracts.py -v` → PASS (extend `test_generation_status_matches_contract` with the two authed-shape cases).
- [ ] **Step 4:** gates `ruff check qwen3_tts tests && mypy qwen3_tts/server`. Register module in `tests/run_batches.py` batch 3. Commit: `feat(server): emit auth-gated progress details on /generation-status`.
- [ ] **Step 5:** restart PM2 (`tts server stop && tts server start`) and verify live: `curl -s -H "Authorization: Bearer $(cat ~/.config/qwen3-tts/.voice_server_token)" localhost:5123/generation-status` during a generation shows the new fields; unauthed curl shows none. Note the restart requirement in the PR body.

### Task 1.1: `ui/theme.py` design layer + single CSS wiring — M

**Files:**
- Create: `qwen3_tts/interface/ui/theme.py` (import-light: stdlib only)
- Modify: `qwen3_tts/interface/ui/_facade.py:233` (Blocks gets `css=`/`theme=`), `:375` (`history_df` gets `elem_classes=["tts-history"]`)
- Modify: `qwen3_tts/interface/ui/shared.py:872-887` (`get_gradio_launch_kwargs` **drops** `theme`/`css` — Blocks is now the single source)
- Test: `tests/test_ui_theme.py` (new; batch 4 "Engine & UI")

**Interfaces (produces):**
- `TOKENS: Mapping[str, str]` — canonical fallback set. Keys: `surface, surface_alt, border, border_strong, accent, accent_soft, text, text_subtle, text_inverse, success, warning, error, info, loading, focus, radius_sm, radius_md, radius_lg, dur_fast, dur_slow`.
- `var(token: str) -> str` — returns `"var(--tts-<token>,<fallback>)"`; ONE canonical fallback source.
- `SEVERITY_CLASS: Mapping[str, str]` — `"error" -> "tts-sev-error"`, etc.
- `BLOCKS_CSS: str` — built with `"".join(f"--tts-{k}:{v};" for k, v in TOKENS.items())` so the constant cannot drift from the stylesheet. Contains: custom-property block mapped onto gradio theme vars (`--block-background-fill`, `--block-border-color`, `--body-text-color`, `--border-color-primary`); radius scale; **distinct** `--tts-loading`; `.tts-num { font-variant-numeric: tabular-nums; font-feature-settings: "tnum"; }`; `:focus-visible` ring via `--tts-focus`; explicit-property transitions only (`transition-property: width, background-color, border-color;` — never `all`); history action-cell rules (`.tts-history td:nth-child(6), .tts-history td:nth-child(7) { cursor:pointer; text-align:center; user-select:none; min-width:44px; }` + hover background); `.tts-history td { font-variant-numeric: tabular-nums; }`; severity classes `.tts-sev-*` that render a Textbox as a banner (border/radius/padding/color from tokens); `.tts-empty` empty-state styling; a `prefers-reduced-motion` block zeroing transitions.

**Steps:**
- [ ] **Step 1 (RED):** `tests/test_ui_theme.py`: parse `--tts-([a-z_]+):` out of `BLOCKS_CSS` and compare to `TOKENS` keys (drift guard); `var("accent") == "var(--tts-accent,<exact-fallback>)"`; `TOKENS["loading"] != TOKENS["info"]`; CSS contains `prefers-reduced-motion`, `.tts-num`, `.tts-history td:nth-child(6)`; **no `transition: all`** (regex); `get_gradio_launch_kwargs()` has no `theme`/`css` keys; `build_ui()` Blocks carries css (headless build, pattern of `tests/test_ui_facade.py`). Run → FAIL.
- [ ] **Step 2 (GREEN):** write `theme.py`; wire `_facade.py` Blocks; strip launch kwargs.
- [ ] **Step 3:** `pytest tests/test_ui_theme.py tests/test_ui_facade.py tests/test_ui_headless.py tests/test_ui_port_flag.py -v` → PASS (update their kwargs assertions).
- [ ] **Step 4:** gates + batch-4 registration. Commit: `feat(ui): design-token CSS layer wired at Blocks level`.

### Task 1.2: `components.py` severity/token consolidation — S

**Files:** Modify `qwen3_tts/interface/ui/components.py`; Test: `tests/test_ui_status_banner.py`, `tests/test_ui_progress_indicator.py` (update).

**Interfaces:** `_SEVERITY_STYLE["loading"]` gets its own color + distinct icon key; colors read `theme.var(...)` (module-style import `qwen3_tts.interface.ui.theme as theme`); `StatusBanner.render` concentric radius (outer `radius_lg`, inner child surfaces `radius_sm`) with inline styles from `theme.var`; `ProgressIndicator` bar colors from tokens (keep explicit `transition:width 200ms ease-out`); new `severity_class(severity: Severity) -> str` returning `"tts-sev-<name>"` (consumed by T1.7).

**Steps:**
- [ ] **Step 1 (RED):** update tests to assert `var(--tts-…)` tokens, distinct loading color, concentric radii, empty-container `min-height` preserved, clamping unchanged. Run → FAIL.
- [ ] **Step 2 (GREEN):** apply the component changes. Run both test files → PASS.
- [ ] **Step 3:** gates. Commit: `refactor(ui): severity tokens onto the theme layer`.

### Task 1.3: `shared.py` format helpers + sweep — S

**Files:** Modify `qwen3_tts/interface/ui/shared.py` + call sites (`model_management.py:264-285`, `:26-120`, `format_status_display` ~`:396`, `tabs_management.py`, `history_panel.py`); Test: `tests/test_ui_shared_format.py` (new; batch 4).

**Interfaces (produces):**
```python
def fmt_duration(seconds: float | None) -> str   # m:ss under 1h, h:mm:ss above; "—" for None/negative/non-finite
def fmt_size(num_bytes: float | int | None) -> str  # B/KB/MB/GB, one decimal; "—" for None
def fmt_eta(seconds: float | None) -> str        # "~12s" / "~1m 20s"; "—" when None — THE single ETA spelling
def fmt_memory_mb(memory_mb: float | int | None) -> str  # "2500 MB" (one decimal <1000); "—" when 0/None
def format_history_time(ts: float, now: float | None = None) -> str  # "14:32:07" today (local); "Sep 6 14:32" otherwise; "—" falsy
```
All pure. Module docstring notes `qwen3_tts/tools/_shared._format_size` is the CLI sibling (canonical implementation becomes `cli_output.fmt_bytes` in T2.1; cross-layer drift-guard test here).

**Steps:**
- [ ] **Step 1 (RED):** `tests/test_ui_shared_format.py` — table-driven cases for all five (incl. `None`/`0`/negative/`inf`/`nan` → `"—"`); `fmt_size(1024**2) == tools._shared._format_size(1024**2)`. Run → FAIL.
- [ ] **Step 2 (GREEN):** implement the five helpers. Run → PASS.
- [ ] **Step 3:** sweep call sites — **one commit per site-group** so a visual regression bisects cleanly: (a) `_badge_from_models_payload`, (b) `get_model_table_data` (`"-"` → `"—"`), (c) `format_status_display` + tab mgmt memory strings, (d) history duration lines. Gates after each.
- [ ] **Step 4:** Commit sequence: `feat(ui): shared duration/size/eta/memory formatters` then the sweep commits.

### Task 1.4: Model badge + Manage Models honesty — S

**Files:** Modify `qwen3_tts/interface/ui/model_management.py`; Test: `tests/test_ui_model_management.py`.

**Details:** `_badge_from_models_payload` (`:264-285`) stops implying the *previous* load time is this load's ETA: `"Loading <type>…"` when `load_time_sec is None`; `"Loading <type>… (last load {fmt_duration(...)})"` otherwise. Elapsed strings via `fmt_duration` (`:152`). Delete the dead `ProgressIndicator` at `:201`.

**Steps:**
- [ ] **Step 1 (RED):** loading badge with/without `load_time_sec`; memory formatting; `"—"` alignment; assert `components.ProgressIndicator` **not constructed** (`mock.patch(..., side_effect=AssertionError)`). Run → FAIL.
- [ ] **Step 2 (GREEN):** apply. Run → PASS. Gates. Commit: `fix(ui): model loading badge no longer presents last-load time as ETA`.

### Task 1.5: History table polish — M

**Files:** Modify `shared.py` (`get_history_data` ~`:561-609`, uses `format_history_time` from T1.3), `_facade.py:375-388` (`elem_classes=["tts-history"]`, done in T1.1), `theme.py` (tune action-cell CSS); Test: `tests/test_ui_history_helpers.py` (update), `tests/test_ui_facade.py`.

**Details:** time cell becomes date-aware via `format_history_time`; Seed/Chunks columns get `.tts-num` via CSS (no data change); **7-column shape unchanged** (columns indices `history_panel.py:25-27` untouched — Duration/Size columns cut per Non-goals). CSS tuning lands against the real DOM; add `::selection` suppression so double-click doesn't select the glyph. Unit tests assert `elem_classes` only — gradio's internal DOM belongs to the Playwright suite.

**Steps:**
- [ ] **Step 1 (RED):** `format_history_time` cases (today/other-day/`0`/future); 7-column test stays green. Run → FAIL (helper missing).
- [ ] **Step 2 (GREEN):** implement + wire. Run → PASS. Gates. Commit: `feat(ui): date-aware history timestamps + action-cell hit areas`.

### Task 1.6: Generation live progress + cancel-confirmation defect fix — L

**Files:** Modify `generation.py`, `tabs_generation.py` (pass-through), `_facade.py` (one shared Timer); Test: `tests/test_ui_generation_progress.py` (new; batch 4).

**Interfaces (produces):**
```python
PROGRESS_POLL_SECONDS = 2.0      # 30 req/min
PROGRESS_POLL_MAX_MINUTES = 20   # hard stop; timer self-disables

def poll_generation_progress() -> str
"""Authed GET /generation-status -> ProgressIndicator HTML. "" when not active /
non-200 / exception (never propagates into a toast). Degrades to elapsed-only when
progress_pct is absent (older server or unauthenticated): NEVER derives a percent
from chunk_index alone."""

def start_progress_timer() -> gr.Timer   # returns gr.Timer(active=True)  — chain step
def stop_progress_timer() -> gr.Timer    # returns gr.Timer(active=False) — chain step
```
Verified against installed gradio: `Timer(value, *, active, ...)` accepts `active`, and returning a Component instance as an output merges as a prop-update — wiring the Timer as an **output** of `.then` steps is the supported toggle. Timer is never visible, so the `visible=False` ban doesn't apply.

**Wiring:** `_build_generate_buttons_and_output` (`:455-502`) adds `progress = gr.HTML(value="")` to the dict; `_wire_generation_tab` (`:504-551`) gains keyword-only `progress=None, progress_timer=None` (defaults keep existing callers/tests working). Chain: `btn.click → … → .then(start_progress_timer, outputs=[progress_timer]) → .then(_generate_server_side, …) → .then(stop+clear, outputs=[progress_timer, progress]) → …` (existing js-load/model-indicator/guard/announcer steps unchanged). `_facade.py` creates `progress_timer = gr.Timer(value=PROGRESS_POLL_SECONDS, active=False)` once and hands the same instance to all three tabs (only one generation in flight — `inference_lock` + guard).

**Budget:** 24–36/min today + 30/min = 54–66/min of the 120/min ceiling; single local client. Guards: test asserts `PROGRESS_POLL_SECONDS >= 2.0`; timer `active=False` at build; poll handler checks `is_server_running` before any HTTP call; exactly-one-Timer introspection test over `demo.blocks`.

**Defect fix:** `_prepare_cancel_confirmation` (`:130-175`) consumes `progress_pct` when present; when absent returns `("Stop generation?\nProgress: unknown\nChunks: n/a\nETA: n/a", False, 0, 0, None)`. Deletes the `chunk_total = status.get("chunk_total", 1)` default and the raw ratio math. The `<10%` fast path requires `progress_pct is not None and progress_pct < 10` — **unknown progress takes the confirm path** (safe default).

**Steps:**
- [ ] **Step 1 (RED):** `_prepare_cancel_confirmation` defect tests first — they fail today (latent bug): `chunk_total` absent → confirm path, message contains `"Progress: unknown"`, never `"300%"`; `progress_pct=7` → fast path; `progress_pct=42` → confirm path. Run → FAIL.
- [ ] **Step 2 (GREEN):** fix `_prepare_cancel_confirmation`. Run → PASS. Commit: `fix(ui): stop-confirm no longer fabricates progress from absent fields`.
- [ ] **Step 3 (RED):** poller tests: `mock.patch("qwen3_tts.core.http_client.server_request")` feeding authed payloads → exact ProgressIndicator HTML (percent, ETA via `fmt_eta`); absent-`progress_pct` → elapsed-only, no percent, never `aria-valuenow > 99`; `active: False`/non-200/exception → `""`; `start/stop_progress_timer` return Timer with `active=True/False`. Run → FAIL.
- [ ] **Step 4 (GREEN):** implement poller + timer plumbing; pass through `tabs_generation.py`; create the shared Timer in `_facade.py`. Run new tests + `tests/test_ui_generation_ext.py` (chain-arity) + `tests/test_ui_facade.py` + `tests/test_ui_headless.py` → PASS.
- [ ] **Step 5:** wiring test (pattern of `tests/test_ui_tab_select_wiring.py`): generate chain contains a `.then` arming the timer and the terminal step disarms it. Gates + batch-4 registration. Commit: `feat(ui): live generation progress via authed generation-status`.

### Task 1.7: Outcome severity routing (on the existing Textbox) — M

**Design decision (D6):** the status Textbox is *styled into* a banner via `elem_classes` — NOT swapped for `gr.HTML`. A `gr.HTML` takes raw innerHTML (today's `f"Error: {e}"` strings are safe only because Textbox escapes), and a swap breaks `_announce_status` input plus ~5 exact-equality assertions. Visual consistency is achieved through **shared tokens** (`StatusBanner` and `.tts-sev-*` classes read the same `theme.var` values), not a shared component.
**Fallback (D6-alt, only if the `gr.update` propagation test fails on both gradio versions):** two outputs — plain message string + banner `gr.HTML` — exact cost: `_generate_server_side` returns a 5-tuple, outputs grow by one, `_announce_status` re-targets the plain output, ~8 assertions in `test_ui_generation_ext.py` + `test_ui_tab_select_wiring.py` wiring change.

**Files:** Modify `generation.py`; Test: `tests/test_ui_generation_ext.py` (update), `tests/test_ui_a11y_announcer.py` (must stay green).

**Interfaces (produces):**
```python
_OUTCOME_SEVERITY: tuple[tuple[Severity, tuple[str, ...]], ...]
# ordered most-specific first: ("error", ("Error:", "Failed:", "Stop failed")),
# ("success", ("Generated", "Saved", "Copied")), ("warning", ...), ("info", STATUS_* constants)

def outcome_severity(message: str) -> Severity: ...
def status_update(message: str, severity: Severity | None = None) -> dict:
# gr.update(value=message, elem_classes=[severity_class(sev)]) — dict return keeps
# Textbox escaping client-side (XSS-safe) and downstream `.then` inputs receive
# the plain string after gradio preprocesses the update.
```

**Steps:**
- [ ] **Step 1 (RED):** `outcome_severity` table-driven cases; `status_update` returns a dict carrying `elem_classes`; **announcer-compat test**: a `status_update(...)` output delivered into `_announce_status`' input yields the plain string (this test gates the whole D6 decision — run it before the swap). Run → FAIL.
- [ ] **Step 2 (GREEN):** implement + swap terminal return sites (`:269`, `:296`, `:310` `gr.Warning`, `:395-415`, `cancel_streaming_generation` `:105-127`, `generate_guard_check` `:71`, `_prepare_streaming_config` `:178-221`). Run `test_ui_generation_ext.py` + `test_ui_a11y_announcer.py` + `test_ui_tab_select_wiring.py` → PASS (if propagation fails → D6-alt, documented cost).
- [ ] **Step 3:** gates. Commit: `feat(ui): generation outcomes carry severity styling via tokens`.

### Task 1.8: WaveSurfer CSS dedup — S

**Files:** Modify `qwen3_tts/interface/wavesurfer_js.py` (`:136-168`, `:196`), `theme.py` (`.ws-btn` rules move into `BLOCKS_CSS`); Test: `tests/test_wavesurfer_js.py`.

**Details:** the four per-instance `<style>` blocks collapse into `.ws-btn` rules in `theme.BLOCKS_CSS` (page-loaded once; JS emits markup only). `.ws-btn`: `min-width:40px; min-height:40px; padding:0 14px; border-radius:var(--tts-radius-md,…); transition-property: transform, background-color;` `:active { transform: scale(0.96); }` `:focus-visible { outline: 2px solid var(--tts-focus,…); outline-offset:2px; }` (replaces hardcoded `#4a9eff`). Time readout span gains `class="tts-num"` (tabular).

**Steps:**
- [ ] **Step 1 (RED):** player JS contains no `<style>` tag; contains `tts-num`, `--tts-focus`, `scale(0.96)`; `BLOCKS_CSS` contains `.ws-btn`. Existing CDN/self-host/security tests (`test_wavesurfer_security.py`, `test_wavesurfer_selfhost.py`) enumerated as must-stay-green. Run → FAIL.
- [ ] **Step 2 (GREEN):** move styles, delete inline blocks. Run all wavesurfer tests → PASS. Gates. Commit: `refactor(ui): wavesurfer styling onto shared tokens`.

### Task 1.9: Empty states, dead code, confirm + loading affordances — M

**Files:** Modify `shared.py` (new `empty_history_rows() -> list[list[str]]` returning one dimmed 7-column placeholder row; consumed by `_facade.py:385` `value=`), `tabs_management.py:178-182` (rename voice wrapped in `ConfirmButton`, pattern of delete at `:185-195`), `voice_management.py:343-392` (preview two-step chain: `.then` sets `status_update(f"Previewing {n}…", "loading")` before the blocking GET; existing return restores; `preview_voice` still returns path/`None` — no generator), delete dead `ProgressIndicator` at `shared.py:146` + `voice_management.py:247` (model_management's already gone in T1.4); Test: `tests/test_ui_confirm_patterns.py`, `tests/test_ui_voice_mgmt.py`, `tests/test_ui_history_helpers.py`.

**Steps:**
- [ ] **Step 1 (RED):** rename arm/execute/timeout confirm tests; preview loading-then-restore test; empty-rows shape test (7 columns). Run → FAIL.
- [ ] **Step 2 (GREEN):** implement rename confirm + preview loading + empty states. Run → PASS.
- [ ] **Step 3 (cleanup, last, own commit):** delete the dead `ProgressIndicator` constructions so a revert is clean. Gates. Commits: `feat(ui): rename confirm + preview loading state + empty states`, then `chore(ui): remove constructed-but-discarded ProgressIndicator instances`.
- [ ] **Step 4 (phase close):** run batch 4 in BOTH envs (`python -m pytest tests/ -m "not e2e" -k ui` in `.venv-310` for the gradio-6.14-shaped env; `conda run -n qwen3-tts-mlx` for 6.20) — R6 mitigation. Update CLAUDE.md UI section. Push → PR.

## Phase 2 — CLI consistency (worktree → branch `feat/cli-output-consistency`)

Order: **T2.1 → T2.2 → T2.3 → {T2.4, T2.5} → T2.6**. Soft cross-phase note: T2.2 should merge before Phase 3's T3.3 so typed `TTSError`s never surface as tracebacks.

### Task 2.1: `qwen3_tts/cli_output.py` (tests first) — M

**Files:** Create `qwen3_tts/cli_output.py` (package root, beside `cli.py`; import-light: stdlib + `click` only). Test: `tests/test_cli_output.py` (new; batch 2 "Voice & CLI").

**Interfaces (produces — canonical formatting for the whole CLI):**
```python
SYMBOLS = {"success": "✓", "warning": "⚠", "error": "✗", "info": "ℹ"}
def color_enabled(stream: TextIO) -> bool          # stream.isatty() and not os.environ.get("NO_COLOR")
def paint(stream, text: str, *styles: str) -> str  # click.style when enabled; zero ANSI escapes otherwise
def success(message, *, file=None) -> None         # stdout
def warn(message, *, file=None) -> None            # stdout
def error(message, *, file=None) -> None           # ALWAYS stderr
def info(message, *, file=None) -> None
def header(title, *, rule="=", width=60, file=None) -> None
def kv_line(key, value, *, key_width=22, file=None) -> None
def fmt_duration(seconds: float | None) -> str     # same spelling rules as UI fmt_duration (T1.3)
def fmt_bytes(num: float | None) -> str            # CANONICAL byte formatter; tools._shared._format_size becomes an alias (T2.3)
def fmt_eta(seconds: float | None) -> str
@dataclass(frozen=True)
class Column: key: str; title: str; width: int; align: str = "left"   # "left"|"right"
def render_table(columns, rows, *, indent="  ", empty_message: str | None = None) -> str
# Pure string. Separator derived from Column.width (fixes model_cache bug). Returns
# empty_message when rows empty and empty_message given. Glyphs are emitted as a
# SEPARATE prefix element so test-pinned substrings stay contiguous.
```

**Steps:**
- [ ] **Step 1 (RED):** `tests/test_cli_output.py` — table-driven `fmt_*`; `render_table` alignment + separator-widths == header-widths + empty message; `color_enabled` with `NO_COLOR` set/unset and a non-tty fake stream; `error()` → stderr, others → stdout (capfd); regex `\x1b\[` absent when color disabled. Run → FAIL.
- [ ] **Step 2 (GREEN):** pure functions first (`fmt_*`, `render_table` — no click needed), then emission wrappers. Run → PASS.
- [ ] **Step 3:** gates + batch-2 registration. Commit: `feat(cli): shared output/formatting module`.

### Task 2.2: Exception boundary + exit-code contract — M

**Files:** Modify `cli.py` (`TTSGroup` ~`:42` gains `invoke`; `_call_generate` `:120-143`; docstrings), `interface/generate.py:740-741`, `interface/cli/srt.py:39-40`, `interface/cli/dialogue.py:201-202`; Test: `tests/test_cli_error_boundary.py` (new; batch 2).

**Details:** `TTSGroup.invoke` = CLI-wide boundary: `TTSError`/`TTSGenericError` → `cli_output.error(...)` (first production caller of `TTSError.format_cli()`) + `SystemExit(1)`; Click's own `UsageError`/`Abort` pass through untouched. `_call_generate` wraps `_gen_main()`; `sys.exit(2)` on `result is True` **unchanged** (contract is deliberate and pinned). Docstrings document: `0` success · `1` error · `2` completed via server (reserved, treat as success). `generate.py` invalid-output-path and `srt.py` no-subtitles convert from print-and-return (exit 0) to `raise TTSError(..., recovery="config")` / `InvalidInputError` (exit 1). `dialogue.py:201-202` loses the `logger.error` double-write.

**Steps:**
- [ ] **Step 1 (RED):** CliRunner cases: `TTSError` from a stubbed `generate.main` → exit 1, stderr contains the `format_cli()` suggestion, `"Traceback" not in result.output`; `TTSGenericError` → exit 1; success → 0; `_call_generate` True → 2 (keeps `test_cli_commands.py:136`); invalid output path → 1; srt with empty file → 1. Run → FAIL.
- [ ] **Step 2 (GREEN):** boundary + raise-site conversions. Run → PASS, plus `tests/test_cli_commands.py`, `test_cli_ext.py`, `test_cli_srt.py`, `test_cli_dialogue.py` → PASS.
- [ ] **Step 3:** gates + registration. Commit: `fix(cli): top-level exception boundary; errors exit 1 on stderr`.

### Task 2.3: Adopt `cli_output` across the hot path + logging config — L (mechanical)

**Files:** `interface/generate.py`, `generate_helpers.py`, `generate_interactive.py`, `generate_server.py`, `cli/batch.py`, `cli/srt.py`, `cli/dialogue.py`, `cli_server.py`, `cli_voice.py`, `cli_config.py`, `tools/_shared.py`, `tools/healthcheck.py`, `tools/model_cache.py`; Tests: existing CLI suites are the regression net + `TTS_LOG_LEVEL` cases in `tests/test_cli_output.py`.

**Pattern (applies identically at every site):** status/error lines swap `print(f"Error: {e}")` → `cli_output.error(f"Error: {e}")` (keeps the pinned `"Error:"`-style substrings contiguous; glyph is a separate prefix element). **Pinned-literal inventory to preserve verbatim or update in-commit:** `"No voice prompts found"`, `"(default)"`, `"Premium CustomVoice speakers"`, `"No presets configured"`, `"=== TTS REPL Mode ==="`, `"Exiting REPL."`, `"Server not running"`. `tools/_shared.py`: `print_success/print_warning/print_header/print_info` become thin delegates (call sites + `tests/test_model_cache.py` stay green); `_format_size` aliases `cli_output.fmt_bytes`. `tools/healthcheck.py:25-31`: raw ANSI constants deleted; `_print_*` delegate (healthcheck becomes tty-guarded for the first time). Logging: inside `TTSGroup.invoke` before dispatch: `logging.basicConfig(level=os.environ.get("TTS_LOG_LEVEL", "WARNING").upper(), format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)` — `tts.cli` stops hitting `lastResort`; invalid value falls back to WARNING.

**Steps:**
- [ ] **Step 1:** land `tools/_shared.py` delegation first (smallest blast radius) → run `tests/test_model_cache_commands.py`. Commit.
- [ ] **Step 2:** `tools/healthcheck.py` → run its suites. Commit.
- [ ] **Step 3:** `interface/generate*.py` + `cli/*.py` → run `-k "generate or srt or dialogue or batch"` after each file. Commit.
- [ ] **Step 4:** `cli_server.py`/`cli_voice.py`/`cli_config.py` + logging config → run `tests/test_cli_commands.py tests/test_cli_ext.py`. Gates. Commit: `refactor(cli): single output idiom via cli_output; configured logging`.

### Task 2.4: CLI progress (consumes T1.0) — L

**Files:** Modify `interface/generate_interactive.py` (`_ProgressPoller` `:179-349`), `generate_server.py` (`:220`, `:244-252`), `generate.py`, `cli/srt.py`, `cli/dialogue.py`; Test: `tests/test_cli_progress.py` (new; batch 2).

**Details:** The poller already polls via `server_request` (authed) — once T1.0 ships it receives `chunk_total/batch_total/progress_pct/eta_sec` with no client change. This task makes it *correct and single-idiom*: `_run_rich`/`_run_fallback` use `progress_pct` for a determinate bar when `batch_total == 1` (today `total=None`); when `progress_pct` is absent, elapsed-only — **never** synthesize a percent from `chunk_index` (D2 ambiguity). New signature `_ProgressPoller(batch_total: int = 1, *, stream: bool = False)` — `stream=True` renders "Streaming…" and never a percent; `--stream` starts the poller (today one static print). Plus `_ProgressPoller.elapsed_only()` classmode for the local path (static `"Generating audio..."` → 1 s elapsed ticker; `run_inference` has no callback, so this is the honest maximum). Single progress idiom for batch/srt/dialogue: item lines stay on stdout, the bar owns stderr — different streams, no interleaving; `quiet_progress: bool = False` kwarg for `--dry-run`/CI. `generate_server.py:244-252` mid-generation `input()` gains `if sys.stdin.isatty()` guard, else `raise TTSError("Model not loaded and stdin is not a tty; run `tts model load clone` first")` — non-interactive scripts can no longer hang.

**Steps:**
- [ ] **Step 1 (RED):** poller tests with mocked `server_request` feeding authed and unauthed payloads (rich and fallback paths); `stream=True` never renders a percent; absent `progress_pct` → no percent; stream separation (capfd: bar stderr-only); `elapsed_only()` ticks; non-tty stdin raises `TTSError`. Run → FAIL.
- [ ] **Step 2 (GREEN):** implement. Run new + `tests/test_generate_interactive_ext.py`, `test_generate_interactive_ext_part2.py`, `test_generate_server.py`, `test_generate_server_fallback.py` → PASS.
- [ ] **Step 3:** gates + registration. Commit: `feat(cli): determinate progress from authed status; streaming + local tickers; tty-guarded load prompt`. (Merges after Phase 1's T1.0; code degrades gracefully either way.)

### Task 2.5: Unify `tts list X` vs legacy `--list-X` — M

**Files:** Create `qwen3_tts/cli_tables.py`; modify `interface/generate.py` (`_handle_info_commands`), `cli.py` (`list_*`), `tools/model_cache.py:160-197`; Test: `tests/test_cli_tables.py` (new; batch 2).

**Interfaces (produces):** `SPEAKER_COLUMNS/PRESET_COLUMNS/PROSODY_COLUMNS/VOICE_PROMPT_COLUMNS/CACHE_COLUMNS/HISTORY_COLUMNS: tuple[Column, ...]` (ONE width per column for both surfaces — kills the 12-vs-18 prosody drift) and `render_speakers/render_presets/render_prosody/render_voice_prompts/render_cache/render_history` (each `-> str`, pure). Click commands call them; `generate._handle_info_commands` delegates to the same functions (argparse surface stops re-implementing). Empty states via `render_table(empty_message=...)` with `"No presets configured"` / `"No voice prompts found"` preserved verbatim. `render_cache` fixes the duplicate `size_formatted` column (`:187` vs `:192`) and builds the separator from `Column.width`.

**Steps:**
- [ ] **Step 1 (RED):** golden-string tests per renderer (speakers grouped by language; presets empty + populated; prosody identical output from both entry points; cache separator widths + no duplicated value; history alignment); **the unification contract test**: `tts list speakers` output == `tts generate --list-speakers` output. Run → FAIL.
- [ ] **Step 2 (GREEN):** implement `cli_tables.py`, rewire both surfaces. Run → PASS + `tests/test_cli_commands.py`, `test_cli_ext.py`, `test_model_cache_commands.py`.
- [ ] **Step 3:** gates + registration. Commit: `refactor(cli): one renderer per resource for list commands and legacy flags`.

### Task 2.6: Help-text pass — S

**Files:** Modify `cli.py` (all command docstrings + `context_settings={"show_subcommand_help": True}` on groups; Click `metavar=` on shared options), `tools/model_cache.py` (`prune` epilog); Test: `tests/test_cli_help.py` (new; batch 2).

**Details:** short help stays one line; `epilog=` carries `Examples:` for ~10 key commands (`generate`, `batch`, `srt`, `dialogue`, `server start/stop`, `voice create/rebuild`, `list speakers`, `config edit`, `history`, `doctor`). Click metvars so `Usage: tts generate [OPTIONS] TEXT` instead of `TEXT...`. `cache prune` epilog: **remove the `30d` claim** (no new flag). `_FLAG_MAP`: add the missing `list_speakers`/`list_presets`/`list_prosody` entries so legacy flags are reachable from Click and delegate to `cli_tables.render_*` (keeps documented flags alive and makes T2.5's both-surfaces test meaningful).

**Steps:**
- [ ] **Step 1 (RED):** `tests/test_cli_help.py`: `--help` exits 0 and contains `Examples:` for the 10 commands; `Usage:` lines contain the metvars; `tts cache prune --help` no longer contains `30d`; **introspection test: every `_FLAG_MAP` value resolves to a real argparse flag on `generate.py`'s parser** (prevents the next gap). Run → FAIL.
- [ ] **Step 2 (GREEN):** apply. Run + `tests/test_voice_features.py`, `test_cli_model_size_choices.py`, `test_package_metadata.py` → PASS.
- [ ] **Step 3:** gates + registration + CLAUDE.md CLI section update. Commit: `docs(cli): help text, metvars, flag-map completeness`. Phase close: full `pytest -m "not e2e"` + push → PR.

## Phase 3 — API consistency (worktree → branch `feat/api-error-contract`)

Order: **T3.1 → T3.2 → T3.3 → T3.5; T3.4 independent.** Contract-test edits land in the same commit as each change (T3.6 is a ledger, not a task).

### Task 3.1: Error envelope as the only HTTP error shape — L

**Files:** Modify `validation.py` (enums + `ErrorBody`/`ErrorEnvelope`; `_error_response` becomes the typed single raise path), `app_generation.py` (`:765-772` family-C fix; `:231` recovery literal; all string-`detail` sites), `app.py`, `app_models.py`, `app_prompts.py` (string-`detail` migration — `grep -n "HTTPException" qwen3_tts/server/*.py` is the worklist), `validation.py:417-431` (`_validate_prompt_name` returns → `_error_response` calls); routes gain `responses={4xx/5xx: {"model": ErrorEnvelope}}`. Test: `tests/test_error_envelope.py` (new; batch 3).

**Interfaces (produces):**
```python
class ErrorCode(str, Enum): INSUFFICIENT_MEMORY/MODEL_NOT_LOADED/MODEL_UNLOADED/ASR_UNLOADED/
    LOAD_FAILED/LOAD_IN_PROGRESS/IMPORT_ERROR/UNKNOWN_ERROR/INVALID_NAME/TRANSCRIPT_REQUIRED/
    UNSUPPORTED_REFERENCE_AUDIO/INVALID_AUDIO/CREATION_FAILED   # (str, Enum) not StrEnum: py>=3.10
class RecoveryAction(str, Enum): RESTART/CONFIG/BUG/RETRY/UNLOAD   # "unload" becomes documented, not stray
class ErrorBody(BaseModel):      # extra="allow" for context like model_type / available_mb
    error: ErrorCode; detail: str = ""; recovery: RecoveryAction = RecoveryAction.RETRY
class ErrorEnvelope(BaseModel): detail: ErrorBody   # the wire shape: {"detail": {...}} — unchanged,
                                                    # so client._error_payload keeps unwrapping
def _error_response(status_code: int, error: ErrorCode | str, detail: str = "",
                    recovery: RecoveryAction | str = RecoveryAction.RETRY, **context: Any) -> NoReturn
```
Family-C `/generate-stream` 503 becomes `_error_response(503, ErrorCode.MODEL_NOT_LOADED, detail=error_msg, recovery=RETRY, model_type=mode)` — `model_type` finally lands where the client reads it. FastAPI's own 422 request-validation body stays as-is, documented as family D. WS flat shapes stay (Non-goals), documented in `websocket.py`'s docstring.

**Steps:**
- [ ] **Step 1 (RED):** `tests/test_error_envelope.py` — parametrized endpoints × failure modes asserting `body["detail"]["error"] ∈ ErrorCode`, `recovery ∈ RecoveryAction`, `model_type` present on the `/generate-stream` 503; sweep test walking `app.openapi()["paths"]` asserting every documented 4xx/5xx references `ErrorEnvelope`; grep-based repo test failing if `HTTPException(` is called with a string `detail=` outside a small allowlist. Run → FAIL.
- [ ] **Step 2 (GREEN):** enums + models + `_error_response` typing; migrate sites family-by-family (generation → models → prompts → app-level), running the sweep test after each family. Contract updates in-commit: `tests/test_fastapi_endpoints.py` (29 `"detail"` string assertions → structured), `tests/test_response_contracts.py`, `tests/test_extract_error.py` (matrix gains the new shape; legacy shapes stay as tolerance rows), `tests/test_ai_regression.py`, `tests/test_e2e_security_rate_limiting.py:241-243`, `tests/test_e2e_queueing.py:167`, `tests/test_error_handling.py`, `tests/test_validation.py`.
- [ ] **Step 3:** full batch 3 + gates + registration. Restart PM2; verify live error shape. Commit: `feat(api): single structured error envelope with code enums`.

### Task 3.2: Status vocabulary + renames + `text`/`texts` — M

**Files:** Modify `validation.py`, `app_models.py`, `app_prompts.py`, `app.py`, `app_generation.py`; Test: `tests/test_response_contracts.py` (extend).

**Details:** per-endpoint `Literal` status unions (`ModelOpResponse.status: Literal["loaded","already_loaded","unloaded","already_unloaded","load_failed"]`, etc.; `/health` `"ok"` and `/ready` `"ready"` pinned as `Literal`) — the 15-value vocabulary collapses to documented per-endpoint sets. **Renames (breaking, enumerated):** `ModelOpResponse.model` → `model_type` (aligns with the error envelope); `StatsResponse.idle_seconds` → `idle_sec` (`validation.py:234`, written `app_models.py:37,60`) — after which the rule "all durations end in `_sec`" holds with `model_load_times`/`load_time_sec`/`elapsed_sec`/`sec_per_char` documented as conforming (`model_load_times: dict[str, float]` typed). `X-Seed`/`X-Sample-Rate` headers kept (headers ≠ body fields), documented on `/generate-stream`. `GenerateRequest`: `texts` canonical; `@model_validator(mode="before") _promote_text_to_texts` normalizes `text=X → texts=[X]`; sending both → 422; `text` marked deprecated in its description. Response shape unchanged (`results: list[GenerateResult]`).

**Steps:**
- [ ] **Step 1 (RED):** `TestStatusVocabulary` (each endpoint's exact status set; unknown value rejected) + `TestNamingConventions` (no response field ends `_seconds`; no model has a bare `model` field) + `text`→`texts` promotion/mutual-exclusion cases. Run → FAIL.
- [ ] **Step 2 (GREEN):** apply + update the pinned status-string assertions in-commit (`test_response_contracts.py:276,336,363,526` etc.; client `models.py` reads the renamed key). Run → PASS.
- [ ] **Step 3:** gates. Commit: `feat(api): per-endpoint status vocabularies; model_type/idle_sec renames`.

### Task 3.3: Client hardening (single raise path) — M

**Files:** Modify `server/client/_base.py`, `models.py`, `generator.py`, `core/config/errors.py`; Test: `tests/test_client.py`, `test_client_models.py`, `test_client_generator.py`, `test_extract_error.py` (extend).

**Interfaces (produces):**
```python
def _raise_for_error(resp, fallback: type[TTSError] = GenerationError, **ctx) -> NoReturn
# Maps ErrorBody.error -> TTSError subclass: model_not_loaded -> ModelNotLoadedError(model_type);
# invalid_* -> InvalidInputError; insufficient_memory -> TTSError(recovery="unload");
# else fallback. detail -> technical_detail. Falls back to _extract_error_message for
# non-envelope shapes (legacy server tolerance).
def _check(resp) -> requests.Response   # 2xx -> resp; else _raise_for_error(resp)
```
`models.py`: `_check(resp)` in `get_models/get_stats/get_health/shutdown/cancel_generation` (a 401/500 HTML page becomes a typed `AuthenticationError`, not `JSONDecodeError`); `unload_model` keeps treating 409 as success, now documented why. `generator.py:404-413`: top-level `model_type` read replaced by `_raise_for_error(resp)` — the stream path finally raises `ModelNotLoadedError`. `core/config/errors.py`: `recovery="unload"` added to the `format_cli`/`format_gradio` suggestion maps. **Phase 3 owns raising; Phase 2 T2.2 owns catching/rendering** — sequencing note: merge T2.2 before this task if phases land in the stated order.

**Steps:**
- [ ] **Step 1 (RED):** per-suite cases: envelope 503 → typed exception; legacy flat shape → still typed; 401 HTML → `AuthenticationError`; unload 409 → dict return; stream 503 → `ModelNotLoadedError`. Run → FAIL.
- [ ] **Step 2 (GREEN):** implement. Run → PASS.
- [ ] **Step 3:** gates. Commit: `feat(client): single error raise path with typed mapping`.

### Task 3.4: OpenAPI metadata + version single-source — S (independent)

**Files:** Modify `pyproject.toml:7` (`dynamic = ["version"]` + `[tool.hatch.version] path = "qwen3_tts/__init__.py"`; remove static version), `qwen3_tts/__init__.py` (`__version__` authoritative), `app.py:267-271` (`version=__version__`, `description=`, `tags=` for `generation/models/prompts/voice/system`), per-route `summary=` on the ~15 highest-traffic routes, `Field(description=...)` for `GenerateRequest/GenerateResult/GenerateResponse/HealthResponse/ModelEntry/ModelsResponse/StatsResponse` (currently zero), `cli.py:152` (`version_option(version=__version__, prog_name="Qwen3-TTS")`); Test: `tests/test_response_contracts.py::TestOpenApiContract` (extend), `tests/test_package_metadata.py`.

**Steps:**
- [ ] **Step 1 (RED):** every route has `tags` + `summary`; spec `version` equals `qwen3_tts.__version__`; `GenerateRequest` fields all carry descriptions; pyproject has no static `version` (parse TOML); `pyproject` dynamic version == `__init__.__version__`. Run → FAIL.
- [ ] **Step 2 (GREEN):** apply. Run → PASS. Gates. Commit: `feat(api): OpenAPI metadata; version single-sourced`.

### Task 3.5: `/health` additive version + uptime — S

**Files:** Modify `app.py` (`health()` ~`:520-572`), `app_lifespan.py` (record `started_at = time.time()` in lifespan), `validation.py:154-171` (`HealthResponse` gains `version: str | None`, `uptime_sec: float | None`, `started_at: float | None` — emitted unconditionally, additive; public endpoint, so no request-derived numbers). Test: `tests/test_response_contracts.py` health contract + `uptime_sec >= 0` and `version == __version__`; `tests/test_fastapi_endpoints.py`.

**Steps:**
- [ ] **Step 1 (RED)** → contract cases fail. **Step 2 (GREEN)** → apply. **Step 3:** gates. Commit: `feat(api): health exposes version and uptime`.
- [ ] **Step 4 (phase close):** full `pytest -m "not e2e"`; restart PM2; `curl` verification of `/health`, a 503 shape, and the OpenAPI doc at `/docs`. Update CLAUDE.md Server API section. Push → PR.

## Dependency graph

```
Phase 1:  T1.0 ─┬─> T1.6 ─> T1.7        T1.7 needs T1.2's severity_class
                │     ▲
         T1.1 ──┴─> T1.2 ─> T1.4, T1.5, T1.8, T1.9
                └──> T1.3 ─> T1.4, T1.5, T1.6
Phase 2:  T2.1 → T2.2 → T2.3 → {T2.4, T2.5} → T2.6
Phase 3:  T3.1 → T3.2 → T3.3 → T3.5;  T3.4 independent
```

Cross-phase constraints (only three, deliberate):
1. **T1.0 → T1.6 and T1.0 → T2.4:** the authed-details contract is owned and pinned by Phase 1 before any second consumer codifies expectations (prevents CLI/UI drift). T2.4's reader degrades gracefully, so merge order is flexible; the *feature* lights up only after T1.0 ships.
2. **T2.2 before T3.3** (soft, UX ordering): typed `TTSError`s from the client must have a catcher in the CLI first.
3. **T1.1 → T2.3** (cosmetic only): the CLI `✓/⚠/✗/ℹ` vocabulary matches the UI severity vocabulary so docs describe one symbol set.

File overlap across phases: only `tests/run_batches.py` (append-only registrations; expect a trivial rebase).

## Risk register

| # | Risk | L/I | Mitigation |
|---|---|---|---|
| R1 | Dataframe cell CSS (`td:nth-child(6/7)`) silently no-ops across gradio versions | H/M | CSS is hit-area/hover only — the Python column routing (`HISTORY_COL_*`) stays the functional path, so a no-op degrades to today's behavior; unit tests assert `elem_classes`, not gradio DOM; one Playwright assertion for action cells; run batch 4 in both envs (6.14-shaped + 6.20) before PR. |
| R2 | `gr.update(elem_classes=…)` propagates a dict into `.then` inputs, breaking `_announce_status` | M/H | The announcer-compat test is written **before** the swap (T1.7 Step 1) and gates the design; verified fallback D6-alt has its exact cost enumerated. |
| R3 | `exclude_unset` semantics leak or drop the new `/generation-status` fields | M/H | Byte-identity test on the unauthenticated key set; auth check behind `secrets.compare_digest` on `verify_auth`'s own token source; negative test for a *wrong* token; `/health` additions are unconditional (no leak class). |
| R4 | Polling budget creep crosses the 120/min ceiling → UI 429s with no visible cause | M/M | One timer only, `active=False` at build, disarmed on completion and at `PROGRESS_POLL_MAX_MINUTES`; tests assert `PROGRESS_POLL_SECONDS >= 2.0` and exactly-one-Timer in `demo.blocks`. |
| R5 | Server changes invisible until PM2 restart → "green tests, broken live" | H/M | Every server-touching task (T1.0, all of Phase 3) names the restart + a live `curl` assertion; CLAUDE.md server-restart rule. |
| R6 | gradio 6.14 (banned, present in `.venv-310`) vs 6.20 (mlx env) behavioral differences | M/M | Timer/`gr.update` mechanisms verified against installed source; `hasattr(gr, "Timer")` guard pattern (as at `_facade.py:250-252`) so a missing Timer degrades to no live progress, never a crash; batch 4 run in both envs as a done criterion. |
| R7 | CLI sweep breaks a pinned literal late (batch 2) | H/L | Glyphs emitted as separate prefix elements so pinned substrings stay contiguous; `pytest -k cli` after each file's swap, not at the end. |
| R8 | A missed string-`detail` site survives the envelope migration | M/M | Sweep test over `app.openapi()["paths"]` + the grep-based repo test with a small allowlist (T3.1 Step 1). |

## Verification (end-to-end, per phase and at the end)

1. **Unit/contract:** `conda run -n qwen3-tts-mlx python -m pytest tests/ -m "not e2e"` (full suite; new modules registered → batch runner picks them up). CI-repo rule: also run touched modules in `.venv-310` (torchless CI proxy).
2. **Static gates:** `ruff check qwen3_tts tests && mypy qwen3_tts/{core,server,interface} && bandit -r qwen3_tts -c pyproject.toml && make check-config-docs`.
3. **Live server (after PM2 restart):**
   - Unauthed `curl -s localhost:5123/generation-status` during a generation → exactly today's keys; authed (token from `~/.config/qwen3-tts/.voice_server_token`) → adds `batch_total/chunk_total/progress_pct/eta_sec` (T1.0).
   - `curl -s localhost:5123/health` → `version` + `uptime_sec` present (T3.5); `curl -s localhost:5123/openapi.json | python -m json.tool | head -40` → tags/summaries/`ErrorEnvelope` refs (T3.4/T3.1).
   - Force one error (e.g. generate with no model loaded) → `{"detail":{"error":"model_not_loaded","detail":…,"recovery":"restart","model_type":…}}` (T3.1).
4. **UI manual pass** (`tts ui`): generate → live progress bar animates; Stop mid-generation shows sane percentages; an error renders red-banner-styled status; history shows date-aware timestamps and hover-able action cells; rename asks for confirm; empty history shows the placeholder row. Verify in the browser once light and once dark theme.
5. **CLI manual pass:** `tts "hello"` → clean progress with ETA, exit 0; `tts "hello" --stream` → streaming prefix, no bogus percent; kill the server and rerun → clean one-line error on stderr, exit 1, no traceback; `NO_COLOR=1 tts doctor` → no ANSI escapes; `tts list speakers` vs `tts generate --list-speakers` → identical output.
6. **Both gradio environments** for UI batches (R6), per-phase before PR.

## Execution handoff

On approval, each phase runs in its own worktree off latest `main` (superpowers:using-git-worktrees), implemented task-by-task with two-gate adversarial review (Gate A: tests alone at RED; Gate B: full diff at GREEN). Recommended execution mode: subagent-driven development (fresh subagent per task, review between tasks). Phases can also be handed to separate sessions — each phase document section is self-contained.
