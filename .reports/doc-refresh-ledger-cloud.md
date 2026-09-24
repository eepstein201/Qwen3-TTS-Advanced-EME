# Doc-refresh evidence ledger (cloud) — base 6c6a39df

Base: `6c6a39dfb38810dcd8e526affe1f84f240840725` ("docs: mark 7B done (PR #347); Open 17 → 15 (#353)"), confirmed with `git rev-parse HEAD`.
Codemap drift base: `59c23ac` (last commit touching `docs/CODEMAPS/architecture.md`, 2026-09-21).
Line counts: `git ls-files 'qwen3_tts/*.py' | xargs wc -l` (tracked files only). This reproduces the previous header exactly (76 .py / 29,825 LOC at 59c23ac), so the old and new numbers are comparable.
CLI facts come from `python -m qwen3_tts <group> --help`. `pip install -e ".[test]"` failed building the `docopt` wheel, so `click` was installed separately. The `tts` entry point was not installed; `python -m qwen3_tts` loads the same `cli` object.

## Factual changes

| # | Doc:line | Old claim | New claim | Evidence | Confidence |
|---|---|---|---|---|---|
| 1 | CODEMAPS/architecture.md:1 | 76 .py (29.8k LOC) | 79 .py (31.3k LOC) | `git ls-files 'qwen3_tts/*.py' \| wc -l` → 79; `… \| xargs cat \| wc -l` → 31342 | verified |
| 2 | CODEMAPS/architecture.md:12-14 | `"vllm"` → engine_vllm + vllm_client "DISABLED since #293: engine ValueError + boundary 400; code retained" | No engine strategy (ValueError). `/generate` uses the server VLLMAdapter only when `vllm.enabled=true`; the default `false` gives a 400 at validation | `core/engine/inference.py:74-85`; `server/validation.py:419-433`; `server/app_generation.py:647-666` (`use_vllm` path); `server/app_lifespan.py:377-398` (`_maybe_start_vllm_adapter`) | verified |
| 3 | CODEMAPS/architecture.md:17-18 | (none) | Entry points: `tts`→`qwen3_tts.cli:cli`; `test*`→`qwen3_tts.test_cli`; app.py; `_facade.build_ui/main` | `pyproject.toml` `[project.scripts]`; `ui/_facade.py:229,606` | verified |
| 4 | CODEMAPS/architecture.md:23-24 | "interface/ — `cli.py` (Click groups) …" | `cli.py`, `cli_*.py`, `cli_output.py` and `cli_tables.py` sit at the package root. `ui/` gains `theme.py` | `git ls-files qwen3_tts \| grep -v /` → `qwen3_tts/cli.py` etc.; `git diff --name-status 59c23ac 6c6a39d` → `A cli_output.py, A cli_tables.py, A interface/ui/theme.py` | verified |
| 5 | CODEMAPS/architecture.md:40 | Long serialization list: "/generate, /ws — outermost holders"; `/update-model-config` not listed | Condensed. `/generate-stream` added as an outermost holder and `/update-model-config` (T5) added as a lock holder | `app_generation.py:1005` (`inference_lock = state.inference_lock` in the stream path); `app.py:809` (`async with state.inference_lock` in update_model_config) | verified |
| 6 | CODEMAPS/architecture.md:47-49 | Heaviest: inference 1847, app_generation 1228, app.py 1180, generate 902, ui/shared 982, generate_interactive 826, app_lifespan 843; note named 5 files over 800 | inference 1849, app_generation 1243, app.py 1223, ui/shared 1102, ui/tabs_generation 1017, generate 891, generate_interactive 885, app_lifespan 870, ui/generation 869; all 9 are over 800 | `git ls-files 'qwen3_tts/*.py' \| xargs wc -l \| sort -rn` | verified |
| 7 | CODEMAPS/architecture.md:51-52 | core 7.7k · server 8.2k · interface 9.5k (ui 5.5k) · tools 2.5k · tests 218 modules, ~3.7k tests | core 7.9k · server 8.3k · interface 10.3k (ui 6.2k) · tools 2.5k · root 2.2k · tests 215 `test_*.py`, ~4.1k `def test_` | per-dir `git ls-files … \| xargs cat \| wc -l` → 7947/8310/10343/6232/2524/2218; `git ls-files tests \| grep -E '(^\|/)test_[^/]*\.py$' \| wc -l` → 215; the `def test_` grep sums to 4110. The old "218" counted every tracked .py under tests/ (`git ls-tree 59c23ac tests` → 218) | verified |
| 8 | CODEMAPS/backend.md:1 | server/ (8.2k LOC), config/pm2.py (128 LOC) | 19 .py (8.4k LOC) = server 18 files / 8310 + pm2.py 127 | `wc -l qwen3_tts/core/config/pm2.py` → 127 | verified |
| 9 | CODEMAPS/backend.md:7-13 | `/cancel-generation` listed under `app_generation.py`/`websocket.py`; `/stats`, `/health`… listed under `app.py`/`app_lifespan.py` | 24 routes, all registered in app.py. Handlers: `/cancel-generation`, `/health`, `/ready`, `/generation-status`, `/queue-status` and `/shutdown` are inline in app.py; `/stats` → `app_models.handle_stats` | `grep -n '@app\.' server/app.py` (24 decorators); `app.py:978-1020` (inline cancel); `app.py:106-115` imports `handle_stats` from `app_models`; `app_models.py:32` | verified |
| 10 | CODEMAPS/backend.md:8-13 | (none) | Rate-limit category for each route (all hybrid) | `app.py:741-1090` `@_rate_limit(...)` lines, in order: model×5 (`/load-model`, `/unload-model`, `/update-model-config`, `/load-asr`, `/unload-asr`), config_ops (`/update-startup-config`), transcribe, prompt_ops×3, generate×2; `_rate_limit(..., strategy="hybrid")` default at `app.py:546` | verified |
| 11 | CODEMAPS/backend.md:15 | (none) | Authed `/generation-status` extras: `batch_total`/`chunk_total` plus `progress_pct`/`eta_sec` | `app.py:674-693` (`request_is_authed`, `_authed_generation_details`); `app_lifespan.py:118,126` | verified |
| 12 | CODEMAPS/backend.md:17 | "All JSON routes carry Pydantic `response_model=` … 18 response models" | 18 routes carry `response_model=` (33 models in total). `/cancel-generation` returns JSON but has no model | `grep -o 'response_model=\w\+' app.py \| sort -u \| wc -l` → 18; `app.py:978` `@app.post("/cancel-generation")` has no `response_model` and returns a dict; `grep -c '^class .*BaseModel' validation.py` → 33 | verified |
| 13 | CODEMAPS/backend.md:20 | (none) | Middleware order: SlowAPI → security_headers → RequestBodySizeLimit → CORS → ExceptionMiddleware | `app.py:436-440` comment "Net order (outer->inner): SlowAPI -> security_headers -> RequestBody -> CORS -> ExceptionMiddleware"; `app.py:543` "Added last → outermost" | verified |
| 14 | CODEMAPS/backend.md:22 | "CORS — configurable allowlist" | Fixed regex (localhost/127.0.0.1, any port; plus `*.gradio.live` in Colab), GET/POST, headers Authorization/Content-Type. There is no config knob | `app.py:324-334` | verified |
| 15 | CODEMAPS/backend.md:23 | (none) | 4 Limiters: global/hybrid/ip/token | `app.py:511-520` | verified |
| 16 | CODEMAPS/backend.md:29 | Client section implied the 900 s timeouts live in `server/client/` | The `*_TIMEOUT_SEC = 900` constants are in `core/http_client.py`. `UNLOAD_MODEL_TIMEOUT_SEC` added | `core/http_client.py:42,50,59,78,88`; `server/client/generator.py:43` `_generation_timeout(text_len)` | verified |
| 17 | CODEMAPS/backend.md:31-43 | Long narrative sections | Condensed to a flow diagram plus a symbol list. All symbols confirmed to exist | `app_generation.py:66,92,156,206`; `generation_state_guard.py:117,219,274,349`; `model_loading.py:86,179,204,322`; `websocket.py:42,57,68`; `prompt_loading.py:12` | verified |
| 18 | CODEMAPS/backend.md:48 | app_generation 1228 · app.py 1180 · app_lifespan 843 · websocket 654 · app_models 604 · app_prompts 561 · validation 549 · model_loading 541 · vllm_client 332 | 1243 · 1223 · 870 · 656 · 600 · 558 · 558 · 536 · 335 (generator 620, guard 371 and prompt_loading 73 unchanged) | `wc -l` | verified |
| 19 | CODEMAPS/frontend.md:1 | interface/ui/ (5.5k LOC) | 11 .py (6.2k LOC) | `git ls-files 'qwen3_tts/interface/ui/*.py' \| xargs cat \| wc -l` → 6232 | verified |
| 20 | CODEMAPS/frontend.md:7-17 | (none) | Tab tree: status bar, Model Settings accordion, Clone/Design/Custom/Create Voice/Manage Voices/Manage Models, Recent Generations dataframe columns | `ui/_facade.py:237-460`; accordions at `tabs_generation.py:412,596,653,861` (inside the clone 367-546, design 547-810 and custom 811+ builders) | verified |
| 21 | CODEMAPS/frontend.md:19 | "Model badges refreshed via shared gr.Timer (5 s, one /models call)" | `status_timer` (`STATUS_POLL_SECONDS=5`) ticks 3 handlers. The badges use one `/models` call. `progress_timer` (2.0 s, built inactive) polls authed `/generation-status` | `_facade.py:99,254-258,309,414-431`; `generation.py:248-249,252-262,773-777`; `model_management.py:321-342` | verified |
| 22 | CODEMAPS/frontend.md:21 | (none) | Per-session `gr.State` inventory | `_facade.py:304,442-449`; `tabs_generation.py:454-466,907-914`; `generation.py:798`; `tabs_management.py:130-131,335` | verified |
| 23 | CODEMAPS/frontend.md:24-33 | shared 982, generation 712, _facade 585, tabs_generation 775, voice_management 494, tabs_management 460, history_panel 420, model_management 387 | 1102, 869, 661, 1017, 489, 502, 436, 383. theme.py 90 added | `wc -l` | verified |
| 24 | CODEMAPS/frontend.md:32 | "model_management.py — … ETA badge during load" and "Recent fixes: ETA badge — model-load ETA is surfaced" | The badge reads "(last load M:SS)" and is never an ETA. The "Recent fixes" block was removed | `model_management.py:273-282`; commit 08c2d83 "model loading badge no longer presents last-load time as ETA" | verified |
| 25 | CODEMAPS/frontend.md:33 | (none) | theme.py: `TOKENS`, `SEVERITY_CLASS`, `var()`, `UI_CSS` reach Gradio through `launch(css=)` | `theme.py:9-10,15,40,45,59`; `shared.py:1066` `"css": theme.UI_CSS` | verified |
| 26 | CODEMAPS/frontend.md:44 | (none) | Generation presets (T1b) added, tested by `tests/test_generation_preset_builder.py` | `tabs_generation.py:215,285,412`; `_facade.py:347-393`; the test file exists | verified |
| 27 | CODEMAPS/data.md:1 | no Files-scanned field | 5 .py (1.8k LOC) | `cat core/config/{io,paths,presets,models}.py server/generation_state_guard.py \| wc -l` → 1777 | verified |
| 28 | CODEMAPS/data.md:13 (old :11) | generation: … `language` (default `"auto"`) | `language` is a top-level key, not under `generation` | `python -c 'get_default_config()'` → top-level `"language": "auto"`; the `generation` dict has no `language`; `core/config/io.py:301` | verified |
| 29 | CODEMAPS/data.md:10-13 | (partial schema) | Added top-level, server, ui, cache, vllm, aliases, prompt_enhancer keys and the remaining advanced/generation defaults | `get_default_config()` dump (see Verification) | verified |
| 30 | CODEMAPS/data.md:13 | clone_speed / trim_icl_echo listed as ordinary generation keys | Both are engine `.get()` fallbacks, not `get_default_config()` keys | Default dump has neither key; CONFIG.md:167 says the same; `inference.py:977,1171` read them via `.get(` | verified |
| 31 | CODEMAPS/data.md:15 | security.rate_limits: generate, model_ops, transcribe, config_ops, global (prompt_ops missing; implied part of the config schema) | Not in `get_default_config()`. `validate_config()` adds 5 keys, including `prompt_ops` 10/min. `global` comes only from app.py | `core/config/io.py:63-69,155-168`; `server/app.py:499`; the default dump `security` = {max_text_length, max_batch_size} | verified |
| 32 | CODEMAPS/data.md:16 | (none) | presets: merged over 8 built-ins; defaults ship `consistent` and `creative` | `core/config/presets.py:19-62,71`; default dump | verified |
| 33 | CODEMAPS/data.md:21,24 | Runtime files listed without a location | `USER_FILES_DIR=~/Qwen3-TTS_UserFiles` holds `.voice_server.{pid,log,lock}` and `voice_prompts/`; config.json resolves there first, then the repo root | `core/config/paths.py:32,35-55,58-63` | verified |
| 34 | CODEMAPS/data.md:25 | (none) | `~/.voice_history.jsonl` (`HISTORY_FILE`) | `paths.py:60`; `generate_helpers.py:266`; `app_lifespan.py:83-86`; `cli_config.py:263-265` | verified |
| 35 | CODEMAPS/dependencies.md:6-10 | "3 distinct models" (no IDs) | Torch, MLX, ASR and vLLM-processor HF IDs | `core/config/models.py:278-346`; `core/engine/asr.py:44,71-72`; `core/engine_vllm.py:112` | verified |
| 36 | CODEMAPS/dependencies.md:15 | `mlx_audio>=0.5.0` | `mlx-audio>=0.5.4` (was already `>=0.5.1` at 59c23ac, so the codemap was wrong before #302) | `pyproject.toml` mlx extra; `git show 59c23ac:pyproject.toml \| grep mlx-audio` → `>=0.5.1` | verified |
| 37 | CODEMAPS/dependencies.md:25 | `uvicorn[standard]>=0.52.4` | `>=0.53.0` | `pyproject.toml` server extra (#339) | verified |
| 38 | CODEMAPS/dependencies.md:14-25 | extras only partly listed | Every extra with its floors: torch, mlx, vllm, cuda, audio, server, ui, rich, prompt-enhancer, dev; base deps | `pyproject.toml:10-120` | verified |
| 39 | CODEMAPS/dependencies.md:29-30 | (none) | External services: HF Hub, Anthropic API (optional), Gradio share | `asr.py:102-108,131-132` (`hf_hub_download`/`snapshot_download`), `ui/shared.py:185-208` (`anthropic.Anthropic`), `shared.py:1036-1082` (share auth) | verified |
| 40 | COMMANDS.md:16 | `make install` — "Quick install with all dependencies" | "Editable install, core dependencies only (`pip install -e .`)" | `Makefile:35-36` | verified |
| 41 | COMMANDS.md:25-26 | (no rows) | `tts say` alias; bare `tts start/stop/restart/status/log` | `cli.py:354-361`; `python -m qwen3_tts --help` lists `say`, `start`, `stop`, `restart`, `status`, `log` | verified |
| 42 | COMMANDS.md:35 | `tts voice create AUDIO` | `[AUDIO]` (optional) plus option list | `voice create --help` → `Usage: … voice create [OPTIONS] [AUDIO]` and its 6 options | verified |
| 43 | COMMANDS.md:39 | `tts voice rebuild NAME` | `[NAME]`; rebuilds all corrupt/missing prompts when NAME is omitted | `voice rebuild --help` → `[NAME]`; `cli_voice.py:278-285` docstring | verified |
| 44 | COMMANDS.md:74 | `tts cache prune` — "unused for N days" | plus `--unused N`, default 30 | `cache prune --help` → `--unused INTEGER … (default: 30)` | verified |
| 45 | COMMANDS.md:120 | (no row) | `make solid-score-fail` (below 35) | `Makefile:127-128` | verified |
| 46 | COMMANDS.md:162 | (no row) | `--list-speakers/--list-presets/--list-prosody` | `say --help`; `interface/generate.py:421-454` (render then `return False`); `cli_voice.py:356,366,397` use the same `cli_tables.render_*` | verified |
| 47 | CONFIG.md:30 | `TTS_LOG_LEVEL` "Server/log verbosity (default INFO)" | Server default INFO; the CLI defaults to WARNING on stderr | `server/app.py:1151`; `cli.py:59-66` | verified |
| 48 | CONFIG.md:31 | (missing) | `TTS_SKIP_WARMUP` | `core/engine/model_loader.py:483-495`; `server/app_lifespan.py:620-623` | verified |
| 49 | CONFIG.md:50 | (missing) | `TTS_UI_USERNAME` / `TTS_UI_PASSWORD` | `interface/ui/shared.py:1068-1082` | verified |
| 50 | CONFIG.md:52-56 | (missing) | `NO_COLOR` | `cli_output.py:32-34` | verified |
| 51 | CONTRIBUTING.md:102 | `make install` — "Install all dependencies" | core dependencies only | `Makefile:35-36` | verified |
| 52 | CONTRIBUTING.md:136-140 | Batch module counts 29/24/67/40/15 | 31/30/71/47/16 | `len(BATCHES[n]['modules'])` from `tests/run_batches.py` → 31, 30, 71, 47, 16, 1 | verified |
| 53 | .reports/codemap-diff.txt | 2026-09-21 diff | Regenerated for 59c23ac..6c6a39d | `git diff --name-status 59c23ac 6c6a39d -- qwen3_tts` (3 A, 41 M); `git log 59c23ac..6c6a39d -- qwen3_tts pyproject.toml` | verified |

**Totals: 53 rows — 53 verified, 0 inferred.**

### >30% rewrites (explanation)
- **backend.md**: `git diff --stat` shows 87 lines changed in a 63-line file. Why: (a) the route grouping by handler module was wrong (row 9); (b) the "all JSON routes typed" claim was false (row 12); (c) at ~3.3k tokens (13.2k chars) the file was far over the ~1000-token budget, and its header's "~890" understated that. The long narratives about the guard, streaming, echo-trim and model-rebind duplicated CLAUDE.md, so they were condensed to verified symbol names plus a pointer. Nothing was added that isn't in the evidence above.
- **frontend.md**: added a tab tree, timers and state (the requested scope) and removed the stale "ETA badge" fix list (row 24).
- Token budgets: after condensing, the files are architecture ~1400, backend ~1500, data ~1250, frontend ~1650 and dependencies ~850 (chars/4). Four are still above ~1000. I stopped trimming rather than drop verified content; the earlier headers also understated their sizes (backend claimed ~890 at ~3.3k).

## Unverified (suspected, left unchanged)
- CODEMAPS/frontend.md "Critical constraints": `gr.Audio` must be cleared with `None`, never `""`; a js-only `.then(fn=None, js=...)` never runs. These are Gradio runtime behaviors and can't be confirmed from repo code.
- CODEMAPS/architecture.md "Per-chunk seeds … 5-layer propagation CLI→request→engine": `seed_lock_chunks` exists (`validation.py:69`), but I did not trace all five layers.
- CODEMAPS/architecture.md "`core/protocols.py` removed (#179)": the file is absent (the `test -e` check below shows MISS, which is expected). I didn't look up the #179 history.
- COMMANDS.md Generation Options ranges ("--temperature (0.7-1.0)", "--top-k (1-50)", "--top-p (0.8-1.0)"): Click `--help` shows no ranges and I found no CLI-side range validation in the Click layer. I didn't trace the argparse layer, so these are left as recommendations.
- `Dockerfile:41` `CMD ["server", "start", "--public"]`: `tts server start` daemonizes by default, which may let the container's PID 1 exit. Not verified at runtime, and RUNBOOK has no Docker section to correct.

## Found wrong, not fixed (hand-written prose / out of scope)
- **CONTRIBUTING.md:131** "**3,200+ tests** across 184 modules": there are 215 `test_*.py` modules with 4,110 `def test_` functions. "3,200+" is technically still true; "184" is wrong. The same claim appears in CLAUDE.md "Testing" ("3,200+ tests across 184 modules").
- **CONTRIBUTING.md:265-266** "mypy … (FastAPI `app.py` and vLLM modules are excluded)": `pyproject.toml [tool.mypy] exclude` lists only `tests/`, `build/`, `colab_notebook.ipynb`, `server/vllm_client.py` and `core/engine_vllm.py`. app.py is type-checked (dependencies.md already says so).
- **CONTRIBUTING.md:190-191** example paths `tests/test_core_config.py` and `tests/test_server_app_generation.py` don't exist (`ls` → No such file).
- **rate-limiting.md:268** "Server stats include rate limit information" (`/stats`): `handle_stats` (`app_models.py:32-118`) and `StatsResponse` have no rate-limit fields.
- **RUNBOOK.md:400-402** tells you to check rate-limit status via `/stats`. Same error as above.
- **RUNBOOK.md:692-696** nginx example sets `X-Real-IP`, but the server only reads `X-Forwarded-For`, and only from `TTS_TRUSTED_PROXIES` peers (`app.py:192-205`). As written, per-IP limits would key on the proxy's IP.
- **CONFIG.md:7** "The main configuration file lives at the repository root: `config.json`": `_resolve_config_path()` prefers `~/Qwen3-TTS_UserFiles/config.json` and falls back to the repo root (`core/config/paths.py:35-55`).
- **CONFIG.md:182** "List them with `tts list presets`" follows text about the 8 merged built-ins, but `tts list presets` prints only `load_config()["presets"]`, without the `DEFAULT_GENERATION_PRESETS` merge (`cli_voice.py:359-366`).
- **CLAUDE.md** (out of scope for this refresh): "JSON routes carry Pydantic `response_model=` contracts" omits the untyped `/cancel-generation`. "live ETA badge, PR #175" is superseded by 08c2d83 ("last load"). The `tts voice {… rebuild [NAME] …}` row is correct.

## Verification

### 1. `python -m qwen3_tts.tools.check_config_docs`
```
$ python -m qwen3_tts.tools.check_config_docs; echo "exit=$?"
OK: CONFIG.md defaults match get_default_config() (66 keys).
exit=0
```

### 2. `test -e` on every path cited in the codemaps
Script: every `*.py|md|json|cjs|toml|lock|txt` token in `docs/CODEMAPS/*.md`, resolved against the repo root, `qwen3_tts/`, its subpackages and `docs/`. Runtime paths (`~/…`, `.voice_server.*`) are excluded.
```
OK   CLAUDE.md -> CLAUDE.md
OK   _facade.py -> qwen3_tts/interface/ui/_facade.py
OK   app.py -> qwen3_tts/server/app.py
OK   app_generation.py -> qwen3_tts/server/app_generation.py
OK   app_lifespan.py -> qwen3_tts/server/app_lifespan.py
OK   app_models.py -> qwen3_tts/server/app_models.py
OK   app_prompts.py -> qwen3_tts/server/app_prompts.py
OK   backend.md -> docs/CODEMAPS/backend.md
OK   cli.py -> qwen3_tts/cli.py
OK   cli_config.py -> qwen3_tts/cli_config.py
OK   cli_output.py -> qwen3_tts/cli_output.py
OK   cli_server.py -> qwen3_tts/cli_server.py
OK   cli_tables.py -> qwen3_tts/cli_tables.py
OK   cli_voice.py -> qwen3_tts/cli_voice.py
OK   client/generator.py -> qwen3_tts/server/client/generator.py
OK   components.py -> qwen3_tts/interface/ui/components.py
OK   core/config/io.py -> qwen3_tts/core/config/io.py
OK   core/config/models.py -> qwen3_tts/core/config/models.py
OK   core/config/paths.py -> qwen3_tts/core/config/paths.py
OK   core/config/pm2.py -> qwen3_tts/core/config/pm2.py
OK   core/config/presets.py -> qwen3_tts/core/config/presets.py
OK   core/engine/asr.py -> qwen3_tts/core/engine/asr.py
OK   core/engine_vllm.py -> qwen3_tts/core/engine_vllm.py
OK   core/http_client.py -> qwen3_tts/core/http_client.py
MISS core/protocols.py
OK   core/stream_protocol.py -> qwen3_tts/core/stream_protocol.py
OK   data.md -> docs/CODEMAPS/data.md
OK   docs/00-Foundations/ARCHITECTURE.md -> docs/00-Foundations/ARCHITECTURE.md
OK   docs/rate-limiting.md -> docs/rate-limiting.md
OK   ecosystem.config.cjs -> ecosystem.config.cjs
OK   engine/inference.py -> qwen3_tts/core/engine/inference.py
OK   engine_vllm.py -> qwen3_tts/core/engine_vllm.py
OK   generate.py -> qwen3_tts/interface/generate.py
OK   generate_interactive.py -> qwen3_tts/interface/generate_interactive.py
OK   generation.py -> qwen3_tts/interface/ui/generation.py
OK   generation_state_guard.py -> qwen3_tts/server/generation_state_guard.py
OK   history_panel.py -> qwen3_tts/interface/ui/history_panel.py
OK   inference.py -> qwen3_tts/core/engine/inference.py
OK   interface/generate_helpers.py -> qwen3_tts/interface/generate_helpers.py
OK   model_loading.py -> qwen3_tts/server/model_loading.py
OK   model_management.py -> qwen3_tts/interface/ui/model_management.py
OK   prompt_loading.py -> qwen3_tts/server/prompt_loading.py
OK   pyproject.toml -> pyproject.toml
OK   qwen3_tts/interface/ui/_facade.py -> qwen3_tts/interface/ui/_facade.py
OK   qwen3_tts/server/app.py -> qwen3_tts/server/app.py
OK   requirements.lock -> requirements.lock
OK   server/app.py -> qwen3_tts/server/app.py
OK   server/app_lifespan.py -> qwen3_tts/server/app_lifespan.py
OK   server/prompt_loading.py -> qwen3_tts/server/prompt_loading.py
OK   server/validation.py -> qwen3_tts/server/validation.py
OK   server/vllm_client.py -> qwen3_tts/server/vllm_client.py
OK   shared.py -> qwen3_tts/interface/ui/shared.py
OK   tabs_generation.py -> qwen3_tts/interface/ui/tabs_generation.py
OK   tabs_management.py -> qwen3_tts/interface/ui/tabs_management.py
OK   tests/test_generation_preset_builder.py -> tests/test_generation_preset_builder.py
OK   tests/test_prosody_preset_builder.py -> tests/test_prosody_preset_builder.py
OK   tests/test_response_contracts.py -> tests/test_response_contracts.py
OK   tests/test_stream_error_frame.py -> tests/test_stream_error_frame.py
OK   tests/test_stream_protocol.py -> tests/test_stream_protocol.py
OK   tests/test_ui_tab_select_wiring.py -> tests/test_ui_tab_select_wiring.py
OK   theme.py -> qwen3_tts/interface/ui/theme.py
OK   ui/_facade.py -> qwen3_tts/interface/ui/_facade.py
OK   ui/generation.py -> qwen3_tts/interface/ui/generation.py
OK   ui/shared.py -> qwen3_tts/interface/ui/shared.py
OK   ui/tabs_generation.py -> qwen3_tts/interface/ui/tabs_generation.py
OK   validation.py -> qwen3_tts/server/validation.py
OK   vllm_client.py -> qwen3_tts/server/vllm_client.py
OK   voice_management.py -> qwen3_tts/interface/ui/voice_management.py
OK   websocket.py -> qwen3_tts/server/websocket.py
checked=69 missing=1
```
The single MISS, `core/protocols.py`, is cited as **removed** ("`core/protocols.py` removed (#179)"), so its absence is expected.

### 3. `git diff --stat 6c6a39df` (staged tree; the ledger is excluded from the stat so it does not measure itself — it appears in the name-only list)
```
$ git diff --cached --stat 6c6a39df -- . ":!.reports/doc-refresh-ledger-cloud.md"
 .reports/codemap-diff.txt     | 72 +++++++++++++++++++++--------------
 docs/CODEMAPS/architecture.md | 32 +++++++---------
 docs/CODEMAPS/backend.md      | 87 ++++++++++++++++++-------------------------
 docs/CODEMAPS/data.md         | 15 +++++---
 docs/CODEMAPS/dependencies.md | 23 ++++++++----
 docs/CODEMAPS/frontend.md     | 46 ++++++++++++++---------
 docs/COMMANDS.md              | 12 ++++--
 docs/CONFIG.md                | 10 ++++-
 docs/CONTRIBUTING.md          | 12 +++---
 9 files changed, 172 insertions(+), 137 deletions(-)

$ git diff --cached --name-only 6c6a39df
.reports/codemap-diff.txt
.reports/doc-refresh-ledger-cloud.md
docs/CODEMAPS/architecture.md
docs/CODEMAPS/backend.md
docs/CODEMAPS/data.md
docs/CODEMAPS/dependencies.md
docs/CODEMAPS/frontend.md
docs/COMMANDS.md
docs/CONFIG.md
docs/CONTRIBUTING.md

$ git diff --cached --name-only 6c6a39df | grep -v -e "^docs/" -e "^\.reports/" | wc -l
0
```
