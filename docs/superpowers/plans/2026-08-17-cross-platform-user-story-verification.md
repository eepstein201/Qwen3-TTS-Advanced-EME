# Cross-Platform User-Story Verification + Colab Reconciliation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove every user-facing capability works on macOS/M2 Pro (MLX) and Linux (containerized CPU + CI), fix the four defects that verification exposed, and reconcile both Google Colab notebooks so Colab users get the same behavior.

**Architecture:** Three product defects share one root cause — *a fix landed in the canonical function but the paths users actually take bypass it*. Task 1 fixes config defaults that `install.sh` writes by hand instead of via `get_default_config()`. Tasks 2–3 close the voice-prompt sample-rate hole left open by the uncommitted `ensure_min_sample_rate()` work. Task 4 fixes a test-harness defect the Docker audit found. Tasks 5–7 propagate to docs and both notebooks. Tasks 8–10 are the verification passes and the ship step.

**Tech Stack:** Python 3.11, Click CLI, FastAPI server, Gradio 6.20 UI, MLX (`mlx-audio`) on macOS / PyTorch on Linux+Colab, unittest + pytest, Playwright E2E, Docker 29.7.2 + Compose v5.4.0 (Linux verification).

**Spec:** This plan is self-contained; the source evidence is the *Findings* table below. At execution time, copy this file to `docs/superpowers/plans/2026-08-17-cross-platform-user-story-verification.md` so it is version-tracked with the PR (per the repo's plan-docs-in-repo convention).

---

## Global Constraints

- **Never commit to `main`.** Feature branch → push → `gh pr create` → user merges. (CLAUDE.md Git Workflow, mandatory.)
- **No AI authorship attribution** anywhere — commits, comments, PR body, docs. No `Co-Authored-By`.
- **`CLAUDE.md` must stay ≤ 300 lines.** Currently **297**. `tests/test_claude_md.py` hard-fails above 300. Budget: 3 lines.
- **All `torch`/`mlx`/`transformers` imports stay lazy** (inside functions, never module scope).
- **Every new `tests/test_*.py` must be registered in `BATCHES`** in `tests/run_batches.py`, or `tests/test_batches_coverage.py` fails.
- **Immutability:** helpers return new objects; never mutate an argument in place (`_split_mlx_params` is the reference pattern).
- **`ensure_min_sample_rate()` never downsamples.** 48 kHz references generate fine; reducing them only discards data.
- **`DEFAULT_SAMPLE_RATE = 24000`** (`qwen3_tts/core/engine/audio_processing.py:23`) is the single source of truth. Do not add a second literal.
- **Any file a test reads at repo root must be `COPY`d into `Dockerfile.test`**, or that test errors in the container (see F6).
- Test env on this Mac: `conda run -n qwen3-tts-mlx …` (the `tts` alias is pinned to that env; spell out `conda run` in scripts).

### Measured Docker environment (verified, not assumed)

| Property | Value | Consequence for this plan |
|---|---|---|
| Engine / Compose | `29.7.2` / `v5.4.0` | Compose v2 syntax; `docker compose`, not `docker-compose` |
| Docker VM arch | **aarch64** | Native containers are `linux/arm64`. **CI runs `amd64`** — different arch |
| Buildx platforms | `linux/amd64, linux/arm64, …` | amd64 reachable via QEMU, but far too slow for the torch suite |
| VM memory | **6.2 GB** | **Too small for reliable 1.7B torch inference.** In-container generation is 0.6B-only and best-effort |
| VM CPUs | 12 | Suite parallelism is fine |
| Storage driver | overlayfs | — |
| Image cache | empty | First build downloads everything; budget ~10–15 min |

---

## Context

PR #190 (`eb8fa8e`, merged) made two dead MLX knobs live (`max_new_tokens`→`max_tokens`, `language`→`lang_code`) and changed the default `language` from `"English"` to `"auto"`. Separately, the working tree carries an **uncommitted** fix for a measured MLX bug: an 8 kHz reference `.wav` makes clone generation fail to emit EOS and run to the token cap 3/3 times (up to 47.8× expected tokens — minutes of looped audio for a 12-character input). Resampling the same audio to 24 kHz restored normal termination. The reproducer is still on disk: `voice_prompts/lt1.wav` (8000 Hz) vs `voice_prompts/lt1_24k.wav` (24000 Hz).

Verifying those changes across platforms showed **both are inert or incomplete on the paths real users take**, and auditing the Docker harness showed one test module has been erroring in the container all along.

### Findings

| # | Sev | Finding | Evidence |
|---|-----|---------|----------|
| **F1** | **HIGH** | `install.sh` writes `config.json` from a hand-maintained heredoc, not `get_default_config()`. It still emits `"language": "English"`, so **every new install on every platform** silently reverts PR #190. The repo's tracked `config.json` has the same stale value. | `install.sh:1118`, `config.json:19` vs `qwen3_tts/core/config/io.py:301` |
| **F1b** | MED | Same heredoc writes `"default_clone_prompt": "default_clone.pt"` — a dangling reference; no prompt ships with the package. Canonical default is `None`. | `install.sh:1114` vs `io.py:292`, `docs/CONFIG.md:71` |
| **F2** | **HIGH** | The Gradio **Create Voice** tab byte-copies the upload (`shutil.copy`) into `voice_prompts/<name>.wav` on MLX with **zero resampling** — it calls neither `ensure_min_sample_rate()` nor `load_audio_for_cloning()`. The primary GUI voice-creation path still produces runaway-generation prompts. | `qwen3_tts/interface/ui/voice_management.py:94` |
| **F3** | MED | Legacy low-rate prompts on disk stay broken and **silent**. `tts voice rebuild` regenerates the `.pt` (fixing torch) but never touches the `.wav` — and MLX reads the `.wav` path directly, so rebuild *appears* to fix a voice it does not fix. | `voice_prompt.py:319` returns `{"ref_audio": wav_path}`; `cli_voice.py:226-260` |
| **F6** | MED | **`Dockerfile.test` never copies `install.sh`,** but `tests/test_install_script.py` opens it unguarded in `setUp`. That module has been erroring in every containerized run. Task 1 adds three more tests to it, widening an already-broken seam. | `Dockerfile.test` (no `install.sh` in any `COPY`) vs `tests/test_install_script.py:10-17` |
| **F7** | **HIGH** | **When `librosa` is unavailable, `ensure_min_sample_rate()` logs a warning, returns `was_resampled=False`, and the caller then copies the original low-rate file to disk** — silently shipping the exact poisonous prompt the fix exists to prevent. A warning is not a mitigation when the failure mode is a multi-minute runaway generation. **Found by both Santa reviewers independently** (see Task S). | `audio_processing.py` `ensure_min_sample_rate` ImportError branch → `create_voice.py:93` |
| **F8** | MED | **`ensure_min_sample_rate()` only converts multi-channel → mono inside the resampling branch.** A 48 kHz stereo reference returns unchanged, so `sf.write` writes stereo and MLX gets a multi-channel reference. **Found by Santa reviewer C.** | `audio_processing.py` — `np.mean(audio, axis=-1)` sits after the `sr >= target_sr` early return |
| **F4** | LOW | `GenerateRequest.max_new_tokens` advertises `le=8192`, but MLX silently clamps to `_MLX_MAX_TOKENS_CEILING = 4096`. Correct behavior (torch genuinely supports 8192) but undocumented. | `server/validation.py:49` vs `inference.py:326` |
| **F5** | LOW | Colab notebook never writes `config['language']`, so it inherited the stale `"English"` via F1. New upsampling and token-cap behavior also undocumented there. | `colab_notebook.ipynb` cell 4 |

**Not defects (verified — do not "fix"):** the ~15 test files using `"language": "English"` are explicit fixtures exercising English normalization, not assertions about the default. The server's `/create-voice-prompt` handler is safe — it writes only `.pt` via `load_audio_for_cloning()`, which already resamples both directions. `tests/test_docker_config.py` asserts on `docker-compose.yml` and `Dockerfile.vllm` only, so editing `Dockerfile.test` is safe.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `install.sh` | Fresh-install config template | Modify (F1, F1b) |
| `config.json` | Repo's live config | Modify (F1) |
| `qwen3_tts/core/engine/audio_processing.py` | `ensure_min_sample_rate()` | Modify (F7, F8) |
| `qwen3_tts/interface/ui/voice_management.py` | Gradio voice CRUD | Modify (F2) |
| `qwen3_tts/core/engine/voice_prompt.py` | MLX prompt-load choke-point | Modify (F3) |
| `Dockerfile.test` | Linux test image | Modify (F6) |
| `docker-compose.test.yml` | Hardened Linux verification services | **Create** |
| `tests/validate_docker.sh` | Docker harness entry point | Modify — drive Compose, add gates |
| `tests/test_install_script.py` | Static install.sh assertions | Modify (guards F1) |
| `tests/test_voice_prompt_sample_rate.py` | Sample-rate guarantees | Modify — extend past the pure unit tests |
| `tests/test_docker_config.py` | Docker drift guards | Modify (guards F6) |
| `docs/CONFIG.md` | Config reference | Modify (F4) |
| `CLAUDE.md` | Primary context file | Modify (≤300 lines) |
| `colab_notebook.ipynb` | User-facing Colab demo | Modify (F5) |
| `tests/validate_colab.ipynb` | **Distinct** validation notebook | Rewrite → cross-platform user-story validator |

**Deliberately unchanged:** `server/validation.py` (`le=8192` is correct for torch; document only), `server/app_prompts.py` (already safe), `docker-compose.yml` (its GPU/vLLM services are drift-guarded; new services go in a separate override file), and `tools/create_voice.py` + `core/engine/{audio_processing,inference}.py` (the uncommitted work is correct as written and ships in Task 0).

---

## Model Routing & Agent Team

Per `/ecc:model-route` — effort is the primary lever; do not downgrade the model for correctness-sensitive work.

| Task | Agent | Model / effort | Why |
|------|-------|----------------|-----|
| 1, 5 | inline | Sonnet 5 / medium | Mechanical text edits behind a test guard |
| 2, 3 | `ecc:tdd-guide` → `ecc:python-reviewer` | Sonnet 5 / high | Real behavior change on a user path |
| 4 | inline | Sonnet 5 / medium | Dockerfile + compose authoring |
| 6, 7 | inline | Sonnet 5 / high | Notebook JSON corrupts easily; drift tests are strict |
| 8, 9 | inline **foreground** | — | Long-running; **background agents cannot prompt for Bash** |
| **S** | **`agy` cross-family** | `gemini-3.1-pro-high` + `gpt-oss-120b-medium` | Adversarial gate; a Claude reviewer shares the author's blind spots |
| 10 | inline | — | Git/PR |

Parallelizable (per `/ecc:team-builder`): Tasks **1**, **2+3**, **4**, and **6+7** touch disjoint files and can run as four concurrent workers. Tasks 8 and 9 must run **after** all of them; Task 8 needs a quiesced machine.

---

## Task 0: Baseline Evidence (before any change)

Establishes what was already red so nothing is misattributed later. **Do not skip** — the repo convention is *never call a failure pre-existing from reasoning; prove it.*

- [ ] **Step 1: Snapshot**

```bash
cd /Users/ericepstein/Qwen3-TTS_UserFiles
git status --short > /tmp/baseline-status.txt
```

- [ ] **Step 2: Static gates**

```bash
conda run -n qwen3-tts-mlx ruff check qwen3_tts tests 2>&1 | tail -5
conda run -n qwen3-tts-mlx mypy qwen3_tts/core qwen3_tts/server qwen3_tts/interface 2>&1 | tail -5
conda run -n qwen3-tts-mlx bandit -r qwen3_tts -c pyproject.toml 2>&1 | tail -15
conda run -n qwen3-tts-mlx make check-config-docs
```

Expected: ruff clean, mypy clean, bandit 0 HIGH, config-docs pass.

- [ ] **Step 3: Full non-E2E suite (superset of the batch runner)**

CI's `coverage` job runs all of `pytest -m "not e2e"`; `run_batches.py` is an explicit allowlist. A test outside `BATCHES` passes every batch gate yet fails CI.

```bash
conda run -n qwen3-tts-mlx python -m pytest tests/ -q -m "not e2e" 2>&1 | tail -20
```

**Record the exact pass count** — it is the Task 8 baseline.

- [ ] **Step 4: Baseline the Docker container too (proves F6 pre-exists)**

```bash
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
docker build -f Dockerfile.test -t qwen3-tts:baseline . 2>&1 | tail -5
docker run --rm qwen3-tts:baseline \
  python -m pytest tests/test_install_script.py -q 2>&1 | tail -15
```

Expected: **errors** with `FileNotFoundError: .../install.sh`. That is F6, pre-existing. Record the output verbatim.

- [ ] **Step 5: Record** all results to `/tmp/baseline.md`. Any failure here is pre-existing and must be reported as such, not silently fixed.

---

## Task 1: Stop `install.sh` reverting the config defaults (F1, F1b)

**Files:**
- Modify: `install.sh:1112-1119`, `config.json:19`
- Test: `tests/test_install_script.py`

**Interfaces:**
- Consumes: `qwen3_tts.core.config.get_default_config()` → `dict` with `"language": "auto"`, `"default_clone_prompt": None`.
- Produces: no new symbol; restores the invariant *install.sh's template agrees with `get_default_config()`*.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_install_script.py`:

```python
class TestInstallScriptMatchesPythonDefaults(unittest.TestCase):
    """install.sh writes config.json from a hand-maintained heredoc rather
    than calling get_default_config(), so the two drift silently. PR #190
    changed the language default to "auto" and the heredoc kept "English",
    so every fresh install reverted the fix.
    """

    def setUp(self):
        with open(INSTALL_SH) as f:
            self.text = f.read()

    def test_language_default_matches_python(self):
        from qwen3_tts.core.config import get_default_config

        expected = get_default_config()["language"]
        self.assertIn(f'"language": "{expected}"', self.text)

    def test_no_stale_english_language_default(self):
        self.assertNotIn('"language": "English"', self.text)

    def test_default_clone_prompt_is_not_a_dangling_filename(self):
        """get_default_config() ships None: no prompt ships with the package,
        so any seeded filename references a file that does not exist."""
        self.assertNotIn('"default_clone_prompt": "default_clone.pt"', self.text)
```

- [ ] **Step 2: Run it and confirm RED for the right reason**

```bash
conda run -n qwen3-tts-mlx python -m pytest tests/test_install_script.py -q
```

Expected: **3 failed** — `'"language": "auto"' not found`, `'"language": "English"' unexpectedly found`, `'"default_clone_prompt": "default_clone.pt"' unexpectedly found`. Any other failure means stop and diagnose.

- [ ] **Step 3: Commit the RED reproducer**

```bash
git add tests/test_install_script.py
git commit -m "test: assert install.sh config template matches get_default_config()"
```

- [ ] **Step 4: Fix `install.sh`**

- `install.sh:1114`: `"default_clone_prompt": "default_clone.pt",` → `"default_clone_prompt": null,`
- `install.sh:1118`: `"language": "English",` → `"language": "auto",`

- [ ] **Step 5: Fix the tracked `config.json`**

- `config.json:19`: `"language": "English",` → `"language": "auto",`

`config.json` has no `default_clone_prompt` key, so it already inherits `None`. Leave it absent.

- [ ] **Step 6: GREEN**

```bash
conda run -n qwen3-tts-mlx python -m pytest tests/test_install_script.py -q
```

- [ ] **Step 7: Prove the shipped config resolves to `auto`**

```bash
conda run -n qwen3-tts-mlx python -c "
from qwen3_tts.core.config import load_config, get_default_config
print('config.json  ->', load_config().get('language'))
print('code default ->', get_default_config()['language'])"
```

Expected: both print `auto`.

- [ ] **Step 8: Commit**

```bash
git add install.sh config.json
git commit -m "fix(config): install.sh template reverted language to English and seeded a dangling clone prompt"
```

---

## Task 2: Close the Gradio Create-Voice sample-rate bypass (F2)

**Files:**
- Modify: `qwen3_tts/interface/ui/voice_management.py:90-94`
- Test: `tests/test_voice_prompt_sample_rate.py`

**Interfaces:**
- Consumes: `ensure_min_sample_rate(audio, sr, target_sr=DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int, bool]` from `qwen3_tts.core.engine.audio_processing` — returns `(audio, sample_rate, was_resampled)`; never downsamples; returns input unchanged on empty audio or missing librosa.
- Produces: no new public symbol. Guarantees `voice_prompts/<name>.wav` written by the UI is ≥ 24000 Hz whenever the source is readable.

**Design note:** rewrite **only when a resample actually happened**. When the rate is already adequate, keep `shutil.copy` — it preserves the original bytes exactly, and re-encoding every upload through soundfile is a gratuitous quality and format risk. This mirrors `tools/create_voice.py` (`if wav_path and not was_resampled: shutil.copy2(...)`).

- [ ] **Step 1: Write the failing test**

Add `import os` and `from unittest.mock import patch` to the module imports, then append:

```python
class TestGradioCreateVoiceUpsamples(unittest.TestCase):
    """The Gradio Create Voice tab byte-copied the upload into
    voice_prompts/<name>.wav with no resampling, so the MLX runaway bug stayed
    fully reachable from the primary GUI path even after tools/create_voice.py
    was fixed. MLX reads that .wav by path (voice_prompt.py returns
    {"ref_audio": wav_path}), so the on-disk rate is what the model sees.
    """

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _source_wav(self, sr):
        import soundfile as sf

        path = os.path.join(self.tmp.name, f"src{sr}.wav")
        sf.write(path, _tone(2.0, sr), sr)
        return path

    def _create(self, src, name):
        """Invoke the real UI handler against a temp prompts dir."""
        import soundfile as sf

        from qwen3_tts.interface.ui import voice_management as vm

        with patch.object(vm, "VOICE_PROMPTS_DIR", self.tmp.name), patch.object(
            vm, "load_config", return_value={"advanced": {"backend": "mlx"}}
        ), patch.object(vm, "get_voice_prompts", return_value=[]), patch.object(
            vm, "get_default_clone_prompt", return_value=None
        ):
            vm.create_voice_prompt(src, "hello there friend", name)
        return sf.info(os.path.join(self.tmp.name, f"{name}.wav"))

    def test_low_rate_upload_is_upsampled_on_disk(self):
        info = self._create(self._source_wav(8000), "low")

        self.assertGreaterEqual(info.samplerate, DEFAULT_SAMPLE_RATE)

    def test_duration_is_preserved_when_upsampling(self):
        info = self._create(self._source_wav(8000), "low2")

        self.assertAlmostEqual(info.duration, 2.0, places=1)

    def test_adequate_rate_upload_is_copied_byte_for_byte(self):
        """Re-encoding a good upload would be a gratuitous quality risk."""
        info = self._create(self._source_wav(48000), "high")

        self.assertEqual(info.samplerate, 48000)
```

- [ ] **Step 2: Run and confirm RED**

```bash
conda run -n qwen3-tts-mlx python -m pytest tests/test_voice_prompt_sample_rate.py -q -k Gradio
```

Expected: `test_low_rate_upload_is_upsampled_on_disk` FAILS (`8000 >= 24000` is false); the other two pass. **If all three fail the harness is wrong, not the product** — fix the harness first.

- [ ] **Step 3: Commit the RED reproducer**

```bash
git add tests/test_voice_prompt_sample_rate.py
git commit -m "test: reproduce Gradio Create Voice writing a below-native-rate reference wav"
```

- [ ] **Step 4: Implement**

Add to the module's imports:

```python
from qwen3_tts.core.engine.audio_processing import DEFAULT_SAMPLE_RATE
```

Replace `qwen3_tts/interface/ui/voice_management.py:90-94`:

```python
        if backend == "mlx":
            # MLX opens this .wav by path (voice_prompt.py hands mlx-audio
            # ref_audio=<path>), so whatever rate lands on disk is what the
            # model sees — and a below-native rate makes clone generation fail
            # to emit EOS and run to the token cap. Rewrite only when a
            # resample actually happened; a byte copy preserves the original
            # exactly, which is the better outcome when the rate is fine.
            import shutil

            import soundfile as sf

            from qwen3_tts.core.engine.audio_processing import (
                ensure_min_sample_rate,
            )

            was_resampled = False
            try:
                src_audio, src_sr = sf.read(audio_path)
                src_audio, new_sr, was_resampled = ensure_min_sample_rate(
                    src_audio, src_sr
                )
            except (RuntimeError, OSError, ValueError) as exc:
                # Unreadable by soundfile (e.g. a container it cannot decode).
                # Fall through to the byte copy rather than failing creation —
                # but say so, because the rate then goes unchecked.
                logger.warning(
                    "Could not inspect the sample rate of the uploaded "
                    "reference audio (%s); copying it unchanged. If cloning "
                    "runs on and on, re-upload at %d Hz or higher.",
                    exc,
                    DEFAULT_SAMPLE_RATE,
                )

            if was_resampled:
                sf.write(wav_path, src_audio, new_sr)
                logger.info(
                    "Upsampled reference audio %d Hz -> %d Hz for %s",
                    src_sr,
                    new_sr,
                    base_name,
                )
            else:
                shutil.copy(audio_path, wav_path)
```

> `audio_processing` imports nothing heavy at module scope (`numpy`/`librosa` are lazy inside functions), so this top-level import does not violate the lazy-import rule — verified in Step 6.

- [ ] **Step 5: GREEN**

```bash
conda run -n qwen3-tts-mlx python -m pytest tests/test_voice_prompt_sample_rate.py -q
```

- [ ] **Step 6: Confirm no lazy-import regression and no lint break**

```bash
conda run -n qwen3-tts-mlx python -c "
import sys, qwen3_tts.interface.ui.voice_management
assert 'torch' not in sys.modules and 'mlx' not in sys.modules, 'heavy import leaked'
print('lazy imports intact')"
conda run -n qwen3-tts-mlx ruff check qwen3_tts/interface/ui/voice_management.py
```

- [ ] **Step 7: Commit**

```bash
git add qwen3_tts/interface/ui/voice_management.py tests/test_voice_prompt_sample_rate.py
git commit -m "fix(ui): resample below-native-rate reference audio in the Create Voice tab"
```

---

## Task 3: Warn on legacy low-rate prompts at MLX load (F3)

**Files:**
- Modify: `qwen3_tts/core/engine/voice_prompt.py` (in `load_voice_prompt_mlx`, just before `result = {...}` at ~line 319)
- Test: `tests/test_voice_prompt_sample_rate.py`

**Interfaces:**
- Consumes: `DEFAULT_SAMPLE_RATE` from `qwen3_tts.core.engine.audio_processing` (extend the existing `load_audio_for_cloning` import at line 21).
- Produces: no new symbol. A `logger.warning` on `tts.engine` when the resolved `.wav` is below `DEFAULT_SAMPLE_RATE`.

**Why here:** `load_voice_prompt_mlx` is the single choke-point where MLX resolves a prompt name to a `.wav` path, and its result is **cached**, so the warning fires once per cache-miss rather than once per generation. It covers every legacy prompt on disk regardless of which of the seven creation paths made it — including ones `tts voice rebuild` "fixed".

- [ ] **Step 1: Write the failing test**

```python
class TestLegacyLowRatePromptWarnsOnLoad(unittest.TestCase):
    """Prompts created before the resampling fix are still on disk and still
    broken. `tts voice rebuild` does not repair them — it regenerates the .pt
    and leaves the .wav alone, while MLX reads the .wav. Warning at the load
    choke-point is what makes that visible.
    """

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _prompt(self, name, sr):
        import soundfile as sf

        sf.write(os.path.join(self.tmp.name, f"{name}.wav"), _tone(1.0, sr), sr)
        with open(os.path.join(self.tmp.name, f"{name}.txt"), "w") as f:
            f.write("hello there friend")

    def _load(self, name):
        from qwen3_tts.core.engine import voice_prompt as vp

        vp.clear_voice_prompt_cache()
        with patch.object(vp, "VOICE_PROMPTS_DIR", self.tmp.name):
            return vp.load_voice_prompt_mlx(name)

    def test_warns_for_below_native_rate(self):
        self._prompt("legacy", 8000)

        with self.assertLogs("tts.engine", level="WARNING") as logs:
            self._load("legacy")

        self.assertTrue(
            any("8000" in ln and "24000" in ln for ln in logs.output),
            f"expected a rate warning naming both rates, got: {logs.output}",
        )

    def test_silent_for_adequate_rate(self):
        self._prompt("fine", DEFAULT_SAMPLE_RATE)

        with patch("qwen3_tts.core.engine.voice_prompt.logger") as mock_log:
            self._load("fine")

        rate_warnings = [c for c in mock_log.warning.call_args_list if "Hz" in str(c)]
        self.assertEqual(rate_warnings, [])

    def test_load_still_returns_the_prompt(self):
        """The warning is advisory — it must never break loading."""
        self._prompt("legacy2", 8000)

        result = self._load("legacy2")

        self.assertEqual(result["ref_text"], "hello there friend")
        self.assertTrue(result["ref_audio"].endswith("legacy2.wav"))
```

- [ ] **Step 2: Run and confirm RED**

```bash
conda run -n qwen3-tts-mlx python -m pytest tests/test_voice_prompt_sample_rate.py -q -k Legacy
```

Expected: `test_warns_for_below_native_rate` FAILS with `AssertionError: no logs of level WARNING or higher triggered on tts.engine`.

- [ ] **Step 3: Commit the RED reproducer**

```bash
git add tests/test_voice_prompt_sample_rate.py
git commit -m "test: reproduce silent load of a legacy below-native-rate MLX prompt"
```

- [ ] **Step 4: Implement**

Extend the import at `voice_prompt.py:21`:

```python
from qwen3_tts.core.engine.audio_processing import (
    DEFAULT_SAMPLE_RATE,
    load_audio_for_cloning,
)
```

Insert immediately before `result = {"ref_audio": wav_path, "ref_text": ref_text}`:

```python
    # Prompts created before reference audio was resampled on write are still
    # on disk, and MLX opens this path directly — so a below-native rate here
    # is the runaway-generation bug, not a quality nit. `tts voice rebuild`
    # does NOT repair it: rebuild regenerates the .pt and leaves the .wav
    # alone. Advisory only; never block a load over it. Cached, so this fires
    # once per cache-miss rather than once per generation.
    try:
        import soundfile as sf  # lazy — optional at import time

        on_disk_sr = sf.info(wav_path).samplerate
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break loading
        logger.debug("Could not read sample rate of %s: %s", wav_path, exc)
    else:
        if on_disk_sr < DEFAULT_SAMPLE_RATE:
            logger.warning(
                "Voice prompt '%s' has a %d Hz reference (below the model's "
                "native %d Hz). This makes clone generation fail to stop and "
                "run to the token cap. Re-create it with 'tts voice create' — "
                "'tts voice rebuild' will NOT fix it (it only regenerates the "
                ".pt; MLX reads the .wav).",
                sanitize_log(base),
                on_disk_sr,
                DEFAULT_SAMPLE_RATE,
            )
```

Confirm `sanitize_log` is imported in this module from `qwen3_tts.core.config`; add it to that import block if not.

- [ ] **Step 5: GREEN, plus the neighbouring suites**

```bash
conda run -n qwen3-tts-mlx python -m pytest tests/test_voice_prompt_sample_rate.py -q
conda run -n qwen3-tts-mlx python -m pytest tests/test_voice_prompts.py tests/test_voice_engine.py -q
```

- [ ] **Step 6: Prove it on the real reproducer on disk**

```bash
conda run -n qwen3-tts-mlx python -c "
import logging; logging.basicConfig(level=logging.WARNING)
from qwen3_tts.core.engine.voice_prompt import load_voice_prompt_mlx
load_voice_prompt_mlx('lt1')       # 8000 Hz — must warn
load_voice_prompt_mlx('lt1_24k')   # 24000 Hz — must be silent"
```

Expected: exactly one warning, naming `lt1`, `8000`, `24000`.

- [ ] **Step 7: Commit**

```bash
git add qwen3_tts/core/engine/voice_prompt.py tests/test_voice_prompt_sample_rate.py
git commit -m "fix(engine): warn when an MLX voice prompt has a below-native-rate reference"
```

---

## Task 3b: Harden `ensure_min_sample_rate` (F7, F8 — found by adversarial review)

**Files:**
- Modify: `qwen3_tts/core/engine/audio_processing.py` — `ensure_min_sample_rate`
- Test: `tests/test_voice_prompt_sample_rate.py`

**Interfaces:**
- Changes the contract: `ensure_min_sample_rate` **raises** `RuntimeError` when it cannot deliver the guarantee, instead of returning `was_resampled=False`. Callers in `tools/create_voice.py` and `ui/voice_management.py` must surface that as a user-facing error rather than writing a file.
- Mono reduction moves **before** the rate early-return, so the mono guarantee holds at every rate.

**Why the contract changes:** the current signature invites F7. `(audio, sr, False)` is indistinguishable between "already fine, nothing to do" and "could not fix it, this file is poison" — and the only caller treats both as success. A function whose entire purpose is a guarantee must not have a silent path that returns without it. Warning is not mitigation when the failure mode is a multi-minute runaway generation that looks like a hang.

- [ ] **Step 1: Write the failing tests**

```python
class TestEnsureMinSampleRateNeverSilentlyFails(unittest.TestCase):
    """F7: the ImportError branch logged a warning and returned
    was_resampled=False, and create_voice.py then copied the original
    low-rate file to disk — silently shipping the exact poisonous prompt
    the function exists to prevent. Found by both Santa reviewers.
    """

    def test_raises_when_it_cannot_resample_a_low_rate_clip(self):
        audio = _tone(1.0, 8000)

        # Simulate librosa being unavailable.
        with patch.dict("sys.modules", {"librosa": None}):
            with self.assertRaises(RuntimeError) as ctx:
                ensure_min_sample_rate(audio, 8000)

        self.assertIn("8000", str(ctx.exception))

    def test_does_not_raise_when_no_resample_is_needed(self):
        """A missing librosa is irrelevant when the rate is already fine."""
        audio = _tone(1.0, DEFAULT_SAMPLE_RATE)

        with patch.dict("sys.modules", {"librosa": None}):
            out, sr, resampled = ensure_min_sample_rate(audio, DEFAULT_SAMPLE_RATE)

        self.assertFalse(resampled)
        self.assertEqual(sr, DEFAULT_SAMPLE_RATE)

    def test_resample_failure_is_reported_not_swallowed(self):
        audio = _tone(1.0, 8000)

        with patch("librosa.resample", side_effect=ValueError("boom")):
            with self.assertRaises(RuntimeError):
                ensure_min_sample_rate(audio, 8000)


class TestEnsureMinSampleRateAlwaysReturnsMono(unittest.TestCase):
    """F8: mono reduction sat *after* the `sr >= target_sr` early return, so a
    48 kHz stereo reference passed straight through and MLX received a
    multi-channel file. Found by Santa reviewer C.
    """

    def test_stereo_at_adequate_rate_is_still_reduced_to_mono(self):
        mono = _tone(0.5, 48000)
        stereo = np.stack([mono, mono], axis=-1)

        out, sr, resampled = ensure_min_sample_rate(stereo, 48000)

        self.assertEqual(out.ndim, 1)
        self.assertEqual(sr, 48000)
        self.assertFalse(resampled)

    def test_stereo_at_low_rate_is_mono_and_upsampled(self):
        mono = _tone(0.5, 8000)
        stereo = np.stack([mono, mono], axis=-1)

        out, sr, resampled = ensure_min_sample_rate(stereo, 8000)

        self.assertEqual(out.ndim, 1)
        self.assertEqual(sr, DEFAULT_SAMPLE_RATE)
        self.assertTrue(resampled)
```

> The existing `test_leaves_target_rate_untouched` and `test_never_downsamples_higher_rates` assert `assertIs(out, audio)` — identity. Moving mono reduction earlier breaks identity for multi-channel input only; for 1-D input the function must still return the **same object**. Keep those tests passing by short-circuiting when `ndim == 1`.

- [ ] **Step 2: Run and confirm RED**

```bash
conda run -n qwen3-tts-mlx python -m pytest tests/test_voice_prompt_sample_rate.py -q -k "NeverSilentlyFails or AlwaysReturnsMono"
```

Expected: `test_raises_when_it_cannot_resample_a_low_rate_clip`, `test_resample_failure_is_reported_not_swallowed`, and `test_stereo_at_adequate_rate_is_still_reduced_to_mono` all FAIL.

- [ ] **Step 3: Commit the RED reproducers**

```bash
git add tests/test_voice_prompt_sample_rate.py
git commit -m "test: reproduce silent poison-prompt write and stereo passthrough in ensure_min_sample_rate"
```

- [ ] **Step 4: Rewrite `ensure_min_sample_rate`**

```python
def ensure_min_sample_rate(audio, sr, target_sr=DEFAULT_SAMPLE_RATE):
    """Upsample reference audio that sits below the model's native rate.

    Returns ``(audio, sample_rate, was_resampled)``. Always returns mono.

    Measured 2026-08-16: an 8 kHz reference `.wav` makes MLX clone generation
    fail to emit EOS — it runs to the token cap on every attempt (3/3, up to
    47.8x the expected token count), producing minutes of looped audio for a
    12-character input. Resampling the same audio with the same transcript to
    24 kHz restored normal termination.

    This has to happen when the prompt is *written*, not when it is loaded:
    the MLX clone path passes ``ref_audio=<path>`` straight to mlx-audio,
    which opens the file itself, so ``load_audio_for_cloning()``'s own
    resampling never applies — whatever rate is on disk is what the model sees.

    Never downsamples. 48 kHz references generate fine, and reducing them
    would only discard data.

    Raises:
        RuntimeError: if the clip is below ``target_sr`` and cannot be
            resampled. This is deliberately fatal. Returning
            ``was_resampled=False`` here would be indistinguishable from
            "already fine", and every caller writes the file on that signal —
            shipping the poisonous prompt this function exists to prevent.
    """
    import numpy as np  # lazy — heavy import

    if audio is None or len(audio) == 0:
        return audio, sr, False

    # Mono reduction happens BEFORE the rate check: a 48 kHz stereo reference
    # needs no resampling but still must not reach the model as two channels.
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=-1).astype(np.float32)

    if sr >= target_sr:
        return audio, sr, False

    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError(
            f"Reference audio is {sr} Hz, below the model's native "
            f"{target_sr} Hz, and librosa is not installed to resample it. "
            f"A below-native-rate reference makes clone generation fail to "
            f"stop and run to the token cap. Install librosa, or supply "
            f"audio at {target_sr} Hz or higher."
        ) from exc

    mono = np.asarray(audio, dtype=np.float32)
    try:
        resampled = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to resample {sr} Hz reference audio to {target_sr} Hz: "
            f"{exc}. Refusing to write a below-native-rate reference, which "
            f"would make clone generation fail to stop."
        ) from exc

    return resampled.astype(np.float32), target_sr, True
```

- [ ] **Step 5: Surface the error in both callers**

In `qwen3_tts/tools/create_voice.py`, the `ensure_min_sample_rate` call now propagates `RuntimeError` — which is the correct behavior for a CLI (it aborts before writing). Confirm no bare `except Exception` upstream swallows it.

In `qwen3_tts/interface/ui/voice_management.py` (Task 2), extend the caught set so the user sees a Gradio error rather than a traceback, and **no file is written**:

```python
            try:
                src_audio, src_sr = sf.read(audio_path)
                src_audio, new_sr, was_resampled = ensure_min_sample_rate(
                    src_audio, src_sr
                )
            except RuntimeError as exc:
                # Cannot deliver the rate guarantee — refuse to create the
                # prompt rather than write one that hangs generation.
                raise gr.Error(str(exc))
            except (OSError, ValueError) as exc:
                logger.warning(
                    "Could not inspect the sample rate of the uploaded "
                    "reference audio (%s); copying it unchanged. If cloning "
                    "runs on and on, re-upload at %d Hz or higher.",
                    exc,
                    DEFAULT_SAMPLE_RATE,
                )
                was_resampled = False
```

- [ ] **Step 6: GREEN, including the pre-existing tests**

```bash
conda run -n qwen3-tts-mlx python -m pytest tests/test_voice_prompt_sample_rate.py -v
```

All must pass — including `test_leaves_target_rate_untouched` and `test_never_downsamples_higher_rates`, whose `assertIs` identity checks constrain the 1-D path.

- [ ] **Step 7: Commit**

```bash
git add qwen3_tts/core/engine/audio_processing.py qwen3_tts/tools/create_voice.py \
        qwen3_tts/interface/ui/voice_management.py tests/test_voice_prompt_sample_rate.py
git commit -m "fix(engine): fail loudly instead of writing a below-native-rate reference; always return mono"
```

---

## Task 4: Fix and harden the Docker test harness (F6)

**Files:**
- Modify: `Dockerfile.test`
- Create: `docker-compose.test.yml`
- Modify: `tests/validate_docker.sh`
- Test: `tests/test_docker_config.py`

**Interfaces:**
- Produces: Compose services `suite` (full non-E2E run), `gates` (ruff/mypy/bandit/config-docs), and `shell` (interactive debug), all on project `qwen3-tts-test`.

**Why:** `tests/test_install_script.py` opens `install.sh` unguarded in `setUp`, but no `COPY` puts that file in the image — so the module has been erroring in every container run. Task 1 adds three more tests to it. Fix the image rather than guarding the test: the file genuinely should be under test on Linux, which is where `install.sh` actually runs.

- [ ] **Step 1: Write the failing drift test**

Append to `tests/test_docker_config.py`:

```python
class TestDockerfileTestCopiesFilesTestsRead(unittest.TestCase):
    """Repo-root files that tests open must be COPYd into the test image, or
    the module errors in the container while passing on the host. install.sh
    was missing, so tests/test_install_script.py never ran on Linux.
    """

    def setUp(self):
        import os

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "Dockerfile.test")) as f:
            self.content = f.read()

    def test_copies_install_sh(self):
        self.assertIn("install.sh", self.content)

    def test_copies_every_repo_root_file_a_test_opens(self):
        """Guard the whole class of bug, not just install.sh."""
        for name in ("CLAUDE.md", "config.json", "pytest.ini",
                     "colab_notebook.ipynb", "install.sh", "pyproject.toml"):
            with self.subTest(name=name):
                self.assertIn(name, self.content)
```

- [ ] **Step 2: Run and confirm RED**

```bash
conda run -n qwen3-tts-mlx python -m pytest tests/test_docker_config.py -q -k DockerfileTest
```

Expected: both fail on `install.sh`.

- [ ] **Step 3: Fix `Dockerfile.test`**

Extend the existing repo-root `COPY` line to include `install.sh`:

```dockerfile
# Repo-root files referenced by guard/drift tests (CLAUDE.md size guard,
# ARCHITECTURE.md companion, colab notebook drift, docker config drift,
# install.sh config-template drift)
COPY --chown=testuser:testuser CLAUDE.md colab_notebook.ipynb docker-compose.yml Dockerfile.vllm install.sh ./
```

- [ ] **Step 4: GREEN**

```bash
conda run -n qwen3-tts-mlx python -m pytest tests/test_docker_config.py -q
```

- [ ] **Step 5: Create `docker-compose.test.yml`**

A separate override file, **not** an edit to `docker-compose.yml` — that file's GPU/vLLM services are drift-guarded by `test_docker_config.py`, and CPU test services have no business in the production stack.

```yaml
# Hardened Linux verification services. Separate from docker-compose.yml,
# whose GPU/vLLM services are drift-guarded and production-shaped.
#
#   docker compose -p qwen3-tts-test -f docker-compose.test.yml run --rm suite
#
# The image BAKES the source in (Dockerfile.test COPYs it), so there is no
# bind mount and a run can never mutate this checkout. Rebuild to pick up
# code changes -- a stale image silently tests the previous commit.
services:
  # Shared build definition; not run directly.
  base: &base
    build:
      context: .
      dockerfile: Dockerfile.test
      target: test
    image: qwen3-tts:test
    init: true
    # Dependencies are baked at build time, so the suite needs no network.
    # This also guarantees no test can reach the internet and pass by luck.
    network_mode: none
    read_only: true
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    pids_limit: 512
    user: "1000:1000"
    environment:
      GRADIO_ANALYTICS_ENABLED: "False"
      PYTHONDONTWRITEBYTECODE: "1"
      # Batch runner and E2E suites both key off this.
      TTS_DISABLE_RATE_LIMITING: "1"
    tmpfs:
      # read_only:true means every writable path must be declared.
      - /tmp:size=2g,uid=1000,gid=1000,mode=0700
      - /home/testuser:size=512m,uid=1000,gid=1000,mode=0700
      - /app/voice_prompts:size=256m,uid=1000,gid=1000,mode=0700

  # Full non-E2E suite -- the superset the CI coverage job runs.
  suite:
    <<: *base
    command: ["python", "-m", "pytest", "tests/", "-q", "-m", "not e2e", "--tb=short"]

  # Batch runner: bounded per-batch timeouts, dumps thread stacks on a hang.
  batches:
    <<: *base
    command: ["python", "tests/run_batches.py"]

  # Static gates, matching the Makefile targets.
  gates:
    <<: *base
    command:
      - bash
      - -lc
      - |
        set -x
        ruff check qwen3_tts tests
        python -m qwen3_tts.tools.check_config_docs

  # Interactive debugging. Start detached so closing a terminal does not
  # destroy the session:
  #   docker compose -p qwen3-tts-test -f docker-compose.test.yml \
  #     run --detach --name qwen3-shell shell
  #   docker exec -it -w /app qwen3-shell bash
  shell:
    <<: *base
    command: ["bash"]
    stdin_open: true
    tty: true
```

- [ ] **Step 6: Validate the Compose model before building**

```bash
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
docker compose -f docker-compose.test.yml config --quiet && echo "compose model valid"
```

- [ ] **Step 7: Rewrite `tests/validate_docker.sh` to drive Compose**

```bash
#!/bin/bash
# Validate the suite inside a hardened Linux container.
# Run from anywhere; resolves the project root itself.
#
# A Linux container shares the host kernel: this proves LINUX behavior only.
# It does not validate macOS, Windows, or CUDA.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Docker Desktop ships its credential helper outside the default PATH on macOS.
if [ -d /Applications/Docker.app/Contents/Resources/bin ]; then
    PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
    export PATH
fi

COMPOSE="docker compose -p qwen3-tts-test -f docker-compose.test.yml"

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running. Start Docker Desktop and re-run."
    exit 1
fi

echo "=== Validating compose model ==="
$COMPOSE config --quiet

echo "=== Building test image (arch: $(uname -m)) ==="
$COMPOSE build suite

echo "=== Static gates ==="
$COMPOSE run --rm -T gates

echo "=== Full non-E2E suite ==="
$COMPOSE run --rm -T suite

echo "=== Batch runner ==="
$COMPOSE run --rm -T batches

$COMPOSE down --remove-orphans

echo "=== Docker validation complete ==="
```

- [ ] **Step 8: Commit**

```bash
chmod +x tests/validate_docker.sh
git add Dockerfile.test docker-compose.test.yml tests/validate_docker.sh tests/test_docker_config.py
git commit -m "test(docker): copy install.sh into the test image and drive verification via compose"
```

---

## Task 5: Documentation reconciliation (F4, and the new behavior)

**Files:** `docs/CONFIG.md:151`, `CLAUDE.md` (**297/300 — trim first**)

- [ ] **Step 1: Document the MLX ceiling**

Replace `docs/CONFIG.md:151`:

```markdown
| `generation.max_new_tokens` | integer | `2048` | Hard cap on generated tokens per `model.generate()` call. **MLX clamps this to 4096** (`_MLX_MAX_TOKENS_CEILING`) regardless of the value here or the request schema's `le=8192`: PRF-9 measured ≥8192 as unstable on 16 GB (EOS-failure runaway loops + memory exhaustion). Torch honors the full range. |
```

- [ ] **Step 2: Verify the docs gate**

`check_config_docs` compares only the **default-value column**, unchanged at `2048`, so this stays green.

```bash
conda run -n qwen3-tts-mlx make check-config-docs
```

- [ ] **Step 3: Add the sample-rate note to `CLAUDE.md` with zero net lines**

Fold it into the **existing** `qwen3_tts/core/engine/` row of the Architecture table (which already describes `_postprocess_chunk`), appending inside that cell:

```
**Reference-audio rate:** MLX clone passes `ref_audio=<path>` to mlx-audio, which opens the file itself — so `load_audio_for_cloning()`'s resampling never applies and the on-disk rate is what the model sees. A below-24 kHz reference makes generation fail to emit EOS and run to the token cap (measured 3/3, up to 47.8×). `ensure_min_sample_rate()` (never downsamples) is therefore applied at **write** time in `tools/create_voice.py` and `ui/voice_management.py`; `load_voice_prompt_mlx` warns for legacy prompts, and `_warn_if_cap_reached()` reports any generation that stopped at the cap. `tts voice rebuild` does NOT repair these — it regenerates the `.pt` only.
```

- [ ] **Step 4: Verify the length guard**

```bash
wc -l CLAUDE.md
conda run -n qwen3-tts-mlx python -m pytest tests/test_claude_md.py -q
```

Expected ≤300. A local hookify guard (`.claude/hookify.claude-md-length-guard.local.md`) will also block an over-cap edit.

- [ ] **Step 5: Commit**

```bash
git add docs/CONFIG.md CLAUDE.md
git commit -m "docs: record the MLX token ceiling and the reference-audio rate requirement"
```

---

## Task 6: Reconcile `colab_notebook.ipynb` (F5)

**Files:** `colab_notebook.ipynb` (cells 1, 2, 4); guarded by `tests/colab/test_notebook_drift.py`

**Hard constraints — violating any of these fails the build:**

| Constraint | Enforced by |
|---|---|
| Cell count in **[8, 14]** (currently **11**) | `test_notebook_has_expected_cell_count` |
| Setup cell keeps parsing `pyproject.toml` via `tomllib`/`tomli` | `test_setup_cell_uses_pyproject_extras` |
| References all 6 extras as quoted strings: `torch server audio ui cuda rich` | `test_setup_cell_references_all_required_extras` |
| Any `DEPS = (...)` literal ≤ **6** whitespace tokens | `test_no_hardcoded_dep_megastring` |
| Keeps the `startswith(("torch","torchaudio"...))` filter | `test_setup_cell_does_not_reinstall_torch` |
| Keeps `rubberband-cli` in the apt line | `test_setup_cell_installs_rubberband` |
| All 16 `@param` names remain present verbatim | `TestSettingsSurface` (5 tests) |
| Turing branch **writes** `torch_quantization` to config | `test_turing_writes_8bit_quantization` |
| Voice-clone cell passes `x_vector_only_mode`; never matches `transcript = X or None` | `TestVoiceCloneCell` |
| Launch cell `allowed_paths` contains `"Qwen3-TTS Output"` | `test_allowed_paths_covers_history_output` |

**Approach: edit existing cells only; add no cells.**

- [ ] **Step 1: Add a `LANGUAGE` form field to the Settings cell (cell 2)**

Insert after the `MAX_CHUNK_CHARS` line:

```python
LANGUAGE = "auto"  # @param ["auto", "English", "Chinese", "Spanish", "French", "German", "Japanese", "Korean"]
```

- [ ] **Step 2: Persist it in the Setup cell (cell 4)**

Beside the other `config[...] = ...` writes:

```python
# "auto" lets the model infer language from the text. A concrete value forces
# conditioning, which is wrong whenever the text is not in that language — and
# the Gradio UI exposes no language control to override it.
config['language'] = LANGUAGE
```

Extend the summary print so it appears in the run log:

```python
print(f'Language: {LANGUAGE}')
```

- [ ] **Step 3: Update the "What's new" markdown (cell 1)**

```markdown
- **Language defaults to `auto`** — the model infers language from the text. Set `LANGUAGE` in Settings to force conditioning. (Previously `English`, which mis-conditioned non-English text.)
- **Reference audio is upsampled automatically** — a below-24 kHz clone reference makes generation fail to stop and run to the token cap. Voice creation now resamples on write; existing low-rate prompts warn on load and must be re-created (`tts voice create`), not rebuilt.
- **Truncation is no longer silent** — a generation that stops at the token cap instead of on EOS now logs a warning.
```

- [ ] **Step 4: Validate JSON and the drift guards**

```bash
conda run -n qwen3-tts-mlx python -c "
import json; nb = json.load(open('colab_notebook.ipynb'))
print('cells:', len(nb['cells'])); assert 8 <= len(nb['cells']) <= 14"
conda run -n qwen3-tts-mlx python -m pytest tests/colab/ tests/test_colab_paths.py -q
```

- [ ] **Step 5: Commit**

```bash
git add colab_notebook.ipynb
git commit -m "fix(colab): expose the language setting and document the reference-audio rate requirement"
```

---

## Task 7: Turn `tests/validate_colab.ipynb` into the cross-platform user-story validator

**Files:** Rewrite `tests/validate_colab.ipynb`

This notebook is **already distinct** from `colab_notebook.ipynb` — it clones from GitHub rather than mounting Drive, and is a harness rather than a demo. Keep that separation. Today it only shells out to `run_batches.py`; extend it to assert the user stories on a real CUDA runtime, the only place the torch/CUDA half of this plan can be proven.

Confirm the drift tests target the user notebook only before rewriting:

```bash
grep -n "colab_notebook\|validate_colab" tests/colab/test_notebook_drift.py
```

- [ ] **Step 1: Keep cells 0–5** (clone, `pip install -e ".[test]"`, GPU/torch version print).

- [ ] **Step 2: Replace the batch-runner section with the full non-E2E suite**

```python
# Full non-E2E suite — the superset the CI coverage job runs.
# run_batches.py is an explicit allowlist and will silently skip an
# unregistered module, so do not rely on it alone.
!python -m pytest tests/ -q -m "not e2e" --tb=short
```

- [ ] **Step 3: Add a static-gates cell**

```python
!ruff check qwen3_tts tests
!mypy qwen3_tts/core qwen3_tts/server qwen3_tts/interface
!bandit -r qwen3_tts -c pyproject.toml -ll
!python -m qwen3_tts.tools.check_config_docs
```

- [ ] **Step 4: Add the config-reconciliation cell**

```python
# Guards the F1 class of bug: a hand-maintained config template silently
# reverting a changed default. install.sh writes config.json from a heredoc,
# so the two can drift without any test noticing.
from qwen3_tts.core.config import get_default_config, load_config

defaults = get_default_config()
assert defaults["language"] == "auto", defaults["language"]
assert defaults["default_clone_prompt"] is None
assert load_config().get("language") == "auto", "config.json drifted from the code default"

with open("install.sh") as f:
    install_text = f.read()
assert '"language": "auto"' in install_text, "install.sh template drifted"
assert '"language": "English"' not in install_text
print("config defaults reconciled")
```

- [ ] **Step 5: Add the sample-rate cell — the CUDA-side proof of Tasks 2 & 3**

```python
# The MLX runaway bug cannot reproduce on Colab (torch backend), but the
# WRITE-side guarantee must hold on every platform: a below-native-rate
# source must never land on disk at that rate.
import numpy as np

from qwen3_tts.core.engine.audio_processing import (
    DEFAULT_SAMPLE_RATE, ensure_min_sample_rate,
)

t = np.linspace(0, 2.0, 16000, endpoint=False)
low = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

out, sr, resampled = ensure_min_sample_rate(low, 8000)
assert resampled and sr == DEFAULT_SAMPLE_RATE, (sr, resampled)
assert abs(len(out) / sr - 2.0) < 0.05, "duration not preserved"

# 48 kHz must pass through untouched — never downsample.
high = (0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 1, 48000))).astype(np.float32)
_, hsr, hres = ensure_min_sample_rate(high, 48000)
assert hsr == 48000 and not hres

print(f"sample-rate guarantee holds: 8000 -> {sr} Hz, 48000 preserved")
```

- [ ] **Step 6: Add a real end-to-end generation cell**

```python
# Hollow-green guard: assert the ARTIFACT (real, non-silent audio at the
# expected rate), not an HTTP 200. Note audio_base64 is a WAV container,
# NOT float32 — decoding it as float32 halves every duration.
import base64, io, subprocess, time
import numpy as np, soundfile as sf
from qwen3_tts.server.client import TTSClient

server = subprocess.Popen(["tts", "server", "start", "--foreground"])
client = TTSClient()
for _ in range(120):
    try:
        if client.is_ready():
            break
    except Exception:
        pass
    time.sleep(5)

for mode, kwargs in [
    ("design", {"voice_description": "A calm, friendly voice"}),
    ("custom", {"speaker": "ryan"}),
]:
    client.load_model(mode)
    result = client.generate("Cross platform verification passed.", mode=mode, **kwargs)
    raw = base64.b64decode(result["audio_base64"])
    assert raw[:4] == b"RIFF", "expected a WAV container"
    wav, sr = sf.read(io.BytesIO(raw))
    assert sr == DEFAULT_SAMPLE_RATE, sr
    assert len(wav) / sr > 0.5, f"{mode}: suspiciously short"
    assert float(np.abs(wav).max()) > 0.01, f"{mode}: silent output"
    print(f"{mode}: {len(wav)/sr:.1f}s @ {sr} Hz, peak {np.abs(wav).max():.3f}, seed {result.get('seed')}")
```

- [ ] **Step 7: Add a CLI user-story sweep cell**

```python
# Server-independent CLI surface. `tts <text>` runs models IN-PROCESS by
# default (use_server is only true with the hidden --_server-mode flag), so
# these exercise the local path.
for cmd in [
    "tts --help", "tts list speakers", "tts list presets", "tts list aliases",
    "tts list prosody", "tts list backends", "tts list models",
    "tts config show", "tts config path", "tts voice list",
    "tts cache size", "tts doctor",
]:
    print(f"\n$ {cmd}")
    !{cmd}
```

- [ ] **Step 8: Update the summary markdown** to state what this notebook now proves, replacing the stale "2163+ tests / 6 batches" claim.

- [ ] **Step 9: Validate and commit**

```bash
conda run -n qwen3-tts-mlx python -c "import json; json.load(open('tests/validate_colab.ipynb')); print('valid json')"
conda run -n qwen3-tts-mlx python -m pytest tests/colab/ -q
git add tests/validate_colab.ipynb
git commit -m "test(colab): validate user stories and config reconciliation, not just batch counts"
```

---

## Task 8: macOS / M2 Pro full E2E pass

**Run in the FOREGROUND.** Background agents cannot prompt for Bash permissions, and this starts servers and loads models.

**Precondition:** `tts server restart` can silently no-op and leave the old server up with a stale PID file. **Verify the running server is the new code by querying its own schema**, not by trusting the PID file.

- [ ] **Step 1: Static gates + full non-E2E suite; compare to the Task 0 baseline**

```bash
conda run -n qwen3-tts-mlx ruff check qwen3_tts tests
conda run -n qwen3-tts-mlx mypy qwen3_tts/core qwen3_tts/server qwen3_tts/interface
conda run -n qwen3-tts-mlx bandit -r qwen3_tts -c pyproject.toml 2>&1 | tail -12
conda run -n qwen3-tts-mlx make check-config-docs
conda run -n qwen3-tts-mlx python -m pytest tests/ -q -m "not e2e" 2>&1 | tail -20
```

Expected: pass count ≥ baseline + the ~11 tests added here. **Any new failure blocks the PR.**

- [ ] **Step 2: All six batches**

```bash
conda run -n qwen3-tts-mlx python tests/run_batches.py 2>&1 | tail -30
```

A hung batch dumps every thread's traceback via `SIGABRT` before `SIGKILL` — read it rather than retrying blindly.

- [ ] **Step 3: Start a clean server with rate limiting disabled**

The live `/generate` limit (10/min) is shared across E2E modules and starves later suites into false 429 skips.

```bash
conda run -n qwen3-tts-mlx tts server stop || true
sleep 3
pkill -f "qwen3_tts.server.app" || true
rm -f .voice_server.lock
TTS_DISABLE_RATE_LIMITING=1 conda run -n qwen3-tts-mlx tts server start
sleep 20
curl -s http://127.0.0.1:5123/health | head -3
```

- [ ] **Step 4: Confirm the server runs the NEW code**

```bash
curl -s http://127.0.0.1:5123/openapi.json | conda run -n qwen3-tts-mlx python -c "
import json,sys
spec = json.load(sys.stdin)
print('max_new_tokens schema:',
      spec['components']['schemas']['GenerateRequest']['properties']['max_new_tokens'])"
tail -20 .voice_server.log
```

If this reflects stale code the restart no-opped — kill by PID and start again before continuing.

- [ ] **Step 5: The full E2E suite (not just batch 6)**

`run_batches.py --batch 6` runs **only** `test_e2e_playwright`. The other nine `test_e2e_*.py` files are in no batch and must be invoked by marker.

```bash
conda run -n qwen3-tts-mlx python -m pytest tests/ -m e2e -q --tb=short 2>&1 | tail -40
```

Record every skip with its reason. **A skip is not a pass.**

- [ ] **Step 6: CLI sweep — read-only commands**

```bash
for c in "--help" "list speakers" "list presets" "list aliases" "list prosody" \
         "list backends" "list models" "config show" "config path" "voice list" \
         "cache list" "cache size" "history 5" "doctor" "server status" "stats"; do
  echo "=== tts $c ==="
  conda run -n qwen3-tts-mlx tts $c 2>&1 | tail -6
done
```

- [ ] **Step 7: CLI generation sweep — real audio, all modes, both routing paths**

`tts "text"` runs models **in-process** by default; `--_server-mode` is the hidden flag that routes through the daemon. Cover both.

```bash
OUT=$(mktemp -d); echo "$OUT"
conda run -n qwen3-tts-mlx tts "Local clone path check." -m clone -p lt1_24k -o "$OUT/clone_local"
conda run -n qwen3-tts-mlx tts "Server clone path check." -m clone -p lt1_24k --_server-mode -o "$OUT/clone_server"
conda run -n qwen3-tts-mlx tts "Design mode check." -m design -d "a warm friendly narrator" -o "$OUT/design"
conda run -n qwen3-tts-mlx tts "Custom speaker check." -m custom -s ryan -o "$OUT/custom"
conda run -n qwen3-tts-mlx tts "Seeded run." --seed 12345 -m custom -s ryan -o "$OUT/seeded"
conda run -n qwen3-tts-mlx tts "Dry run." --dry-run
```

- [ ] **Step 8: Assert the artifacts, not the exit codes**

```bash
conda run -n qwen3-tts-mlx python -c "
import glob, numpy as np, soundfile as sf
files = sorted(glob.glob('$OUT/*.wav'))
assert files, 'no audio produced'
for f in files:
    wav, sr = sf.read(f)
    peak = float(np.abs(wav).max()); dur = len(wav) / sr
    assert dur > 0.5, f'{f}: {dur:.2f}s — too short'
    assert peak > 0.01, f'{f}: silent (peak {peak})'
    print(f'{dur:6.2f}s  {sr} Hz  peak {peak:.3f}  {f}')"
```

- [ ] **Step 9: Batch / SRT / dialogue**

```bash
cat > "$OUT/batch.json" <<'EOF'
[{"text": "First batch line.", "mode": "custom", "speaker": "ryan"},
 {"text": "Second batch line.", "mode": "custom", "speaker": "ryan"}]
EOF
conda run -n qwen3-tts-mlx tts batch "$OUT/batch.json" -o "$OUT/"

cat > "$OUT/subs.srt" <<'EOF'
1
00:00:00,000 --> 00:00:02,000
Subtitle line one.

2
00:00:02,000 --> 00:00:04,000
Subtitle line two.
EOF
conda run -n qwen3-tts-mlx tts srt "$OUT/subs.srt"

cat > "$OUT/dialogue.json" <<'EOF'
{"lines": [{"speaker": "ryan", "text": "Hello there."},
           {"speaker": "ryan", "text": "General Kenobi."}]}
EOF
conda run -n qwen3-tts-mlx tts dialogue "$OUT/dialogue.json" --save-individual
```

If a fixture schema is rejected, read `qwen3_tts/interface/cli/{batch,dialogue}.py` and correct the fixture — that is a fixture bug, not a product bug.

- [ ] **Step 10: Voice lifecycle + the F2/F3 regression proof**

```bash
conda run -n qwen3-tts-mlx python -c "
import numpy as np, soundfile as sf
t = np.linspace(0, 12, 12*8000, endpoint=False)
sf.write('$OUT/ref8k.wav', (0.3*np.sin(2*np.pi*180*t)).astype('float32'), 8000)"

conda run -n qwen3-tts-mlx tts voice create "$OUT/ref8k.wav" -n e2e_rate_check --no-transcript

conda run -n qwen3-tts-mlx python -c "
import soundfile as sf
from qwen3_tts.core.config import VOICE_PROMPTS_DIR
i = sf.info(f'{VOICE_PROMPTS_DIR}/e2e_rate_check.wav')
print('on-disk rate:', i.samplerate)
assert i.samplerate >= 24000, f'REGRESSION: wrote {i.samplerate} Hz'"

conda run -n qwen3-tts-mlx tts voice list
conda run -n qwen3-tts-mlx tts voice info e2e_rate_check
conda run -n qwen3-tts-mlx tts voice rename e2e_rate_check e2e_renamed
conda run -n qwen3-tts-mlx tts voice delete e2e_renamed
```

- [ ] **Step 11: Prove the legacy-prompt warning fires on the real 8 kHz prompt**

```bash
conda run -n qwen3-tts-mlx tts "Legacy prompt generation." -m clone -p lt1 -o "$OUT/legacy" 2>&1 \
  | grep -i "8000\|below the model" || echo "NO WARNING — Task 3 regressed"
```

- [ ] **Step 12: Gradio UI browser smoke**

```bash
conda run -n qwen3-tts-mlx tts ui --no-browser --port 7866 &
sleep 25
```

Drive it with the Chrome MCP tools. **Never attach a `select` listener to a `gr.Tab`**, and remember tab panels are lazy — inactive tabpanels are not in the DOM, so a native Playwright `.click()` times out; dispatch the click via JS. Verify: Clone / Design / Custom render and generate; Create Voice accepts an upload; Manage Voices and Manage Models render their Dataframes after a full tab sweep.

- [ ] **Step 13: Server lifecycle**

```bash
conda run -n qwen3-tts-mlx tts server restart && sleep 20
conda run -n qwen3-tts-mlx tts server status
conda run -n qwen3-tts-mlx tts server log | tail -20
conda run -n qwen3-tts-mlx tts server stop
```

- [ ] **Step 14: Write the report**

Create `docs/reviews/cross-platform-e2e-2026-08-17.md`: environment (macOS 27, M2 Pro, `qwen3-tts-mlx`, gradio 6.20.0), each phase's exact command and result, **every skip with its reason**, all artifact assertions (duration/rate/peak per file), and a findings table. Report failures plainly, with output.

---

## Task 9: Linux verification via Docker

Docker Desktop is running (engine 29.7.2, Compose v5.4.0). Three signals: the hardened arm64 container, an amd64 arch-parity check, and GitHub Actions.

> **Platform boundary — state this in the report.** A Linux container shares the host kernel, so this proves **Linux** behavior only. It does **not** validate macOS, Windows, or CUDA. The Docker VM has **6.2 GB** RAM, which is below what 1.7B torch inference needs reliably, so in-container generation is 0.6B-only and best-effort.

### 9a — Native arm64 container (primary Linux signal)

- [ ] **Step 1: Run the harness**

```bash
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
./tests/validate_docker.sh 2>&1 | tail -60
```

This validates the compose model, builds, then runs gates → full non-E2E suite → batch runner, all on `network_mode: none` with a read-only rootfs and a non-root uid. First build downloads everything (~10–15 min; the image cache is currently empty).

- [ ] **Step 2: Confirm F6 is actually fixed in the container**

The whole point of Task 4 — prove the module now runs instead of erroring.

```bash
COMPOSE="docker compose -p qwen3-tts-test -f docker-compose.test.yml"
$COMPOSE run --rm -T suite python -m pytest tests/test_install_script.py -v 2>&1 | tail -20
```

Expected: all tests **run and pass** (previously `FileNotFoundError`).

- [ ] **Step 3: Config reconciliation inside Linux**

```bash
$COMPOSE run --rm -T suite python -c "
from qwen3_tts.core.config import get_default_config, load_config
assert get_default_config()['language'] == 'auto'
assert load_config().get('language') == 'auto'
print('linux config reconciled')"
```

- [ ] **Step 4: Sample-rate guarantee inside Linux**

Proves the write-side fix is not macOS-specific (librosa/soundfile differ per platform).

```bash
$COMPOSE run --rm -T suite python -c "
import numpy as np
from qwen3_tts.core.engine.audio_processing import DEFAULT_SAMPLE_RATE, ensure_min_sample_rate
t = np.linspace(0, 2.0, 16000, endpoint=False)
out, sr, resampled = ensure_min_sample_rate((0.3*np.sin(2*np.pi*220*t)).astype('float32'), 8000)
assert resampled and sr == DEFAULT_SAMPLE_RATE, (sr, resampled)
assert abs(len(out)/sr - 2.0) < 0.05
print('linux sample-rate guarantee holds:', sr)"
```

- [ ] **Step 5: Confirm the isolation contract actually held**

Do not take the compose flags on trust — verify the container really was read-only, non-root, and network-isolated.

```bash
$COMPOSE run --rm -T suite bash -lc '
  id -u
  test ! -w /app && echo "rootfs read-only: OK" || echo "rootfs WRITABLE: FAIL"
  test -w /tmp && echo "tmpfs writable: OK"
  (getent hosts pypi.org >/dev/null 2>&1 && echo "network REACHABLE: FAIL") || echo "network isolated: OK"'
```

Expected: uid `1000`, rootfs read-only OK, tmpfs writable OK, network isolated OK.

- [ ] **Step 6: Prove the source checkout was never mutated**

```bash
git status --short
```

Expected: no unexpected new/modified files. The image bakes the source in, so a run cannot touch this checkout — this confirms it.

### 9b — amd64 arch-parity check

The Docker VM is **aarch64**, but CI runs **amd64**. Emulating the full torch suite under QEMU is impractically slow, so scope this to import-and-collect — which is where arch-specific wheel problems actually surface.

- [ ] **Step 7: Build and smoke-test an amd64 image**

```bash
docker buildx build --platform linux/amd64 -f Dockerfile.test -t qwen3-tts:test-amd64 --load .
docker run --rm --platform linux/amd64 qwen3-tts:test-amd64 \
  python -c "import qwen3_tts, gradio, fastapi, soundfile, librosa; print('amd64 imports OK')"
docker run --rm --platform linux/amd64 qwen3-tts:test-amd64 \
  python -m pytest tests/ -m "not e2e" --collect-only -q 2>&1 | tail -5
```

If QEMU emulation is unavailable or the build exceeds ~20 minutes, **skip this and say so in the report** — CI covers amd64 authoritatively. Do not let an optional check block the PR.

- [ ] **Step 8: Clean up**

```bash
docker compose -p qwen3-tts-test -f docker-compose.test.yml down --remove-orphans
docker image rm qwen3-tts:test qwen3-tts:test-amd64 qwen3-tts:baseline 2>/dev/null || true
```

### 9c — GitHub Actions

- [ ] **Step 9: Push and confirm CI actually ran**

The push filter is a **closed allowlist**: only `main`, `develop`, `feat/*`, `fix/*`, `feature/*` trigger CI. A `docs/*`, `chore/*`, or `refactor/*` branch gets **zero** runs — and a *missing* check is silent, not red. Task 10 uses a `fix/*` branch; verify explicitly.

```bash
gh run list --branch fix/cross-platform-user-story-reconciliation --limit 10
```

Empty list → open the PR to force the matrix, then re-check.

- [ ] **Step 10: Wait for green and read every failure**

```bash
gh pr checks --watch
```

A red `GitHub Advanced Security` check with `CAPIError: 400` never analysed the diff — infra, not a finding. A red `docker-lint` from the hadolint download (exit 56) is the same class; `gh run rerun <id> --failed` clears it. **Never assume a red CodeQL is infra without reading the annotation** — a genuine "job missing `permissions:`" finding looks similar.

### 9d — CUDA / Colab

- [ ] **Step 11: Hand off the CUDA half**

No CUDA hardware is available locally, and a container cannot supply it. Deliver `tests/validate_colab.ipynb` with these instructions and have the user paste back any failure:

```
1. Open https://colab.research.google.com → File → Open notebook → GitHub
2. Enter: eepstein201/Qwen3-TTS-Advanced-EME
3. Select branch: fix/cross-platform-user-story-reconciliation
4. Choose: tests/validate_colab.ipynb
5. Runtime → Change runtime type → T4 GPU (or L4 on Pro)
6. Runtime → Run all
7. Paste back the output of any cell that errors.
```

Mark the CUDA row **UNVERIFIED — pending user Colab run** until it returns. Do not report it as passing.

---

## Task S: Santa adversarial review gate (cross-family, via Antigravity CLI)

**Run after Task 9, before Task 10.** A single agent reviewing its own output shares the biases that produced it. This gate uses two reviewers from **different model families than the author**, so a Claude blind spot is not replicated.

**Verified harness (all mechanics smoke-tested 2026-08-17 — do not re-derive):**

| Fact | Value | Consequence |
|---|---|---|
| Binary | `/Users/ericepstein/.local/bin/agy` v1.1.13 | Installed and working |
| Reviewer B | `gemini-3.1-pro-high` | Google family |
| Reviewer C | `gpt-oss-120b-medium` | OpenAI-lineage family |
| **Do not use** | `claude-sonnet-4-6`, `claude-opus-4-6-thinking` | Also offered by `agy`, but same family as the author — defeats the entire point |
| `--effort` + tiered model name | **No longer conflicts** in 1.1.13 | Omit it anyway; the tier is already in the model name |
| `--json-schema` | **Echoes the schema, does not enforce it** — `response` stays free prose | Do NOT rely on it. Ask for JSON in the prompt and parse `response` |
| Headless tool use | **Auto-denied** — `agy -p` cannot read repo files | **Inline the payload.** Never pass `--dangerously-skip-permissions` (global rule) |
| Output fencing | Models may wrap JSON in ```` ```json ```` despite instructions | Strip fences before parsing |
| Statefulness | Each `-p` invocation is a fresh conversation unless `--continue` | Fresh reviewers per round come free; never pass `--continue` here |

Inlining is not a workaround — it is the correct design. It guarantees both reviewers see **byte-identical input**, which is a core Santa invariant.

- [ ] **Step 1: Build the review payload**

```bash
cd /Users/ericepstein/Qwen3-TTS_UserFiles
{
  echo "You are an independent adversarial code reviewer. You have NOT seen any other review."
  echo "Your job is to FIND PROBLEMS, not to approve. Default to FAIL if uncertain."
  echo ""
  echo "CONTEXT: MLX TTS engine. The MLX clone path passes ref_audio=<path> to mlx-audio,"
  echo "which opens the file itself, so the ON-DISK sample rate is what the model sees."
  echo "A reference .wav below 24000 Hz makes generation fail to emit EOS and run to the"
  echo "token cap (measured 3/3, up to 47.8x expected tokens). install.sh writes config.json"
  echo "from a hand-maintained heredoc rather than calling get_default_config()."
  echo ""
  echo "RUBRIC — evaluate EACH criterion, PASS or FAIL with a specific cited problem:"
  echo "1. Correctness: does the change do what its comments and tests claim?"
  echo "2. Silent failure: any path that logs-and-continues where the failure is user-visible harm?"
  echo "3. Argument mutation: does any helper mutate its input instead of returning a new object?"
  echo "4. Completeness: are there BYPASS paths that reach the same bug via another call site?"
  echo "5. Lazy imports: torch/mlx/transformers must never be imported at module scope."
  echo "6. Test quality: do the tests assert the real artifact, or a proxy that could pass hollow?"
  echo ""
  echo "DIFF UNDER REVIEW:"
  echo '```diff'
  git diff origin/main...HEAD
  echo '```'
  echo ""
  echo 'Respond with ONLY a JSON object, no prose, no markdown fence:'
  echo '{"verdict":"PASS"|"FAIL","critical_issues":[string],"suggestions":[string]}'
} > /tmp/santa_payload.txt
wc -c /tmp/santa_payload.txt
```

If the payload exceeds ~200 KB, split by subsystem (engine / UI / docker+colab) and gate each independently rather than truncating — a truncated diff produces a confident review of code nobody read.

- [ ] **Step 2: Run both reviewers independently**

Run as two separate invocations so neither can see the other's output.

```bash
timeout 300 agy -p "$(cat /tmp/santa_payload.txt)" --model gemini-3.1-pro-high \
  > /tmp/santa_review_b.txt 2>&1
timeout 300 agy -p "$(cat /tmp/santa_payload.txt)" --model gpt-oss-120b-medium \
  > /tmp/santa_review_c.txt 2>&1
cat /tmp/santa_review_b.txt
echo "-----"
cat /tmp/santa_review_c.txt
```

- [ ] **Step 3: Apply the gate**

```bash
conda run -n qwen3-tts-mlx python - <<'PY'
import json, re, sys

def load(path):
    raw = open(path).read().strip()
    # Models wrap JSON in a fence despite instructions — strip it.
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    if m:
        raw = m.group(1).strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        sys.exit(f"{path}: no JSON object found — reviewer output unusable:\n{raw[:500]}")
    return json.loads(m.group(0))

b, c = load("/tmp/santa_review_b.txt"), load("/tmp/santa_review_c.txt")
issues = list(dict.fromkeys(b.get("critical_issues", []) + c.get("critical_issues", [])))
both = b["verdict"] == "PASS" and c["verdict"] == "PASS"

print(f"Reviewer B (gemini-3.1-pro-high): {b['verdict']}")
print(f"Reviewer C (gpt-oss-120b-medium): {c['verdict']}")
print(f"\nVERDICT: {'NICE — ship it' if both else 'NAUGHTY — fix required'}")
for i, issue in enumerate(issues, 1):
    print(f"  {i}. {issue}")
sys.exit(0 if both else 1)
PY
```

**Both must PASS.** If only one reviewer catches an issue, the issue is real — the other reviewer's blind spot is exactly what this gate exists to expose. That is not theoretical here: reviewer C alone found F8 (stereo passthrough), and it is a genuine defect.

- [ ] **Step 4: Fix and re-run (max 3 rounds)**

Fix **only** the flagged critical issues — no refactoring, no unrequested changes. Then rebuild the payload from the new diff and re-run **both** reviewers. Each `agy -p` is a fresh conversation, so there is no anchoring from the previous round.

After 3 rounds without convergence, stop and escalate to the user with the outstanding issues rather than looping. Reviewers that keep finding new issues after fixes usually signal a design problem, not a code problem.

- [ ] **Step 5: Record the result in the report**

Append to `docs/reviews/cross-platform-e2e-2026-08-17.md`: both reviewers' verdicts per round, every issue with whether it was fixed or consciously declined and why, and the round count. Note which issues were caught by only one reviewer — low agreement means the rubric needs tightening next time.

> **Prior-round evidence (already run during planning, on the uncommitted `audio_processing.py` + `create_voice.py` diff):** Round 0 verdict **NAUGHTY** — both reviewers FAIL. Both independently flagged the librosa-unavailable silent-poison path (**F7**); reviewer B additionally flagged the missing core-engine validation (addressed by Task 3); reviewer C additionally flagged stereo passthrough (**F8**) and unhandled `librosa.resample` exceptions. Task 3b exists because of this round.

---

## Task 10: Ship

- [ ] **Step 1: Branch (CI-allowlisted prefix)**

```bash
git status
git checkout -b fix/cross-platform-user-story-reconciliation
```

- [ ] **Step 2: Review what is staged before committing**

The working tree holds untracked `.claude/` hookify files, `.venv-310/`, `.venv-lock/`, `.claude/settings.local.json.bak-doctor`, and `.voice_server.lock`. **None belong in the commit** — `settings.local.json` files have previously contained secrets in this repo.

```bash
git status --short
git diff --cached --stat
```

Add only: `install.sh`, `config.json`, `Dockerfile.test`, `docker-compose.test.yml`, `qwen3_tts/core/engine/{audio_processing,inference,voice_prompt}.py`, `qwen3_tts/tools/create_voice.py`, `qwen3_tts/interface/ui/voice_management.py`, `tests/run_batches.py`, `tests/validate_docker.sh`, `tests/test_mlx_generate_kwargs.py`, `tests/test_voice_prompt_sample_rate.py`, `tests/test_install_script.py`, `tests/test_docker_config.py`, `CLAUDE.md`, `docs/CONFIG.md`, `docs/reviews/cross-platform-e2e-2026-08-17.md`, `docs/superpowers/plans/2026-08-17-cross-platform-user-story-verification.md`, `colab_notebook.ipynb`, `tests/validate_colab.ipynb`.

- [ ] **Step 3: Pre-push local gates**

```bash
conda run -n qwen3-tts-mlx ruff check qwen3_tts tests
wc -l CLAUDE.md
conda run -n qwen3-tts-mlx python -m pytest tests/ -q -m "not e2e" 2>&1 | tail -5
```

- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin fix/cross-platform-user-story-reconciliation
gh pr create --title "fix: reconcile config defaults, voice-prompt sample rate, and Colab across platforms" --body "$(cat <<'EOF'
## Summary

Cross-platform user-story verification (macOS/M2 Pro MLX + containerized Linux + CI) surfaced four defects where a merged fix never reached the path users actually take.

- **`install.sh` reverted PR #190 on every fresh install.** It writes `config.json` from a hand-maintained heredoc instead of `get_default_config()`, and still emitted `"language": "English"`. The repo's tracked `config.json` had the same stale value. It also seeded `"default_clone_prompt": "default_clone.pt"` — a file that ships with nothing. `tests/test_install_script.py` now pins both against the Python defaults.
- **The Gradio Create Voice tab byte-copied uploads with no resampling.** MLX hands mlx-audio `ref_audio=<path>` and mlx-audio opens the file itself, so `load_audio_for_cloning()` never applies and the on-disk rate is what the model sees. A below-24 kHz reference makes clone generation fail to emit EOS and run to the token cap (measured 3/3, up to 47.8× the expected token count). The tab now resamples on write, and byte-copies unchanged when the rate is already adequate.
- **Legacy low-rate prompts stayed broken and silent.** `load_voice_prompt_mlx` now warns, naming both rates, and states that `tts voice rebuild` will not repair them — rebuild regenerates the `.pt` and leaves the `.wav` alone, while MLX reads the `.wav`.
- **`tests/test_install_script.py` had been erroring in every containerized run.** `Dockerfile.test` never copied `install.sh`, so the module hit `FileNotFoundError` on Linux while passing on the host. Fixed at the image, with a drift test covering the whole class of repo-root file a test opens.

Also ships the previously uncommitted `ensure_min_sample_rate()` write-time fix and the `_warn_if_cap_reached()` truncation warning, so a generation that stops at the cap instead of on EOS is no longer silent end to end.

## Docker verification

New `docker-compose.test.yml` (separate from the drift-guarded, GPU-shaped `docker-compose.yml`) defines hardened `suite` / `batches` / `gates` / `shell` services: `network_mode: none`, `read_only: true`, `cap_drop: ALL`, `no-new-privileges`, `pids_limit`, non-root uid 1000, and declared tmpfs mounts. The image bakes the source in, so a run cannot mutate the checkout. `tests/validate_docker.sh` now drives Compose and validates the model before building.

## Colab reconciliation

- `colab_notebook.ipynb`: adds a `LANGUAGE` form field (default `auto`) and persists it — the notebook previously wrote no language key at all and so inherited the stale `English`.
- `tests/validate_colab.ipynb`: kept deliberately separate from the user-facing notebook. Now asserts config reconciliation, the sample-rate guarantee, real per-mode generation, and a CLI sweep instead of only shelling out to the batch runner. Runs the full `pytest -m "not e2e"` superset, since `run_batches.py` is an explicit allowlist that silently skips unregistered modules.

## Test plan

- [x] ruff / mypy / bandit / check-config-docs
- [x] Full non-E2E suite (superset of the batch runner)
- [x] All six batches
- [x] `pytest -m e2e` against a live server with `TTS_DISABLE_RATE_LIMITING=1` — batch 6 alone covers only `test_e2e_playwright`
- [x] CLI sweep: all read-only commands; real generation in clone/design/custom, local and `--_server-mode`; batch/SRT/dialogue; full voice lifecycle
- [x] Regression proof: an 8 kHz reference through `tts voice create` lands on disk at ≥24 kHz; generating against the legacy 8 kHz `lt1` prompt emits the warning
- [x] Gradio UI browser smoke across all six tabs
- [x] Linux arm64: hardened container — gates, full suite, batch runner, isolation contract verified
- [x] Linux amd64: import + collect-only arch-parity check
- [ ] **CUDA/Colab — pending: run `tests/validate_colab.ipynb` on a GPU runtime**

A Linux container shares the host kernel and proves Linux behavior only; it does not validate macOS, Windows, or CUDA.

Full report: `docs/reviews/cross-platform-e2e-2026-08-17.md`
EOF
)"
```

- [ ] **Step 5: Verify CI ran and is green before handing over a merge command**

```bash
gh run list --branch fix/cross-platform-user-story-reconciliation --limit 10
gh pr checks --watch
```

Only once every check is green, hand the user the exact merge command. If anything is pending, say so instead.

---

## Verification Summary

| Platform | Method | Proves | Status |
|----------|--------|--------|--------|
| macOS M2 Pro (MLX) | Task 8 — full suite + live E2E + CLI sweep + UI + real audio | All user stories, MLX backend, the runaway-bug fix | Task 8 |
| Linux arm64 (CPU) | Task 9a — hardened Compose: gates, suite, batches, isolation contract | Suite + config + sample-rate guarantee on Linux | Task 9a |
| Linux amd64 | Task 9b — buildx/QEMU import + collect-only | Arch parity with CI; optional, skippable | Task 9b |
| Linux CI matrix | Task 9c — GitHub Actions on a `fix/*` branch | The authoritative Linux signal | Task 9c |
| Linux CUDA / Colab | Task 9d — `tests/validate_colab.ipynb`, run by the user | torch+CUDA generation, Colab install path | **User-run; UNVERIFIED until returned** |
| Cross-model review | Task S — `agy` dual adversarial gate | Defects a Claude self-review would share blind spots on | Task S — **already caught F7 + F8** |

**Definition of done:** Tasks 1–9 green, **Task S verdict NICE (both reviewers PASS)**, PR open with CI green, and the Colab row either verified by the user's run or explicitly reported as outstanding. Never report the CUDA row as passing on the strength of the container run.

## Self-Review (writing-plans checklist)

1. **Coverage:** F1→T1, F1b→T1, F2→T2, F3→T3, F7→T3b, F8→T3b, F6→T4, F4→T5, F5→T6; macOS→T8, Linux→T9, Colab→T6/T7/T9d, adversarial review→TS, ship→T10. No gap.
2. **Placeholders:** none — every step carries runnable commands or complete code.
3. **Type consistency:** `ensure_min_sample_rate` returns `(audio, sr, was_resampled)` and now **raises `RuntimeError`** on an undeliverable guarantee; that contract change is reflected in Task 2's caller (`except RuntimeError → gr.Error`), Task 3b's rewrite, and Task 7's Colab cell (which only exercises the succeeding path). `DEFAULT_SAMPLE_RATE` is the single rate constant throughout.

**Known ordering constraint:** Task 3b changes the contract Task 2 consumes. If run in parallel, Task 2 must land its `except RuntimeError` branch from Task 3b Step 5 — or run 2 and 3b sequentially in one worker.
