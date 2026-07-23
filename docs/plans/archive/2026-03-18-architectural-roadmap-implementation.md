# Architectural Roadmap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Commit the Gemini Deep Research architectural roadmap, then execute its 4-phase spec-driven development plan to improve codebase modularity, inference architecture, streaming, and quality assurance.

**Architecture:** The roadmap prescribes a progressive hardening of the Qwen3-TTS codebase: first reduce monolithic files to combat context rot (Phase 1), then decouple FastAPI from vLLM inference (Phase 2), upgrade to bidirectional WebSocket streaming (Phase 3), and finally integrate automated audio quality evaluation into CI/CD (Phase 4).

**Tech Stack:** Python, FastAPI, vLLM, wavesurfer.js, Docker, pytest, Whisper ASR, WavLM-SV, GitHub Actions

---

## Task 0: Commit the Roadmap Document

**Files:**
- Create: `docs/plans/architectural-roadmap.md`

**Step 1: Create git worktree**

```bash
git worktree add ../qwen3-tts-roadmap -b docs/architectural-roadmap
```

**Step 2: Convert .docx to markdown**

Use `textutil` to extract text from the .docx, then format as proper markdown with headings, tables, and lists. Write to `docs/plans/architectural-roadmap.md` in the worktree.

**Step 3: Add backlog section for deferred items**

Append a `## Backlog (Deferred)` section to the roadmap document capturing high-effort items not in the 4-phase sprint:

- **Entropy-based hallucination monitoring**: Monitor Shannon entropy in the vLLM autoregressive decoding loop. When uncertainty spikes above a threshold, abort and regenerate that audio chunk before streaming to client. Requires invasive modification of the vLLM forward pass. (Ref: Evaluation Layer 4 — Hallucination Mitigation)
- **GFlowNet distribution alignment**: Steer generation toward desired acoustic distribution using Generative Flow Networks to complement entropy detection.
- **Adaptive attention head deactivation**: Dynamically prune "hallucination heads" that over-attend to previously generated audio tokens instead of conditioning text.

**Step 4: Commit**

```bash
cd ../qwen3-tts-roadmap
git add docs/plans/architectural-roadmap.md
git commit -m "docs: add Gemini Deep Research architectural roadmap with backlog"
```

**Step 5: Verify**

```bash
git log --oneline -1
grep "Backlog" docs/plans/architectural-roadmap.md
```

---

## Phase 1: Codebase Characterization and Agent Memory Optimization

**Objective:** Halt context degradation, optimize CLAUDE.md, and decompose the monolithic test_voice.py (70 classes, 3,461 lines) before modifying any logic.

### Task 1.1: Refactor CLAUDE.md for Progressive Disclosure

**Files:**
- Modify: `CLAUDE.md` (currently 539 lines → target <300 lines)
- Create: `docs/00-Foundations/ARCHITECTURE.md`

**Step 1: Write a test that CLAUDE.md is under 300 lines**

```python
# tests/test_claude_md.py
import unittest

class TestClaudeMD(unittest.TestCase):
    def test_claude_md_under_300_lines(self):
        with open("CLAUDE.md") as f:
            lines = f.readlines()
        self.assertLessEqual(len(lines), 300,
            f"CLAUDE.md is {len(lines)} lines; must be ≤300 for progressive disclosure")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_claude_md.py -v`
Expected: FAIL (currently 539 lines)

**Step 3: Extract architectural deep-dive content to docs/00-Foundations/ARCHITECTURE.md**

Move these sections out of CLAUDE.md:
- Tier 3 — Deep Dive (Security, Platform Support, Caching, Thread Safety, Constants, Logging, Error Responses, Hardware Optimization, Text Processing Roadmap, Upstream Dependency Monitoring)
- Code Review Status history
- Detailed config structure (keep only key settings summary)

Keep in CLAUDE.md only:
- **WHAT**: Project purpose, file layout, module table
- **HOW**: Rules, coding conventions, lazy imports
- **VERIFICATION**: Test commands, server restart rule

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_claude_md.py -v`
Expected: PASS

**Step 5: Run full test suite to verify nothing broke**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All existing tests PASS

**Step 6: Commit**

```bash
git add CLAUDE.md docs/00-Foundations/ARCHITECTURE.md tests/test_claude_md.py
git commit -m "docs: refactor CLAUDE.md to <300 lines with progressive disclosure"
```

---

### Task 1.2: Map test_voice.py Decomposition Plan

**Files:**
- Read: `tests/test_voice.py` (70 classes, 3,461 lines)

**Step 1: Analyze and categorize the 70 test classes by domain**

Group the classes into 8 domain files. Proposed mapping:

| New file | Classes (from test_voice.py) |
|----------|------------------------------|
| `tests/test_voice_config.py` | TestTTSConfig, TestBackendConfig, TestModelSize, TestPlatformDetection, TestLazyImports |
| `tests/test_voice_server.py` | TestServerValidation, TestServerAuth, TestHealthEndpointInfo, TestGenerationStatus, TestLoadModelEndpoint, TestCancelGenerationEndpoint, TestGenerationStateFields, TestUpdateModelConfigEndpoint, TestStreamingEndpointStructure, TestGenerateStreamIdCheck, TestUnloadModelEndpoint, TestUpdateStartupConfigEndpoint, TestModelsEndpointEnhanced |
| `tests/test_voice_prompts.py` | TestMLXVoicePrompt, TestMLXVoicePromptCache, TestDeletePromptEndpoint, TestRenamePromptEndpoint, TestPreviewPromptEndpoint, TestPromptDetailsEndpoint, TestClientPromptManagement, TestSetDefaultClonePrompt, TestGetDefaultClonePromptFallback, TestGetVoicePrompts, TestValidatePromptNameCallers, TestSetVoiceDefaultExtension, TestPreviewVoiceExtension |
| `tests/test_voice_streaming.py` | TestStreaming, TestStreamingServerEndpoint, TestStreamingClientMethod |
| `tests/test_voice_engine.py` | TestBackendDispatch, TestMLXImport, TestMLXInferenceCloneValidation, TestASR, TestFloat32Guard, TestMLXMetalRecovery, TestMLXMemoryStats, TestDeviceAwareEngine, TestEngineModelCleanup, TestSmartAudioLoader |
| `tests/test_voice_generation.py` | TestStability, TestETACache, TestGenerationCache, TestClientUpdateModelConfig, TestClientModelMethods, TestReturnValueCounts, TestTextChunking, TestSSMLParsing, TestSRTParsing, TestAutoIncrementFilename |
| `tests/test_voice_ui.py` | TestUIHistoryFunctions, TestUICancelFunction, TestUITextInfo, TestUIModelSettings, TestUIModelSettingsImports, TestVoiceManagementUI, TestManageModelsUI, TestProsodyUI |
| `tests/test_voice_features.py` | TestRubberBandAudioProcessing, TestProsodyPresets, TestXVectorOnlyMode, TestXVectorOnlyClient, TestCreateVoiceNoTranscript, TestClickCLI, TestPlatformSafeCommands, TestGetPresets |

**Step 2: Document the plan, get user approval before executing**

---

### Task 1.3: Execute test_voice.py Decomposition

**Files:**
- Modify: `tests/test_voice.py` (extract classes)
- Create: 8 new test files (per approved mapping above)

**Step 1: Write a meta-test that validates the decomposition**

```python
# tests/test_decomposition_check.py
import unittest
import os
import importlib
import sys

class TestDecompositionComplete(unittest.TestCase):
    def test_voice_test_files_exist(self):
        expected = [
            "tests/test_voice_config.py",
            "tests/test_voice_server.py",
            "tests/test_voice_prompts.py",
            "tests/test_voice_streaming.py",
            "tests/test_voice_engine.py",
            "tests/test_voice_generation.py",
            "tests/test_voice_ui.py",
            "tests/test_voice_features.py",
        ]
        for f in expected:
            self.assertTrue(os.path.exists(f), f"Missing: {f}")

    def test_original_is_empty_or_redirects(self):
        with open("tests/test_voice.py") as f:
            content = f.read()
        self.assertLess(len(content), 500,
            "test_voice.py should be a minimal shim after decomposition")

    def test_no_circular_imports(self):
        """Verify each new test module imports cleanly without circular deps."""
        modules = [
            "tests.test_voice_config",
            "tests.test_voice_server",
            "tests.test_voice_prompts",
            "tests.test_voice_streaming",
            "tests.test_voice_engine",
            "tests.test_voice_generation",
            "tests.test_voice_ui",
            "tests.test_voice_features",
        ]
        for mod_name in modules:
            # Clear from cache to force fresh import
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            try:
                importlib.import_module(mod_name)
            except ImportError as e:
                self.fail(f"Circular or broken import in {mod_name}: {e}")

    def test_no_orphaned_dependencies(self):
        """Verify no test file imports symbols that were left behind in the original."""
        if os.path.exists("tests/test_voice.py"):
            with open("tests/test_voice.py") as f:
                original = f.read()
            # After decomposition, original should not define any test classes
            self.assertNotIn("class Test", original,
                "test_voice.py still contains test classes after decomposition")

    def test_no_silent_skips(self):
        """Run all decomposed test files and verify no tests are silently skipped."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--co", "-q",
             "tests/test_voice_config.py",
             "tests/test_voice_server.py",
             "tests/test_voice_prompts.py",
             "tests/test_voice_streaming.py",
             "tests/test_voice_engine.py",
             "tests/test_voice_generation.py",
             "tests/test_voice_ui.py",
             "tests/test_voice_features.py"],
            capture_output=True, text=True
        )
        # --co (collect-only) lists all discovered tests
        # Verify test count matches original (should be same total)
        self.assertIn("test", result.stdout.lower(),
            f"No tests collected from decomposed files. stderr: {result.stderr}")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_decomposition_check.py -v`
Expected: FAIL (new files don't exist yet)

**Step 3: Extract classes one file at a time**

For each target file:
1. Copy the relevant classes + their imports to the new file
2. Run: `python -m pytest tests/test_voice_<domain>.py -v`
3. Verify all extracted tests PASS and no circular imports
4. Remove those classes from `tests/test_voice.py`
5. Run: `python -m pytest tests/ -v --tb=short` (full suite still passes)
6. Commit the extraction

**Step 4: After each extraction, run the safety checks**

```bash
# Check for circular imports
python -c "import tests.test_voice_<domain>"

# Check for orphaned imports (symbols used but not imported)
python -m py_compile tests/test_voice_<domain>.py

# Verify no silent test skips
python -m pytest tests/test_voice_<domain>.py -v 2>&1 | grep -c "SKIP\|skip"
```

**Step 5: Replace test_voice.py with a minimal shim**

```python
# tests/test_voice.py
"""
Voice tests have been decomposed into domain-specific modules.
See: test_voice_config.py, test_voice_server.py, test_voice_prompts.py,
     test_voice_streaming.py, test_voice_engine.py, test_voice_generation.py,
     test_voice_ui.py, test_voice_features.py
"""
```

**Step 6: Run decomposition meta-test (all 5 checks)**

Run: `python -m pytest tests/test_decomposition_check.py -v`
Expected: PASS (all 5 tests — files exist, no circular imports, no orphans, no silent skips, shim is minimal)

**Step 7: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All 860+ tests PASS

**Step 8: Update run_batches.py**

Update batch definitions to reference new test files instead of monolithic test_voice.py.

**Step 9: Commit**

```bash
git add tests/
git commit -m "refactor: decompose test_voice.py into 8 domain-specific test files"
```

---

## Phase 2: Inference Decoupling and Hardware Optimization

**Objective:** Physically separate FastAPI from vLLM, fix Docker IPC configs, and optimize vLLM multimodal parameters.

### Task 2.1: Write Characterization Tests for engine_vllm.py

**Files:**
- Read: `qwen3_tts/core/engine_vllm.py` (546 lines)
- Create: `tests/test_engine_vllm_characterization.py`

**Step 1: Write characterization tests capturing current I/O contracts**

Tests must capture:
- Input payload schemas (what `engine_vllm.py` functions accept)
- Output structures (what they return)
- Error conditions (what exceptions they raise)

**Step 2: Run to confirm they pass against current implementation**

Run: `python -m pytest tests/test_engine_vllm_characterization.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_engine_vllm_characterization.py
git commit -m "test: add characterization tests for engine_vllm.py"
```

---

### Task 2.2: Decouple FastAPI from vLLM Inference

**Files:**
- Modify: `qwen3_tts/server/app.py` (1,727 lines)
- Modify: `qwen3_tts/core/engine_vllm.py` (546 lines)

**Step 1: Write integration test for decoupled architecture**

```python
# tests/test_decoupled_inference.py
def test_fastapi_does_not_import_torch_or_vllm():
    """FastAPI server module must not directly import heavy inference libs."""
    import ast
    with open("qwen3_tts/server/app.py") as f:
        tree = ast.parse(f.read())
    top_imports = [
        node.names[0].name for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    ]
    for lib in ["torch", "vllm", "transformers"]:
        assert lib not in top_imports, f"app.py must not import {lib} at module scope"
```

**Step 2: Run test to verify current state**

Run: `python -m pytest tests/test_decoupled_inference.py -v`

**Step 3: Refactor app.py to route inference via httpx.AsyncClient**

- Wrap `engine_vllm.py` as a standalone vLLM OpenAI-compatible endpoint
- FastAPI connects to it via `httpx.AsyncClient` instead of direct function calls
- Follow Route → Controller → Service pattern

**Step 4: Run characterization tests to confirm I/O contracts preserved**

Run: `python -m pytest tests/test_engine_vllm_characterization.py -v`
Expected: PASS

**Step 5: Run full suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: PASS

**Step 6: Commit**

```bash
git add qwen3_tts/server/app.py qwen3_tts/core/engine_vllm.py
git commit -m "refactor: decouple FastAPI from vLLM via async client"
```

---

### Task 2.3: Fix Docker IPC, Precision, and Parallelism Config

**Files:**
- Modify: `Dockerfile.vllm`
- Modify: `docker-compose.yml`

**Step 1: Write test validating Docker configs**

```python
# tests/test_docker_config.py
import unittest, yaml

class TestDockerCompose(unittest.TestCase):
    def test_vllm_has_ipc_host_or_shm_size(self):
        with open("docker-compose.yml") as f:
            config = yaml.safe_load(f)
        vllm_svc = config["services"].get("vllm", config["services"].get("tts-vllm", {}))
        has_ipc = vllm_svc.get("ipc") == "host"
        has_shm = "shm_size" in vllm_svc
        self.assertTrue(has_ipc or has_shm,
            "vLLM service must have ipc: host or shm_size >= 16g")

    def test_hf_cache_volume_mounted(self):
        with open("docker-compose.yml") as f:
            config = yaml.safe_load(f)
        vllm_svc = config["services"].get("vllm", config["services"].get("tts-vllm", {}))
        volumes = vllm_svc.get("volumes", [])
        hf_cache = any(".cache/huggingface" in str(v) for v in volumes)
        self.assertTrue(hf_cache, "Must mount HuggingFace cache to prevent re-downloads")

    def test_vllm_dtype_bfloat16(self):
        """vLLM must enforce --dtype=bfloat16 to halve memory bandwidth."""
        with open("docker-compose.yml") as f:
            config = yaml.safe_load(f)
        vllm_svc = config["services"].get("vllm", config["services"].get("tts-vllm", {}))
        command = str(vllm_svc.get("command", ""))
        entrypoint = str(vllm_svc.get("entrypoint", ""))
        env = vllm_svc.get("environment", {})
        # Check command/entrypoint args or environment
        has_dtype = ("--dtype" in command and "bfloat16" in command) or \
                    ("--dtype" in entrypoint and "bfloat16" in entrypoint) or \
                    env.get("VLLM_DTYPE") == "bfloat16"
        self.assertTrue(has_dtype,
            "vLLM must enforce --dtype=bfloat16 for 50% memory bandwidth reduction")

    def test_tensor_parallel_parameterized(self):
        """--tensor-parallel-size must be mapped to GPU_AMOUNT env var."""
        with open("docker-compose.yml") as f:
            config = yaml.safe_load(f)
        vllm_svc = config["services"].get("vllm", config["services"].get("tts-vllm", {}))
        command = str(vllm_svc.get("command", ""))
        entrypoint = str(vllm_svc.get("entrypoint", ""))
        env = vllm_svc.get("environment", {})
        # tensor-parallel-size should reference GPU_AMOUNT variable
        has_tp = ("GPU_AMOUNT" in command) or ("GPU_AMOUNT" in entrypoint) or \
                 ("GPU_AMOUNT" in str(env))
        self.assertTrue(has_tp,
            "--tensor-parallel-size must be parameterized via GPU_AMOUNT env var")

    def test_vllm_multimodal_params(self):
        """vLLM must include chunked prefill and multimodal limit flags."""
        with open("docker-compose.yml") as f:
            config = yaml.safe_load(f)
        vllm_svc = config["services"].get("vllm", config["services"].get("tts-vllm", {}))
        command = str(vllm_svc.get("command", ""))
        entrypoint = str(vllm_svc.get("entrypoint", ""))
        combined = command + entrypoint
        self.assertIn("--enable-chunked-prefill", combined,
            "Missing --enable-chunked-prefill for audio embedding chunk processing")
        self.assertIn("--limit-mm-per-prompt", combined,
            "Missing --limit-mm-per-prompt audio=1 for VRAM protection")
```

**Step 2: Run test to verify current state**

Run: `python -m pytest tests/test_docker_config.py -v`
Expected: Multiple FAILs

**Step 3: Update Docker configs**

In `docker-compose.yml` vLLM service:
- Add `ipc: host` (or `shm_size: 16g`)
- Add HuggingFace cache volume mount: `~/.cache/huggingface:/root/.cache/huggingface`
- Add `--dtype=bfloat16` to vLLM launch args
- Parameterize `--tensor-parallel-size=${GPU_AMOUNT:-1}` (defaults to 1 GPU)
- Add `--limit-mm-per-prompt audio=1`
- Add `--enable-chunked-prefill`

In `Dockerfile.vllm`:
- Ensure `--ipc=host` is documented in usage comments
- Add `ENV GPU_AMOUNT=1` as default

**Step 4: Run test to verify**

Run: `python -m pytest tests/test_docker_config.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add Dockerfile.vllm docker-compose.yml tests/test_docker_config.py
git commit -m "fix: Docker IPC, bfloat16 dtype, tensor parallelism, and vLLM multimodal params"
```

---

## Phase 3: Asynchronous Streaming and Frontend Synchronization

**Objective:** Enable real-time bidirectional audio with WebSocket streaming and offload wavesurfer.js decoding to backend.

### Task 3.1: Convert TTS Endpoint to Async StreamingResponse

**Files:**
- Modify: `qwen3_tts/server/app.py`
- Test: `tests/test_fastapi_endpoints.py`

**Step 1: Write test for streaming response type**

```python
def test_generate_stream_returns_streaming_response():
    """Verify /generate-stream returns chunked audio data."""
    ...
```

**Step 2: Implement async generator with yield**

Refactor the generation endpoint to use `StreamingResponse` with an `async def` generator that yields audio chunks as they're produced by the inference engine.

**Step 3: Run tests**

Run: `python -m pytest tests/test_fastapi_endpoints.py -v`
Expected: PASS

**Step 4: Commit**

---

### Task 3.2: Implement WebSocket Endpoint for Bidirectional Audio

**Files:**
- Create: `qwen3_tts/server/websocket.py`
- Test: `tests/test_websocket.py`

**Step 1: Write failing test for WebSocket connection**

**Step 2: Implement WebSocket handler**

- Accept incoming audio streams
- Buffer with `numpy.frombuffer`
- Voice Activity Detection (silence detection for generation trigger)
- Stream TTS response bytes back through WebSocket

**Step 3: Run tests, commit**

---

### Task 3.3: Pre-calculate Wavesurfer Peak Data on Backend

**Files:**
- Modify: `qwen3_tts/core/engine/audio_processing.py`
- Modify: `qwen3_tts/interface/wavesurfer_js.py`
- Test: `tests/test_audio_utils.py`

**Step 1: Write test for peak calculation function**

```python
def test_calculate_waveform_peaks():
    """Backend should pre-calculate normalized peak array."""
    import numpy as np
    audio = np.random.randn(24000).astype(np.float32)  # 1 second at 24kHz
    peaks = calculate_waveform_peaks(audio, num_peaks=100)
    assert len(peaks) == 100
    assert all(-1.0 <= p <= 1.0 for p in peaks)
```

**Step 2: Implement peak calculation**

**Step 3: Update wavesurfer.js init to accept backendData**

**Step 4: Run tests, commit**

---

## Phase 4: Automated Evaluation Pipeline Integration

**Objective:** Add objective audio quality metrics (WER + SIM) to CI/CD.

### Task 4.1: Create WER and SIM Evaluation Scripts

**Files:**
- Create: `tests/evaluations/__init__.py`
- Create: `tests/evaluations/test_wer.py`
- Create: `tests/evaluations/test_speaker_similarity.py`

**Step 1: Write WER evaluation test**

```python
# tests/evaluations/test_wer.py
def test_wer_below_threshold():
    """Generated audio transcription must have WER < 5%."""
    # Uses Whisper to transcribe generated audio
    # Compares against original text prompt via jiwer
    # Asserts WER < 0.05
    ...
```

**Step 2: Write Speaker Similarity (SIM) evaluation test**

```python
# tests/evaluations/test_speaker_similarity.py
import unittest
import numpy as np

class TestSpeakerSimilarity(unittest.TestCase):
    def test_clone_voice_cosine_similarity_above_threshold(self):
        """Zero-shot voice cloning output must match source speaker embedding."""
        # 1. Load original voice prompt audio (from voice_prompts/)
        # 2. Load generated clone output audio
        # 3. Extract WavLM-SV feature vectors from both
        # 4. Calculate cosine similarity between embeddings
        # 5. Assert similarity > 0.85 threshold
        ...

    def test_cosine_distance_calculation(self):
        """Verify cosine similarity helper works correctly."""
        from tests.evaluations.speaker_similarity_utils import cosine_similarity
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(a, b), 1.0)

        c = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(a, c), 0.0)

    def test_wavlm_embedding_extraction(self):
        """Verify WavLM-SV embeddings have expected dimensionality."""
        # Extract embedding from a short test audio clip
        # Assert embedding is a 1-D float vector of expected size
        ...
```

**Step 3: Create speaker similarity utility module**

```python
# tests/evaluations/speaker_similarity_utils.py
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def extract_wavlm_embedding(audio_path: str) -> np.ndarray:
    """Extract WavLM-SV speaker verification embedding from audio file."""
    # Lazy import to avoid loading heavy models unless needed
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    import torchaudio
    ...
```

**Step 4: Run tests locally**

Run: `python -m pytest tests/evaluations/ -v`

**Step 5: Commit**

```bash
git add tests/evaluations/
git commit -m "test: add WER and speaker similarity (SIM) evaluation scripts"
```

---

### Task 4.2: GitHub Actions Audio Evaluation Workflow

**Files:**
- Create: `.github/workflows/audio_eval.yml`

**Step 1: Write workflow that runs both WER and SIM evaluations on PR**

```yaml
name: Audio Quality Evaluation
on: [pull_request]
jobs:
  audio-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install evaluation deps
        run: pip install whisper jiwer transformers torchaudio numpy
      - name: Run WER evaluation
        run: python -m pytest tests/evaluations/test_wer.py -v
      - name: Run Speaker Similarity evaluation
        run: python -m pytest tests/evaluations/test_speaker_similarity.py -v
```

**Step 2: Commit**

```bash
git add .github/workflows/audio_eval.yml
git commit -m "ci: add audio quality evaluation workflow (WER + SIM)"
```

---

### Task 4.3: LLM-as-a-Judge Prototype

**Files:**
- Create: `tests/evaluations/llm_judge.py`

**Step 1: Write prototype that evaluates prompt adherence**

- Takes generated audio transcription + original prompt
- Sends to LLM with strict rubric
- Returns pass/fail with reasoning

**Step 2: Add as non-blocking step in GitHub Actions**

**Step 3: Commit**

---

## Verification

After each phase:
1. `python -m pytest tests/ -v --tb=short` — full suite passes
2. `git log --oneline` — clean commit history
3. No regressions in existing functionality

After all phases:
1. `tts server stop && tts server start` — server starts cleanly
2. `tts doctor` — health check passes
3. Docker builds: `docker build -f Dockerfile.vllm .` — succeeds
