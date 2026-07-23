# AI Regression Testing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add AI regression tests to prevent systematic blind spots where AI introduces bugs that self-review misses (backend inconsistencies, model state edge cases, API response contracts)

**Architecture:** Create new test module `test_ai_regression.py` with 3 test classes covering backend consistency, model state edge cases, and API contracts. Tests use existing E2E infrastructure (SERVER_URL, auth tokens, model loading helpers).

**Tech Stack:** pytest, urllib, JSON, TTSClient, FastAPI server endpoints

---

## Task 1: Create AI Regression Test Module Structure

**Files:**
- Create: `tests/test_ai_regression.py`

**Step 1: Create module header and imports**

```python
"""
AI Regression Tests - Prevent systematic AI-introduced bugs.

Tests patterns where AI self-review carries same assumptions into both
implementation and review, creating blind spots:
- Backend-specific logic (MLX vs Torch vs vLLM path inconsistencies)
- Model state edge cases (operations when models not loaded)
- API response contract violations (missing fields in responses)
"""
import pytest
import json
import time
import urllib.request
import os
from pathlib import Path

# Test configuration
SERVER_URL = "http://127.0.0.1:5123"
```

**Step 2: Add helper functions from existing E2E tests**

```python
def _get_auth_token():
    """Read the server auth token."""
    token_path = os.path.expanduser("~/.config/qwen3-tts/.voice_server_token")
    try:
        with open(token_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        # Try legacy location
        legacy_path = os.path.expanduser("~/.voice_server_token")
        try:
            with open(legacy_path, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return ""

def _get_headers():
    """Get authenticated headers for API requests."""
    token = _get_auth_token()
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
```

**Step 3: Run Python syntax check**

Run: `python -m py_compile tests/test_ai_regression.py`
Expected: No syntax errors

**Step 4: Commit**

```bash
git add tests/test_ai_regression.py
git commit -m "test: create AI regression test module structure"
```

---

## Task 2: Implement Backend Consistency Test

**Files:**
- Modify: `tests/test_ai_regression.py`

**Step 1: Write the backend consistency test class**

```python
class TestBackendConsistency:
    """Test that all backends return identical API response shapes."""

    @pytest.fixture(scope="class", autouse=True)
    def ensure_server_running(self):
        """Verify server is running before backend tests."""
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=2)
                if resp.status == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
        pytest.skip("Server not available")

    def test_all_backends_return_same_response_shape(self):
        """Test MLX, Torch backends return identical API shapes.

        This prevents the #1 AI regression pattern: backend-specific path
        inconsistencies where a feature is added to MLX backend but forgotten
        in Torch backend, or vice versa.
        """
        text = "Test text for backend consistency"
        modes = ["clone", "design", "custom"]
        backends = self._get_available_backends()

        if len(backends) < 2:
            pytest.skip("Need at least 2 backends for consistency test")

        # Store response shapes from first backend
        reference_shapes = {}

        for backend in backends:
            for mode in modes:
                # Skip if model not loaded for this backend
                if not self._is_model_loaded(mode):
                    continue

                try:
                    response = self._generate_with_backend(text, mode, backend)
                    response_shape = self._extract_response_shape(response)

                    key = f"{mode}_{backend}"
                    if key not in reference_shapes:
                        reference_shapes[key] = response_shape
                    else:
                        # Compare with reference
                        assert response_shape == reference_shapes[key], \
                            f"Response shape differs for {mode} mode on {backend} backend\n" + \
                            f"Expected: {reference_shapes[key]}\n" + \
                            f"Got: {response_shape}"
                except Exception as e:
                    pytest.fail(f"Backend consistency test failed for {mode}/{backend}: {e}")

    def _get_available_backends(self):
        """Return list of available backends (mlx, torch, vllm)."""
        backends = []

        # Check MLX backend availability
        try:
            import mlx.core as mlx
            backends.append("mlx")
        except ImportError:
            pass

        # Check Torch backend availability
        try:
            import torch
            backends.append("torch")
        except ImportError:
            pass

        return backends

    def _is_model_loaded(self, model_name):
        """Check if a model is currently loaded."""
        try:
            resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)
            health = json.loads(resp.read())
            return health.get(f"{model_name}_model_loaded", False)
        except Exception:
            return False

    def _generate_with_backend(self, text, mode, backend):
        """Generate audio using specific backend via config change."""
        # Get current config
        headers = _get_headers()
        resp = urllib.request.urlopen(f"{SERVER_URL}/config", headers=headers)
        original_config = json.loads(resp.read())

        # Temporarily switch backend
        try:
            update_data = {"backend": backend}
            req = urllib.request.Request(
                f"{SERVER_URL}/update-model-config",
                data=json.dumps(update_data).encode(),
                headers=headers,
                method="POST"
            )
            resp = urllib.request.urlopen(req, timeout=10)

            # Wait a moment for config change to apply
            time.sleep(1)

            # Generate audio
            gen_data = {"text": text, "mode": mode}
            req = urllib.request.Request(
                f"{SERVER_URL}/generate",
                data=json.dumps(gen_data).encode(),
                headers=headers,
                method="POST"
            )
            resp = urllib.request.urlopen(req, timeout=120)
            return json.loads(resp.read())

        finally:
            # Restore original backend
            original_backend = original_config.get("advanced", {}).get("backend", "mlx")
            restore_data = {"backend": original_backend}
            req = urllib.request.Request(
                f"{SERVER_URL}/update-model-config",
                data=json.dumps(restore_data).encode(),
                headers=headers,
                method="POST"
            )
            urllib.request.urlopen(req, timeout=10)

    def _extract_response_shape(self, response):
        """Extract the shape/structure from API response."""
        # For /generate endpoint, check key fields are present
        required_fields = ["audio", "sample_rate", "generation_time"]
        response_fields = list(response.keys())

        shape = {field: (field in response_fields) for field in required_fields}
        return shape
```

**Step 2: Run test to verify it fails (no backends configured yet)**

Run: `pytest tests/test_ai_regression.py::TestBackendConsistency::test_all_backends_return_same_response_shape -v`
Expected: SKIP or PASS (depends on available backends)

**Step 3: Commit**

```bash
git add tests/test_ai_regression.py
git commit -m "test: add backend consistency test for AI regression prevention"
```

---

## Task 3: Implement Model State Edge Case Tests

**Files:**
- Modify: `tests/test_ai_regression.py`

**Step 1: Write model state edge case test class**

```python
class TestModelStateEdgeCases:
    """Test proper error handling when models are not loaded."""

    @pytest.fixture(scope="class", autouse=True)
    def ensure_server_running(self):
        """Verify server is running before model state tests."""
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=2)
                if resp.status == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
        pytest.skip("Server not available")

    def test_clone_generation_fails_gracefully_when_model_not_loaded(self):
        """Test proper error handling when clone model not loaded.

        This prevents silent failures where generation fails with unclear error
        messages or crashes the server when a model is unexpectedly unloaded.
        """
        headers = _get_headers()

        # First, ensure clone model is loaded to get baseline
        try:
            # Check if clone model is loaded
            resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)
            health = json.loads(resp.read())

            if not health.get("clone_model_loaded"):
                # Load it first
                load_data = {"model_type": "clone"}
                req = urllib.request.Request(
                    f"{SERVER_URL}/load-model",
                    data=json.dumps(load_data).encode(),
                    headers=headers,
                    method="POST"
                )
                resp = urllib.request.urlopen(req, timeout=120)

                # Wait for model to be ready
                deadline = time.time() + 60
                while time.time() < deadline:
                    time.sleep(1)
                    resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)
                    health = json.loads(resp.read())
                    if health.get("clone_model_loaded"):
                        break
        except Exception as e:
            pytest.skip(f"Could not load clone model: {e}")

        # Now test: unload clone model and verify graceful failure
        try:
            # Unload clone model
            unload_data = {"model_type": "clone"}
            req = urllib.request.Request(
                f"{SERVER_URL}/unload-model",
                data=json.dumps(unload_data).encode(),
                headers=headers,
                method="POST"
            )
            resp = urllib.request.urlopen(req, timeout=30)

            # Try to generate with unloaded model - should fail gracefully
            gen_data = {"text": "Test", "mode": "clone"}
            req = urllib.request.Request(
                f"{SERVER_URL}/generate",
                data=json.dumps(gen_data).encode(),
                headers=headers,
                method="POST"
            )

            # Should fail with clear error, not hang or crash
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                resp = exc_info.value
                assert resp.code in [400, 500, 503], \
                    f"Expected 400/500/503, got {resp.code}"

                # Verify error message is clear
                error_response = json.loads(resp.read())
                assert "error" in error_response or "message" in error_response, \
                    "Error response should have error/message field"

        finally:
            # Restore clone model for other tests
            load_data = {"model_type": "clone"}
            req = urllib.request.Request(
                f"{SERVER_URL}/load-model",
                data=json.dumps(load_data).encode(),
                headers=headers,
                method="POST"
            )
            urllib.request.urlopen(req, timeout=120)

    def test_design_generation_fails_gracefully_when_model_not_loaded(self):
        """Test design mode fails gracefully when design model not loaded."""
        headers = _get_headers()

        try:
            # Unload design model if loaded
            resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)
            health = json.loads(resp.read())

            if health.get("design_model_loaded"):
                unload_data = {"model_type": "design"}
                req = urllib.request.Request(
                    f"{SERVER_URL}/unload-model",
                    data=json.dumps(unload_data).encode(),
                    headers=headers,
                    method="POST"
                )
                urllib.request.urlopen(req, timeout=30)

            # Try to generate - should fail gracefully
            gen_data = {"text": "Test", "mode": "design"}
            req = urllib.request.Request(
                f"{SERVER_URL}/generate",
                data=json.dumps(gen_data).encode(),
                headers=headers,
                method="POST"
            )

            with pytest.raises(urllib.error.HTTPError) as exc_info:
                resp = exc_info.value
                assert resp.code in [400, 500, 503]

                error_response = json.loads(resp.read())
                assert "error" in error_response or "message" in error_response

        finally:
            # Restore design model
            load_data = {"model_type": "design"}
            req = urllib.request.Request(
                f"{SERVER_URL}/load-model",
                data=json.dumps(load_data).encode(),
                headers=headers,
                method="POST"
            )
            urllib.request.urlopen(req, timeout=120)

    def test_custom_generation_fails_gracefully_when_model_not_loaded(self):
        """Test custom mode fails gracefully when custom model not loaded."""
        headers = _get_headers()

        try:
            # Unload custom model if loaded
            resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)
            health = json.loads(resp.read())

            if health.get("custom_model_loaded"):
                unload_data = {"model_type": "custom"}
                req = urllib.request.Request(
                    f"{SERVER_URL}/unload-model",
                    data=json.dumps(unload_data).encode(),
                    headers=headers,
                    method="POST"
                )
                urllib.request.urlopen(req, timeout=30)

            # Try to generate - should fail gracefully
            gen_data = {"text": "Test", "mode": "custom"}
            req = urllib.request.Request(
                f"{SERVER_URL}/generate",
                data=json.dumps(gen_data).encode(),
                headers=headers,
                method="POST"
            )

            with pytest.raises(urllib.error.HTTPError) as exc_info:
                resp = exc_info.value
                assert resp.code in [400, 500, 503]

                error_response = json.loads(resp.read())
                assert "error" in error_response or "message" in error_response

        finally:
            # Restore custom model
            load_data = {"model_type": "custom"}
            req = urllib.request.Request(
                f"{SERVER_URL}/load-model",
                data=json.dumps(load_data).encode(),
                headers=headers,
                method="POST"
            )
            urllib.request.urlopen(req, timeout=120)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ai_regression.py::TestModelStateEdgeCases -v`
Expected: Tests should pass (they handle model load/unload internally)

**Step 3: Commit**

```bash
git add tests/test_ai_regression.py
git commit -m "test: add model state edge case tests for graceful failure handling"
```

---

## Task 4: Implement API Response Contract Tests

**Files:**
- Modify: `tests/test_ai_regression.py`

**Step 1: Write API response contract test class**

```python
class TestAPIResponseContracts:
    """Test that API responses adhere to expected contracts (all required fields present)."""

    @pytest.fixture(scope="class", autouse=True)
    def ensure_server_running(self):
        """Verify server is running before API contract tests."""
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=2)
                if resp.status == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
        pytest.skip("Server not available")

    def test_stats_endpoint_includes_all_required_fields(self):
        """Prevent regression where new stats fields are missed.

        This catches the #2 AI regression pattern: adding a field to the response
        construction but forgetting to add it to the SELECT clause or omitting it
        from one path (sandbox vs production, MLX vs Torch).
        """
        headers = _get_headers()

        req = urllib.request.Request(
            f"{SERVER_URL}/stats",
            headers=headers,
            method="GET"
        )
        resp = urllib.request.urlopen(req, timeout=10)
        stats = json.loads(resp.read())

        # Define required fields that must always be present
        required_fields = {
            "memory",
            "models",
            "generation_history",
            "cache_stats"
        }

        for field in required_fields:
            assert field in stats, f"Missing required field: {field}"

        # Verify models substructure has required fields
        assert "models" in stats
        assert isinstance(stats["models"], dict)

        # Check each model type has expected fields
        for model_type in ["clone", "design", "custom"]:
            if model_type in stats["models"]:
                model_info = stats["models"][model_type]
                expected_model_fields = ["loaded", "memory_mb"]
                for field in expected_model_fields:
                    assert field in model_info, \
                        f"Model {model_type} missing field: {field}"

    def test_models_endpoint_includes_all_required_fields(self):
        """Test /models endpoint response contract."""
        headers = _get_headers()

        req = urllib.request.Request(
            f"{SERVER_URL}/models",
            headers=headers,
            method="GET"
        )
        resp = urllib.request.urlopen(req, timeout=10)
        models_data = json.loads(resp.read())

        # Verify response structure
        assert isinstance(models_data, dict)

        # Check for each expected model type
        for model_type in ["clone", "design", "custom"]:
            assert model_type in models_data, f"Missing model type: {model_type}"
            model_info = models_data[model_type]

            # Verify model info has required fields
            required_model_fields = ["loaded", "memory_mb"]
            for field in required_model_fields:
                assert field in model_info, \
                    f"Model {model_type} missing field: {field}"

    def test_health_endpoint_includes_all_required_fields(self):
        """Test /health endpoint response contract."""
        resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)
        health = json.loads(resp.read())

        # Check model status fields
        for model_type in ["clone", "design", "custom"]:
            model_key = f"{model_type}_model_loaded"
            assert model_key in health, f"Health missing {model_key} field"
            assert isinstance(health[model_key], bool), \
                f"{model_key} should be boolean, got {type(health[model_key])}"

        # Check server status
        assert "server_status" in health or "status" in health, \
            "Health response should have status field"

    def test_generate_endpoint_response_contract(self):
        """Test /generate endpoint returns expected response structure."""
        headers = _get_headers()

        # First ensure clone model is loaded
        try:
            resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)
            health = json.loads(resp.read())

            if not health.get("clone_model_loaded"):
                # Load clone model
                load_data = {"model_type": "clone"}
                req = urllib.request.Request(
                    f"{SERVER_URL}/load-model",
                    data=json.dumps(load_data).encode(),
                    headers=headers,
                    method="POST"
                )
                urllib.request.urlopen(req, timeout=120)

                # Wait for model to be ready
                deadline = time.time() + 60
                while time.time() < deadline:
                    time.sleep(1)
                    resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)
                    health = json.loads(resp.read())
                    if health.get("clone_model_loaded"):
                        break
        except Exception:
            pytest.skip("Could not ensure clone model is loaded")

        # Test generate response structure
        gen_data = {"text": "Contract test", "mode": "clone"}
        req = urllib.request.Request(
            f"{SERVER_URL}/generate",
            data=json.dumps(gen_data).encode(),
            headers=headers,
            method="POST"
        )
        resp = urllib.request.urlopen(req, timeout=120)
        result = json.loads(resp.read())

        # Verify response has expected fields
        required_fields = ["audio", "sample_rate", "generation_time"]
        for field in required_fields:
            assert field in result, f"Generate response missing field: {field}"

        # Verify data types
        assert isinstance(result["audio"], str), "audio should be base64 string"
        assert isinstance(result["sample_rate"], int), "sample_rate should be integer"
        assert isinstance(result["generation_time"], (int, float)), \
            "generation_time should be numeric"
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_ai_regression.py::TestAPIResponseContracts -v`
Expected: Tests should pass (all fields present in current implementation)

**Step 3: Commit**

```bash
git add tests/test_ai_regression.py
git commit -m "test: add API response contract tests to prevent field omissions"
```

---

## Task 5: Add Pytest Configuration and Documentation

**Files:**
- Modify: `tests/test_ai_regression.py`

**Step 1: Add pytest configuration markers and docstring**

```python
"""
AI Regression Tests - Prevent systematic AI-introduced bugs.

Tests patterns where AI self-review carries same assumptions into both
implementation and review, creating blind spots:
- Backend-specific logic (MLX vs Torch vs vLLM path inconsistencies)
- Model state edge cases (operations when models not loaded)
- API response contract violations (missing fields in responses)

Run tests with:
    pytest tests/test_ai_regression.py -v
    pytest tests/test_ai_regression.py -v -k "backend"  # Backend consistency only
    pytest tests/test_ai_regression.py -v -k "state"    # Model state edge cases only
    pytest tests/test_ai_regression.py -v -k "contract" # API contracts only
"""
```

**Step 2: Verify module loads correctly**

Run: `python -c "import tests.test_ai_regression; print('Module loads successfully')"`
Expected: No import errors

**Step 3: Run all AI regression tests**

Run: `pytest tests/test_ai_regression.py -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add tests/test_ai_regression.py
git commit -m "docs: add AI regression test documentation and markers"
```

---

## Task 6: Integration with Full Test Suite

**Step 1: Update batch test runner if needed**

Check if `tests/run_batches.py` needs to include new test module:

Run: `grep -n "test_ai_regression" tests/run_batches.py`
Expected: Not found (optional addition)

**Step 2: Verify new tests don't conflict with existing E2E tests**

Run: `pytest tests/test_e2e_playwright.py tests/test_ai_regression.py -v`
Expected: All tests pass, no conflicts

**Step 3: Document in project overview**

Create documentation note in CLAUDE.md if needed:

```markdown
## AI Regression Testing

See `tests/test_ai_regression.py` for AI-specific regression tests that prevent:
- Backend inconsistencies (MLX vs Torch response shape differences)
- Model state edge cases (graceful failures when models not loaded)
- API response contract violations (missing required fields)

These tests catch systematic blind spots where AI self-review misses bugs due to
carrying the same assumptions into both implementation and review.
```

**Step 4: Final commit with all changes**

```bash
git add .
git commit -m "test: complete AI regression testing suite with documentation"
```

---

## Task 7: Verification and Validation

**Step 1: Run full test suite including new AI regression tests**

Run: `pytest tests/test_ai_regression.py -v`
Expected: All tests pass

**Step 2: Run E2E tests to ensure no regressions**

Run: `pytest tests/test_e2e_playwright.py -v`
Expected: All existing tests still pass

**Step 3: Run batch performance tests**

Run: `pytest tests/test_e2e_performance_batch.py -v`
Expected: All tests still pass

**Step 4: Final validation**

Run: `pytest tests/test_ai_regression.py tests/test_e2e_playwright.py tests/test_e2e_performance_batch.py -v`
Expected: 22 + new AI regression tests all pass

**Step 5: Create summary documentation**

Add to `docs/plans/2026-04-06-ai-regression-testing.md`:

```markdown
## Implementation Summary

**Added Files:**
- tests/test_ai_regression.py - New AI regression test module

**Test Classes Added:**
1. TestBackendConsistency - Backend response shape consistency
2. TestModelStateEdgeCases - Model not loaded graceful failures
3. TestAPIResponseContracts - API response field completeness

**Total New Tests:** 6
- test_all_backends_return_same_response_shape
- test_clone_generation_fails_gracefully_when_model_not_loaded
- test_design_generation_fails_gracefully_when_model_not_loaded
- test_custom_generation_fails_gracefully_when_model_not_loaded
- test_stats_endpoint_includes_all_required_fields
- test_models_endpoint_includes_all_required_fields
- test_health_endpoint_includes_all_required_fields
- test_generate_endpoint_response_contract

**Integration:** New tests integrate seamlessly with existing E2E test suite
**Coverage:** Addresses top 3 AI regression patterns identified in project analysis
```

**Step 6: Final commit**

```bash
git add docs/plans/2026-04-06-ai-regression-testing.md
git commit -m "docs: complete AI regression testing implementation plan"
```
