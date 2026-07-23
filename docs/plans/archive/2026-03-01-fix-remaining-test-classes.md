# Fix Remaining test_voice.py Test Classes for FastAPI

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update remaining test classes in test_voice.py to use the `_setup_fastapi_app_state()` helper for proper FastAPI compatibility.

**Architecture:** Mechanical refactor — replace manual app.state initialization in setUpClass methods with calls to the existing `_setup_fastapi_app_state()` helper function. The helper already initializes all required FastAPI state attributes (auth_token, models, model_load_times, generation_lock, request_queue, etc.).

**Tech Stack:** Python unittest, FastAPI TestClient

---

## Context

The helper function `_setup_fastapi_app_state()` was added in commit `f6ed576` to properly initialize FastAPI app.state. Two test classes (TestServerValidation, TestServerAuth) were updated to use it. Fourteen test classes remain with manual/incomplete app.state setup.

**Test classes to fix:**
1. TestStreamingServerEndpoint
2. TestHealthEndpointInfo
3. TestGenerationStatus
4. TestLoadModelEndpoint
5. TestCancelGenerationEndpoint
6. TestUpdateModelConfigEndpoint
7. TestStreamingEndpointStructure
8. TestETACache
9. TestDeletePromptEndpoint
10. TestRenamePromptEndpoint
11. TestPreviewPromptEndpoint
12. TestPromptDetailsEndpoint
13. TestUnloadModelEndpoint
14. TestUpdateStartupConfigEndpoint
15. TestModelsEndpointEnhanced

---

## Task 1: Update TestStreamingServerEndpoint setUpClass

**Files:**
- Modify: `tests/test_voice.py:950-970` (approximately)

**Step 1: Read current setUpClass**

Read: `tests/test_voice.py` around line 950-970
Expected: See manual app.state initialization

**Step 2: Replace manual setup with helper call**

OLD pattern:
```python
@classmethod
def setUpClass(cls):
    from fastapi.testclient import TestClient
    from qwen3_tts.server.app import app
    app.state.auth_token = "test_token"  # nosec B105
    app.state.server_config = {...}
    app.state.models = {"clone": None, "design": None, "custom": None}
    # ... other manual setup ...
    cls.client = TestClient(app)
    cls.auth = {"Authorization": "Bearer test_token"}
```

NEW pattern:
```python
@classmethod
def setUpClass(cls):
    from fastapi.testclient import TestClient
    from qwen3_tts.server.app import app
    _setup_fastapi_app_state(app, server_config={...})
    app.state.models_loaded.set()
    cls.client = TestClient(app)
    cls.auth = {"Authorization": "Bearer test_token"}
```

**Step 3: Run the test class to verify**

Run: `pytest tests/test_voice.py::TestStreamingServerEndpoint -v --tb=short`
Expected: All tests in class pass

---

## Task 2: Update TestHealthEndpointInfo setUpClass

**Files:**
- Modify: `tests/test_voice.py:1240-1260` (approximately)

**Step 1: Read and update setUpClass**

Same pattern as Task 1 — replace manual setup with `_setup_fastapi_app_state(app)` call.

**Step 2: Run tests**

Run: `pytest tests/test_voice.py::TestHealthEndpointInfo -v --tb=short`
Expected: All tests pass

---

## Task 3: Update TestGenerationStatus setUpClass

**Files:**
- Modify: `tests/test_voice.py:1290-1310` (approximately)

**Step 1: Read and update setUpClass**

Replace manual setup with helper call.

**Step 2: Run tests**

Run: `pytest tests/test_voice.py::TestGenerationStatus -v --tb=short`
Expected: All tests pass

---

## Task 4: Update TestLoadModelEndpoint setUpClass

**Files:**
- Modify: `tests/test_voice.py:1335-1355` (approximately)

**Step 1: Read and update setUpClass**

Replace manual setup with helper call.

**Step 2: Run tests**

Run: `pytest tests/test_voice.py::TestLoadModelEndpoint -v --tb=short`
Expected: All tests pass

---

## Task 5: Update TestCancelGenerationEndpoint setUpClass

**Files:**
- Modify: `tests/test_voice.py:1390-1410` (approximately)

**Step 1: Read and update setUpClass**

Replace manual setup with helper call.

**Step 2: Run tests**

Run: `pytest tests/test_voice.py::TestCancelGenerationEndpoint -v --tb=short`
Expected: All tests pass

---

## Task 6: Update TestUpdateModelConfigEndpoint setUpClass

**Files:**
- Modify: `tests/test_voice.py:1415-1435` (approximately)

**Step 1: Read and update setUpClass**

Replace manual setup with helper call.

**Step 2: Run tests**

Run: `pytest tests/test_voice.py::TestUpdateModelConfigEndpoint -v --tb=short`
Expected: All tests pass

---

## Task 7: Update TestStreamingEndpointStructure setUpClass

**Files:**
- Modify: `tests/test_voice.py:1755-1775` (approximately)

**Step 1: Read and update setUpClass**

Replace manual setup with helper call.

**Step 2: Run tests**

Run: `pytest tests/test_voice.py::TestStreamingEndpointStructure -v --tb=short`
Expected: All tests pass

---

## Task 8: Update TestETACache setUpClass

**Files:**
- Modify: `tests/test_voice.py:2080-2100` (approximately)

**Step 1: Read and update setUpClass**

Note: This test class may only need partial state (eta_cache). Verify what's actually needed.

**Step 2: Run tests**

Run: `pytest tests/test_voice.py::TestETACache -v --tb=short`
Expected: All tests pass

---

## Task 9: Update TestDeletePromptEndpoint setUpClass

**Files:**
- Modify: `tests/test_voice.py:2200-2220` (approximately)

**Step 1: Read and update setUpClass**

Replace manual setup with helper call. Note: This class also has a setUp method for temp directory.

**Step 2: Run tests**

Run: `pytest tests/test_voice.py::TestDeletePromptEndpoint -v --tb=short`
Expected: All tests pass

---

## Task 10: Update TestRenamePromptEndpoint setUpClass

**Files:**
- Modify: `tests/test_voice.py:2260-2280` (approximately)

**Step 1: Read and update setUpClass**

Replace manual setup with helper call.

**Step 2: Run tests**

Run: `pytest tests/test_voice.py::TestRenamePromptEndpoint -v --tb=short`
Expected: All tests pass

---

## Task 11: Update TestPreviewPromptEndpoint setUpClass

**Files:**
- Modify: `tests/test_voice.py:2340-2360` (approximately)

**Step 1: Read and update setUpClass**

Replace manual setup with helper call.

**Step 2: Run tests**

Run: `pytest tests/test_voice.py::TestPreviewPromptEndpoint -v --tb=short`
Expected: All tests pass

---

## Task 12: Update TestPromptDetailsEndpoint setUpClass

**Files:**
- Modify: `tests/test_voice.py:2400-2420` (approximately)

**Step 1: Read and update setUpClass**

Replace manual setup with helper call.

**Step 2: Run tests**

Run: `pytest tests/test_voice.py::TestPromptDetailsEndpoint -v --tb=short`
Expected: All tests pass

---

## Task 13: Update TestUnloadModelEndpoint setUpClass

**Files:**
- Modify: `tests/test_voice.py:2670-2690` (approximately)

**Step 1: Read and update setUpClass**

Replace manual setup with helper call.

**Step 2: Run tests**

Run: `pytest tests/test_voice.py::TestUnloadModelEndpoint -v --tb=short`
Expected: All tests pass

---

## Task 14: Update TestUpdateStartupConfigEndpoint setUpClass

**Files:**
- Modify: `tests/test_voice.py:2790-2810` (approximately)

**Step 1: Read and update setUpClass**

Replace manual setup with helper call.

**Step 2: Run tests**

Run: `pytest tests/test_voice.py::TestUpdateStartupConfigEndpoint -v --tb=short`
Expected: All tests pass

---

## Task 15: Update TestModelsEndpointEnhanced setUpClass

**Files:**
- Modify: `tests/test_voice.py:2880-2900` (approximately)

**Step 1: Read and update setUpClass**

Replace manual setup with helper call.

**Step 2: Run tests**

Run: `pytest tests/test_voice.py::TestModelsEndpointEnhanced -v --tb=short`
Expected: All tests pass

---

## Task 16: Full verification and commit

**Files:**
- Test: `tests/test_voice.py`

**Step 1: Run all test_voice.py tests**

Run: `pytest tests/test_voice.py --tb=no -q`
Expected: Significantly fewer failures (most tests now pass)

**Step 2: Run specific server test classes**

Run: `pytest tests/test_voice.py -k "Server or Endpoint" -v --tb=no`
Expected: All server endpoint tests pass

**Step 3: Git commit**

```bash
git add tests/test_voice.py
git commit -m "fix: update remaining test classes to use _setup_fastapi_app_state helper

- Update 14 test classes with setUpClass to use helper function
- Ensures consistent FastAPI app.state initialization across all tests
- Fixes AttributeError: 'State' object has no attribute 'request_queue' and similar"
```

---

## Verification Summary

After completing all tasks:

```bash
# Run all server-related tests
pytest tests/test_voice.py -k "Server or Endpoint" -v

# Expected: All server/endpoint tests pass (0 failures)
```
