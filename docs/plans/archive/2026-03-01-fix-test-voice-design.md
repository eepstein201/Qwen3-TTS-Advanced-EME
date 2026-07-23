# Fix test_voice.py for FastAPI Compatibility

## Context

After Flask to FastAPI migration, 10 tests in `test_voice.py` fail due to:
- Error format mismatch: tests expect `{"error": "..."}` but FastAPI uses `{"detail": "..."}`
- Missing state attribute: `model_load_times` not initialized in test setup
- Mock patches targeting functions that are no longer module-level

## Design

### 1. Fix Error Response Assertions

Replace all `resp.json()["error"]` with `resp.json()["detail"]` to match FastAPI's native error format.

### 2. Initialize `model_load_times` in Test Setup

Add to all `setUpClass` methods that need it:
```python
app.state.model_load_times = {}
```

### 3. Rewrite Mock-Dependent Tests as Integration Tests

**Tests to rewrite:**
- `test_generate_generic_exception_returns_sanitized_detail`
- `test_load_model_exception_returns_sanitized_detail`

**Approach:** Trigger actual errors instead of mocking internal functions. This tests at the HTTP boundary and is more robust.

## Implementation Steps

1. Global replace `["error"]` with `["detail"]` in error assertions
2. Add `app.state.model_load_times = {}` to `TestServerValidation.setUpClass`
3. Rewrite the 2 mock-dependent tests to trigger real errors
4. Run `pytest tests/test_voice.py::TestServerValidation -v` to verify

## Files to Modify

- `tests/test_voice.py`

## Verification

```bash
pytest tests/test_voice.py::TestServerValidation -v
# Expect: 12 passing tests
```
