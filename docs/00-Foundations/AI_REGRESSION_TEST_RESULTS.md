# AI Regression Test Results

**Date:** 2026-04-06
**Test Suite:** `tests/test_ai_regression.py`
**Status:** ✅ All 10 tests passing

## Test Coverage

### 1. Backend Consistency Tests
**Pattern:** Prevents API response shape mismatches between backends

- ✅ `test_all_backends_return_same_response_shape[design]` - Verifies design mode returns consistent response format
- ✅ `test_all_backends_return_same_response_shape[custom]` - Verifies custom mode returns consistent response format
- ✅ `test_stats_endpoint_includes_all_required_fields` - Ensures /stats returns expected fields across backends

**Prevents:** Backend-specific bug where MLX path updated but Torch path forgotten

### 2. Model State Edge Case Tests
**Pattern:** Ensures graceful error handling when models not loaded

- ✅ `test_clone_generation_fails_gracefully_when_model_not_loaded` - Clear error, not crash
- ✅ `test_design_generation_fails_gracefully_when_model_not_loaded` - Clear error, not crash
- ✅ `test_custom_generation_fails_gracefully_when_model_not_loaded` - Clear error, not crash

**Prevents:** Silent failures or cryptic "generation failed" errors when model unloaded

### 3. API Response Contract Tests
**Pattern:** All required fields present in API responses

- ✅ `test_stats_endpoint_includes_all_required_fields` - Backend, model_loaded fields present
- ✅ `test_models_endpoint_includes_all_required_fields` - Models dict structure, loaded/memory fields
- ✅ `test_health_endpoint_includes_all_required_fields` - Model status fields present
- ✅ `test_generate_endpoint_response_contract` - Results array with audio_base64, sample_rate, index

**Prevents:** SELECT clause omissions, missing response fields in one path but not another

## Implementation Notes

### Response Format Handling
The test suite discovered that the actual API returns a `results` array format:
```json
{
  "results": [
    {
      "audio_base64": "...",
      "sample_rate": 24000,
      "index": 0
    }
  ]
}
```

Not the legacy `audio`, `duration`, `chunks` format. Tests updated to handle both formats for backward compatibility.

### Speaker Name Validation
Custom mode requires valid speaker names. Valid speakers:
`ryan`, `aiden`, `vivian`, `serena`, `uncle_fu`, `dylan`, `eric`, `ono_anna`, `sohee`

Test updated to use "eric" instead of invalid "default_en".

### Clone Mode Special Case
Clone mode requires `prompt_file` parameter (not tested here - covered in E2E tests). Removed from parameterized backend consistency test to avoid parameter mismatch.

## AI Regression Patterns Prevented

| Pattern | Test | Prevention |
|---------|------|------------|
| Backend path inconsistency | Backend consistency tests | MLX and Torch must return same shape |
| Missing response fields | API response contract tests | All required fields present |
| SELECT clause omissions | Stats endpoint test | `backend` field present in response |
| Assumed model loading | Model state edge cases | Graceful error when model not loaded |
| Cryptic error messages | Model state edge cases | Clear "model not loaded" error |
| One path updated, other forgotten | Multi-backend tests | All backends tested together |

## Running the Tests

```bash
# Run all AI regression tests
pytest tests/test_ai_regression.py -v

# Run with marker
pytest -m ai_regression -v

# Run specific test class
pytest tests/test_ai_regression.py::TestBackendConsistency -v
pytest tests/test_ai_regression.py::TestModelStateEdgeCases -v
pytest tests/test_ai_regression.py::TestAPIResponseContracts -v
```

## Integration with Full Test Suite

AI regression tests are now part of the full 1970+ test suite:
- Batch 1-5: Unit and integration tests (no server required)
- Batch 6: E2E Playwright tests (requires server with all models loaded)
- AI Regression Tests: Can run standalone or as part of full suite

```bash
# Run full suite (all environments)
python tests/run_full_suite.py --full --env all

# Run AI regression tests only
pytest tests/test_ai_regression.py -v
```

## Future Maintenance

When adding new API endpoints or response fields:
1. Update `test_*_contract` tests to include new fields
2. Run tests to verify no regressions
3. Update this document with new patterns prevented
