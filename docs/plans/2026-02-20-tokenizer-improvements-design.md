# Tokenizer Improvements Design — 2026-02-20

## Context

The Qwen3-TTS application currently uses character-based text chunking (500 char default) with a simple regex for sentence splitting. This causes three problems:

1. **Audio artifacts** — The regex `(?<=[.!?])\s+` splits on "Dr. Smith" and decimal points, causing unnatural sentence boundaries that the model receives as separate fragments.
2. **Unnormalized text** — Numbers like "42", dates like "2026-02-20", and abbreviations like "Dr." reach the model as-is, producing inconsistent or awkward speech.
3. **Silent truncation risk** — Chunks are sized in characters, not tokens, so long or dense text can silently exceed the model's token context window.

**User goals:** audio quality, text normalization (always-on), and token-aware chunking for reliability.

---

## Approach: pySBD + num2words + token-aware chunking

Two new lightweight dependencies: `pySBD>=0.3.4` and `num2words>=0.5.13`. No runtime downloads required.

---

## Component 1: `_normalize_text(text, language)` — new function in `engine.py`

Called at the start of `run_inference()`, before text reaches `_split_text()`. All imports are lazy (inside the function).

**Normalizes (in order):**
1. URLs/emails: `user@example.com` → `user at example dot com`, `https://example.com` → `example dot com`
2. Phone numbers: `555-1234`, `(800) 555-1234` → spelled digit-by-digit
3. Dates: `2026-02-20`, `02/20/2026`, `February 20th, 2026` → `February twentieth, two thousand and twenty-six`
4. Currencies: `$5.00`, `€10` → `five dollars`, `ten euros`
5. Ordinal numbers: `3rd`, `21st` → `third`, `twenty-first`
6. Cardinal numbers: `42` → `forty-two` (via `num2words(n, lang=lang)`)
7. Abbreviation lookup table:
   - `Dr.` → `Doctor`, `Mr.` → `Mister`, `Mrs.` → `Missus`, `Prof.` → `Professor`
   - `yrs` / `yrs.` → `years`, `approx.` → `approximately`, `etc.` → `et cetera`
   - `e.g.` → `for example`, `i.e.` → `that is`, `vs.` → `versus`

Language mapping: existing `language` string (e.g. `"English"`) is mapped to ISO code for `num2words` (`"en"`). Falls back to `"en"` on unmapped languages.

**Error handling:** Any normalization step that raises catches the exception and returns the original token — normalization never breaks generation.

---

## Component 2: pySBD sentence splitting — modify `_split_text()` in `engine.py`

**Current code (lines 67-69, 75-143):**
```python
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
# ...
sentences = _SENTENCE_SPLIT_RE.split(text)
```

**Change:** Replace the `_SENTENCE_SPLIT_RE.split(text)` call with pySBD:
```python
import pysbd  # lazy, inside function
segmenter = pysbd.Segmenter(language=_map_language(language), clean=False)
sentences = segmenter.segment(text)
```

`_SENTENCE_SPLIT_RE` constant can be removed; `_PARAGRAPH_SPLIT_RE` remains for the paragraph-split pass. The clause/word-level fallback logic downstream is unchanged.

A small `_map_language(language)` helper maps full language names to pySBD's 2-letter codes (`en`, `es`, `fr`, `de`, `it`, `ja`, `zh`, `ru`, `nl`, `pl`); falls back to `"en"`.

---

## Component 3: Token-aware chunking — modify `_split_text()` in `engine.py`

**`_split_text()` signature change:**
```python
def _split_text(text, max_chars=500, language="English", tokenizer=None, max_tokens=None):
```

When `tokenizer` and `max_tokens` are both provided, all `len(chunk)` comparisons are replaced by `len(tokenizer.encode(chunk, add_special_tokens=False))`. When tokenizer is absent (MLX backend), behavior is unchanged.

**Threading the tokenizer through:**
- `model.tokenizer` is already set in `load_model()` at engine.py:432-435
- `run_inference()` (torch path) passes `tokenizer=model.tokenizer` and `max_tokens=cfg_max_chunk_tokens` to `_split_text()`
- MLX `run_inference()` path omits `tokenizer` → falls back to char-based

**New config key:** `generation.max_chunk_tokens: 200` (200 tokens ≈ 500 chars for English prose).

**New helper:** `_get_max_chunk_tokens()` in `engine.py` reads `generation.max_chunk_tokens` from config (analogous to existing `_get_max_chunk_chars()` at line 892-898).

---

## Component 4: Config changes

**`config.json`:** Add under `"generation"`:
```json
"max_chunk_tokens": 200
```

**CLAUDE.md config table:** Add row for `generation.max_chunk_tokens`.

---

## Component 5: Dependencies

Add to `requirements-mlx.txt`, `requirements-cuda.txt`, and `pyproject.toml`:
```
pySBD>=0.3.4
num2words>=0.5.13
```

---

## Component 6: Roadmap documentation

Add `## Text Processing Roadmap` section to **CLAUDE.md** (under Recent Significant Changes) and **README.md**:

> **Current:** pySBD sentence splitting, num2words normalization, token-aware chunking (torch backend).
>
> **Future options (not yet implemented):**
> - **NLTK punkt tokenizer** — Moderate-weight alternative to pySBD; requires punkt data download at first use. Good for multi-language academic text.
> - **NVIDIA NeMo text processing** — Production-grade normalization covering dates, times, measures, addresses, financial data. ~500MB+ in new dependencies; suitable for high-volume or broadcast-quality TTS.

---

## Files Modified

| File | Change |
|------|--------|
| `qwen3_tts/core/engine.py` | Add `_normalize_text()`, `_map_language()`, `_get_max_chunk_tokens()`; modify `_split_text()`, `run_inference()` |
| `config.json` | Add `generation.max_chunk_tokens: 200` |
| `requirements-mlx.txt` | Add pySBD, num2words |
| `requirements-cuda.txt` | Add pySBD, num2words |
| `pyproject.toml` | Add pySBD, num2words to dependencies |
| `CLAUDE.md` | Update config table; add roadmap section |
| `README.md` | Add roadmap section |

---

## Verification

1. `python -m pytest tests/ -v` — all tests pass (no regressions)
2. Manual: `tts "The U.S.A. spent $42 on 3 items for Dr. Smith on 2026-02-20."` — output should say "The United States spent forty-two dollars on three items for Doctor Smith on February twentieth, two thousand and twenty-six"
3. Confirm token-aware chunking: generate a 2000+ char text and verify chunk sizes don't exceed ~200 tokens
4. Check MLX backend still works (falls back to char-based chunking, no pySBD language error)
