# Tokenizer Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace naïve regex sentence-splitting with pySBD, add always-on text normalization (numbers, dates, abbreviations) via num2words, and introduce token-aware chunking on the torch backend so chunks never silently exceed the model's context window.

**Architecture:** Three components added to `qwen3_tts/core/engine.py`: `_map_language()` maps language names to ISO codes; `_normalize_text()` expands numbers/dates/abbreviations before chunking; `_split_text()` gains pySBD sentence splitting and an optional `tokenizer`/`max_tokens` fast-path.  All imports are lazy (inside functions). MLX backend falls back to char-based chunking unchanged.

**Tech Stack:** pySBD≥0.3.4 (sentence boundary detection), num2words≥0.5.13 (number expansion), existing HuggingFace tokenizer already attached to the torch model at `model.tokenizer`.

---

### Task 1: Save design doc and create docs directory structure

**Files:**
- Create: `docs/plans/2026-02-20-tokenizer-improvements-design.md` (the design doc from the previous session)

**Step 1: Confirm docs/plans directory exists**

```bash
ls docs/plans/
```
Expected: directory listed without error.

**Step 2: Commit the design doc**

```bash
git add docs/plans/2026-02-20-tokenizer-improvements-design.md
git commit -m "docs: add tokenizer improvements design doc"
```

---

### Task 2: Add pySBD and num2words to all dependency files

**Files:**
- Modify: `requirements-mlx.txt`
- Modify: `requirements-cuda.txt`
- Modify: `pyproject.toml`

**Step 1: Verify current dependency files**

```bash
grep -n "pySBD\|num2words" requirements-mlx.txt requirements-cuda.txt pyproject.toml
```
Expected: no matches (deps not yet present).

**Step 2: Add to requirements-mlx.txt**

In `requirements-mlx.txt`, append after the `click>=8.0` line:
```
pySBD>=0.3.4
num2words>=0.5.13
```

**Step 3: Add to requirements-cuda.txt**

In `requirements-cuda.txt`, append after the `click>=8.0` line:
```
pySBD>=0.3.4
num2words>=0.5.13
```

**Step 4: Add to pyproject.toml**

Change the `dependencies` line in `[project]`:
```toml
dependencies = ["click>=8.0", "pySBD>=0.3.4", "num2words>=0.5.13"]
```

**Step 5: Verify changes look correct**

```bash
grep -n "pySBD\|num2words" requirements-mlx.txt requirements-cuda.txt pyproject.toml
```
Expected: 2 lines each in both requirements files, 1 line in pyproject.toml.

**Step 6: Install the new deps in the active conda environment**

```bash
pip install "pySBD>=0.3.4" "num2words>=0.5.13"
python -c "import pysbd; import num2words; print('OK')"
```
Expected: `OK`.

**Step 7: Commit**

```bash
git add requirements-mlx.txt requirements-cuda.txt pyproject.toml
git commit -m "deps: add pySBD and num2words for sentence splitting and text normalization"
```

---

### Task 3: Add `_map_language()` helper + tests

**Files:**
- Modify: `qwen3_tts/core/engine.py` (add after `_CLAUSE_SPLIT_RE` at line 69)
- Modify: `tests/test_audio_utils.py` (add new test class)

**Step 1: Write the failing tests**

Add this class to `tests/test_audio_utils.py` before `if __name__ == "__main__":`:

```python
class TestMapLanguage(unittest.TestCase):
    """Tests for _map_language() language-code helper."""

    def test_english_returns_en(self):
        from voice_engine import _map_language
        self.assertEqual(_map_language("English"), "en")

    def test_case_insensitive(self):
        from voice_engine import _map_language
        self.assertEqual(_map_language("ENGLISH"), "en")
        self.assertEqual(_map_language("english"), "en")

    def test_spanish(self):
        from voice_engine import _map_language
        self.assertEqual(_map_language("Spanish"), "es")

    def test_french(self):
        from voice_engine import _map_language
        self.assertEqual(_map_language("French"), "fr")

    def test_german(self):
        from voice_engine import _map_language
        self.assertEqual(_map_language("German"), "de")

    def test_japanese(self):
        from voice_engine import _map_language
        self.assertEqual(_map_language("Japanese"), "ja")

    def test_chinese(self):
        from voice_engine import _map_language
        self.assertEqual(_map_language("Chinese"), "zh")

    def test_unknown_language_returns_en(self):
        from voice_engine import _map_language
        self.assertEqual(_map_language("Klingon"), "en")

    def test_none_returns_en(self):
        from voice_engine import _map_language
        self.assertEqual(_map_language(None), "en")

    def test_empty_string_returns_en(self):
        from voice_engine import _map_language
        self.assertEqual(_map_language(""), "en")
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_audio_utils.py::TestMapLanguage -v
```
Expected: FAIL with `ImportError: cannot import name '_map_language'`.

**Step 3: Add `_map_language()` to engine.py**

In `qwen3_tts/core/engine.py`, insert after line 69 (after `_CLAUSE_SPLIT_RE`):

```python
# ---------------------------------------------------------------------------
# Language mapping
# ---------------------------------------------------------------------------

_PYSBD_LANG_MAP = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "japanese": "ja",
    "chinese": "zh",
    "russian": "ru",
    "dutch": "nl",
    "polish": "pl",
}


def _map_language(language):
    """Map a full language name to a pySBD/num2words 2-letter ISO code.

    Returns 'en' for any unrecognized or missing language.
    """
    return _PYSBD_LANG_MAP.get((language or "english").lower(), "en")
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_audio_utils.py::TestMapLanguage -v
```
Expected: all 10 tests PASS.

**Step 5: Commit**

```bash
git add qwen3_tts/core/engine.py tests/test_audio_utils.py
git commit -m "feat: add _map_language() helper for pySBD and num2words language codes"
```

---

### Task 4: Add `_normalize_text()` + tests

**Files:**
- Create: `tests/test_text_processing.py`
- Modify: `qwen3_tts/core/engine.py` (add after `_map_language()`)

**Step 1: Write the failing tests**

Create `tests/test_text_processing.py`:

```python
#!/usr/bin/env python3
"""Tests for _normalize_text() text normalization in engine.py.

No GPU, models, or running server required.
Tests require: pySBD>=0.3.4, num2words>=0.5.13
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNormalizeText(unittest.TestCase):
    """Tests for _normalize_text() text normalization."""

    def setUp(self):
        try:
            import pysbd  # noqa: F401
            import num2words  # noqa: F401
        except ImportError:
            self.skipTest("pySBD or num2words not installed")

    def _normalize(self, text, language="English"):
        from voice_engine import _normalize_text
        return _normalize_text(text, language)

    # --- Cardinal numbers ---

    def test_cardinal_integer(self):
        result = self._normalize("There are 42 items.")
        self.assertIn("forty-two", result)
        self.assertNotIn("42", result)

    def test_cardinal_zero(self):
        result = self._normalize("I have 0 apples.")
        self.assertIn("zero", result)

    def test_cardinal_large(self):
        result = self._normalize("Population is 1000000.")
        self.assertIn("one million", result)

    # --- Ordinal numbers ---

    def test_ordinal_3rd(self):
        result = self._normalize("It was the 3rd time.")
        self.assertIn("third", result)
        self.assertNotIn("3rd", result)

    def test_ordinal_21st(self):
        result = self._normalize("It was the 21st century.")
        self.assertIn("twenty-first", result)

    def test_ordinal_2nd(self):
        result = self._normalize("2nd place.")
        self.assertIn("second", result)

    def test_ordinal_4th(self):
        result = self._normalize("4th floor.")
        self.assertIn("fourth", result)

    # --- Abbreviations ---

    def test_abbrev_dr(self):
        result = self._normalize("Dr. Smith arrived.")
        self.assertIn("Doctor", result)
        self.assertNotIn("Dr.", result)

    def test_abbrev_mr(self):
        result = self._normalize("Mr. Jones called.")
        self.assertIn("Mister", result)

    def test_abbrev_mrs(self):
        result = self._normalize("Mrs. Brown is here.")
        self.assertIn("Missus", result)

    def test_abbrev_prof(self):
        result = self._normalize("Prof. Lee gave a lecture.")
        self.assertIn("Professor", result)

    def test_abbrev_eg(self):
        result = self._normalize("e.g. apples and oranges.")
        self.assertIn("for example", result)

    def test_abbrev_ie(self):
        result = self._normalize("i.e. the main point.")
        self.assertIn("that is", result)

    def test_abbrev_vs(self):
        result = self._normalize("cats vs. dogs.")
        self.assertIn("versus", result)

    def test_abbrev_etc(self):
        result = self._normalize("fruits, vegetables, etc.")
        self.assertIn("et cetera", result)

    def test_abbrev_approx(self):
        result = self._normalize("approx. 5 miles away.")
        self.assertIn("approximately", result)

    def test_abbrev_yrs(self):
        result = self._normalize("She is 30 yrs old.")
        self.assertIn("years", result)

    # --- Currencies ---

    def test_currency_dollars(self):
        result = self._normalize("It costs $5.")
        self.assertIn("dollar", result)
        self.assertNotIn("$5", result)

    def test_currency_euros(self):
        result = self._normalize("Price is €10.")
        self.assertIn("euro", result)

    def test_currency_with_cents(self):
        result = self._normalize("Pay $5.00 now.")
        self.assertIn("dollar", result)

    # --- URLs / emails ---

    def test_email_expansion(self):
        result = self._normalize("Email user@example.com for info.")
        self.assertIn("at", result)
        self.assertNotIn("@", result)

    def test_url_expansion(self):
        result = self._normalize("Visit https://example.com today.")
        self.assertIn("dot com", result)

    # --- Error resilience ---

    def test_empty_string(self):
        result = self._normalize("")
        self.assertEqual(result, "")

    def test_plain_text_unchanged(self):
        """Text with no special tokens should be returned nearly unchanged."""
        text = "Hello world, how are you today?"
        result = self._normalize(text)
        # Core words preserved
        self.assertIn("Hello", result)
        self.assertIn("world", result)

    def test_normalization_never_raises(self):
        """Normalization should not raise even on unusual input."""
        from voice_engine import _normalize_text
        unusual = "!@#$%^&*() 123 Dr. 0th $0 €€€ http://x"
        try:
            _normalize_text(unusual, "English")
        except Exception as e:
            self.fail(f"_normalize_text raised {e}")


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_text_processing.py -v
```
Expected: FAIL with `ImportError: cannot import name '_normalize_text'`.

**Step 3: Add `_normalize_text()` to engine.py**

In `qwen3_tts/core/engine.py`, insert after `_map_language()` (before the `# MPS bfloat16 safety patch` section):

```python
# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

# Abbreviation expansion table — order matters (longer first to avoid partial matches)
_ABBREV_TABLE = [
    (r'\bProf\.', "Professor"),
    (r'\bDr\.', "Doctor"),
    (r'\bMrs\.', "Missus"),
    (r'\bMr\.', "Mister"),
    (r'\be\.g\.', "for example"),
    (r'\bi\.e\.', "that is"),
    (r'\bvs\.', "versus"),
    (r'\betc\.', "et cetera"),
    (r'\bapprox\.', "approximately"),
    (r'\byrs\.', "years"),
    (r'\byrs\b', "years"),
]

# ISO 4217 currency symbols → (singular, plural) names
_CURRENCY_MAP = {
    "$": ("dollar", "dollars"),
    "€": ("euro", "euros"),
    "£": ("pound", "pounds"),
    "¥": ("yen", "yen"),
}


def _normalize_text(text, language="English"):
    """Normalize text for TTS: expand numbers, dates, abbreviations, and URLs.

    Called before chunking in run_inference(). All normalization steps are
    wrapped in try/except so a failure never blocks generation.

    Args:
        text: Raw input text.
        language: Language name string (e.g. "English").

    Returns:
        Normalized text string.
    """
    import re as _re

    if not text:
        return text

    lang = _map_language(language)

    try:
        from num2words import num2words as _n2w
    except ImportError:
        _n2w = None

    # 1. Emails: user@example.com → "user at example dot com"
    try:
        def _expand_email(m):
            addr = m.group()
            local, _, domain = addr.partition("@")
            domain_parts = domain.split(".")
            return local + " at " + " dot ".join(domain_parts)
        text = _re.sub(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b',
                       _expand_email, text)
    except Exception:
        pass

    # 2. URLs: https://example.com → "example dot com"
    try:
        def _expand_url(m):
            url = m.group()
            # Strip scheme
            url = _re.sub(r'^https?://', '', url)
            url = _re.sub(r'^www\.', '', url)
            # Replace dots and slashes
            url = url.replace(".", " dot ").replace("/", " slash ").rstrip()
            return url
        text = _re.sub(r'https?://\S+', _expand_url, text)
    except Exception:
        pass

    # 3. Currencies: $5.00 → "five dollars", €10 → "ten euros"
    try:
        def _expand_currency(m):
            symbol = m.group(1)
            amount_str = m.group(2)
            singular, plural = _CURRENCY_MAP.get(symbol, ("unit", "units"))
            try:
                amount = float(amount_str)
                whole = int(amount)
                cents = round((amount - whole) * 100)
                if _n2w:
                    words = _n2w(whole, lang=lang)
                else:
                    words = str(whole)
                label = singular if whole == 1 else plural
                if cents:
                    return f"{words} {label} and {cents} cents"
                return f"{words} {label}"
            except Exception:
                return m.group()
        symbols_pattern = "[" + _re.escape("".join(_CURRENCY_MAP.keys())) + "]"
        text = _re.sub(rf'({symbols_pattern})(\d+(?:\.\d+)?)', _expand_currency, text)
    except Exception:
        pass

    # 4. Ordinals: 3rd, 21st, 2nd, 4th
    try:
        def _expand_ordinal(m):
            n_str = m.group(1)
            try:
                n = int(n_str)
                if _n2w:
                    return _n2w(n, lang=lang, to="ordinal")
                return m.group()
            except Exception:
                return m.group()
        text = _re.sub(r'\b(\d+)(?:st|nd|rd|th)\b', _expand_ordinal, text)
    except Exception:
        pass

    # 5. Dates: YYYY-MM-DD and MM/DD/YYYY
    try:
        def _expand_date_iso(m):
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                import calendar
                month_name = calendar.month_name[month]
                if _n2w:
                    day_word = _n2w(day, lang=lang, to="ordinal")
                    year_word = _n2w(year, lang=lang)
                else:
                    day_word = str(day)
                    year_word = str(year)
                return f"{month_name} {day_word}, {year_word}"
            except Exception:
                return m.group()
        text = _re.sub(r'\b(\d{4})-(\d{2})-(\d{2})\b', _expand_date_iso, text)
    except Exception:
        pass

    try:
        def _expand_date_us(m):
            month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                import calendar
                month_name = calendar.month_name[month]
                if _n2w:
                    day_word = _n2w(day, lang=lang, to="ordinal")
                    year_word = _n2w(year, lang=lang)
                else:
                    day_word = str(day)
                    year_word = str(year)
                return f"{month_name} {day_word}, {year_word}"
            except Exception:
                return m.group()
        text = _re.sub(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', _expand_date_us, text)
    except Exception:
        pass

    # 6. Abbreviations (applied before cardinal expansion)
    try:
        import re as _re2
        for pattern, replacement in _ABBREV_TABLE:
            text = _re2.sub(pattern, replacement, text)
    except Exception:
        pass

    # 7. Cardinals: standalone integers
    if _n2w:
        try:
            def _expand_cardinal(m):
                try:
                    n = int(m.group())
                    return _n2w(n, lang=lang)
                except Exception:
                    return m.group()
            # Only expand bare integers not preceded/followed by letters (avoids version numbers like 3.14)
            text = _re.sub(r'(?<![.\w])\b\d+\b(?![.\w])', _expand_cardinal, text)
        except Exception:
            pass

    return text
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_text_processing.py -v
```
Expected: all tests PASS (or SKIP if pySBD/num2words not installed).

**Step 5: Run existing tests to check for regressions**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```
Expected: same pass count as before (431).

**Step 6: Commit**

```bash
git add qwen3_tts/core/engine.py tests/test_text_processing.py
git commit -m "feat: add _normalize_text() for number/date/abbreviation expansion"
```

---

### Task 5: Replace regex sentence splitting with pySBD in `_split_text()`

**Files:**
- Modify: `qwen3_tts/core/engine.py` (lines 67, 93 — `_SENTENCE_SPLIT_RE` and its call)
- Modify: `tests/test_audio_utils.py` (update `TestTextChunkingEdgeCases`)

**Step 1: Add a pySBD-specific test that would fail with the old regex**

Add to `TestTextChunkingEdgeCases` in `tests/test_audio_utils.py`:

```python
def test_split_does_not_split_dr_smith(self):
    """pySBD should NOT split 'Dr. Smith' mid-sentence (old regex did)."""
    try:
        import pysbd  # noqa: F401
    except ImportError:
        self.skipTest("pySBD not installed")
    from voice_engine import _split_text
    # "Dr. Smith arrived." is ONE sentence — must stay as one chunk
    text = "Dr. Smith arrived. She was happy."
    result = _split_text(text, max_chars=500)
    # Both sentences fit in 500 chars — should be a single chunk
    self.assertEqual(len(result), 1)
    self.assertIn("Dr. Smith", result[0])

def test_split_does_not_split_decimal(self):
    """pySBD should NOT split on '3.14' as a sentence boundary."""
    try:
        import pysbd  # noqa: F401
    except ImportError:
        self.skipTest("pySBD not installed")
    from voice_engine import _split_text
    text = "Pi is 3.14 approximately. That is a fact."
    result = _split_text(text, max_chars=500)
    # Should be a single chunk (fits in 500 chars)
    self.assertEqual(len(result), 1)
```

**Step 2: Run to verify the new tests fail with current regex splitting**

```bash
python -m pytest tests/test_audio_utils.py::TestTextChunkingEdgeCases::test_split_does_not_split_dr_smith tests/test_audio_utils.py::TestTextChunkingEdgeCases::test_split_does_not_split_decimal -v
```
Expected: FAIL — the regex does split on "Dr." and "3.14".

**Step 3: Modify `_split_text()` in engine.py**

Replace the `_SENTENCE_SPLIT_RE` usage. In `qwen3_tts/core/engine.py`:

1. Remove line 67 (`_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')`). Keep `_PARAGRAPH_SPLIT_RE` and `_CLAUSE_SPLIT_RE`.

2. Change the function signature at line 75 and its docstring to:
```python
def _split_text(text, max_chars=500, tokenizer=None, max_tokens=None):
    """Split text into chunks at sentence boundaries.

    Uses pySBD for accurate sentence segmentation when available (falling back
    to a simple regex). If tokenizer and max_tokens are both provided, chunk
    sizes are measured in tokens rather than characters (torch backend only).

    Args:
        text: Input text to split.
        max_chars: Maximum characters per chunk (used when tokenizer is absent).
        tokenizer: Optional HuggingFace tokenizer for token-based sizing.
        max_tokens: Maximum tokens per chunk (used only when tokenizer is provided).

    Returns:
        List of text chunks. Returns [text] unchanged if it fits in one chunk.
    """
```

3. Replace the `len(text) <= max_chars` early-return and `_SENTENCE_SPLIT_RE.split(text)` call:

```python
    text = text.strip()

    def _measure(chunk):
        """Return size of chunk in the active unit (tokens or chars)."""
        if tokenizer is not None and max_tokens is not None:
            return len(tokenizer.encode(chunk, add_special_tokens=False))
        return len(chunk)

    limit = max_tokens if (tokenizer is not None and max_tokens is not None) else max_chars

    if _measure(text) <= limit:
        return [text]

    # Sentence splitting: pySBD when available, regex fallback
    try:
        import pysbd
        # Language is not available here; default to 'en'.
        # Callers that know the language can pass pre-segmented sentences.
        segmenter = pysbd.Segmenter(language="en", clean=False)
        sentences = segmenter.segment(text)
    except ImportError:
        sentences = re.split(r'(?<=[.!?])\s+', text)
```

4. In the loop body, replace every `len(current_chunk)` and `len(sentence)` / `len(clause)` / `len(word)` comparison with `_measure(...)`, and every `> max_chars` with `> limit`.  The full modified loop becomes:

```python
    # Also split on paragraph breaks (multiple newlines)
    expanded = []
    for s in sentences:
        parts = _PARAGRAPH_SPLIT_RE.split(s)
        expanded.extend(p.strip() for p in parts if p.strip())
    sentences = expanded

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if current_chunk and _measure(current_chunk) + 1 + _measure(sentence) > limit:
            chunks.append(current_chunk.strip())
            current_chunk = ""

        if _measure(sentence) > limit:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            clauses = _CLAUSE_SPLIT_RE.split(sentence)

            for clause in clauses:
                if current_chunk and _measure(current_chunk) + 1 + _measure(clause) > limit:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                if _measure(clause) > limit:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    words = clause.split()
                    for word in words:
                        if current_chunk and _measure(current_chunk) + 1 + _measure(word) > limit:
                            chunks.append(current_chunk.strip())
                            current_chunk = ""
                        current_chunk = (current_chunk + " " + word).strip() if current_chunk else word
                else:
                    current_chunk = (current_chunk + " " + clause).strip() if current_chunk else clause
        else:
            current_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]
```

**Step 4: Run the new pySBD tests**

```bash
python -m pytest tests/test_audio_utils.py::TestTextChunkingEdgeCases -v
```
Expected: all tests PASS including the two new ones.

**Step 5: Run the full test suite**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```
Expected: same pass count as before (no regressions).

**Step 6: Commit**

```bash
git add qwen3_tts/core/engine.py tests/test_audio_utils.py
git commit -m "feat: replace regex sentence splitting with pySBD in _split_text()"
```

---

### Task 6: Add `_get_max_chunk_tokens()` + config key + tests

**Files:**
- Modify: `config.json` (add `generation.max_chunk_tokens`)
- Modify: `qwen3_tts/core/engine.py` (add `_get_max_chunk_tokens()` after `_get_max_chunk_chars()`)
- Modify: `tests/test_core_infra.py` (add test for the new helper)

**Step 1: Write the failing test**

Search for existing config-reading tests in `tests/test_core_infra.py` and append a new class (or add to an existing config test class):

```python
class TestGetMaxChunkTokens(unittest.TestCase):
    """Tests for _get_max_chunk_tokens() config reader."""

    def test_returns_config_value(self):
        from unittest.mock import patch
        from voice_engine import _get_max_chunk_tokens
        fake_config = {"generation": {"max_chunk_tokens": 150}}
        with patch("qwen3_tts.core.engine.load_config", return_value=fake_config):
            self.assertEqual(_get_max_chunk_tokens(), 150)

    def test_returns_default_when_key_missing(self):
        from unittest.mock import patch
        from voice_engine import _get_max_chunk_tokens
        with patch("qwen3_tts.core.engine.load_config", return_value={}):
            self.assertEqual(_get_max_chunk_tokens(), 200)

    def test_returns_default_on_exception(self):
        from unittest.mock import patch
        from voice_engine import _get_max_chunk_tokens
        with patch("qwen3_tts.core.engine.load_config", side_effect=RuntimeError("disk error")):
            self.assertEqual(_get_max_chunk_tokens(), 200)
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_core_infra.py::TestGetMaxChunkTokens -v
```
Expected: FAIL with `ImportError: cannot import name '_get_max_chunk_tokens'`.

**Step 3: Add `max_chunk_tokens` to config.json**

In `config.json`, inside the `"generation"` object, add after `"max_chunk_chars": 500`:
```json
"max_chunk_tokens": 200,
```

**Step 4: Add `_get_max_chunk_tokens()` to engine.py**

Insert after `_get_max_chunk_chars()` (engine.py line 892–898):

```python
def _get_max_chunk_tokens():
    """Read max_chunk_tokens from config, defaulting to 200."""
    try:
        config = load_config()
        return config.get("generation", {}).get("max_chunk_tokens", 200)
    except Exception:
        return 200
```

**Step 5: Run test to verify it passes**

```bash
python -m pytest tests/test_core_infra.py::TestGetMaxChunkTokens -v
```
Expected: all 3 tests PASS.

**Step 6: Run full suite**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -5
```
Expected: no regressions.

**Step 7: Commit**

```bash
git add qwen3_tts/core/engine.py config.json tests/test_core_infra.py
git commit -m "feat: add max_chunk_tokens config key and _get_max_chunk_tokens() helper"
```

---

### Task 7: Wire `_normalize_text()` and token-aware chunking into `run_inference()`

**Files:**
- Modify: `qwen3_tts/core/engine.py` — `run_inference()` and `run_inference_streaming()`

**Step 1: Write integration tests**

Add to `tests/test_text_processing.py`:

```python
class TestNormalizeIntegration(unittest.TestCase):
    """Verify _normalize_text is called within run_inference via mock."""

    def setUp(self):
        try:
            import num2words  # noqa: F401
        except ImportError:
            self.skipTest("num2words not installed")

    def test_normalize_called_before_inference(self):
        """run_inference should normalize text; '42' becomes 'forty-two'."""
        from unittest.mock import patch, MagicMock
        from voice_engine import run_inference

        captured = []

        def fake_run_single(model, text, *args, **kwargs):
            captured.append(text)
            return (MagicMock(), 24000)

        fake_model = MagicMock()
        fake_model.tokenizer = None  # no tokenizer → char-based chunking

        gen_params = {"temperature": 0.7, "top_k": 50, "top_p": 0.95,
                      "repetition_penalty": 1.05, "max_new_tokens": 2048}

        with patch("qwen3_tts.core.engine._run_inference_single", side_effect=fake_run_single):
            run_inference(
                fake_model, "There are 42 items.", "clone",
                gen_params, language="English",
                voice_prompt=MagicMock(),
                max_chunk_chars=0,  # disable chunking, single chunk
            )

        self.assertTrue(len(captured) >= 1)
        joined = " ".join(captured)
        self.assertIn("forty-two", joined)
        self.assertNotIn("42", joined)
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_text_processing.py::TestNormalizeIntegration -v
```
Expected: FAIL — "42" still in captured text, "forty-two" not present.

**Step 3: Modify `run_inference()` in engine.py**

At the top of `run_inference()`, right after the `if max_chunk_chars is None:` block, add normalization and token-aware chunking:

Current (lines 927–934):
```python
    if max_chunk_chars is None:
        max_chunk_chars = _get_max_chunk_chars()

    # Split into chunks if text is long enough
    if max_chunk_chars > 0 and len(text) > max_chunk_chars:
        chunks = _split_text(text, max_chars=max_chunk_chars)
    else:
        chunks = [text]
```

Replace with:
```python
    if max_chunk_chars is None:
        max_chunk_chars = _get_max_chunk_chars()

    # Normalize text (expand numbers, dates, abbreviations) before chunking
    text = _normalize_text(text, language)

    # Resolve tokenizer for token-aware chunking (torch backend only)
    tokenizer = getattr(model, "tokenizer", None)
    max_tokens = _get_max_chunk_tokens() if tokenizer is not None else None

    # Split into chunks if text is long enough
    if tokenizer is not None and max_tokens is not None:
        # Token-aware path: measure by tokens
        from qwen3_tts.core.engine import _split_text as _st  # local ref avoids confusion
        # Inline check using tokenizer
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count > max_tokens:
            chunks = _split_text(text, max_chars=max_chunk_chars,
                                 tokenizer=tokenizer, max_tokens=max_tokens)
        else:
            chunks = [text]
    elif max_chunk_chars > 0 and len(text) > max_chunk_chars:
        chunks = _split_text(text, max_chars=max_chunk_chars)
    else:
        chunks = [text]
```

**Note:** The `from qwen3_tts.core.engine import _split_text as _st` line is unnecessary — `_split_text` is already in scope. Remove that line and use `_split_text` directly.  Correct version:

```python
    if max_chunk_chars is None:
        max_chunk_chars = _get_max_chunk_chars()

    # Normalize text (expand numbers, dates, abbreviations) before chunking
    text = _normalize_text(text, language)

    # Resolve tokenizer for token-aware chunking (torch backend only)
    tokenizer = getattr(model, "tokenizer", None)
    max_tokens = _get_max_chunk_tokens() if tokenizer is not None else None

    # Split into chunks
    if tokenizer is not None and max_tokens is not None:
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count > max_tokens:
            chunks = _split_text(text, max_chars=max_chunk_chars,
                                 tokenizer=tokenizer, max_tokens=max_tokens)
        else:
            chunks = [text]
    elif max_chunk_chars > 0 and len(text) > max_chunk_chars:
        chunks = _split_text(text, max_chars=max_chunk_chars)
    else:
        chunks = [text]
```

**Step 4: Apply the same normalization to `run_inference_streaming()` (torch path)**

In `run_inference_streaming()` at the `else:` branch (torch fallback, line ~1077), add normalization and token-aware chunking mirror to the non-streaming version:

Current:
```python
        if max_chunk_chars is None:
            max_chunk_chars = _get_max_chunk_chars()

        if max_chunk_chars > 0 and len(text) > max_chunk_chars:
            chunks = _split_text(text, max_chars=max_chunk_chars)
        else:
            chunks = [text]
```

Replace with:
```python
        if max_chunk_chars is None:
            max_chunk_chars = _get_max_chunk_chars()

        text = _normalize_text(text, language)
        tokenizer = getattr(model, "tokenizer", None)
        max_tokens = _get_max_chunk_tokens() if tokenizer is not None else None

        if tokenizer is not None and max_tokens is not None:
            token_count = len(tokenizer.encode(text, add_special_tokens=False))
            if token_count > max_tokens:
                chunks = _split_text(text, max_chars=max_chunk_chars,
                                     tokenizer=tokenizer, max_tokens=max_tokens)
            else:
                chunks = [text]
        elif max_chunk_chars > 0 and len(text) > max_chunk_chars:
            chunks = _split_text(text, max_chars=max_chunk_chars)
        else:
            chunks = [text]
```

**Step 5: Run integration test**

```bash
python -m pytest tests/test_text_processing.py::TestNormalizeIntegration -v
```
Expected: PASS.

**Step 6: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -10
```
Expected: no regressions, same or higher pass count.

**Step 7: Commit**

```bash
git add qwen3_tts/core/engine.py tests/test_text_processing.py
git commit -m "feat: wire _normalize_text() and token-aware chunking into run_inference()"
```

---

### Task 8: Update CLAUDE.md and README.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Step 1: Add `max_chunk_tokens` row to CLAUDE.md config table**

In the `### Key Settings` table in CLAUDE.md, add a row after the `generation.max_chunk_chars` row (or after the existing rows):

```markdown
| `generation.max_chunk_tokens` | positive integer | `200` (torch backend only; char-based when absent) |
```

**Step 2: Add Text Processing Roadmap section to CLAUDE.md**

After the `### 2026-02-20 — Bug fixes: server startup timeout + Colab flash-attn stall` section, add:

```markdown
### 2026-02-20 — Tokenizer improvements: pySBD + num2words + token-aware chunking

**Goal:** Higher audio quality, always-on text normalization, and reliable chunking.

**Changes:**
- `qwen3_tts/core/engine.py`: Added `_map_language()`, `_normalize_text()`, `_get_max_chunk_tokens()`; modified `_split_text()` (pySBD + token-aware); modified `run_inference()` and `run_inference_streaming()` to normalize text and use token-aware chunking on torch backend.
- `config.json`: Added `generation.max_chunk_tokens: 200`.
- `requirements-mlx.txt`, `requirements-cuda.txt`, `pyproject.toml`: Added `pySBD>=0.3.4`, `num2words>=0.5.13`.
- New test file: `tests/test_text_processing.py`.

## Text Processing Roadmap

**Current (implemented):** pySBD sentence splitting, num2words text normalization, token-aware chunking (torch backend).

**Future options (not yet implemented):**
- **NLTK punkt tokenizer** — Moderate-weight alternative to pySBD; requires punkt data download at first use. Good for multi-language academic text.
- **NVIDIA NeMo text processing** — Production-grade normalization covering dates, times, measures, addresses, financial data. ~500MB+ in new dependencies; suitable for high-volume or broadcast-quality TTS.
```

**Step 3: Add roadmap section to README.md**

Before the `## License` section, add:

```markdown
## Text Processing Roadmap

**Current:** pySBD sentence splitting prevents false breaks on "Dr. Smith" and decimals. num2words expands numbers, dates, currencies, ordinals, and common abbreviations before synthesis. Token-aware chunking (torch backend) ensures chunks never silently exceed the model's context window.

**Future options (not yet implemented):**
- **NLTK punkt tokenizer** — Moderate-weight alternative to pySBD; requires punkt data download at first use. Good for multi-language academic text.
- **NVIDIA NeMo text processing** — Production-grade normalization covering dates, times, measures, addresses, financial data. ~500MB+ in new dependencies; suitable for high-volume or broadcast-quality TTS.
```

**Step 4: Run tests to make sure docs changes didn't break anything**

```bash
python -m pytest tests/ --tb=short -q 2>&1 | tail -5
```
Expected: clean.

**Step 5: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: update CLAUDE.md and README.md with tokenizer improvements and roadmap"
```

---

### Task 9: Write design doc and save to docs/plans/

**Files:**
- Create: `docs/plans/2026-02-20-tokenizer-improvements-design.md`

**Step 1: Save the design document**

The content of this file is the design document that was used to create this plan. Save it at the path above, then commit:

```bash
git add docs/plans/
git commit -m "docs: add tokenizer improvements design doc"
```

---

### Task 10: Final verification

**Step 1: Run complete test suite**

```bash
python -m pytest tests/ -v 2>&1 | tail -20
```
Expected: all 431+ tests PASS (new tests added by this work bring the total higher).

**Step 2: Quick smoke test — manual normalization check**

```bash
python -c "
from qwen3_tts.core.engine import _normalize_text
result = _normalize_text('The U.S.A. spent \$42 on 3 items for Dr. Smith on 2026-02-20.', 'English')
print(result)
"
```
Expected output contains: `forty-two dollars`, `three items`, `Doctor Smith`, `February`, `twentieth`.

**Step 3: Verify pySBD does not split on Dr.**

```bash
python -c "
from qwen3_tts.core.engine import _split_text
chunks = _split_text('Dr. Smith arrived. She was pleased.', max_chars=500)
print(len(chunks), chunks)
"
```
Expected: `1 ['Dr. Smith arrived. She was pleased.']`

**Step 4: Verify MLX backend still works (char-based fallback)**

```bash
python -c "
from qwen3_tts.core.engine import _split_text
# No tokenizer → char-based, same as before
chunks = _split_text('First sentence. Second sentence.', max_chars=20)
print(chunks)
"
```
Expected: 2 chunks.

**Step 5: Final commit if any stragglers**

If all clean:
```bash
git log --oneline -8
```
Confirm all feature commits are present.

---

## Summary of Commits

1. `deps: add pySBD and num2words for sentence splitting and text normalization`
2. `feat: add _map_language() helper for pySBD and num2words language codes`
3. `feat: add _normalize_text() for number/date/abbreviation expansion`
4. `feat: replace regex sentence splitting with pySBD in _split_text()`
5. `feat: add max_chunk_tokens config key and _get_max_chunk_tokens() helper`
6. `feat: wire _normalize_text() and token-aware chunking into run_inference()`
7. `docs: update CLAUDE.md and README.md with tokenizer improvements and roadmap`
8. `docs: add tokenizer improvements design doc`
