#!/usr/bin/env python3
"""Text processing utilities for TTS: normalization, chunking, and language mapping.

Base utility module — imports only from config.py, never from other engine submodules.
"""

import logging
import re

logger = logging.getLogger("tts.engine")

# ---------------------------------------------------------------------------
# Pre-compiled regex patterns for text chunking
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
_PARAGRAPH_SPLIT_RE = re.compile(r'\n+')
_CLAUSE_SPLIT_RE = re.compile(r'(?<=[,;:\u2014])\s+')

# ---------------------------------------------------------------------------
# Language mapping helpers
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


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

# Abbreviation table — longer patterns first to avoid partial matches
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

# Currency symbols → (singular, plural)
_CURRENCY_MAP = {
    "$": ("dollar", "dollars"),
    "€": ("euro", "euros"),
    "£": ("pound", "pounds"),
    "¥": ("yen", "yen"),
}

# Pre-compiled regex patterns for _normalize_text()
_EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b')
_URL_RE = re.compile(r'https?://\S+')
_URL_PROTO_RE = re.compile(r'^https?://')
_URL_WWW_RE = re.compile(r'^www\.')
_PHONE_RE = re.compile(r'(?:\(\d{3}\)\s*|\d{3}[-.])\d{3}[-.]?\d{4}')
_PHONE_NONDIGIT_RE = re.compile(r'\D')
_ORDINAL_RE = re.compile(r'\b(\d+)(?:st|nd|rd|th)\b')
_ISO_DATE_RE = re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b')
_US_DATE_RE = re.compile(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b')
_CARDINAL_RE = re.compile(r'(?<![.\w])\b\d+\b(?![.\w])')

# Pre-compile abbreviation table
_ABBREV_TABLE_COMPILED = [(re.compile(pat), repl) for pat, repl in _ABBREV_TABLE]

# Currency pattern (depends on _CURRENCY_MAP, so built here)
_CURRENCY_RE = re.compile(
    rf'([{"".join(re.escape(s) for s in _CURRENCY_MAP.keys())}])(\d+(?:\.\d+)?)'
)


def _normalize_text(text, language="English"):
    """Normalize text for TTS: expand numbers, dates, abbreviations, and URLs.

    Called before chunking in run_inference(). All steps are wrapped in
    try/except so a failure never blocks generation.

    Args:
        text: Raw input text.
        language: Language name string (e.g. "English").

    Returns:
        Normalized text string.
    """
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
        text = _EMAIL_RE.sub(_expand_email, text)
    except Exception:
        pass

    # 2. URLs: https://example.com → "example dot com"
    try:
        def _expand_url(m):
            url = m.group()
            url = _URL_PROTO_RE.sub('', url)
            url = _URL_WWW_RE.sub('', url)
            url = url.replace(".", " dot ").rstrip()
            return url
        text = _URL_RE.sub(_expand_url, text)
    except Exception:
        pass

    # 3. Phone numbers: (800) 555-1234 or 555-1234 → "8 0 0 5 5 5 1 2 3 4"
    try:
        def _expand_phone(m):
            digits = _PHONE_NONDIGIT_RE.sub('', m.group())
            return " ".join(digits)
        text = _PHONE_RE.sub(_expand_phone, text)
    except Exception:
        pass

    # 4. Currencies: $5.00 → "five dollars"
    try:
        def _expand_currency(m):
            symbol = m.group(1)
            amount_str = m.group(2)
            singular, plural = _CURRENCY_MAP.get(symbol, ("unit", "units"))
            try:
                amount = float(amount_str)
                whole = int(amount)
                if _n2w:
                    words = _n2w(whole, lang=lang)
                else:
                    words = str(whole)
                label = singular if whole == 1 else plural
                return f"{words} {label}"
            except Exception:
                return m.group()
        text = _CURRENCY_RE.sub(_expand_currency, text)
    except Exception:
        pass

    # 5. Ordinals: 3rd, 21st, etc.
    try:
        def _expand_ordinal(m):
            try:
                n = int(m.group(1))
                return _n2w(n, lang=lang, to="ordinal") if _n2w else m.group()
            except Exception:
                return m.group()
        text = _ORDINAL_RE.sub(_expand_ordinal, text)
    except Exception:
        pass

    # 6. ISO dates: YYYY-MM-DD
    try:
        def _expand_iso_date(m):
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
        text = _ISO_DATE_RE.sub(_expand_iso_date, text)
    except Exception:
        pass

    # 7. US dates: MM/DD/YYYY
    try:
        def _expand_us_date(m):
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
        text = _US_DATE_RE.sub(_expand_us_date, text)
    except Exception:
        pass

    # 8. Abbreviations
    try:
        for pattern, replacement in _ABBREV_TABLE_COMPILED:
            text = pattern.sub(replacement, text)
    except Exception:
        pass

    # 9. Cardinal numbers (standalone integers)
    if _n2w:
        try:
            def _expand_cardinal(m):
                try:
                    return _n2w(int(m.group()), lang=lang)
                except Exception:
                    return m.group()
            text = _CARDINAL_RE.sub(_expand_cardinal, text)
        except Exception:
            pass

    return text


# ---------------------------------------------------------------------------
# Text chunking for long-form reliability
# ---------------------------------------------------------------------------

def _split_text(text, max_chars=500, language="English", tokenizer=None, max_tokens=None):
    """Split text into chunks at sentence boundaries.

    Splits on sentence-ending punctuation (. ! ?) followed by whitespace,
    or on newlines. If a single sentence exceeds the limit, falls back to
    clause boundaries (, ; — :). Never splits mid-word.

    When tokenizer and max_tokens are provided, uses token counts instead of
    character counts to measure chunk sizes (torch backend only).

    Args:
        text: Input text to split.
        max_chars: Maximum characters per chunk (used when tokenizer is None).
        language: Language name string for pySBD segmenter.
        tokenizer: Optional tokenizer for token-aware chunking.
        max_tokens: Maximum tokens per chunk (used when tokenizer is provided).

    Returns:
        List of text chunks. Returns [text] unchanged if it fits in one chunk.
    """
    text = text.strip()

    def _measure(chunk):
        if tokenizer is not None and max_tokens is not None:
            return len(tokenizer.encode(chunk, add_special_tokens=False))
        return len(chunk)

    limit = max_tokens if (tokenizer is not None and max_tokens is not None) else max_chars

    if _measure(text) <= limit:
        return [text]

    # Sentence splitting: pySBD when available, regex fallback
    try:
        import pysbd
        segmenter = pysbd.Segmenter(language=_map_language(language), clean=False)
        sentences = segmenter.segment(text)
    except ImportError:
        sentences = _SENTENCE_SPLIT_RE.split(text)

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
