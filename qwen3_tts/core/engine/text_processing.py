#!/usr/bin/env python3
"""Text processing utilities for TTS: normalization, chunking, and language mapping.

Base utility module — imports only from config.py, never from other engine submodules.
"""

import logging
import re

logger = logging.getLogger("tts.engine")

# ---------------------------------------------------------------------------
# Cached imports — loaded once on first use
# ---------------------------------------------------------------------------

_n2w_cached = None
_n2w_loaded = False
_SEGMENTER_CACHE = {}

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
_SUBUNIT_MAP = {
    "$": ("cent", "cents"),
    "€": ("cent", "cents"),
    "£": ("penny", "pence"),
    "¥": (None, None),
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


def _safe_transform(text: str, step_name: str, transform_fn) -> str:
    """Apply transform_fn to text; on failure log a warning and return original text."""
    try:
        return transform_fn(text)
    except Exception as e:
        logger.warning("Text normalization (%s) failed: %s", step_name, e)
        return text


# ---------------------------------------------------------------------------
# Normalization step helpers (H7 — extracted from _normalize_text closure)
# ---------------------------------------------------------------------------

def _expand_email_match(m) -> str:
    """Replace email address match with spoken form: 'user at example dot com'."""
    addr = m.group()
    local, _, domain = addr.partition("@")
    domain_parts = domain.split(".")
    return local + " at " + " dot ".join(domain_parts)


def _expand_url_match(m) -> str:
    """Replace URL match with spoken form: 'example dot com'."""
    url = m.group()
    url = _URL_PROTO_RE.sub('', url)
    url = _URL_WWW_RE.sub('', url)
    url = url.replace(".", " dot ").rstrip()
    return url


def _expand_phone_match(m) -> str:
    """Replace phone number match with digit-by-digit spoken form."""
    digits = _PHONE_NONDIGIT_RE.sub('', m.group())
    return " ".join(digits)


def _expand_currency_match(m, n2w, lang: str) -> str:
    """Replace currency match (e.g. '$5.99') with spoken form."""
    symbol = m.group(1)
    amount_str = m.group(2)
    singular, plural = _CURRENCY_MAP.get(symbol, ("unit", "units"))
    try:
        amount = float(amount_str)
        whole = int(amount)
        frac = round((amount - whole) * 100)
        whole_words = n2w(whole, lang=lang) if n2w else str(whole)
        label = singular if whole == 1 else plural
        if frac > 0:
            sub_singular, sub_plural = _SUBUNIT_MAP.get(symbol, ("cent", "cents"))
            if sub_singular:
                frac_words = n2w(frac, lang=lang) if n2w else str(frac)
                sub_label = sub_singular if frac == 1 else sub_plural
                return f"{whole_words} {label} and {frac_words} {sub_label}"
        return f"{whole_words} {label}"
    except Exception as e:
        logger.warning("Currency expansion failed for '%s': %s", m.group(), e)
        return m.group()


def _expand_ordinal_match(m, n2w, lang: str) -> str:
    """Replace ordinal match (e.g. '3rd') with spoken form."""
    try:
        n = int(m.group(1))
        return n2w(n, lang=lang, to="ordinal") if n2w else m.group()
    except Exception as e:
        logger.debug("Ordinal expansion failed for '%s': %s", m.group(), e)
        return m.group()


def _expand_date_components(month: int, day: int, year: int, n2w, lang: str) -> str:
    """Format date components into spoken form: 'March third, two thousand one'."""
    import calendar
    month_name = calendar.month_name[month]
    if n2w:
        day_word = n2w(day, lang=lang, to="ordinal")
        year_word = n2w(year, lang=lang)
    else:
        day_word = str(day)
        year_word = str(year)
    return f"{month_name} {day_word}, {year_word}"


def _expand_iso_date_match(m, n2w, lang: str) -> str:
    """Replace ISO date match (YYYY-MM-DD) with spoken form."""
    try:
        return _expand_date_components(int(m.group(2)), int(m.group(3)), int(m.group(1)), n2w, lang)
    except Exception as e:
        logger.debug("ISO date expansion failed for '%s': %s", m.group(), e)
        return m.group()


def _expand_us_date_match(m, n2w, lang: str) -> str:
    """Replace US date match (MM/DD/YYYY) with spoken form."""
    try:
        return _expand_date_components(int(m.group(1)), int(m.group(2)), int(m.group(3)), n2w, lang)
    except Exception as e:
        logger.debug("US date expansion failed for '%s': %s", m.group(), e)
        return m.group()


def _apply_abbreviations(t: str) -> str:
    """Expand abbreviations (Dr., Mr., etc.) in text."""
    for pattern, replacement in _ABBREV_TABLE_COMPILED:
        t = pattern.sub(replacement, t)
    return t


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

    global _n2w_cached, _n2w_loaded
    if not _n2w_loaded:
        try:
            from num2words import num2words
            _n2w_cached = num2words
        except ImportError:
            _n2w_cached = None
        _n2w_loaded = True
    _n2w = _n2w_cached

    text = _safe_transform(text, "email", lambda t: _EMAIL_RE.sub(_expand_email_match, t))
    text = _safe_transform(text, "url", lambda t: _URL_RE.sub(_expand_url_match, t))
    text = _safe_transform(text, "phone", lambda t: _PHONE_RE.sub(_expand_phone_match, t))
    text = _safe_transform(text, "currency", lambda t: _CURRENCY_RE.sub(
        lambda m: _expand_currency_match(m, _n2w, lang), t))
    text = _safe_transform(text, "ordinal", lambda t: _ORDINAL_RE.sub(
        lambda m: _expand_ordinal_match(m, _n2w, lang), t))
    text = _safe_transform(text, "iso_date", lambda t: _ISO_DATE_RE.sub(
        lambda m: _expand_iso_date_match(m, _n2w, lang), t))
    text = _safe_transform(text, "us_date", lambda t: _US_DATE_RE.sub(
        lambda m: _expand_us_date_match(m, _n2w, lang), t))
    text = _safe_transform(text, "abbreviation", _apply_abbreviations)
    if _n2w:
        text = _safe_transform(text, "cardinal", lambda t: _CARDINAL_RE.sub(
            lambda m: _n2w(int(m.group()), lang=lang) if _n2w else m.group(), t))

    return text


# ---------------------------------------------------------------------------
# Text chunking for long-form reliability
# ---------------------------------------------------------------------------

def _pack_words(words: list, limit: int, measure) -> tuple:
    """Pack words into chunks respecting limit.

    Returns:
        (chunks, current_chunk) — completed chunks and the in-progress remainder.
    """
    chunks = []
    current = ""
    for word in words:
        if current and measure(current) + 1 + measure(word) > limit:
            chunks.append(current.strip())
            current = ""
        current = (current + " " + word).strip() if current else word
    return chunks, current


def _pack_clauses(clauses: list, limit: int, measure) -> tuple:
    """Pack clauses into chunks, splitting oversized clauses at word boundaries.

    Returns:
        (chunks, current_chunk) — completed chunks and the in-progress remainder.
    """
    chunks = []
    current = ""
    for clause in clauses:
        if current and measure(current) + 1 + measure(clause) > limit:
            chunks.append(current.strip())
            current = ""
        if measure(clause) > limit:
            if current:
                chunks.append(current.strip())
                current = ""
            word_chunks, current = _pack_words(clause.split(), limit, measure)
            chunks.extend(word_chunks)
        else:
            current = (current + " " + clause).strip() if current else clause
    return chunks, current


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
        lang_code = _map_language(language)
        if lang_code not in _SEGMENTER_CACHE:
            _SEGMENTER_CACHE[lang_code] = pysbd.Segmenter(language=lang_code, clean=False)
        segmenter = _SEGMENTER_CACHE[lang_code]
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

        if _measure(sentence) <= limit:
            current_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence
            continue

        # Sentence exceeds limit — flush current and split at clause boundaries
        if current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = ""
        clause_chunks, current_chunk = _pack_clauses(
            _CLAUSE_SPLIT_RE.split(sentence), limit, _measure
        )
        chunks.extend(clause_chunks)

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]
