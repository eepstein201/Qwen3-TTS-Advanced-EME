"""Canonical CLI output and formatting (interface-quality plan, Phase 2 T2.1).

One output idiom for every CLI surface: severity wrappers whose glyphs are
separate prefix elements (pinned literal substrings stay contiguous),
tty-guarded color, section headers, key/value lines, pure formatters
spelling-identical to the UI helpers in ``qwen3_tts.interface.ui.shared``,
and a ``Column``/``render_table`` pair whose separator widths derive from
``Column.width``. Import-light by design: stdlib + click only.
"""

import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TextIO

import click

EMPTY_VALUE = "—"  # the same unknown marker the UI helpers render

SYMBOLS = {"success": "✓", "warning": "⚠", "error": "✗", "info": "ℹ"}

_SEVERITY_COLOR = {
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "info": "blue",
}


def color_enabled(stream: TextIO) -> bool:
    """Color reaches only a tty that has not set a non-empty NO_COLOR."""
    return bool(stream.isatty()) and not os.environ.get("NO_COLOR")


def paint(stream: TextIO, text: str, *styles: str) -> str:
    """Styled ``text`` when ``color_enabled(stream)``; zero ANSI otherwise.

    Styles are click style names — color words map to ``fg``, everything
    else ("bold", "underline", …) to its boolean flag.
    """
    if not styles or not color_enabled(stream):
        return text
    kwargs: dict[str, object] = {}
    for style in styles:
        if style in ("bold", "dim", "italic", "underline"):
            kwargs[style] = True
        else:
            kwargs.setdefault("fg", style)
    return click.style(text, **kwargs)


def _line(severity: str, message: str, file: TextIO | None) -> None:
    stream = file or (sys.stderr if severity == "error" else sys.stdout)
    glyph = paint(stream, SYMBOLS[severity], _SEVERITY_COLOR[severity])
    print(f"{glyph} {message}", file=stream)


def success(message: str, *, file: TextIO | None = None) -> None:
    """Success line (glyph prefix, message contiguous) on stdout."""
    _line("success", message, file)


def warn(message: str, *, file: TextIO | None = None) -> None:
    """Warning line (glyph prefix, message contiguous) on stdout."""
    _line("warning", message, file)


def error(message: str, *, file: TextIO | None = None) -> None:
    """Error line (glyph prefix, message contiguous) — ALWAYS on stderr."""
    _line("error", message, file)


def info(message: str, *, file: TextIO | None = None) -> None:
    """Info line (glyph prefix, message contiguous) on stdout."""
    _line("info", message, file)


def header(
    title: str, *, rule: str = "=", width: int = 60, file: TextIO | None = None
) -> None:
    """Section header; byte-identical to ``tools._shared.print_header``.

    That function becomes a thin delegate of this one in T2.3, so the
    leading blank line and two-space title indent are load-bearing.
    """
    stream = file or sys.stdout
    bar = rule * width
    print(f"\n{bar}\n  {title}\n{bar}", file=stream)


def kv_line(
    key: str, value: str, *, key_width: int = 22, file: TextIO | None = None
) -> None:
    """``key`` padded to ``key_width`` then ``value``; keys are never truncated."""
    stream = file or sys.stdout
    print(f"{key.ljust(key_width)}{value}", file=stream)


def _valid_amount(value: float | int | None) -> bool:
    return value is not None and math.isfinite(value) and value >= 0


def fmt_duration(seconds: float | None) -> str:
    """``m:ss`` under an hour, ``h:mm:ss`` above; "—" when unknown."""
    if not _valid_amount(seconds):
        return EMPTY_VALUE
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def fmt_bytes(num: float | int | None) -> str:
    """Canonical byte size (B/KB/MB/GB/TB, one decimal); "—" when unknown.

    ``tools._shared._format_size`` aliases this in T2.3; the drift is
    pinned by ``tests/test_cli_output.py``.
    """
    if not _valid_amount(num):
        return EMPTY_VALUE
    size = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def fmt_eta(seconds: float | None) -> str:
    """The single ETA spelling: ``~12s`` / ``~1m 20s`` / ``~1h 5m``; "—" when unknown."""
    if not _valid_amount(seconds):
        return EMPTY_VALUE
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"~{hours}h {minutes}m" if minutes else f"~{hours}h"
    if minutes:
        return f"~{minutes}m {secs}s" if secs else f"~{minutes}m"
    return f"~{secs}s"


@dataclass(frozen=True)
class Column:
    """Table column spec; alignment and separator width derive from this."""

    key: str
    title: str
    width: int
    align: str = "left"  # "left" | "right"


def _cell(text: str, column: Column) -> str:
    if column.align == "right":
        return text.rjust(column.width)
    return text.ljust(column.width)


def render_table(
    columns: Sequence[Column],
    rows: Sequence[Mapping[str, object]],
    *,
    indent: str = "  ",
    header: bool = True,
    empty_message: str | None = None,
) -> str:
    """Render a table as a pure string; never prints.

    Rows are mappings keyed by ``Column.key``. Empty ``rows`` yield
    ``empty_message`` verbatim when given, else ``""``. ``header=False``
    skips the title/separator lines for list-style output with no header row.
    """
    if not rows:
        return empty_message or ""
    lines = []
    if header:
        lines.append(
            indent + "  ".join(_cell(column.title, column) for column in columns)
        )
        lines.append(indent + "  ".join("-" * column.width for column in columns))
    for row in rows:
        lines.append(
            indent
            + "  ".join(_cell(str(row[column.key]), column) for column in columns)
        )
    return "\n".join(lines)
