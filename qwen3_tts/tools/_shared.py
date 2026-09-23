"""Shared utilities for the tools package.

The print_* helpers and _format_size are thin delegates over
qwen3_tts.cli_output (interface-quality plan T2.3) — one output idiom
for the whole CLI. cli_output.color_enabled tty-guards the glyphs, so
ANSI never reaches a piped stream.
"""

from qwen3_tts import cli_output


def _format_size(bytes_size: float | int | None) -> str:
    """Format bytes into human readable size (aliases cli_output.fmt_bytes)."""
    return cli_output.fmt_bytes(bytes_size)


def print_header(text: str) -> None:
    """Print a formatted section header."""
    cli_output.header(text)


def print_success(text: str) -> None:
    """Print a success message."""
    cli_output.success(text)


def print_warning(text: str) -> None:
    """Print a warning message."""
    cli_output.warn(text)


def print_info(label: str, status: str = "", details: str = "") -> None:
    """Print an info line with optional status and details."""
    message = label
    if status:
        message += f" [{status}]"
    if details:
        message += f" - {details}"
    cli_output.info(message)
