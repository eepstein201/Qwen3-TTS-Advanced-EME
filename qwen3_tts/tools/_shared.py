"""Shared utilities for the tools package."""


def _format_size(bytes_size: int) -> str:
    """Format bytes into human readable size."""
    for unit in ("B", "KB", "MB", "GB"):
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def print_header(text: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print("=" * 60)


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"  ✓ {text}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"  ⚠ {text}")


def print_info(label: str, status: str = "", details: str = "") -> None:
    """Print an info line with optional status and details."""
    line = f"  {label}"
    if status:
        line += f" [{status}]"
    if details:
        line += f" - {details}"
    print(line)


def print_check(label: str, status: bool, details: str = "") -> None:
    """Print a check result with status indicator."""
    status_str = "✓" if status else "✗"
    print_info(label, status_str, details)
