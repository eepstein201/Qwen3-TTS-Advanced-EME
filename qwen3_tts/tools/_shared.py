"""Shared utilities for the tools package."""


def _format_size(bytes_size: int) -> str:
    """Format bytes into human readable size."""
    for unit in ("B", "KB", "MB", "GB"):
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"
