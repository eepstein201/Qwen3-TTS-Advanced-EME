#!/usr/bin/env python3
"""TTS error class hierarchy.

No torch, numpy, or heavy imports.

CONFIG_PATH (paths.py) and get_model_size/MODEL_INFO (models.py) are
resolved via a lazy per-call import from ``qwen3_tts.core.config`` (the
package facade) — see qwen3_tts/core/config/__init__.py for the rationale.
"""


class TTSError(Exception):
    """Base error for all TTS operations.

    Attributes:
        user_message:     Short message safe to show end-users.
        technical_detail: Optional developer-facing detail.
        recovery:         Hint — "restart" | "config" | "bug" | "retry".
    """

    def __init__(self, user_message, technical_detail=None, recovery="restart"):
        self.user_message = user_message
        self.technical_detail = technical_detail
        self.recovery = recovery
        super().__init__(user_message)

    def format_cli(self):
        """Format for terminal display."""
        from qwen3_tts.core.config import CONFIG_PATH

        parts = [f"Error: {self.user_message}"]
        if self.technical_detail:
            parts[0] += f" [{self.technical_detail}]"
        suggestions = {
            "restart": "Try restarting the server with 'tts server start'.",
            "config": f"Check your configuration in {CONFIG_PATH}.",
            "bug": "This is an unexpected error — please report it.",
            "retry": "Try again; the issue may be transient.",
        }
        hint = suggestions.get(self.recovery, "")
        if hint:
            parts.append(f"  Suggestion: {hint}")
        return "\n".join(parts)

    def format_gradio(self):
        """Format for Gradio UI display."""
        from qwen3_tts.core.config import CONFIG_PATH

        color = {
            "restart": "#c0392b",
            "config": "#e67e22",
            "bug": "#8e44ad",
            "retry": "#2980b9",
        }.get(self.recovery, "#333")
        html = (
            f'<span style="color:{color};font-weight:bold;">{self.user_message}</span>'
        )
        if self.technical_detail:
            html += f'<br><small style="color:#666;">{self.technical_detail}</small>'
        suggestions = {
            "restart": "Try restarting the server with <code>tts server start</code>.",
            "config": f"Check your configuration in <code>{CONFIG_PATH}</code>.",
            "bug": "This is an unexpected error — please report it.",
            "retry": "Try again; the issue may be transient.",
        }
        hint = suggestions.get(self.recovery, "")
        if hint:
            html += f"<br><em>{hint}</em>"
        return html


class ServerConnectionError(TTSError):
    """Server is unreachable."""

    def __init__(self, detail=None):
        super().__init__(
            "Cannot connect to TTS server.",
            technical_detail=detail,
            recovery="restart",
        )


class ModelNotLoadedError(TTSError):
    """Required model is not loaded on the server."""

    def __init__(self, model_type, detail=None):
        from qwen3_tts.core.config import MODEL_INFO, get_model_size

        if not detail:
            try:
                size = get_model_size()
            except Exception:
                size = "1.7B"
            detail = MODEL_INFO.get(size, {}).get(model_type, {}).get("description", "")
        super().__init__(
            f"The '{model_type}' model is not loaded.",
            technical_detail=detail,
            recovery="restart",
        )
        self.model_type = model_type


class InvalidInputError(TTSError):
    """User-provided input failed validation."""

    def __init__(self, detail):
        super().__init__(detail, recovery="config")


class GenerationError(TTSError):
    """Generation failed at runtime."""

    def __init__(self, detail=None):
        super().__init__(
            "Audio generation failed.",
            technical_detail=detail,
            recovery="restart",
        )


class AuthenticationError(TTSError):
    """Authentication with the server failed."""

    def __init__(self, detail=None):
        super().__init__(
            "Authentication failed.",
            technical_detail=detail
            or "Cannot authenticate. Run 'tts server start' to generate auth token.",
            recovery="restart",
        )


class VoicePromptError(TTSError):
    """Voice prompt operation failed."""

    def __init__(self, operation: str, detail=None):
        """Initialize VoicePromptError.

        Args:
            operation: Operation that failed (e.g., "delete", "rename", "preview")
            detail: Error details
        """
        self.operation = operation
        super().__init__(
            f"Voice prompt {operation} failed.",
            technical_detail=detail,
            recovery="retry" if operation in ("list", "preview") else "config",
        )


class ModelError(TTSError):
    """Model operation failed."""

    def __init__(self, model_type: str, operation: str, detail=None):
        """Initialize ModelError.

        Args:
            model_type: Model type (clone, design, custom)
            operation: Operation that failed (e.g., "load", "unload")
            detail: Error details
        """
        self.model_type = model_type
        self.operation = operation
        super().__init__(
            f"Model {operation} failed for {model_type}.",
            technical_detail=detail,
            recovery="restart",
        )
