"""Design tokens and the UI stylesheet — the one source of visual constants.

``TOKENS`` holds each token's canonical fallback. ``var()`` (inline styles)
and ``UI_CSS`` (the launch stylesheet) both derive from it, so a component can
never reference a token the stylesheet does not declare. Surface/text/border
tokens map onto Gradio theme variables so dark mode keeps working; severity
colors are fixed mid-tones legible on both light and dark surfaces.

Gradio 6 takes ``css``/``theme`` in ``launch()``, not ``gr.Blocks()`` — wire
``UI_CSS`` through ``shared.get_gradio_launch_kwargs``. Stdlib only.
"""

from types import MappingProxyType

TOKENS = MappingProxyType(
    {
        "surface": "var(--block-background-fill,#ffffff)",
        "surface_alt": "var(--background-fill-secondary,#f7f7f8)",
        "border": "var(--block-border-color,#e5e7eb)",
        "border_strong": "var(--border-color-primary,#d1d5db)",
        "accent": "var(--color-accent,#6366f1)",
        "accent_soft": "var(--color-accent-soft,#eef2ff)",
        "text": "var(--body-text-color,#1f2937)",
        "text_subtle": "var(--body-text-color-subdued,#6b7280)",
        "text_inverse": "#ffffff",
        "success": "#0c5d00",
        "warning": "#7d4f00",
        "error": "#9b1c1c",
        "info": "#1a4480",
        "loading": "#5b3fa0",
        "focus": "#2563eb",
        "radius_sm": "4px",
        "radius_md": "8px",
        "radius_lg": "12px",
        "dur_fast": "120ms",
        "dur_slow": "240ms",
    }
)

SEVERITY_CLASS = MappingProxyType(
    {s: f"tts-sev-{s}" for s in ("info", "success", "warning", "error", "loading")}
)


def var(token: str) -> str:
    """Return ``var(--tts-<token>,<fallback>)``; unknown tokens raise KeyError."""
    return f"var(--tts-{token},{TOKENS[token]})"


def _severity_rules() -> str:
    return "".join(
        f".{cls}{{border:1px solid {var(sev)};border-left-width:4px;"
        f"border-radius:{var('radius_md')};padding:4px 8px;}}"
        f".{cls} textarea{{color:{var(sev)};}}"
        for sev, cls in SEVERITY_CLASS.items()
    )


UI_CSS = (
    ":root{" + "".join(f"--tts-{k}:{v};" for k, v in TOKENS.items()) + "}"
    ".gr-hidden{display:none !important;height:0 !important;overflow:hidden !important;}"
    '.tts-num{font-variant-numeric:tabular-nums;font-feature-settings:"tnum";}'
    f":focus-visible{{outline:2px solid {var('focus')};outline-offset:2px;}}"
    ".tts-history td{font-variant-numeric:tabular-nums;}"
    ".tts-history td:nth-child(6),.tts-history td:nth-child(7){cursor:pointer;"
    "text-align:center;user-select:none;min-width:44px;"
    "transition-property:background-color;"
    f"transition-duration:{var('dur_fast')};}}"
    ".tts-history td:nth-child(6):hover,.tts-history td:nth-child(7):hover"
    f"{{background-color:{var('accent_soft')};}}"
    ".tts-history td:nth-child(6)::selection,.tts-history td:nth-child(7)::selection"
    "{background:transparent;}"
    + _severity_rules()
    + f".tts-empty{{color:{var('text_subtle')};text-align:center;padding:16px;"
    f"border:1px dashed {var('border')};border-radius:{var('radius_md')};}}"
    "@media (prefers-reduced-motion:reduce){*,*::before,*::after{"
    "transition:none !important;animation:none !important;}}"
)
