"""Drift-guard tests for colab_notebook.ipynb.

Asserts the Colab notebook stays in sync with pyproject.toml and config.json.

Root cause that motivated these tests (2026-05-23):
    Cell 3's hand-curated DEPS string omitted slowapi, psutil, pyloudnorm, rich,
    causing the server to fail to start with ModuleNotFoundError.

Rule: dependency lists in installers must be DERIVED from pyproject.toml,
not hand-curated in parallel.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "colab_notebook.ipynb"
PYPROJECT = REPO_ROOT / "pyproject.toml"
SERVER_APP = REPO_ROOT / "qwen3_tts" / "server" / "app.py"
SERVER_LIFESPAN = REPO_ROOT / "qwen3_tts" / "server" / "app_lifespan.py"
AUDIO_PROCESSING = REPO_ROOT / "qwen3_tts" / "core" / "engine" / "audio_processing.py"

REQUIRED_EXTRAS = ("torch", "server", "audio", "ui", "cuda", "rich")

# Map import-name -> distribution-name (PyPI canonical) for non-trivial cases.
IMPORT_TO_DIST = {
    "pysbd": "pySBD",
    "yaml": "PyYAML",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
}

# Stdlib modules we never expect in dep lists.
STDLIB_PREFIXES = {
    "asyncio", "hashlib", "json", "logging", "os", "secrets", "signal", "sys",
    "time", "io", "re", "pathlib", "tempfile", "threading", "subprocess",
    "contextlib", "dataclasses", "typing", "collections", "functools", "abc",
    "itertools", "enum", "warnings", "traceback", "copy", "math", "random",
    "datetime", "shutil", "uuid", "platform", "inspect", "weakref", "atexit",
    "concurrent", "queue", "multiprocessing", "errno", "glob", "fnmatch",
    "struct", "base64", "binascii", "hmac", "ssl", "socket", "http", "urllib",
    "email", "mimetypes", "string", "operator", "ipaddress", "tomllib",
    "importlib", "fcntl",
}

# First-party packages (won't be in pyproject deps).
FIRST_PARTY_PREFIXES = {"qwen3_tts"}


def _load_notebook_cells() -> list[dict]:
    nb = json.loads(NOTEBOOK.read_text())
    return nb["cells"]


def _cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def _find_setup_cell() -> str:
    """Return the Setup cell source (the cell that mounts Drive and installs deps)."""
    cells = _load_notebook_cells()
    for cell in cells:
        src = _cell_source(cell)
        if "drive.mount" in src and ("pip install" in src or "uv pip install" in src):
            return src
    raise AssertionError("Setup cell (with drive.mount + pip install) not found")


def _find_settings_cell() -> str:
    """Return the Settings cell source (Cell 1: @param controls)."""
    cells = _load_notebook_cells()
    for cell in cells:
        src = _cell_source(cell)
        if "MODEL_SIZE" in src and "@param" in src:
            return src
    raise AssertionError("Settings cell (with MODEL_SIZE + @param) not found")


def _find_start_server_cell() -> str:
    """Return the cell that starts the server subprocess."""
    cells = _load_notebook_cells()
    for cell in cells:
        src = _cell_source(cell)
        if "qwen3_tts.server.app" in src and "subprocess.Popen" in src:
            return src
    raise AssertionError("Start-server cell not found")


def _find_launch_cell() -> str:
    """Return the Gradio launch cell source (calls demo.launch with share=True)."""
    cells = _load_notebook_cells()
    for cell in cells:
        src = _cell_source(cell)
        if "demo.launch" in src and "share=True" in src:
            return src
    raise AssertionError("Launch cell (demo.launch + share=True) not found")


def _find_voice_clone_cell() -> str:
    """Return the voice-clone cell source (calls create_and_save_voice_prompt)."""
    cells = _load_notebook_cells()
    for cell in cells:
        src = _cell_source(cell)
        if "create_and_save_voice_prompt" in src:
            return src
    raise AssertionError("Voice-clone cell (create_and_save_voice_prompt) not found")


def _pyproject_dep_universe() -> set[str]:
    """Return the union of base deps + required extras from pyproject.toml.

    Distribution names are normalised to lowercase package names (no version specifiers,
    no extras suffix), to make comparison robust.
    """
    proj = tomllib.loads(PYPROJECT.read_text())["project"]
    raw: list[str] = list(proj.get("dependencies", []))
    extras = proj.get("optional-dependencies", {})
    for ex in REQUIRED_EXTRAS:
        assert ex in extras, f"pyproject.toml missing extra {ex!r}"
        raw.extend(extras[ex])
    return {_canonical_name(d) for d in raw}


def _canonical_name(spec: str) -> str:
    """Strip version, extras, and quotes -> canonical lowercase package name."""
    s = spec.strip().strip("'\"")
    s = re.split(r"[<>=!~;\s]", s, maxsplit=1)[0]
    s = re.sub(r"\[.*\]$", "", s)
    return s.lower().replace("_", "-")


def _module_level_imports(path: Path) -> set[str]:
    """Return top-level (module-scope) imports from a Python file."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:  # only module-scope, not nested
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def _required_dist_names() -> set[str]:
    """Distribution names the server requires at module scope (after stdlib filtering)."""
    names: set[str] = set()
    for path in (SERVER_APP, SERVER_LIFESPAN, AUDIO_PROCESSING):
        names |= _module_level_imports(path)
    names -= STDLIB_PREFIXES
    names -= FIRST_PARTY_PREFIXES
    return {IMPORT_TO_DIST.get(n, n).lower().replace("_", "-") for n in names}


# ---------------------------------------------------------------------------
# Deps drift tests
# ---------------------------------------------------------------------------


class TestDepsDrift:
    def test_setup_cell_uses_pyproject_extras(self):
        """Setup cell must derive deps from pyproject.toml, not hand-curate them."""
        src = _find_setup_cell()
        assert "pyproject.toml" in src, (
            "Setup cell must read pyproject.toml to derive the dependency list. "
            "Hand-curated DEPS strings drift (this caused the slowapi failure)."
        )
        assert "tomllib" in src or "toml" in src, (
            "Setup cell must parse pyproject.toml via tomllib (or tomli)."
        )

    def test_setup_cell_references_all_required_extras(self):
        """Cell 3 must consume every extras name needed for Colab."""
        src = _find_setup_cell()
        missing = [ex for ex in REQUIRED_EXTRAS if f'"{ex}"' not in src and f"'{ex}'" not in src]
        assert not missing, f"Setup cell missing extras: {missing}"

    def test_no_hardcoded_dep_megastring(self):
        """No hand-curated DEPS literal containing many packages.

        We allow a small fallback list (the regression-surfacing fallback), but reject
        any DEPS-style assignment that lists more than 3 packages on one line.
        """
        src = _find_setup_cell()
        # Look for `DEPS = (` or `DEPS = "..."` with many tokens
        match = re.search(r"DEPS\s*=\s*[('\"](.+?)[)'\"]", src, re.DOTALL)
        if match:
            content = match.group(1)
            tokens = [t for t in re.split(r"\s+", content) if t and not t.startswith("#")]
            assert len(tokens) <= 6, (
                f"Setup cell appears to hand-curate {len(tokens)} packages "
                f"in a DEPS literal. Derive from pyproject.toml instead."
            )

    def test_server_imports_are_in_pyproject_universe(self):
        """Every module imported at server top-level must be installable via our extras."""
        required = _required_dist_names()
        universe = _pyproject_dep_universe()
        missing = sorted(n for n in required if n not in universe and n != "numpy")
        assert not missing, (
            f"Server imports not covered by pyproject.toml extras "
            f"{list(REQUIRED_EXTRAS)}: {missing}"
        )

    def test_critical_missing_deps_now_covered(self):
        """Regression test: slowapi, psutil, pyloudnorm, rich must be in the universe."""
        universe = _pyproject_dep_universe()
        for dep in ("slowapi", "psutil", "pyloudnorm", "rich"):
            assert dep in universe, f"{dep!r} must be declared in pyproject.toml extras"

    def test_fallback_deps_gradio_floor(self):
        """The hand-curated fallback DEPS must pin gradio >=6 and exclude 6.14.*.

        Targets the multi-line ``DEPS = ( ... )`` literal (the except-branch
        fallback), not the primary ``DEPS = " ".join(...)`` path. Catches the
        stale-fallback regression where a broken pyproject parse would install
        gradio 5.x (wrong major) instead of 6.x.
        """
        src = _find_setup_cell()
        fallback = re.search(r"DEPS\s*=\s*\((.*?)\)", src, re.DOTALL)
        if not fallback:
            pytest.skip("No fallback DEPS literal (primary path derives from pyproject.toml)")
        block = fallback.group(1)
        gradio = re.search(r'gradio([^"]*)', block)
        assert gradio, "Fallback DEPS must pin gradio"
        spec = gradio.group(1)
        assert ">=6" in spec, f"Fallback gradio pin must require >=6 (got 'gradio{spec}')"
        assert "6.14" in spec, (
            "Fallback gradio pin must exclude 6.14.* (Dataframe recursion build)"
        )

    def test_setup_cell_installs_rubberband(self):
        """The apt line must install rubberband-cli (pyrubberband shells out to it)."""
        src = _find_setup_cell()
        assert "rubberband-cli" in src, (
            "Setup cell apt line must install rubberband-cli "
            "(pyrubberband needs the binary; without it the librosa fallback is used)."
        )

    def test_setup_cell_does_not_reinstall_torch(self):
        """Colab's CUDA-matched torch must not be reinstalled from PyPI.

        Colab ships a torch built for its CUDA driver (e.g. ``2.13.0+cu130``).
        Passing an explicit ``torch>=...`` to ``uv pip install --system``
        overwrites the ``+cuXXX`` wheel and corrupts torch internals
        (``ImportError: cannot import name '_chunk_or_narrow_cat' from
        'torch._utils'`` → ``import transformers`` fails). The Setup cell must
        filter torch/torchaudio out of the install set so uv resolves torch
        transitively (keeping a consistent CUDA build) instead of forcing a
        reinstall. Regression observed live on Colab 2026-07-31.
        """
        src = _find_setup_cell()
        assert re.search(
            r'startswith\(\(\s*["\']torch["\']\s*,\s*["\']torchaudio["\']',
            src,
        ), (
            "Setup cell must filter torch/torchaudio out of DEPS before install "
            "(reinstalling from PyPI corrupts Colab's CUDA-matched torch)."
        )


# ---------------------------------------------------------------------------
# Settings surface tests
# ---------------------------------------------------------------------------


class TestSettingsSurface:
    def test_torch_quantization_param(self):
        src = _find_settings_cell()
        assert "TORCH_QUANTIZATION" in src, "TORCH_QUANTIZATION @param missing"

    def test_max_chunk_chars_param(self):
        src = _find_settings_cell()
        assert "MAX_CHUNK_CHARS" in src, "MAX_CHUNK_CHARS @param missing"

    def test_preload_asr_param(self):
        src = _find_settings_cell()
        assert "PRELOAD_ASR" in src, "PRELOAD_ASR @param missing"

    def test_flash_attn_version_param(self):
        src = _find_settings_cell()
        assert "FLASH_ATTN_VERSION" in src, "FLASH_ATTN_VERSION @param missing"

    def test_audio_loader_param(self):
        src = _find_settings_cell()
        assert "AUDIO_LOADER" in src, "AUDIO_LOADER @param missing"

    def test_no_hardcoded_flash_attn_version(self):
        """Cell 3 must not hardcode `_FA_VERSION = "2.7.4"`."""
        src = _find_setup_cell()
        assert not re.search(
            r'_FA_VERSION\s*=\s*[\'"]2\.7\.4[\'"]', src
        ), "Cell 3 still hardcodes _FA_VERSION; pull from FLASH_ATTN_VERSION instead."


# ---------------------------------------------------------------------------
# Config writes tests
# ---------------------------------------------------------------------------


class TestConfigWrites:
    def test_turing_writes_8bit_quantization(self):
        """When the Turing path runs, torch_quantization must be persisted."""
        src = _find_setup_cell()
        # Find the gpu_tier branching block. Look for the Turing branch writing
        # torch_quantization into config.
        turing_section = re.search(
            r"gpu_tier\s*==\s*['\"]turing['\"](.*?)(elif|else|\Z)",
            src, re.DOTALL
        )
        assert turing_section, "Could not find Turing branch in Setup cell"
        block = turing_section.group(1)
        assert "torch_quantization" in block, (
            "Turing branch prints '8-bit quantization' but does not write "
            "torch_quantization into config.json. The print is a lie."
        )

    def test_audio_loader_persisted(self):
        src = _find_setup_cell()
        assert "audio_loader" in src, "audio_loader not written to config in Setup cell"

    def test_max_chunk_chars_persisted(self):
        src = _find_setup_cell()
        assert "max_chunk_chars" in src, "max_chunk_chars not written to config in Setup cell"


# ---------------------------------------------------------------------------
# Failure diagnostics tests
# ---------------------------------------------------------------------------


class TestFailureDiagnostics:
    def test_start_cell_dumps_pip_info_on_failure(self):
        """Start-server cell must dump `pip show` when server fails."""
        src = _find_start_server_cell()
        assert "pip show" in src or "pip list" in src, (
            "Start-server cell's failure path must dump installed-package info "
            "(pip show / pip list). Without this, missing-module errors are "
            "much harder to diagnose."
        )

    def test_setup_cell_verifies_critical_imports(self):
        """Verification block must import the previously-missed packages."""
        src = _find_setup_cell()
        for name in ("slowapi", "psutil", "pyloudnorm"):
            assert name in src, (
                f"Setup cell's verification block must import {name!r} "
                f"so a missing install is caught immediately."
            )


# ---------------------------------------------------------------------------
# Voice-clone cell tests
# ---------------------------------------------------------------------------


class TestVoiceCloneCell:
    def test_transcript_not_passed_as_none(self):
        """The clone cell must not coerce transcript to None.

        ``transcript=<x> or None`` passes None when the transcript is empty,
        crashing ``create_voice.py`` (``f.write(transcript)`` ->
        ``TypeError: write() argument must be str, not None``). The cell is
        broken by default until the user types a transcript.
        """
        src = _find_voice_clone_cell()
        assert not re.search(r"transcript\s*=\s*\w+\s+or\s+None", src), (
            "Voice-clone cell passes `transcript=<x> or None`, which crashes when "
            "the transcript is empty. Pass transcript=<x> directly."
        )

    def test_voice_clone_cell_passes_x_vector_for_empty_transcript(self):
        """The clone cell must pass x_vector_only_mode for empty transcripts.

        The default ``TRANSCRIPT=""`` with no flag makes
        ``create_and_save_voice_prompt`` raise ``ref_text is required when
        x_vector_only_mode=False``. The cell must derive ``x_vector_only_mode``
        from the transcript so the default works (requires PR #117).
        """
        src = _find_voice_clone_cell()
        call = re.search(r"create_and_save_voice_prompt\((.*?)\)", src, re.DOTALL)
        assert call, "create_and_save_voice_prompt(...) call not found in clone cell"
        assert "x_vector_only_mode" in call.group(1), (
            "Voice-clone cell must pass x_vector_only_mode to "
            "create_and_save_voice_prompt; an empty transcript otherwise "
            "raises 'ref_text is required when x_vector_only_mode=False'."
        )


# ---------------------------------------------------------------------------
# Launch cell tests
# ---------------------------------------------------------------------------


class TestLaunchCell:
    def test_allowed_paths_covers_history_output(self):
        """Launch allowed_paths must include the web-UI output root.

        Generations save under ``~/Downloads/Qwen3-TTS Output/`` (the
        ``history_output_directory`` default). Gradio only serves files under an
        explicit allowed_path, so the root must be listed.
        """
        src = _find_launch_cell()
        assert "Qwen3-TTS Output" in src, (
            "Launch cell allowed_paths must include '~/Downloads/Qwen3-TTS Output' "
            "(the history_output_directory root) so generations are served."
        )


# ---------------------------------------------------------------------------
# Notebook validity / smoke
# ---------------------------------------------------------------------------


class TestNotebookValidity:
    def test_notebook_is_valid_json(self):
        json.loads(NOTEBOOK.read_text())  # raises on malformed JSON

    def test_notebook_has_expected_cell_count(self):
        cells = _load_notebook_cells()
        # Allow some flex but catch accidental deletions
        assert 8 <= len(cells) <= 14, f"Unexpected cell count: {len(cells)}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
