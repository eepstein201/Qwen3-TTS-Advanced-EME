"""inference_lock must not be held during WAV encode / peaks calculation.

GEN-1: handle_generate used to wrap the ENTIRE per-text loop in
`async with state.inference_lock:`, but only the inference call (and the
state update + voice-prompt load that immediately precede it) need GPU
serialization. WAV encode (`sf.write`), cache-file write, gen_cache update,
and `calculate_waveform_peaks` are all CPU-only operating on the local
`wav` array, so holding the lock through them serializes unrelated requests
unnecessarily.

This test enforces the lock-narrowing refactor at the source level using
AST inspection (matching the convention in tests/test_generation_offload.py
and tests/test_streaming_and_peaks.py). Driving the full pipeline in a unit
test is impractical (needs a real model + engine), so we assert structural
properties of the source instead.
"""

import ast
import inspect
import unittest

from qwen3_tts.server import app_generation


def _locked_identifiers(src: str) -> set[str]:
    """Return the set of identifier names referenced anywhere inside an
    `async with state.inference_lock:` block.

    Matches context-manager expressions of the form `state.inference_lock`
    (an ``ast.Attribute`` whose ``.attr == "inference_lock"``). Collects
    every ``ast.Name.id`` and ``ast.Attribute.attr`` in the body so we catch
    identifiers whether they are called directly, passed as arguments to
    ``asyncio.to_thread(...)``, imported, or merely referenced.
    """
    tree = ast.parse(src)
    names: set[str] = set()

    def is_inference_lock_cm(node: ast.withitem) -> bool:
        ctx = node.context_expr
        return isinstance(ctx, ast.Attribute) and ctx.attr == "inference_lock"

    def collect_identifiers(node: ast.AST) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
            elif isinstance(child, ast.Attribute):
                names.add(child.attr)

    # Walk every AsyncWith; if its context manager is inference_lock, collect
    # every identifier nested anywhere inside its body (NOT the items, since
    # those contain the `state.inference_lock` reference itself).
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncWith) and any(
            is_inference_lock_cm(item) for item in node.items
        ):
            for stmt in node.body:
                collect_identifiers(stmt)

    return names


class TestInferenceLockScope(unittest.TestCase):
    def setUp(self):
        self.src = inspect.getsource(app_generation)
        self.locked = _locked_identifiers(self.src)

    def test_ast_walker_finds_locked_region(self):
        """Sanity check: the lock region exists and we found calls in it.

        Without this, the assertions below could pass vacuously if the AST
        walker were broken (e.g. matched nothing). run_inference is a known
        locked call today and must remain under the lock — it is passed as
        a Name arg to asyncio.to_thread, so the walker must collect Names
        (not just Call.func) for this sanity check to fire.
        """
        self.assertIn(
            "run_inference",
            self.locked,
            "AST walker should find run_inference inside inference_lock; "
            "if this fails the walker is broken and the other assertions "
            "may be passing vacuously.",
        )

    def test_wav_encode_not_under_inference_lock(self):
        """sf.write (WAV encode + cache file write) must run AFTER the lock
        releases. It is CPU-only and operates on the local wav array.

        ``sf.write`` is referenced as an ``ast.Attribute`` with ``attr``
        equal to ``"write"`` and is passed as an argument to
        ``asyncio.to_thread``. Banning the ``write`` attribute name inside
        the lock body catches both ``sf.write(...)`` direct calls and
        ``asyncio.to_thread(sf.write, ...)`` offload patterns.
        """
        self.assertNotIn(
            "write",
            self.locked,
            "sf.write must not be referenced inside an inference_lock "
            "block; WAV encode and cache-file write are CPU-only and must "
            "run after the lock is released.",
        )

    def test_calculate_waveform_peaks_not_under_inference_lock(self):
        """calculate_waveform_peaks is CPU-only and must run after lock release."""
        self.assertNotIn(
            "calculate_waveform_peaks",
            self.locked,
            "calculate_waveform_peaks must not be referenced inside an "
            "inference_lock block; it is CPU-only and must run after the "
            "lock is released.",
        )


if __name__ == "__main__":
    unittest.main()
