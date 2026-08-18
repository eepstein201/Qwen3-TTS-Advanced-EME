#!/usr/bin/env python3
"""Every GitHub Actions job must declare a `timeout-minutes`.

A job without one inherits GitHub's 6-hour default. That is not a theoretical
cost: the `coverage` job on PR #195 hung for 6h00m16s and reported only
"cancelled", while the identical job passed in 4m47s on the push trigger and
2m07s on a re-run. The suite has a known unbounded-block hazard (starlette's
TestClient websocket `receive()` takes no timeout), so a hang is a live failure
mode -- and six hours of wall-clock buys strictly less information than a
fast, legible timeout.

This is a *static* gate rather than a review habit for the same reason the
missing-`permissions:` alerts were: the failure is invisible until a job
actually hangs, and by then it has already cost the six hours.

Deliberately dependency-free -- the workflow files are scanned line-wise rather
than through PyYAML, which is only a transitive of `gradio` and is not declared
in any extra. Importing it here would make this gate silently skippable in an
environment where CI installs `.[test]`.

Run: python -m pytest tests/test_workflow_timeouts.py -q
"""
import re
import unittest
from pathlib import Path

WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / ".github" / "workflows"

# Jobs live at exactly two spaces under `jobs:`; their keys at exactly four.
_JOB_HEADER = re.compile(r"^ {2}([A-Za-z0-9_][A-Za-z0-9_-]*):\s*(?:#.*)?$")
_JOB_KEY = re.compile(r"^ {4}([A-Za-z0-9_-]+):\s*(.*?)\s*(?:#.*)?$")

# GitHub's own default is 360. Anything near it defeats the purpose of setting
# one at all, so the gate rejects it -- the point is a fast failure, not a
# marginally faster six hours.
MAX_TIMEOUT_MINUTES = 60


def _workflow_files():
    return sorted(WORKFLOWS_DIR.glob("*.yml")) + sorted(WORKFLOWS_DIR.glob("*.yaml"))


def _iter_jobs(path):
    """Yield ``(job_name, {top-level key: raw value})`` for each job in a workflow.

    Only keys at the job's own indent level are collected, so a step-level
    `timeout-minutes` (which is indented deeper) cannot be mistaken for the
    job-level one this gate is about.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    in_jobs = False
    job_name = None
    keys = {}

    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        if not line.startswith(" "):  # top-level key: jobs:, on:, permissions:
            if job_name is not None:
                yield job_name, keys
                job_name, keys = None, {}
            in_jobs = line.startswith("jobs:")
            continue

        if not in_jobs:
            continue

        header = _JOB_HEADER.match(line)
        if header:
            if job_name is not None:
                yield job_name, keys
            job_name, keys = header.group(1), {}
            continue

        if job_name is None:
            continue

        key = _JOB_KEY.match(line)
        if key:
            keys[key.group(1)] = key.group(2)

    if job_name is not None:
        yield job_name, keys


class TestWorkflowFilesAreDiscovered(unittest.TestCase):
    """Guard the guard: a parser that finds nothing would pass vacuously."""

    def test_workflow_directory_exists(self):
        self.assertTrue(
            WORKFLOWS_DIR.is_dir(), f"no workflows directory at {WORKFLOWS_DIR}"
        )

    def test_workflow_files_are_found(self):
        self.assertGreaterEqual(len(_workflow_files()), 4)

    def test_jobs_are_parsed_from_every_workflow(self):
        for path in _workflow_files():
            with self.subTest(workflow=path.name):
                jobs = list(_iter_jobs(path))
                self.assertTrue(jobs, f"{path.name}: parsed zero jobs")

    def test_parser_finds_the_known_jobs_in_test_yml(self):
        """Pin the parser against jobs that actually exist, so a regex that
        silently stops matching is caught here rather than by passing the
        timeout assertions with an empty set."""
        jobs = dict(_iter_jobs(WORKFLOWS_DIR / "test.yml"))

        for expected in ("test", "lint", "coverage", "test-docker", "test-minimal"):
            self.assertIn(expected, jobs)

    def test_parser_ignores_step_level_keys(self):
        """`steps:` entries are indented deeper and must not leak into the job
        key map -- otherwise a step-level timeout would satisfy this gate."""
        jobs = dict(_iter_jobs(WORKFLOWS_DIR / "test.yml"))

        self.assertNotIn("name", jobs["lint"])  # `- name:` lives under steps


class TestEveryJobHasATimeout(unittest.TestCase):
    def test_every_job_declares_timeout_minutes(self):
        for path in _workflow_files():
            for job_name, keys in _iter_jobs(path):
                with self.subTest(workflow=path.name, job=job_name):
                    if "uses" in keys:
                        # Reusable-workflow calls: GitHub rejects
                        # `timeout-minutes` on them outright, so the bound has
                        # to live in the called workflow.
                        continue
                    self.assertIn(
                        "timeout-minutes",
                        keys,
                        f"{path.name}: job '{job_name}' has no timeout-minutes; "
                        f"it would inherit GitHub's 6-hour default",
                    )

    def test_timeouts_are_sane_positive_bounds(self):
        for path in _workflow_files():
            for job_name, keys in _iter_jobs(path):
                raw = keys.get("timeout-minutes")
                if raw is None:
                    continue
                with self.subTest(workflow=path.name, job=job_name):
                    minutes = int(raw)
                    self.assertGreater(minutes, 0)
                    self.assertLessEqual(
                        minutes,
                        MAX_TIMEOUT_MINUTES,
                        f"{path.name}: job '{job_name}' allows {minutes} min; "
                        f"cap is {MAX_TIMEOUT_MINUTES}",
                    )

    def test_coverage_job_is_bounded(self):
        """The specific job that burned six hours on PR #195."""
        jobs = dict(_iter_jobs(WORKFLOWS_DIR / "test.yml"))

        self.assertIn("timeout-minutes", jobs["coverage"])


if __name__ == "__main__":
    unittest.main()
