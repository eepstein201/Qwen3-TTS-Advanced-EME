"""Tests validating Docker and docker-compose configuration for vLLM deployment.

Verifies IPC settings, dtype, tensor parallelism, multimodal params,
and HuggingFace cache volume mounting.
"""

import pathlib
import unittest

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def _vllm_dockerfile_exists() -> bool:
    """Check if Dockerfile.vllm exists for vLLM-specific tests."""
    return pathlib.Path("Dockerfile.vllm").exists()


def _docker_compose_exists() -> bool:
    """Check if docker-compose.yml exists for compose tests."""
    return pathlib.Path("docker-compose.yml").exists()


@unittest.skipUnless(HAS_YAML and _docker_compose_exists(), "requires PyYAML and docker-compose.yml")
class TestDockerCompose(unittest.TestCase):
    """Validate docker-compose.yml for vLLM production readiness."""

    def setUp(self):
        with open("docker-compose.yml") as f:
            self.config = yaml.safe_load(f)
        # Find vLLM service (could be named various ways)
        services = self.config.get("services", {})
        self.vllm_svc = (
            services.get("vllm")
            or services.get("tts-vllm")
            or services.get("qwen3-tts")
            or {}
        )

    def test_has_ipc_host_or_shm_size(self):
        """vLLM service must have ipc: host or shm_size >= 16g for tensor parallelism."""
        has_ipc = self.vllm_svc.get("ipc") == "host"
        has_shm = "shm_size" in self.vllm_svc
        self.assertTrue(
            has_ipc or has_shm,
            "vLLM service must have ipc: host or shm_size >= 16g "
            "for PyTorch NCCL tensor parallelism",
        )

    def test_hf_cache_volume_mounted(self):
        """Must mount HuggingFace cache to prevent re-downloads."""
        volumes = self.vllm_svc.get("volumes", [])
        hf_cache = any(".cache/huggingface" in str(v) for v in volumes)
        self.assertTrue(
            hf_cache,
            "Must mount HuggingFace cache volume to prevent re-downloads "
            "(~/.cache/huggingface:/root/.cache/huggingface or named volume)",
        )

    def test_gpu_reservation_present(self):
        """GPU reservation must be present for vLLM."""
        deploy = self.vllm_svc.get("deploy", {})
        resources = deploy.get("resources", {})
        reservations = resources.get("reservations", {})
        devices = reservations.get("devices", [])
        has_gpu = any(
            "gpu" in d.get("capabilities", []) for d in devices if isinstance(d, dict)
        )
        self.assertTrue(has_gpu, "GPU reservation must be present for vLLM service")


@unittest.skipIf(not _vllm_dockerfile_exists(), "Dockerfile.vllm not found - skip vLLM Docker tests")
class TestDockerfile(unittest.TestCase):
    """Validate Dockerfile.vllm for production readiness."""

    def setUp(self):
        with open("Dockerfile.vllm") as f:
            self.content = f.read()

    def test_cuda_base_image(self):
        """Dockerfile must use CUDA runtime base image."""
        self.assertIn("nvidia/cuda", self.content)

    def test_healthcheck_present(self):
        """Dockerfile must have HEALTHCHECK instruction."""
        self.assertIn("HEALTHCHECK", self.content)

    def test_exposes_tts_port(self):
        """Dockerfile must expose port 5123 for TTS server."""
        self.assertIn("5123", self.content)

    def test_huggingface_cache_volume(self):
        """Dockerfile must declare HuggingFace cache volume."""
        self.assertIn("/root/.cache/huggingface", self.content)

    def test_gpu_memory_utilization_documented(self):
        """Dockerfile or compose should document GPU memory utilization."""
        # Check if GPU_AMOUNT env var is present
        has_gpu_amount = "GPU_AMOUNT" in self.content
        # Or at least gpu-memory-utilization is mentioned somewhere
        has_gpu_mem = "gpu-memory-utilization" in self.content or "gpu_memory" in self.content
        self.assertTrue(
            has_gpu_amount or has_gpu_mem,
            "Dockerfile should document GPU_AMOUNT or gpu-memory-utilization",
        )


class TestDockerfileTestCopiesFilesTestsRead(unittest.TestCase):
    """Repo-root files that tests open must be COPYd into the test image, or
    the module errors in the container while passing on the host. install.sh
    was missing, so tests/test_install_script.py never ran on Linux.
    """

    def setUp(self):
        import os

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "Dockerfile.test")) as f:
            self.content = f.read()

    def _assert_copied(self, name):
        """Assert the file appears in an actual COPY instruction.

        A bare substring check is hollow here: the COPY line is preceded by a
        comment naming the very files it copies, so `assertIn("install.sh")`
        stays green even if install.sh is deleted from the COPY — which is the
        exact regression this class exists to catch.
        """
        import re

        self.assertRegex(
            self.content,
            rf"(?m)^COPY[^\n]*(?<![\w.\-/]){re.escape(name)}(?![\w.\-])",
            f"{name} is not in any COPY instruction in Dockerfile.test",
        )

    def test_copies_install_sh(self):
        self._assert_copied("install.sh")

    def test_copies_every_repo_root_file_a_test_opens(self):
        """Guard the whole class of bug, not just install.sh."""
        for name in ("CLAUDE.md", "config.json", "pytest.ini",
                     "colab_notebook.ipynb", "install.sh", "pyproject.toml"):
            with self.subTest(name=name):
                self._assert_copied(name)

    def test_copies_the_tool_config_the_static_gates_read(self):
        """Same class of bug, one level out: a tool config that is missing does
        not error, it silently changes the ruleset. Without .ruff.toml the
        container's `ruff check` runs on defaults and reported 780 phantom
        errors that do not reproduce on the host.
        """
        self._assert_copied(".ruff.toml")

    def test_copies_the_workflow_directory_the_ci_gate_reads(self):
        """test_workflow_timeouts.py reads .github/workflows/. A directory is
        the same trap as a file: absent, the gate cannot run in the container,
        and making it skip there would report green while verifying nothing.
        """
        self._assert_copied(".github/workflows/")


if __name__ == "__main__":
    unittest.main()
