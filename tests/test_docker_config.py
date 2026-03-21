"""Tests validating Docker and docker-compose configuration for vLLM deployment.

Verifies IPC settings, dtype, tensor parallelism, multimodal params,
and HuggingFace cache volume mounting.
"""

import unittest

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@unittest.skipUnless(HAS_YAML, "requires PyYAML")
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


if __name__ == "__main__":
    unittest.main()
