"""Voice prompt tests extracted from test_voice.py."""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.voice_test_helpers import (
    _skip_server, _skip_client, _make_test_client,
)


class TestMLXVoicePrompt(unittest.TestCase):
    """Test MLX voice prompt loading (file-based, no mlx import needed)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_dir = None

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_voice_prompt_mlx_success(self):
        """Load MLX prompt from .wav + .txt pair."""
        from qwen3_tts.core.engine import load_voice_prompt_mlx
        # Create fake wav and txt
        wav_path = os.path.join(self.tmpdir, "test_voice.wav")
        txt_path = os.path.join(self.tmpdir, "test_voice.txt")
        with open(wav_path, "wb") as f:
            f.write(b"RIFF\x00\x00\x00\x00WAVEfmt ")
        with open(txt_path, "w") as f:
            f.write("Hello, this is a test transcript.")

        with patch("qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR", self.tmpdir):
            result = load_voice_prompt_mlx("test_voice.pt")

        self.assertIsInstance(result, dict)
        self.assertEqual(result["ref_audio"], wav_path)
        self.assertEqual(result["ref_text"], "Hello, this is a test transcript.")

    def test_load_voice_prompt_mlx_strips_pt(self):
        """Prompt name with .pt extension is handled correctly."""
        from qwen3_tts.core.engine import load_voice_prompt_mlx
        wav_path = os.path.join(self.tmpdir, "voice.wav")
        txt_path = os.path.join(self.tmpdir, "voice.txt")
        with open(wav_path, "wb") as f:
            f.write(b"fake")
        with open(txt_path, "w") as f:
            f.write("transcript")

        with patch("qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR", self.tmpdir):
            result = load_voice_prompt_mlx("voice.pt")
        self.assertEqual(result["ref_audio"], wav_path)

    def test_load_voice_prompt_mlx_missing_files(self):
        """Raises FileNotFoundError when wav/txt missing."""
        from qwen3_tts.core.engine import load_voice_prompt_mlx
        with patch("qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR", self.tmpdir):
            with self.assertRaises(FileNotFoundError):
                load_voice_prompt_mlx("nonexistent")

    def test_load_voice_prompt_mlx_pt_only_error(self):
        """Clear error when only .pt exists (no MLX-compatible files)."""
        from qwen3_tts.core.engine import load_voice_prompt_mlx
        pt_path = os.path.join(self.tmpdir, "legacy.pt")
        with open(pt_path, "wb") as f:
            f.write(b"fake tensor data")

        with patch("qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR", self.tmpdir):
            with self.assertRaises(FileNotFoundError) as ctx:
                load_voice_prompt_mlx("legacy")
            self.assertIn("only has a .pt file", str(ctx.exception))
            self.assertIn("tts voice create", str(ctx.exception))

    def test_load_voice_prompt_dispatch_torch(self):
        """load_voice_prompt dispatches to torch backend."""
        from qwen3_tts.core.engine import load_voice_prompt
        with patch("qwen3_tts.core.engine.voice_prompt.get_backend", return_value="torch"):
            with patch("qwen3_tts.core.engine.voice_prompt._load_voice_prompt_torch", return_value="mock_tensor") as mock:
                result = load_voice_prompt("test.pt")
        mock.assert_called_once_with("test.pt")
        self.assertEqual(result, "mock_tensor")

    def test_load_voice_prompt_dispatch_mlx(self):
        """load_voice_prompt dispatches to MLX backend."""
        from qwen3_tts.core.engine import load_voice_prompt
        mock_result = {"ref_audio": "/fake/path.wav", "ref_text": "text"}
        with patch("qwen3_tts.core.engine.voice_prompt.get_backend", return_value="mlx"):
            with patch("qwen3_tts.core.engine.voice_prompt.load_voice_prompt_mlx", return_value=mock_result) as mock:
                result = load_voice_prompt("test.pt")
        mock.assert_called_once_with("test.pt")
        self.assertEqual(result, mock_result)


class TestMLXVoicePromptCache(unittest.TestCase):
    """Test MLX voice prompt caching in voice_engine."""

    def setUp(self):
        # Clear cache before each test (earlier test classes may have populated it)
        from qwen3_tts.core.engine.voice_prompt import _mlx_prompt_cache
        _mlx_prompt_cache.clear()
        self.tmpdir = tempfile.mkdtemp()
        # Create fake wav and txt
        for name in ("voice_a", "voice_b"):
            with open(os.path.join(self.tmpdir, f"{name}.wav"), "wb") as f:
                f.write(b"RIFF\x00\x00\x00\x00WAVEfmt ")
            with open(os.path.join(self.tmpdir, f"{name}.txt"), "w") as f:
                f.write(f"Transcript for {name}")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        # Clear cache between tests
        from qwen3_tts.core.engine.voice_prompt import _mlx_prompt_cache
        _mlx_prompt_cache.clear()

    def test_mlx_cache_returns_consistent_results(self):
        """Cached result is identical to first load."""
        from qwen3_tts.core.engine import load_voice_prompt_mlx
        with patch("qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR", self.tmpdir):
            first = load_voice_prompt_mlx("voice_a")
            second = load_voice_prompt_mlx("voice_a")
        self.assertIs(first, second)  # Same object from cache

    def test_mlx_cache_stores_entries(self):
        """Loading a prompt adds it to the cache."""
        from qwen3_tts.core.engine import load_voice_prompt_mlx
        from qwen3_tts.core.engine.voice_prompt import _mlx_prompt_cache
        with patch("qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR", self.tmpdir):
            load_voice_prompt_mlx("voice_a")
        self.assertIn("voice_a", _mlx_prompt_cache)

    def test_clear_voice_prompt_cache_clears_mlx(self):
        """clear_voice_prompt_cache clears MLX cache."""
        from qwen3_tts.core.engine import load_voice_prompt_mlx, clear_voice_prompt_cache
        from qwen3_tts.core.engine.voice_prompt import _mlx_prompt_cache
        with patch("qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR", self.tmpdir):
            load_voice_prompt_mlx("voice_a")
        self.assertEqual(len(_mlx_prompt_cache), 1)
        clear_voice_prompt_cache()
        self.assertEqual(len(_mlx_prompt_cache), 0)

    def test_mlx_cache_info_returns_currsize(self):
        """voice_prompt_cache_info returns MLX cache size."""
        from qwen3_tts.core.engine import load_voice_prompt_mlx, voice_prompt_cache_info
        with patch("qwen3_tts.core.engine.voice_prompt.get_backend", return_value="mlx"):
            with patch("qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR", self.tmpdir):
                load_voice_prompt_mlx("voice_a")
            info = voice_prompt_cache_info()
        self.assertEqual(info.currsize, 1)


@_skip_server
class TestDeletePromptEndpoint(unittest.TestCase):
    """Test POST /delete-prompt endpoint."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={
            "security": {"max_text_length": 100, "max_batch_size": 3},
            "auto_shutdown_minutes": 0,
        })
        app.state.auth_token = "test_secret_token"  # nosec B105
        app.state.models_loaded.set()
        cls.auth = {"Authorization": "Bearer test_secret_token"}

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create fake voice files
        for ext in (".pt", ".wav", ".txt"):
            with open(os.path.join(self.tmpdir, f"test_voice{ext}"), "w") as f:
                f.write("fake")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_delete_requires_auth(self):
        """POST /delete-prompt requires authentication."""
        resp = self.client.post("/delete-prompt", json={"name": "test"})
        self.assertEqual(resp.status_code, 401)

    def test_delete_path_traversal(self):
        """POST /delete-prompt rejects path traversal."""
        resp = self.client.post("/delete-prompt", json={"name": "../etc/passwd"},
                                headers=self.auth)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid prompt name", resp.json()["detail"])

    def test_delete_nonexistent(self):
        """POST /delete-prompt returns 404 for missing prompt."""
        with patch("qwen3_tts.server.app.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.post("/delete-prompt", json={"name": "nonexistent"},
                                    headers=self.auth)
        self.assertEqual(resp.status_code, 404)

    def test_delete_success(self):
        """POST /delete-prompt deletes all format files."""
        with patch("qwen3_tts.server.app.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.post("/delete-prompt", json={"name": "test_voice"},
                                    headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "deleted")
        self.assertEqual(len(data["files_removed"]), 3)
        # Verify files are gone
        for ext in (".pt", ".wav", ".txt"):
            self.assertFalse(os.path.exists(os.path.join(self.tmpdir, f"test_voice{ext}")))


@_skip_server
class TestRenamePromptEndpoint(unittest.TestCase):
    """Test POST /rename-prompt endpoint."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={
            "security": {"max_text_length": 100, "max_batch_size": 3},
            "auto_shutdown_minutes": 0,
        })
        app.state.models_loaded.set()
        cls.auth = {"Authorization": "Bearer test_token"}

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        for ext in (".pt", ".wav", ".txt"):
            with open(os.path.join(self.tmpdir, f"old_voice{ext}"), "w") as f:
                f.write("fake")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_rename_requires_auth(self):
        """POST /rename-prompt requires authentication."""
        resp = self.client.post("/rename-prompt", json={"old_name": "a", "new_name": "b"})
        self.assertEqual(resp.status_code, 401)

    def test_rename_path_traversal(self):
        """POST /rename-prompt rejects path traversal in both names."""
        resp = self.client.post("/rename-prompt",
                                json={"old_name": "../bad", "new_name": "ok"},
                                headers=self.auth)
        self.assertEqual(resp.status_code, 400)
        resp = self.client.post("/rename-prompt",
                                json={"old_name": "ok", "new_name": "../bad"},
                                headers=self.auth)
        self.assertEqual(resp.status_code, 400)

    def test_rename_collision(self):
        """POST /rename-prompt returns 409 when new name already exists."""
        # Create collision target
        with open(os.path.join(self.tmpdir, "existing.wav"), "w") as f:
            f.write("fake")
        with patch("qwen3_tts.server.app.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.post("/rename-prompt",
                                    json={"old_name": "old_voice", "new_name": "existing"},
                                    headers=self.auth)
        self.assertEqual(resp.status_code, 409)

    def test_rename_success(self):
        """POST /rename-prompt renames all format files."""
        with patch("qwen3_tts.server.app.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.post("/rename-prompt",
                                    json={"old_name": "old_voice", "new_name": "new_voice"},
                                    headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "renamed")
        # Old files should be gone, new files should exist
        for ext in (".pt", ".wav", ".txt"):
            self.assertFalse(os.path.exists(os.path.join(self.tmpdir, f"old_voice{ext}")))
            self.assertTrue(os.path.exists(os.path.join(self.tmpdir, f"new_voice{ext}")))

    def test_rename_not_found(self):
        """POST /rename-prompt returns 404 for missing prompt."""
        with patch("qwen3_tts.server.app.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.post("/rename-prompt",
                                    json={"old_name": "nonexistent", "new_name": "new"},
                                    headers=self.auth)
        self.assertEqual(resp.status_code, 404)


@_skip_server
class TestPreviewPromptEndpoint(unittest.TestCase):
    """Test GET /preview-prompt endpoint."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={
            "security": {"max_text_length": 100, "max_batch_size": 3},
            "auto_shutdown_minutes": 0,
        })
        app.state.models_loaded.set()  # simulate models ready for tests that need a live server
        app.state.auth_token = "test_secret_token"  # nosec B105
        cls.auth = {"Authorization": "Bearer test_secret_token"}

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        with open(os.path.join(self.tmpdir, "test_voice.wav"), "wb") as f:
            f.write(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_preview_requires_auth(self):
        """GET /preview-prompt requires authentication."""
        resp = self.client.get("/preview-prompt?name=test")
        self.assertEqual(resp.status_code, 401)

    def test_preview_not_found(self):
        """GET /preview-prompt returns 404 for missing prompt."""
        with patch("qwen3_tts.server.app.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.get("/preview-prompt?name=nonexistent",
                                   headers=self.auth)
        self.assertEqual(resp.status_code, 404)

    def test_preview_returns_audio(self):
        """GET /preview-prompt returns audio/wav content."""
        with patch("qwen3_tts.server.app.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.get("/preview-prompt?name=test_voice",
                                   headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("audio/wav", resp.headers.get("content-type", ""))


@_skip_server
class TestPromptDetailsEndpoint(unittest.TestCase):
    """Test GET /prompt-details endpoint."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={
            "security": {"max_text_length": 100, "max_batch_size": 3},
            "auto_shutdown_minutes": 0,
        })
        app.state.auth_token = "test_secret_token"  # nosec B105
        app.state.models_loaded.set()  # simulate models ready for tests that need a live server
        cls.auth = {"Authorization": "Bearer test_secret_token"}

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        for ext in (".pt", ".wav", ".txt"):
            with open(os.path.join(self.tmpdir, f"voice_a{ext}"), "w") as f:
                f.write("fake data here")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_details_requires_auth(self):
        """GET /prompt-details requires authentication."""
        resp = self.client.get("/prompt-details?name=test")
        self.assertEqual(resp.status_code, 401)

    def test_details_single_prompt(self):
        """GET /prompt-details?name=X returns metadata for one prompt."""
        with patch("qwen3_tts.server.app.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.get("/prompt-details?name=voice_a",
                                   headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["name"], "voice_a")
        self.assertIn(".pt", data["formats"])
        self.assertIn(".wav", data["formats"])
        self.assertIn(".txt", data["formats"])
        self.assertGreater(data["size_bytes"], 0)

    def test_details_all_prompts(self):
        """GET /prompt-details without name returns all prompts."""
        with patch("qwen3_tts.server.app.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.get("/prompt-details", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("prompts", data)
        self.assertEqual(len(data["prompts"]), 1)
        self.assertEqual(data["prompts"][0]["name"], "voice_a")


@_skip_client
class TestClientPromptManagement(unittest.TestCase):
    """Test voice_client prompt management method signatures."""

    def test_delete_prompt_method_exists(self):
        """TTSClient has delete_prompt method."""
        from qwen3_tts.server.client import TTSClient
        self.assertTrue(hasattr(TTSClient, "delete_prompt"))

    def test_rename_prompt_method_exists(self):
        """TTSClient has rename_prompt method."""
        from qwen3_tts.server.client import TTSClient
        self.assertTrue(hasattr(TTSClient, "rename_prompt"))

    def test_preview_prompt_method_exists(self):
        """TTSClient has preview_prompt method."""
        from qwen3_tts.server.client import TTSClient
        self.assertTrue(hasattr(TTSClient, "preview_prompt"))

    def test_get_prompt_details_method_exists(self):
        """TTSClient has get_prompt_details method."""
        from qwen3_tts.server.client import TTSClient
        self.assertTrue(hasattr(TTSClient, "get_prompt_details"))

    def test_list_prompts_uses_server(self):
        """list_prompts calls server /prompts when running."""
        import inspect
        from qwen3_tts.server.client import TTSClient
        source = inspect.getsource(TTSClient.list_prompts)
        self.assertIn("/prompts", source)
        self.assertIn("is_server_running", source)


class TestSetDefaultClonePrompt(unittest.TestCase):
    """Test set_default_clone_prompt config helper."""

    def test_set_default_writes_config(self):
        """set_default_clone_prompt updates config.json."""
        from qwen3_tts.core.config import set_default_clone_prompt, load_config, save_config, CONFIG_PATH
        # Save original config
        original = load_config()
        try:
            set_default_clone_prompt("test_voice.pt")
            config = load_config()
            self.assertEqual(config["default_clone_prompt"], "test_voice.pt")
        finally:
            # Restore original config
            save_config(original)


class TestGetDefaultClonePromptFallback(unittest.TestCase):
    """Tests for get_default_clone_prompt() backend-aware fallback."""

    @patch("qwen3_tts.core.config.get_backend", return_value="mlx")
    @patch("qwen3_tts.core.config.os.listdir")
    @patch("qwen3_tts.core.config.os.path.exists")
    @patch("qwen3_tts.core.config.load_config", return_value={})
    def test_mlx_fallback_returns_wav(self, mock_cfg, mock_exists, mock_ls, mock_be):
        mock_ls.return_value = ["a.wav", "a.txt", "b.pt"]
        mock_exists.side_effect = lambda p: True  # all files exist
        from qwen3_tts.core.config import get_default_clone_prompt
        result = get_default_clone_prompt()
        self.assertEqual(result, "a.wav")

    @patch("qwen3_tts.core.config.get_backend", return_value="torch")
    @patch("qwen3_tts.core.config.os.listdir")
    @patch("qwen3_tts.core.config.os.path.exists", return_value=False)
    @patch("qwen3_tts.core.config.load_config", return_value={})
    def test_torch_fallback_returns_pt(self, mock_cfg, mock_exists, mock_ls, mock_be):
        mock_ls.return_value = ["a.pt", "a.wav", "a.txt"]
        from qwen3_tts.core.config import get_default_clone_prompt
        result = get_default_clone_prompt()
        self.assertEqual(result, "a.pt")

    @patch("qwen3_tts.core.config.get_backend", return_value="vllm")
    @patch("qwen3_tts.core.config.os.listdir")
    @patch("qwen3_tts.core.config.os.path.exists", return_value=False)
    @patch("qwen3_tts.core.config.load_config", return_value={})
    def test_vllm_fallback_returns_pt(self, mock_cfg, mock_exists, mock_ls, mock_be):
        """vLLM backend uses .pt files like torch."""
        mock_ls.return_value = ["a.pt", "a.wav", "a.txt"]
        from qwen3_tts.core.config import get_default_clone_prompt
        result = get_default_clone_prompt()
        self.assertEqual(result, "a.pt")


class TestGetVoicePrompts(unittest.TestCase):
    """Tests for get_voice_prompts() backend-aware filtering."""

    @patch("qwen3_tts.interface.ui.shared.get_backend", return_value="mlx")
    @patch("qwen3_tts.interface.ui.shared.os.listdir")
    def test_mlx_returns_only_wav_with_matching_txt(self, mock_ls, mock_backend):
        mock_ls.return_value = ["v1.pt", "v1.wav", "v1.txt", "v2.wav", "v2.txt", "orphan.wav"]
        from qwen3_tts.interface.ui.shared import get_voice_prompts
        result = get_voice_prompts()
        self.assertEqual(result, ["v1.wav", "v2.wav"])
        self.assertNotIn("v1.pt", result)
        self.assertNotIn("orphan.wav", result)

    @patch("qwen3_tts.interface.ui.shared.get_backend", return_value="torch")
    @patch("qwen3_tts.interface.ui.shared.os.listdir")
    def test_torch_returns_only_pt_files(self, mock_ls, mock_backend):
        mock_ls.return_value = ["v1.pt", "v1.wav", "v1.txt", "v2.pt"]
        from qwen3_tts.interface.ui.shared import get_voice_prompts
        result = get_voice_prompts()
        self.assertEqual(result, ["v1.pt", "v2.pt"])
        self.assertNotIn("v1.wav", result)

    @patch("qwen3_tts.interface.ui.shared.get_backend", return_value="vllm")
    @patch("qwen3_tts.interface.ui.shared.os.listdir")
    def test_vllm_returns_only_pt_files(self, mock_ls, mock_backend):
        """vLLM backend uses .pt files like torch."""
        mock_ls.return_value = ["v1.pt", "v1.wav", "v1.txt", "v2.pt"]
        from qwen3_tts.interface.ui.shared import get_voice_prompts
        result = get_voice_prompts()
        self.assertEqual(result, ["v1.pt", "v2.pt"])


class TestValidatePromptNameCallers(unittest.TestCase):
    """Tests that validate_prompt_name callers handle return values correctly."""

    @patch("qwen3_tts.interface.ui.voice_management.load_config", return_value={"advanced": {"backend": "mlx"}})
    @patch("qwen3_tts.interface.ui.voice_management.validate_prompt_name", return_value=None)
    def test_create_voice_valid_name_does_not_crash(self, mock_validate, mock_cfg):
        """Valid name (returns None) must not raise TypeError from unpacking."""
        from qwen3_tts.interface.ui.voice_management import create_voice_prompt
        # Pass a real audio_path so we reach validate_prompt_name (not the audio check)
        # Should proceed past validation, then hit the backend logic
        try:
            create_voice_prompt("/tmp/fake_audio.wav", "transcript", "valid_name")
        except TypeError as e:
            self.fail(f"Crashed unpacking validate_prompt_name return value: {e}")
        except Exception:
            pass  # Any other error (file not found, etc.) is fine

    @patch("qwen3_tts.interface.ui.voice_management.validate_prompt_name")
    def test_create_voice_invalid_name_raises_error(self, mock_validate):
        """Invalid name must raise gr.Error with the error message."""
        mock_validate.return_value = ({"error": "Invalid prompt name", "recovery": "config"}, 400)
        from qwen3_tts.interface.ui.voice_management import create_voice_prompt
        import gradio as gr
        with self.assertRaises(gr.Error) as ctx:
            create_voice_prompt("/tmp/fake_audio.wav", "transcript", "bad_name")
        self.assertIn("Invalid", str(ctx.exception))

    @patch("qwen3_tts.interface.ui.voice_management.validate_prompt_name", return_value=None)
    @patch("qwen3_tts.interface.ui.voice_management.is_server_running", return_value=True)
    @patch("qwen3_tts.interface.ui.voice_management.load_config", return_value={})
    @patch("qwen3_tts.interface.ui.voice_management.get_server_url", return_value="http://127.0.0.1:5123")
    @patch("qwen3_tts.interface.ui.voice_management.auth_headers", return_value={})
    def test_rename_voice_valid_name_does_not_crash(self, *mocks):
        """Valid new name (returns None) must not raise TypeError from unpacking."""
        from qwen3_tts.interface.ui.voice_management import rename_voice
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        with patch("requests.post", return_value=mock_resp):
            try:
                rename_voice("old", "new_name")
            except TypeError as e:
                self.fail(f"Crashed unpacking validate_prompt_name return value: {e}")
            except Exception:
                pass  # Other errors (network, etc.) are fine


class TestSetVoiceDefaultExtension(unittest.TestCase):
    """Tests that set_voice_default uses backend-appropriate extension."""

    @patch("qwen3_tts.interface.ui.voice_management.get_prompt_table_data")
    @patch("qwen3_tts.interface.ui.voice_management.set_default_clone_prompt")
    @patch("qwen3_tts.interface.ui.voice_management.get_backend", return_value="mlx")
    def test_mlx_uses_wav_extension(self, mock_be, mock_set, mock_table):
        from qwen3_tts.interface.ui.voice_management import set_voice_default
        set_voice_default("my_voice")
        mock_set.assert_called_once_with("my_voice.wav")

    @patch("qwen3_tts.interface.ui.voice_management.get_prompt_table_data")
    @patch("qwen3_tts.interface.ui.voice_management.set_default_clone_prompt")
    @patch("qwen3_tts.interface.ui.voice_management.get_backend", return_value="torch")
    def test_torch_uses_pt_extension(self, mock_be, mock_set, mock_table):
        from qwen3_tts.interface.ui.voice_management import set_voice_default
        set_voice_default("my_voice")
        mock_set.assert_called_once_with("my_voice.pt")

    @patch("qwen3_tts.interface.ui.voice_management.get_prompt_table_data")
    @patch("qwen3_tts.interface.ui.voice_management.set_default_clone_prompt")
    @patch("qwen3_tts.interface.ui.voice_management.get_backend", return_value="vllm")
    def test_vllm_uses_pt_extension(self, mock_be, mock_set, mock_table):
        """vLLM backend uses .pt files like torch."""
        from qwen3_tts.interface.ui.voice_management import set_voice_default
        set_voice_default("my_voice")
        mock_set.assert_called_once_with("my_voice.pt")


class TestPreviewVoiceExtension(unittest.TestCase):
    """Tests that preview_voice does not force .pt extension."""

    @patch("qwen3_tts.interface.ui.voice_management.is_server_running", return_value=True)
    @patch("qwen3_tts.interface.ui.voice_management.load_config", return_value={})
    @patch("qwen3_tts.interface.ui.voice_management.get_server_url", return_value="http://127.0.0.1:5123")
    @patch("qwen3_tts.interface.ui.voice_management.auth_headers", return_value={})
    def test_does_not_force_pt_extension(self, mock_auth, mock_url, mock_cfg, mock_running):
        """Name sent to server should not have .pt forced onto it."""
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"RIFF" + b"\x00" * 100
        with patch("requests.get", return_value=mock_resp) as mock_get:
            from qwen3_tts.interface.ui.voice_management import preview_voice
            preview_voice("my_voice")
            # Check the name param sent to server
            call_args = mock_get.call_args
            params = call_args.kwargs.get("params", {})
            self.assertNotEqual(params.get("name"), "my_voice.pt",
                                "preview_voice should not force .pt extension")


if __name__ == "__main__":
    unittest.main()
