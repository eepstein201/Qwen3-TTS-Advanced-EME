import unittest
from unittest.mock import patch, MagicMock
import requests as _requests


class TestServerRequest(unittest.TestCase):
    def _import(self):
        from qwen3_tts.core.http_client import server_request
        return server_request

    @patch("qwen3_tts.core.http_client.requests.request")
    @patch("qwen3_tts.core.http_client.load_config")
    def test_01_allowed_host_succeeds(self, mock_load, mock_req):
        mock_load.return_value = {"server": {"host": "127.0.0.1", "port": 5123}}
        mock_req.return_value = MagicMock(status_code=200)
        fn = self._import()
        resp = fn("GET", "/health")
        self.assertEqual(resp.status_code, 200)
        # URL passed must be the validated base + path (positional arg index 1)
        actual_url = mock_req.call_args[0][1]   # positional: (method, url, ...)
        self.assertEqual(actual_url, "http://127.0.0.1:5123/health")

    @patch("qwen3_tts.core.http_client.load_config")
    def test_02_disallowed_host_raises(self, mock_load):
        mock_load.return_value = {"server": {"host": "evil.example.com", "port": 5123}}
        fn = self._import()
        with self.assertRaises(ValueError):
            fn("GET", "/health")

    @patch("qwen3_tts.core.http_client.load_config")
    def test_03_disallowed_zero_host_raises(self, mock_load):
        mock_load.return_value = {"server": {"host": "0.0.0.0", "port": 5123}}
        fn = self._import()
        with self.assertRaises(ValueError):
            fn("GET", "/health")

    @patch("qwen3_tts.core.http_client.load_config")
    def test_04_path_with_scheme_raises(self, mock_load):
        mock_load.return_value = {"server": {"host": "127.0.0.1", "port": 5123}}
        fn = self._import()
        with self.assertRaises(ValueError):
            fn("GET", "http://evil.example.com/steal")

    @patch("qwen3_tts.core.http_client.load_config")
    def test_05_path_without_leading_slash_raises(self, mock_load):
        mock_load.return_value = {"server": {"host": "127.0.0.1", "port": 5123}}
        fn = self._import()
        with self.assertRaises(ValueError):
            fn("GET", "health")

    @patch("qwen3_tts.core.http_client.requests.request")
    @patch("qwen3_tts.core.http_client.auth_headers")
    @patch("qwen3_tts.core.http_client.load_config")
    def test_06_auth_header_attached(self, mock_load, mock_auth, mock_req):
        mock_load.return_value = {"server": {"host": "127.0.0.1", "port": 5123}}
        mock_auth.return_value = {"Authorization": "Bearer test-token"}
        mock_req.return_value = MagicMock(status_code=200)
        fn = self._import()
        fn("GET", "/health")
        sent_headers = mock_req.call_args.kwargs.get("headers", {})
        self.assertEqual(sent_headers.get("Authorization"), "Bearer test-token")

    @patch("qwen3_tts.core.http_client.requests.request")
    @patch("qwen3_tts.core.http_client.load_config")
    def test_07_request_exception_propagates(self, mock_load, mock_req):
        mock_load.return_value = {"server": {"host": "127.0.0.1", "port": 5123}}
        mock_req.side_effect = _requests.ConnectionError("boom")
        fn = self._import()
        with self.assertRaises(_requests.ConnectionError):
            fn("GET", "/health")


    @patch("qwen3_tts.core.http_client.load_config")
    def test_08_path_with_query_string_raises(self, mock_load):
        mock_load.return_value = {"server": {"host": "127.0.0.1", "port": 5123}}
        fn = self._import()
        with self.assertRaises(ValueError):
            fn("GET", "/health?token=leak")

    @patch("qwen3_tts.core.http_client.load_config")
    def test_09_path_with_fragment_raises(self, mock_load):
        mock_load.return_value = {"server": {"host": "127.0.0.1", "port": 5123}}
        fn = self._import()
        with self.assertRaises(ValueError):
            fn("GET", "/health#inject")

    @patch("qwen3_tts.core.http_client.load_config")
    def test_10_invalid_method_raises(self, mock_load):
        mock_load.return_value = {"server": {"host": "127.0.0.1", "port": 5123}}
        fn = self._import()
        with self.assertRaises(ValueError):
            fn("BADMETHOD", "/health")

    @patch("qwen3_tts.core.http_client.requests.request")
    @patch("qwen3_tts.core.http_client.load_config")
    def test_11_stream_flag_forwarded(self, mock_load, mock_req):
        mock_load.return_value = {"server": {"host": "127.0.0.1", "port": 5123}}
        mock_req.return_value = MagicMock(status_code=200)
        fn = self._import()
        fn("GET", "/generate-stream", stream=True)
        self.assertTrue(mock_req.call_args.kwargs.get("stream"))


if __name__ == "__main__":
    unittest.main()
