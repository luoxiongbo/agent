import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from google.genai import errors


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class McpAgentErrorTest(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("agent.mcp.agent")

    def test_create_gemini_client_uses_ai_studio_api_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-key"}, clear=True):
            with patch.object(self.module.genai, "Client") as client_cls:
                client = self.module.create_gemini_client()

        self.assertIs(client, client_cls.return_value)
        client_cls.assert_called_once_with(api_key="gemini-key")

    def test_create_gemini_client_requires_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                self.module.create_gemini_client()

    def test_create_mcp_server_parameters_uses_server_next_to_agent_file(self):
        server_parameters = self.module.create_mcp_server_parameters()

        self.assertEqual(server_parameters.command, sys.executable)
        self.assertEqual(
            Path(server_parameters.args[0]),
            Path(self.module.__file__).with_name("server.py"),
        )

    def test_create_function_response_part_matches_installed_google_genai_sdk(self):
        part = self.module.create_function_response_part(
            "get_weather",
            {
                "content": '{"city": "Beijing", "temperature": 25}',
            },
        )

        self.assertEqual(part.function_response.name, "get_weather")
        self.assertEqual(
            part.function_response.response["result"]["content"],
            '{"city": "Beijing", "temperature": 25}',
        )

    def test_formats_gemini_connection_error(self):
        message = self.module.format_gemini_call_error(
            httpx.ConnectError("All connection attempts failed")
        )

        self.assertIn("Gemini API 网络连接失败", message)
        self.assertIn("All connection attempts failed", message)

    def test_formats_vertex_api_disabled_error_with_actionable_hint(self):
        error = errors.ClientError(
            403,
            {
                "error": {
                    "code": 403,
                    "message": "Agent Platform API has not been used",
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                            "reason": "SERVICE_DISABLED",
                            "domain": "googleapis.com",
                            "metadata": {
                                "consumer": "projects/demo-project",
                                "service": "aiplatform.googleapis.com",
                                "activationUrl": "https://console.example/enable",
                            },
                        }
                    ],
                }
            },
            None,
        )

        message = self.module.format_google_client_error(error)

        self.assertIn("Vertex AI API", message)
        self.assertIn("demo-project", message)
        self.assertIn("aiplatform.googleapis.com", message)
        self.assertIn("https://console.example/enable", message)


if __name__ == "__main__":
    unittest.main()
