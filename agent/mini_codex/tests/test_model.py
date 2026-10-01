import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError

from mini_codex.model import ChatModel, NoRedirect
from mini_codex.runtime import Runtime
from mini_codex.store import Store
from mini_codex.tools import WorkspaceTools


def response(message):
    return io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": message}]}).encode())


class ModelContractTests(unittest.TestCase):
    def test_adapter_and_harness_round_trip(self):
        model = ChatModel("test-model", "https://example.org/v1", "test-secret")
        calls = [{"id": "call_read", "type": "function", "function": {
            "name": "read_file", "arguments": '{"path":"hello.py"}'}}]
        model.opener = Mock()
        model.opener.open.side_effect = [
            response({"role": "assistant", "content": None, "tool_calls": calls,
                      "reasoning_content": "provider-specific state"}),
            response({"role": "assistant", "content": "Read complete"}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hello.py").write_text("print(42)\n")
            store = Store(root)
            try:
                sid = store.create()
                tools = WorkspaceTools(root, lambda *a: False)
                self.assertEqual(Runtime(model, tools, store, sid).run("Read hello.py").answer, "Read complete")
                first = model.opener.open.call_args_list[0].args[0]
                self.assertEqual(first.full_url, "https://example.org/v1/chat/completions")
                self.assertEqual(first.get_header("Authorization"), "Bearer test-secret")
                payload = json.loads(first.data)
                self.assertEqual(payload["model"], "test-model")
                self.assertEqual(payload["tool_choice"], "auto")
                second = json.loads(model.opener.open.call_args_list[1].args[0].data)
                observation = second["messages"][-1]
                self.assertEqual(observation["tool_call_id"], "call_read")
                self.assertIn("print(42)", json.loads(observation["content"])["content"])
                self.assertEqual(second["messages"][-2]["reasoning_content"], "provider-specific state")
                self.assertNotIn("test-secret", json.dumps(store.load(sid)))
            finally:
                store.close()

    def test_http_errors_do_not_echo_provider_body(self):
        model = ChatModel("test", "https://example.org/v1", "test-secret")
        model.opener = Mock()
        model.opener.open.side_effect = HTTPError(model.url, 401, "test-secret", {}, io.BytesIO(b"test-secret"))
        with self.assertRaisesRegex(RuntimeError, "HTTP 401") as caught:
            model.complete([], [])
        self.assertNotIn("test-secret", str(caught.exception))

    def test_truncated_model_output_is_not_executed(self):
        model = ChatModel("test", "https://example.org/v1", "")
        model.opener = Mock()
        model.opener.open.return_value = io.BytesIO(json.dumps({"choices": [{"finish_reason": "length"}]}).encode())
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            model.complete([], [])

    def test_authorization_is_not_forwarded_on_redirect(self):
        self.assertIsNone(NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.example"))


if __name__ == "__main__":
    unittest.main()
