"""Model adapters. The harness owns tools; the model only requests calls."""

from __future__ import annotations

import json
import os
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class Model(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> dict: ...


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward the Authorization header to a different origin.
        return None


class ChatModel:
    """OpenAI-compatible Chat Completions, using Python's standard library."""

    def __init__(self, model: str, base_url: str, api_key: str, timeout: float = 90):
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            raise ValueError("base_url must be an HTTP(S) URL")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Use HTTPS for remote model endpoints")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain credentials, a query or a fragment")
        if not model.strip():
            raise ValueError("Set MINI_CODEX_MODEL or pass --model")
        self.model, self.api_key, self.timeout = model, api_key, timeout
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.opener = build_opener(NoRedirect())

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        payload = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = Request(self.url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise RuntimeError("Model response exceeded 2 MB")
            data = json.loads(raw)
            choice = data["choices"][0]
            if choice.get("finish_reason") in {"length", "content_filter"}:
                raise RuntimeError("Model response incomplete: " + choice["finish_reason"])
            message = choice["message"]
            if message.get("role") != "assistant":
                raise ValueError("Expected an assistant message")
            result = {"role": "assistant", "content": message.get("content")}
            for key in ("tool_calls", "reasoning_content", "refusal"):
                if message.get(key) is not None:
                    result[key] = message[key]
            return result
        except HTTPError as exc:
            # Provider error bodies can echo user data; avoid logging them or secrets.
            raise RuntimeError(f"Model endpoint returned HTTP {exc.code}; check URL, model and API key") from None
        except (URLError, TimeoutError):
            raise RuntimeError("Model request failed or timed out; the session is saved") from None
        except (KeyError, IndexError, TypeError, ValueError):
            raise RuntimeError("Invalid Chat Completions response from the model endpoint") from None


class DemoModel:
    """Deterministic tool round-trip, deliberately not a language model."""

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        start = max(i for i, m in enumerate(messages) if m["role"] == "user")
        observations = [json.loads(m["content"]) for m in messages[start:] if m["role"] == "tool"]
        if observations and not observations[-1].get("ok"):
            return {"role": "assistant", "content": "离线演示停止：" + str(observations[-1].get("error"))}
        steps = [
            ("write_file", {"path": "mini_codex_demo.py", "content": 'def greet(name):\n    return f"Hello, {name}!"\n', "expected_sha256": "new"}),
            ("read_file", {"path": "mini_codex_demo.py"}),
            ("edit_file", {"path": "mini_codex_demo.py", "old_text": "Hello,", "new_text": "你好，"}),
            ("read_file", {"path": "mini_codex_demo.py"}),
        ]
        index = len(observations)
        if index >= len(steps):
            return {"role": "assistant", "content": "离线演示完成：创建、读取、修改并重新读取了 mini_codex_demo.py。没有调用真实模型，也没有运行代码测试。"}
        name, arguments = steps[index]
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"demo_{os.urandom(6).hex()}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }]}
