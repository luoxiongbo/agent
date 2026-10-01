from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mini_codex.cli import main
from mini_codex.model import ChatModel, DemoModel
from mini_codex.runtime import Runtime
from mini_codex.store import Store
from mini_codex.tools import WorkspaceTools, digest


def call(name, args, ident="c1"):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": ident, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.inputs = []

    def complete(self, messages, tools):
        self.inputs.append(messages)
        reply = next(self.replies)
        if isinstance(reply, BaseException):
            raise reply
        return reply


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tools = WorkspaceTools(self.root, lambda kind, preview: True)
        self.store = Store(self.root)
        self.sid = self.store.create()

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_demo_performs_real_file_changes_and_restores_session(self):
        runtime = Runtime(DemoModel(), self.tools, self.store, self.sid)
        result = runtime.run("离线演示")
        self.assertEqual(result.status, "completed")
        self.assertIn("你好，", (self.root / "mini_codex_demo.py").read_text(encoding="utf-8"))
        restored = Runtime(ScriptedModel([{"role": "assistant", "content": "继续完成"}]), self.tools, self.store, self.sid)
        self.assertEqual(len(restored.messages), 10)
        self.assertEqual(restored.run("继续").answer, "继续完成")

    def test_paths_hidden_files_and_disabled_commands(self):
        for path in ("../outside.txt", ".env", ".git/config", "a.txt:stream", str(self.root / "x")):
            with self.subTest(path=path):
                result = self.tools.invoke("write_file", {"path": path, "content": "x", "expected_sha256": "new"})
                self.assertFalse(result["ok"])
        self.assertFalse(self.tools.invoke("run_command", {"argv": [sys.executable, "-V"]})["ok"])

    def test_symlink_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as other:
            outside = Path(other) / "private.txt"
            outside.write_text("private")
            try:
                (self.root / "link.txt").symlink_to(outside)
            except OSError:
                self.skipTest("Symlink creation unavailable on this Windows account")
            self.assertFalse(self.tools.invoke("read_file", {"path": "link.txt"})["ok"])

    def test_edit_approval_and_stale_writes(self):
        target = self.root / "a.py"
        target.write_text("print(1)\n", encoding="utf-8")
        original_hash = digest(target.read_bytes())
        previews = []
        self.tools.approve = lambda kind, preview: previews.append(preview) or False
        result = self.tools.invoke("edit_file", {"path": "a.py", "old_text": "1", "new_text": "2"})
        self.assertFalse(result["ok"])
        self.assertIn("+print(2)", previews[0])
        self.assertEqual(target.read_text(), "print(1)\n")
        target.write_text("print(3)\n")
        result = self.tools.invoke("write_file", {"path": "a.py", "content": "bad", "expected_sha256": original_hash})
        self.assertFalse(result["ok"])

    def test_edit_rechecks_after_approval(self):
        target = self.root / "a.txt"
        target.write_text("one")
        def approve(kind, preview):
            target.write_text("external edit")
            return True
        self.tools.approve = approve
        result = self.tools.invoke("edit_file", {"path": "a.txt", "old_text": "one", "new_text": "two"})
        self.assertFalse(result["ok"])
        self.assertEqual(target.read_text(), "external edit")

    def test_read_only_and_ambiguous_edit(self):
        (self.root / "a.txt").write_text("one one")
        result = self.tools.invoke("edit_file", {"path": "a.txt", "old_text": "one", "new_text": "two"})
        self.assertFalse(result["ok"])
        readonly = WorkspaceTools(self.root, lambda *a: True, read_only=True, auto_approve_edits=True, allow_shell=True)
        self.assertEqual({s["function"]["name"] for s in readonly.schemas()}, {"read_file", "search", "list_files"})
        self.assertFalse(readonly.invoke("write_file", {"path": "new", "content": "x", "expected_sha256": "new"})["ok"])

    def test_search_and_line_ranges(self):
        (self.root / "a.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
        match = self.tools.invoke("search", {"query": "two"})["matches"][0]
        self.assertEqual((match["path"], match["line"]), ("a.py", 2))
        result = self.tools.invoke("read_file", {"path": "a.py", "start_line": 2, "end_line": 2})
        self.assertEqual(result["content"], "2: two")
        self.assertTrue(result["truncated"])
        self.assertFalse(self.tools.invoke("read_file", {"path": "a.py", "start_line": True})["ok"])

    def test_budget_resolves_every_call_without_executing_excess(self):
        message = call("write_file", {"path": "first", "content": "ok", "expected_sha256": "new"})
        message["tool_calls"] += call("write_file", {"path": "second", "content": "no", "expected_sha256": "new"}, "c2")["tool_calls"]
        result = Runtime(ScriptedModel([message]), self.tools, self.store, self.sid, max_calls=1).run("写文件")
        self.assertEqual(result.status, "stopped")
        self.assertTrue((self.root / "first").exists())
        self.assertFalse((self.root / "second").exists())
        self.assertEqual(len([m for m in self.store.load(self.sid) if m["role"] == "tool"]), 2)

    def test_recovery_never_replays_pending_calls(self):
        pending = call("write_file", {"path": "never.txt", "content": "x", "expected_sha256": "new"})
        self.store.save(self.sid, [{"role": "user", "content": "write"}, pending], "running", "crash", {})
        runtime = Runtime(DemoModel(), self.tools, self.store, self.sid)
        self.assertFalse((self.root / "never.txt").exists())
        self.assertEqual(json.loads(runtime.messages[-1]["content"])["error_type"], "Interrupted")

    def test_tool_json_error_reaches_model(self):
        bad = call("list_files", {})
        bad["tool_calls"][0]["function"]["arguments"] = "{broken"
        model = ScriptedModel([bad, {"role": "assistant", "content": "Handled"}])
        Runtime(model, self.tools, self.store, self.sid).run("test")
        observation = json.loads(model.inputs[-1][-1]["content"])
        self.assertFalse(observation["ok"])

    def test_model_error_preserves_session(self):
        runtime = Runtime(ScriptedModel([RuntimeError("offline")]), self.tools, self.store, self.sid)
        with self.assertRaisesRegex(RuntimeError, "offline"):
            runtime.run("remember this")
        self.assertEqual(self.store.load(self.sid)[0]["content"], "remember this")
        self.assertEqual(self.store.list_sessions()[0]["status"], "failed")

    def test_context_trims_complete_old_turns_only(self):
        messages = [{"role": "user", "content": "old" * 5000}, call("list_files", {}),
                    {"role": "tool", "tool_call_id": "c1", "content": "{}"},
                    {"role": "assistant", "content": "done"}]
        self.store.save(self.sid, messages, "idle", "seed", {})
        model = ScriptedModel([{"role": "assistant", "content": "new answer"}])
        Runtime(model, self.tools, self.store, self.sid, context_chars=8000).run("new question")
        self.assertEqual(len(model.inputs[0]), 2)
        self.assertIn("omitted", model.inputs[0][0]["content"])
        self.assertEqual(len(self.store.load(self.sid)), 6)

    def test_command_success_failure_and_timeout(self):
        approvals = []
        tools = WorkspaceTools(self.root, lambda kind, preview: approvals.append(kind) or True, allow_shell=True)
        result = tools.invoke("run_command", {"argv": [sys.executable, "-c", "print('verified')"]})
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("verified", result["output"])
        result = tools.invoke("run_command", {"argv": [sys.executable, "-c", "raise SystemExit(7)"]})
        self.assertEqual(result["exit_code"], 7)
        result = tools.invoke("run_command", {"argv": [sys.executable, "-c", "import time; time.sleep(10)"], "timeout_seconds": 1})
        self.assertTrue(result["timed_out"])
        self.assertEqual(approvals, ["command"] * 3)

    def test_workspace_lock_prevents_second_process(self):
        script = "from pathlib import Path; from mini_codex.store import Store; import sys; s=Store(Path(sys.argv[1]));\nwith s.exclusive(): print('acquired')"
        with self.store.exclusive():
            result = subprocess.run([sys.executable, "-c", script, str(self.root)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Another Mini Codex", result.stderr)

    def test_cli_demo_json_events(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["demo", "--workspace", str(self.root), "--demo", "--auto-approve-edits", "--json"])
        self.assertEqual(code, 0, stderr.getvalue())
        events = [json.loads(line)["event"] for line in stdout.getvalue().splitlines()]
        self.assertEqual(events.count("tool_completed"), 4)
        self.assertEqual(events[-1], "turn_completed")

    def test_remote_endpoint_requires_https(self):
        with self.assertRaises(ValueError):
            ChatModel("model", "http://example.org/v1", "secret")
        self.assertEqual(ChatModel("local", "http://localhost:11434/v1", "").url,
                         "http://localhost:11434/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
