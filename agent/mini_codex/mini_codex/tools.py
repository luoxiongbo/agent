"""Workspace tools and explicit, reviewable mutation approvals.

Path checks protect these file tools. Approved commands run with the user's OS
permissions and are NOT confined by these checks. This is not an OS sandbox.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from typing import Callable


MAX_FILE = 256_000
MAX_OUTPUT = 16_000
MAX_COMMAND_LOG = 2_000_000
SKIP = {".git", ".mini-codex", ".venv", "venv", "node_modules", "__pycache__", ".idea"}
Approval = Callable[[str, str], bool]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required,
                           "additionalProperties": False}}}


STRING = {"type": "string"}
SCHEMAS = [
    schema("list_files", "List workspace files; hidden/internal files are excluded. Results may be truncated.",
           {"path": STRING}, []),
    schema("read_file", "Read UTF-8 lines and obtain sha256 for a later write_file. Lines start at 1.",
           {"path": STRING, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, ["path"]),
    schema("search", "Search a literal string in UTF-8 workspace files; returns paths and line numbers.",
           {"query": STRING, "path": STRING}, ["query"]),
    schema("write_file", "Create/replace a UTF-8 file. Use expected_sha256='new' for a new file; otherwise use the hash from read_file. Shows diff for approval.",
           {"path": STRING, "content": STRING, "expected_sha256": STRING}, ["path", "content", "expected_sha256"]),
    schema("edit_file", "Replace exactly one nonempty literal occurrence in an existing UTF-8 file. Shows diff for approval.",
           {"path": STRING, "old_text": STRING, "new_text": STRING}, ["path", "old_text", "new_text"]),
    schema("run_command", "Run an argv command after explicit human approval. No shell interpolation. Uses workspace as cwd. Not an OS sandbox.",
           {"argv": {"type": "array", "items": STRING}, "timeout_seconds": {"type": "integer"}}, ["argv"]),
]


class WorkspaceTools:
    def __init__(self, root: Path, approve: Approval, *, allow_shell: bool = False,
                 auto_approve_edits: bool = False, read_only: bool = False):
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Workspace must be a directory")
        self.approve = approve
        self.allow_shell = allow_shell and not read_only
        self.auto_approve_edits = auto_approve_edits and not read_only
        self.read_only = read_only

    def schemas(self) -> list[dict]:
        return [s for s in SCHEMAS if
                (self.allow_shell or s["function"]["name"] != "run_command") and
                (not self.read_only or s["function"]["name"] not in {"write_file", "edit_file"})]

    def path(self, value: str) -> Path:
        # Resolve links and Windows junctions before testing the boundary.
        candidate = Path(value)
        if candidate.is_absolute() or candidate.drive or ":" in value:
            raise ValueError("Use a workspace-relative path without a drive or stream")
        candidate = (self.root / candidate).resolve()
        if not candidate.is_relative_to(self.root):
            raise PermissionError("Path escapes the workspace")
        parts = candidate.relative_to(self.root).parts
        if any(part.lower() in SKIP or part.startswith(".") for part in parts):
            raise PermissionError("Hidden and internal paths are excluded")
        if candidate.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
            raise PermissionError("Private-key file types are excluded")
        return candidate

    def invoke(self, name: str, args: dict) -> dict:
        allowed = {s["function"]["name"]: s["function"]["parameters"] for s in self.schemas()}
        try:
            if name not in allowed:
                raise PermissionError("Unknown or disabled tool: " + name)
            spec = allowed[name]
            if not isinstance(args, dict):
                raise ValueError("Arguments must be an object")
            if set(args) - set(spec["properties"]) or set(spec["required"]) - set(args):
                raise ValueError("Unknown or missing tool arguments")
            for key, value in args.items():
                kind = spec["properties"][key]["type"]
                expected = {"string": str, "integer": int, "array": list}[kind]
                if type(value) is not expected:
                    raise ValueError(f"{key} must be {kind}")
                if kind == "array" and any(not isinstance(v, str) for v in value):
                    raise ValueError(f"{key} must contain strings")
            return {"ok": True, **getattr(self, name)(**args)}
        except (OSError, ValueError, TypeError, UnicodeError, subprocess.SubprocessError) as exc:
            return {"ok": False, "error": str(exc)[:2000], "error_type": type(exc).__name__}

    def _files(self, path: str):
        base = self.path(path)
        if base.is_file():
            yield base
            return
        if not base.is_dir():
            raise FileNotFoundError(path)
        visited = 0
        for directory, dirs, files in os.walk(base, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d.lower() not in SKIP
                             and not (Path(directory) / d).is_symlink()
                             and not (hasattr(Path(directory) / d, "is_junction") and (Path(directory) / d).is_junction()))
            for name in sorted(files):
                visited += 1
                if visited > 5000:
                    return
                try:
                    item = self.path(str((Path(directory) / name).relative_to(self.root)))
                except (OSError, ValueError):
                    continue
                yield item

    def list_files(self, path: str = ".") -> dict:
        files = []
        for item in self._files(path):
            files.append(item.relative_to(self.root).as_posix())
            if len(files) >= 300 or sum(map(len, files)) > MAX_OUTPUT:
                return {"files": files, "truncated": True}
        return {"files": files, "truncated": False, "scan_limit": 5000}

    @staticmethod
    def _read(path: Path) -> bytes:
        with path.open("rb") as stream:
            data = stream.read(MAX_FILE + 1)
        if len(data) > MAX_FILE:
            raise ValueError(f"File exceeds {MAX_FILE} bytes")
        if b"\x00" in data:
            raise ValueError("Binary files are unsupported")
        data.decode("utf-8")
        return data

    def read_file(self, path: str, start_line: int = 1, end_line: int = 200) -> dict:
        if start_line < 1 or end_line < start_line or end_line - start_line > 1000:
            raise ValueError("Invalid line range (maximum 1001 lines)")
        data = self._read(self.path(path))
        lines = data.decode("utf-8").splitlines()
        selected = "\n".join(f"{i + 1}: {lines[i]}" for i in range(start_line - 1, min(end_line, len(lines))))
        return {"path": path, "sha256": digest(data), "total_lines": len(lines),
                "content": selected[:MAX_OUTPUT], "truncated": len(selected) > MAX_OUTPUT or end_line < len(lines)}

    def search(self, query: str, path: str = ".") -> dict:
        if not query or len(query) > 1000:
            raise ValueError("query must contain 1 to 1000 characters")
        matches = []
        scanned = 0
        for item in self._files(path):
            scanned += 1
            if scanned > 500:
                return {"matches": matches, "truncated": True}
            try:
                content = self._read(item).decode("utf-8")
            except (OSError, ValueError, UnicodeError):
                continue
            for number, line in enumerate(content.splitlines(), 1):
                if query in line:
                    matches.append({"path": item.relative_to(self.root).as_posix(), "line": number, "text": line[:300]})
                    if len(matches) >= 40:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}

    def _change(self, value: str, before: bytes | None, after: str) -> dict:
        path = self.path(value)
        data = after.encode("utf-8")
        if len(data) > MAX_FILE or b"\x00" in data:
            raise ValueError("New content is too large or contains NUL")
        previous = (before or b"").decode("utf-8")
        diff_lines = difflib.unified_diff(previous.splitlines(keepends=True), after.splitlines(keepends=True),
                                         fromfile=value + " (before)", tofile=value + " (after)")
        diff = "".join(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
                       for line in diff_lines)
        if before == data:
            return {"path": value, "changed": False, "sha256": digest(data)}
        # Full diff is shown, with file size bounded above; approval must cover all edits.
        if not self.auto_approve_edits and not self.approve("edit", diff or f"Create empty file: {value}"):
            raise PermissionError("User denied file change")
        path = self.path(value)
        current = self._read(path) if path.exists() else None
        if current != before:
            raise ValueError("File changed during approval; read it again before editing")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=".mini-codex-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if path.exists():
                os.chmod(temp, path.stat().st_mode)
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        return {"path": value, "changed": True, "sha256": digest(data), "diff": diff[:MAX_OUTPUT]}

    def write_file(self, path: str, content: str, expected_sha256: str) -> dict:
        target = self.path(path)
        before = self._read(target) if target.exists() else None
        actual = "new" if before is None else digest(before)
        if actual != expected_sha256:
            raise ValueError("File hash mismatch; read the file before replacing it")
        return self._change(path, before, content)

    def edit_file(self, path: str, old_text: str, new_text: str) -> dict:
        before = self._read(self.path(path))
        content = before.decode("utf-8")
        if not old_text or content.count(old_text) != 1:
            raise ValueError("old_text must match exactly once; include more surrounding text")
        return self._change(path, before, content.replace(old_text, new_text, 1))

    def run_command(self, argv: list[str], timeout_seconds: int = 30) -> dict:
        if not argv or not argv[0].strip() or len(argv) > 100 or any("\x00" in arg for arg in argv):
            raise ValueError("Invalid argv")
        if not 1 <= timeout_seconds <= 120:
            raise ValueError("timeout_seconds must be between 1 and 120")
        preview = json.dumps({"argv": argv, "cwd": str(self.root), "timeout_seconds": timeout_seconds}, ensure_ascii=False)
        if not self.approve("command", preview + "\nThis command has your OS permissions, including access outside the workspace."):
            raise PermissionError("User denied command")
        env = {key: value for key, value in os.environ.items()
               if not any(word in key.upper() for word in ("TOKEN", "SECRET", "PASSWORD", "API_KEY"))}
        kwargs = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NO_WINDOW}
        # Use a file instead of PIPE so a verbose child cannot exhaust parent RAM.
        with tempfile.TemporaryFile() as output:
            proc = subprocess.Popen(argv, cwd=self.root, env=env, stdin=subprocess.DEVNULL,
                                    stdout=output, stderr=subprocess.STDOUT, shell=False, **kwargs)
            timed_out = False
            output_limit_reached = False
            try:
                deadline = time.monotonic() + timeout_seconds
                while proc.poll() is None:
                    timed_out = time.monotonic() >= deadline
                    output_limit_reached = os.fstat(output.fileno()).st_size > MAX_COMMAND_LOG
                    if timed_out or output_limit_reached:
                        self._terminate(proc)
                        break
                    time.sleep(0.02)
            except BaseException:
                self._terminate(proc)
                raise
            output.seek(0)
            raw = output.read(MAX_OUTPUT + 1)
        return {"exit_code": proc.returncode, "timed_out": timed_out, "output_limit_reached": output_limit_reached,
                "output": raw[:MAX_OUTPUT].decode("utf-8", errors="replace"), "truncated": len(raw) > MAX_OUTPUT}

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        if os.name == "nt":
            try:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
            except (OSError, subprocess.SubprocessError):
                pass
            finally:
                if proc.poll() is None:
                    proc.kill()
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait(timeout=5)
