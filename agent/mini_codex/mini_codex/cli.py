"""Terminal UI. No API keys are read from project files or written to sessions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sqlite3
import sys

from .model import ChatModel, DemoModel
from .runtime import Runtime
from .store import Store
from .tools import WorkspaceTools


def display(text: str) -> str:
    # Source files/model text must not inject terminal control sequences.
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", "", str(text))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Mini Codex — 本地编程 Agent（Python 3.11+，零运行依赖）")
    result.add_argument("prompt", nargs="?", help="单次任务；省略则进入交互模式")
    result.add_argument("--workspace", type=Path, default=Path.cwd(), help="工作目录（必须已存在）")
    result.add_argument("--model", default=os.getenv("MINI_CODEX_MODEL", os.getenv("OPENAI_MODEL", "")))
    result.add_argument("--base-url", default=os.getenv("MINI_CODEX_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")))
    result.add_argument("--demo", action="store_true", help="离线演示：创建、读取、修改演示文件")
    result.add_argument("--resume", metavar="SESSION_ID", help="恢复工作目录中的会话")
    result.add_argument("--list-sessions", action="store_true", help="列出最近 50 个会话")
    result.add_argument("--read-only", action="store_true", help="仅向模型开放读取和搜索工具")
    result.add_argument("--auto-approve-edits", action="store_true", help="自动批准文件编辑；命令仍需逐次批准")
    result.add_argument("--allow-shell", action="store_true", help="启用命令工具，每次执行前展示 argv 并询问；不是 OS 沙箱")
    result.add_argument("--max-steps", type=int, default=20)
    result.add_argument("--max-tool-calls", type=int, default=40)
    result.add_argument("--context-chars", type=int, default=80_000, help="模型请求上下文的字符上限（不是 token 数）")
    result.add_argument("--json", action="store_true", help="stdout 输出 JSONL 事件；交互与审批在 stderr")
    return result


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parser().parse_args(argv)
    store = None

    def emit(kind: str, data: dict):
        if args.json:
            print(json.dumps({"event": kind, **data}, ensure_ascii=False), flush=True)
        elif kind == "assistant" and data.get("content"):
            print(display(data["content"]), flush=True)
        elif kind == "tool_started":
            print(f"  > {data['name']}", file=sys.stderr, flush=True)
        elif kind == "tool_completed":
            result = data["result"]
            if not result["ok"]:
                print("  ! " + display(result.get("error", "")), file=sys.stderr)
            elif "exit_code" in result:
                print(f"  exit={result['exit_code']} timeout={result['timed_out']} output_limit={result['output_limit_reached']}", file=sys.stderr)
                print(display(result["output"]), file=sys.stderr)
        elif kind in {"recovered", "context_trimmed"}:
            print(f"  [{kind}] {data}", file=sys.stderr)

    def approve(kind: str, preview: str) -> bool:
        print(f"\n[{kind}]\n{display(preview)}", file=sys.stderr)
        # Batch jobs fail closed. --auto-approve-edits is explicit and edits-only.
        if not sys.stdin.isatty():
            print("非交互输入：拒绝需要确认的操作。", file=sys.stderr)
            return False
        print("批准本次操作？[y/N] ", end="", file=sys.stderr, flush=True)
        return sys.stdin.readline().strip().lower() in {"y", "yes"}

    try:
        root = args.workspace.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Workspace must be a directory")
        store = Store(root)
        with store.exclusive():
            if args.list_sessions:
                for item in store.list_sessions():
                    if args.json:
                        emit("session", item)
                    else:
                        print(f"{item['id']}  {item['status']:12}  {item['updated_at']}")
                return 0
            model = DemoModel() if args.demo else ChatModel(
                args.model, args.base_url, os.getenv("MINI_CODEX_API_KEY", os.getenv("OPENAI_API_KEY", "")))
            sid = args.resume or store.create()
            tools = WorkspaceTools(root, approve, allow_shell=args.allow_shell,
                                   auto_approve_edits=args.auto_approve_edits, read_only=args.read_only)
            runtime = Runtime(model, tools, store, sid, emit=emit, max_steps=args.max_steps,
                              max_calls=args.max_tool_calls, context_chars=args.context_chars)
            emit("session_started", {"session_id": sid, "workspace": str(root), "demo": args.demo})
            if not args.json:
                print(f"Mini Codex | {'离线演示' if args.demo else args.model}\n会话：{sid}\n工作目录：{root}", file=sys.stderr)
            if args.prompt:
                result = runtime.run(args.prompt)
                if result.status != "completed" and not args.json:
                    print(result.answer)
                return 0 if result.status == "completed" else 2
            if not sys.stdin.isatty():
                raise ValueError("Non-interactive mode requires a prompt argument")
            print("输入任务开始；/exit 退出，/session 显示会话 ID。Ctrl+C 停止并保存。", file=sys.stderr)
            while True:
                print("\n你 > ", end="", file=sys.stderr, flush=True)
                line = sys.stdin.readline()
                if not line or line.strip() == "/exit":
                    return 0
                prompt = line.strip()
                if prompt == "/session":
                    print(sid, file=sys.stderr)
                elif prompt:
                    result = runtime.run(prompt)
                    if result.status != "completed" and not args.json:
                        print(result.answer)
    except KeyboardInterrupt:
        print("\n已停止，会话已保存。使用 --resume 恢复。", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        if args.json:
            emit("error", {"message": str(exc)})
        else:
            print("错误：" + display(exc), file=sys.stderr)
        return 1
    finally:
        if store:
            store.close()
