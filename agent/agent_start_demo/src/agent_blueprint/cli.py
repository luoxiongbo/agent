from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .llm import DemoLLM, OpenAIResponsesLLM
from .memory import SQLiteMemory
from .policy import AgentPolicy
from .runtime import AgentRuntime
from .tools import build_default_tools
from .tracing import JsonlTracer


def _approval(tool_name: str, arguments: dict[str, Any]) -> bool:
    print(f"\n需要审批的工具：{tool_name}")
    print(f"参数：{arguments}")
    answer = input("批准执行？[y/N] ").strip().lower()
    return answer in {"y", "yes"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Agent Blueprint.")
    parser.add_argument("goal", help="The goal for the agent.")
    parser.add_argument("--demo", action="store_true", help="Use the offline DemoLLM.")
    parser.add_argument("--namespace", default="default", help="Memory namespace.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--db", default=os.getenv("AGENT_DB", ".agent/agent.sqlite3"))
    parser.add_argument(
        "--workspace",
        default=os.getenv("AGENT_WORKSPACE", ".agent/workspace"),
    )
    parser.add_argument(
        "--trace",
        default=os.getenv("AGENT_TRACE", ".agent/traces.jsonl"),
    )
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-tool-calls", type=int, default=10)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Approve side-effect tools without prompting. Use carefully.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    llm = DemoLLM() if args.demo else OpenAIResponsesLLM(model=args.model)
    tools = build_default_tools(Path(args.workspace))
    approval = (lambda _name, _args: True) if args.yes else _approval

    runtime = AgentRuntime(
        llm=llm,
        memory=SQLiteMemory(args.db),
        tools=tools,
        policy=AgentPolicy(
            allowed_tools=tools.names(),
            max_steps=args.max_steps,
            max_tool_calls=args.max_tool_calls,
        ),
        tracer=JsonlTracer(args.trace, echo=False),
        approval_callback=approval,
    )

    result = runtime.run(goal=args.goal, namespace=args.namespace)
    print(f"\n状态：{result.status}")
    print(f"run_id：{result.run_id}")
    print(f"步骤：{result.step_count}，工具调用：{result.tool_call_count}")
    print(f"\n{result.final_answer or '没有最终答案。'}")


if __name__ == "__main__":
    main()
