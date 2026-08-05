from pathlib import Path

from agent_blueprint.llm import DemoLLM
from agent_blueprint.memory import SQLiteMemory
from agent_blueprint.policy import AgentPolicy
from agent_blueprint.runtime import AgentRuntime
from agent_blueprint.tools import build_default_tools
from agent_blueprint.tracing import JsonlTracer


def test_demo_runtime_completes(tmp_path: Path) -> None:
    tools = build_default_tools(tmp_path / "workspace")
    runtime = AgentRuntime(
        llm=DemoLLM(),
        memory=SQLiteMemory(tmp_path / "agent.sqlite3"),
        tools=tools,
        policy=AgentPolicy(
            allowed_tools=tools.names(),
            max_steps=5,
            max_tool_calls=3,
        ),
        tracer=JsonlTracer(tmp_path / "trace.jsonl"),
        approval_callback=lambda _name, _args: True,
    )

    result = runtime.run(goal="计算 (17 + 5) * 3", namespace="test")

    assert result.status == "completed"
    assert result.tool_call_count == 1
    assert result.observations[-1].output == 66
    assert "66" in (result.final_answer or "")
