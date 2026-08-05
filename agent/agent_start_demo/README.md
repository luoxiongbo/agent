# Agent Blueprint

一个**教学型、可直接运行、尽量少依赖**的 Agent 完整实现。它不依赖 LangGraph、CrewAI
或 AutoGen，目的是把 Agent 的核心机制完整摊开，方便阅读、调试和二次开发。

## 已实现的能力

- Goal-driven：围绕目标持续运行
- Plan：生成和调整高层计划
- ReAct：选择动作、调用工具、观察结果
- Reflection：每一步后检查是否继续、重规划、完成或阻塞
- Working state：结构化运行状态和检查点
- Long-term memory：SQLite 长期记忆、检索、过期和置信度
- Tools：工具注册、参数校验、文件沙箱
- Policy：工具白名单、调用预算、副作用审批
- Observability：JSONL 轨迹、运行状态、错误记录
- Termination：最大步数、最大工具调用数、完成与阻塞判断
- Offline demo：无 API Key 也能运行演示和测试
- OpenAI adapter：通过 Responses API 使用真实模型

## 目录

```text
src/agent_blueprint/
├── models.py      # 状态、计划、动作、观察、反思等数据结构
├── llm.py         # LLM 接口、OpenAI 适配器、离线演示模型
├── memory.py      # SQLite 记忆与检查点
├── tools.py       # 工具注册、参数验证、安全文件工具、计算器
├── policy.py      # 权限、预算和审批
├── tracing.py     # JSONL 可观测性
├── runtime.py     # Plan-Act-Observe-Reflect 主循环
└── cli.py         # 命令行入口
```

## 快速运行：离线模式

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

agent-blueprint "计算 (17 + 5) * 3" --demo
```

离线模式使用一个确定性的 `DemoLLM`，用于理解流程、跑测试，不代表通用智能。

## 使用真实模型

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-5.5"

agent-blueprint "读取 notes.txt，总结内容并保存到 summary.txt"
```

默认工作目录是 `.agent/workspace`。文件工具不能访问该目录之外的路径。

## Python 调用

```python
from pathlib import Path

from agent_blueprint.llm import OpenAIResponsesLLM
from agent_blueprint.memory import SQLiteMemory
from agent_blueprint.policy import AgentPolicy
from agent_blueprint.runtime import AgentRuntime
from agent_blueprint.tools import build_default_tools
from agent_blueprint.tracing import JsonlTracer

workspace = Path(".agent/workspace")
runtime = AgentRuntime(
    llm=OpenAIResponsesLLM(model="gpt-5.5"),
    memory=SQLiteMemory(".agent/agent.sqlite3"),
    tools=build_default_tools(workspace),
    policy=AgentPolicy(
        allowed_tools={"calculator", "list_files", "read_file", "write_file"},
        max_steps=12,
        max_tool_calls=10,
    ),
    tracer=JsonlTracer(".agent/traces.jsonl"),
)

result = runtime.run(
    goal="计算 23 * 19，并把结果写入 result.txt",
    namespace="demo-user",
)
print(result.final_answer)
```

## 人工审批

`write_file` 被标记为有副作用。CLI 默认会询问是否批准。程序中可注入审批函数：

```python
def approve(tool_name: str, arguments: dict) -> bool:
    return tool_name == "write_file" and arguments.get("path") == "result.txt"
```

然后把它传给 `AgentRuntime(..., approval_callback=approve)`。

## 记忆模型

记忆包含：

- `kind`：semantic / episodic / procedural / preference
- `content`
- `importance`
- `confidence`
- `namespace`
- `metadata`
- `expires_at`

当前实现用轻量的词项重叠进行检索，便于零额外基础设施运行。生产系统可把
`SQLiteMemory.search()` 替换为 embedding + 向量数据库，并保留相同接口。

## 安全边界

此项目展示了基础防线，但不是完整安全沙箱：

- 文件路径被限制在 workspace 内
- 计算器只允许受限 AST，不使用 `eval`
- 工具必须显式注册
- 副作用工具可以要求人工审批
- 有步骤数和调用数预算

生产环境还应增加容器隔离、网络出口策略、密钥代理、数据权限、审计、内容安全、
提示注入防护、幂等键和补偿事务。

## 测试

```bash
pytest
```

## 设计原则

LLM 只负责提出计划、选择动作和生成总结；真实执行由工具层完成，权限由策略层决定，
状态由运行时维护，记忆由外部存储管理，结果由 Reflection 和确定性限制共同约束。
