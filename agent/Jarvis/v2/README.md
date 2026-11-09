# Jarvis v2

将 `agent/learn-claude-code/s15_integrated_harness/code.py` 按功能归为 **8 个主要 Python 文件**，让相关逻辑集中阅读，保留原有函数、工具 schema、提示词、权限检查和运行流程。原 s15 文件仍作为课程示例和回归对照。另有 `__init__.py`、`__main__.py` 两个很小的包入口文件。

## 运行

需要 Python 3.10+，依赖与原版一致：

```powershell
python -m pip install anthropic python-dotenv pyyaml
```

沿用原来的 `.env` 配置：`ANTHROPIC_API_KEY`、`MODEL_ID`，以及可选的 `ANTHROPIC_BASE_URL`、`FALLBACK_MODEL_ID`。

在仓库根目录运行：

```powershell
python -m agent.Jarvis.v2
# 或保留 code.py 启动方式：
python agent/Jarvis/v2/code.py
# 也支持直接运行 __main__.py（包括 IDE 的运行按钮）：
python agent/Jarvis/v2/__main__.py
```

也可以在目标工作目录中，通过绝对路径运行 `v2/code.py`。工作目录仍然取自启动时的 `Path.cwd()`，任务、记忆、日志等数据仍写入该工作目录下原有的 `.tasks`、`.memory`、`.transcripts` 等位置。终端的 `s15` 提示符及退出命令 `q` / `exit` 保持原样。

## 模块导航

| 模块 | 职责 |
| --- | --- |
| `code.py` | 启动入口、交互输入、终端输出、异步事件唤醒 |
| `config.py` | 环境变量、模型客户端、工作目录与运行常量 |
| `runtime.py` | 主 Agent 循环、重试与恢复、工具 schema 和处理函数注册 |
| `tools.py` | 文件与 Shell 工具、待办列表、权限钩子、模拟 MCP 服务及工具发现 |
| `tasks.py` | 持久化任务、依赖、归属、worktree 绑定与目录使用权 |
| `teams.py` | 消息总线、团队协议、计划审批、常驻队友及一次性子 Agent |
| `context.py` | 记忆、技能、系统提示词、上下文压缩及实时状态 |
| `scheduler.py` | 后台 Shell、完成通知、cron 与持久化投递 |

建议从 `code.main()` → `runtime.agent_loop()` 开始阅读；工具表也在 `runtime.py`，找到处理函数后进入对应功能文件。各文件内保留功能分段，便于定位。

## 状态与依赖

- 配置和模型客户端由 `config.py` 创建，各功能模块复用。任务归属留在 `tasks.py`，团队协议状态留在 `teams.py`，后台任务留在 `scheduler.py`，各自仍使用原来的锁。
- 终端状态和输入代理统一归 `code.py` 管理；MCP 工具权限表归 `tools.py` 管理，避免跨模块赋值产生旧副本。
- 涉及双向依赖的操作在相应函数内导入，例如任务查询团队审批状态、worktree 查询后台任务、MCP 读取工具注册表。
- `context.py` 仍依赖仓库中的 `agent/learn-claude-code/s09_memory/code.py`，加载路径已适配新目录；没有复制或改写记忆算法。

## 验证

```powershell
python -m pytest agent/Jarvis/v2/tests -q
python -m compileall -q agent/Jarvis/v2
```

同一组行为测试分别运行原 s15 与 v2，使用临时工作目录和模拟模型响应，不读取真实凭据或调用在线模型。覆盖任务依赖与归属、计划审批、消息投递、文件工具、技能与记忆适配、MCP 权限、cron 投递、上下文压缩、后台执行及主循环。
