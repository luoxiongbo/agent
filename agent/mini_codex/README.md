# Mini Codex

一个可运行、便于阅读和扩展的本地编程 Agent。Python 3.11+，**零第三方运行依赖**，
通过 OpenAI 兼容的 Chat Completions 接口调用模型。它是独立的学习和开发项目，不依赖
Codex CLI，也不包含 OpenAI 的模型或内部实现。

## 先运行一次（无需 API Key）

在 PowerShell 中：

```powershell
cd D:\AI\code\agent\agent\mini_codex
New-Item -ItemType Directory -Force .demo-workspace
python -m mini_codex "演示文件创建和修改" --demo --workspace .demo-workspace --auto-approve-edits
```

演示会实际调用工具，依次创建 `mini_codex_demo.py`、读取、把问候语改成中文、再读取。
`--demo` 使用确定性的脚本模型，不会访问网络，不具备通用编程能力。
文件已存在时演示会报告哈希冲突，避免覆盖；重跑可指定一个新的空目录。

不加 `--auto-approve-edits` 时，每次修改都会显示完整 diff，让你决定是否执行。

## 接入真实模型

需要一个支持 **Chat Completions + function tools** 的模型接口。把下面三个示例值换成
你自己的服务地址、模型 ID 和 Key；不要将真实 Key 提交到 Git。

```powershell
$env:MINI_CODEX_BASE_URL = "https://你的服务地址/v1"
$env:MINI_CODEX_MODEL = "你的模型ID"
$env:MINI_CODEX_API_KEY = "你的API-Key"

# 交互模式：连续输入任务，/exit 退出，/session 显示会话 ID
python -m mini_codex --workspace "D:\你的代码目录"

# 单次任务
python -m mini_codex "阅读项目，解释入口和主要模块" --workspace "D:\你的代码目录" --read-only

# 开放执行测试的能力：每条命令仍需确认
python -m mini_codex "定位并修复测试失败，运行相关测试验证" --workspace "D:\你的代码目录" --allow-shell
```

配置优先级：CLI 参数 > `MINI_CODEX_*` 环境变量 > 对应的 `OPENAI_*` 环境变量。
默认 `base_url` 为 `https://api.openai.com/v1`，**模型 ID 无默认值**，需明确配置。
传入的地址应是 API 根地址，不要包含 `/chat/completions`；适配器会自动追加它。
项目不会自动加载 `.env`，不会把客户端持有的 API Key 写入会话。

本地兼容服务可使用 `http://localhost:端口/v1`，不需要 Key 时可留空。
远程地址要求 HTTPS，HTTP 重定向被禁用，避免转发认证信息。
支持返回 `reasoning_content` 的兼容服务透传该字段；其他服务特有参数尚未适配。
此版每次接收完整模型响应，终端展示执行进度，暂未实现逐 token 流式显示。

从其他目录直接使用，可选安装入口（运行本身不需要安装）：

```powershell
python -m pip install -e .
mini-codex --workspace "D:\你的代码目录"
```

## 工具与权限

| 工具 | 行为 |
| --- | --- |
| `list_files` | 有上限地遍历代码文件，排除隐藏目录和常见依赖目录 |
| `read_file` | 按行读取 UTF-8 文本，返回文件 SHA-256 |
| `search` | 按字面字符串搜索，返回文件名和行号 |
| `write_file` | 创建文件或带哈希校验覆盖，先预览 diff |
| `edit_file` | 替换恰好匹配一次的文本，先预览 diff |
| `run_command` | 接收 argv，在工作目录执行，返回退出码、输出和超时状态 |

默认开放文件读取和需确认的编辑，命令工具关闭。
`--read-only` 关闭代码修改与命令工具，但会话数据库仍会保存在工作目录。
`--auto-approve-edits` 只自动批准文件修改。
`--allow-shell` 开放命令工具，但每条命令都需要显式批准；没有自动批准命令的选项。
非交互输入会拒绝需人工确认的操作，因此批处理中的命令不会悄悄执行。

**命令执行不是操作系统沙箱。** 获准执行的进程拥有当前用户的系统权限，可以访问
工作目录外的文件和网络。文件工具的路径限制不适用于这些进程。运行不可信项目时，
应把整个 Mini Codex 放进容器或虚拟机中。

文件工具拒绝越界路径、解析后越界的符号链接/Windows junction、隐藏路径和常见私钥
扩展名；单文件限制 256 KB。它们不是面对恶意并发文件系统操作的完整安全边界。
编辑使用同目录临时文件和原子替换，并在审批后重新检查内容，避免覆盖正常的并发编辑。
不会自动删除文件、提交 Git 或回滚你的工作目录。

命令默认超时 30 秒，最多 120 秒，stdin 关闭。终端展示最多 16 KB 输出；后台定期检查
日志，超过约 2 MB 时尝试终止进程树。日志限额有轮询延迟，不是硬磁盘配额。
超时或 Ctrl+C 时，Windows 使用 `taskkill /T`，POSIX 使用进程组终止。
此版面向前台测试/构建命令，不支持长期服务、交互式终端或可靠管理脱离进程组的后台进程。
命令环境会剔除名称包含 API_KEY、TOKEN、SECRET、PASSWORD 的变量；这不阻止进程从其他
来源获取凭据。命令输出和模型可见文件内容会进入本地记录，工具提供的代码内容会发送到
你配置的模型服务。

## 保存、恢复和事件

每个工作目录的 `.mini-codex/sessions.sqlite3` 保存会话、工具结果和执行事件。
把 `.mini-codex/` 加入被处理项目的 `.gitignore`。同一工作目录一次只允许一个 CLI 实例。

```powershell
python -m mini_codex --workspace "D:\你的代码目录" --list-sessions
python -m mini_codex "继续，先检查上次的修改" --workspace "D:\你的代码目录" --resume "会话ID"
```

保存顺序是：模型调用请求 → 工具开始事件 → 执行工具 → 工具结果。
如果进程在执行工具后、保存结果前崩溃，恢复时无法断定操作是否成功，因此会补充
`Interrupted / result unknown` 工具结果，**不会自动重放**。模型需要先检查当前文件状态。
恢复历史不等于恢复一个运行中的进程，也不保证外部副作用恰好执行一次。

`--json` 在 stdout 输出 JSONL 事件，审批/诊断信息进入 stderr，可供将来的网页 UI 消费：

```powershell
python -m mini_codex "演示" --demo --workspace .demo-workspace --auto-approve-edits --json
```

事件包括 `session_started`、`turn_started`、`model_started`、`assistant`、`tool_started`、
`tool_completed`、`turn_completed`、`turn_stopped`、`turn_failed` 和 `recovered`。
`completed` 表示模型结束了本轮，不代表系统独立证明用户目标已完成；应查看模型报告及工具证据。
进程退出码：`0` 本轮正常结束，`1` 错误，`2` 预算耗尽，`130` 用户中断。

## 上下文与预算

- 根目录 `AGENTS.md` 在每次模型请求前重新读取；过大或不可读时停止。
- 系统指令要求模型读取适用的下级 `AGENTS.md`，尚未实现自动分层发现和强制校验。
- 默认每轮最多 20 次模型请求、40 次工具调用，可通过 CLI 参数调整。
- 默认上下文字符预算 80,000，包括消息与工具定义，是近似保护，不是 tokenizer。
- 超预算时仅从模型输入中移除完整的旧轮次，保留工具调用/结果配对，SQLite 保留完整历史。
- 不做语义摘要；旧轮次中的偏好可能不在后续模型输入中。长期约定请放在 `AGENTS.md`。
- 当前轮次单独超过预算时停止并保存，不截断当前工具调用，也不循环重试。

## 阅读代码

```text
mini_codex/
├── cli.py       # 参数、交互、审批 diff、JSONL 事件
├── model.py     # Chat Completions 适配器 + 离线脚本模型
├── runtime.py   # 工具调用循环、预算、上下文、恢复逻辑
├── tools.py     # 文件/搜索/修改/命令工具及权限检查
└── store.py     # SQLite 检查点、事件记录、进程锁
tests/
├── test_harness.py
└── test_model.py
```

建议按 `cli.main → Runtime.run → WorkspaceTools.invoke → Store.save` 阅读。
`Runtime` 只依赖 `Model.complete(messages, tools)`，以后可以增加 Responses、原生 Ollama 等
适配器，而不用重写执行循环。当前使用的工具协议参考
[OpenAI Function calling 官方文档](https://developers.openai.com/api/docs/guides/function-calling)。

## 验证

```powershell
python -m unittest discover -s tests -v
```

测试无需网络或 API Key，覆盖真实文件创建/编辑、diff 拒绝、哈希冲突、路径越界、
符号链接、只读模式、搜索、预算、异常恢复、命令执行/失败/超时、进程锁、CLI JSONL，
以及模拟 HTTP 响应下的模型适配器和工具调用往返。
这些测试不能证明某个真实供应商/模型的工具调用质量；配置接口后需做一次实际任务验证。

## 后续扩展

适合继续增加：逐 token 流式输出、自动加载分层 AGENTS.md/Skills、上下文摘要、
Git diff/review 工作流、容器执行后端、MCP 工具、跨会话检索和网页 UI。
这版先完整实现单 Agent 编程循环；多 Agent 调度和与正式 Codex 的功能等价性不在本版范围。
