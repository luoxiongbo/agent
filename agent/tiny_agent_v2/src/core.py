import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from utils import function_to_json

SYSTEM_PROMPT = """
你是一个叫不要葱姜蒜的人工智能助手。你的输出应该与用户的语言保持一致。
当用户的问题需要调用工具时，你可以从提供的工具列表中调用适当的工具函数。
"""


class Agent:
    def __init__(
        self,
        client: OpenAI,
        model: str = "deepseek-v4-flash",
        tools: Optional[List] = None,
        memory_manager=None,
        user_id: str = "default_user",
        verbose: bool = True,
        max_history_messages: int = 20,
    ):
        self.client = client
        self.tools = tools or []
        self.model = model
        self.memory_manager = memory_manager
        self.user_id = user_id
        self.max_history_messages = max_history_messages
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        self.verbose = verbose

    def _sanitize_text(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return value.encode("utf-8", errors="replace").decode("utf-8")

    def _sanitize_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = {}
        for key, value in message.items():
            if isinstance(value, dict):
                sanitized[key] = {
                    nested_key: self._sanitize_text(nested_value)
                    if not isinstance(nested_value, (dict, list))
                    else nested_value
                    for nested_key, nested_value in value.items()
                }
            elif isinstance(value, list):
                sanitized_list = []
                for item in value:
                    if isinstance(item, dict):
                        sanitized_list.append(self._sanitize_message(item))
                    else:
                        sanitized_list.append(self._sanitize_text(item))
                sanitized[key] = sanitized_list
            else:
                sanitized[key] = self._sanitize_text(value)
        return sanitized

    def _sanitize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._sanitize_message(message) for message in messages]

    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        allow_tool_messages = False

        for message in messages:
            role = message.get("role")

            if role == "tool":
                if allow_tool_messages:
                    normalized.append(message)
                elif self.verbose:
                    print("跳过孤立的 tool 消息：", message)
                continue

            normalized.append(message)
            allow_tool_messages = bool(
                role == "assistant" and message.get("tool_calls")
            )

        return normalized

    def get_tool_schema(self) -> List[Dict[str, Any]]:
        # 获取所有工具的 JSON 模式
        return [function_to_json(tool) for tool in self.tools]

    def handle_tool_call(self, tool_call):
        # 处理工具调用
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments or "{}")
        function_id = tool_call.id

        tool_map = {tool.__name__: tool for tool in self.tools}
        if function_name not in tool_map:
            function_call_content = f"工具调用失败：未找到工具 {function_name}"
        else:
            try:
                function_call_content = tool_map[function_name](**function_args)
            except Exception as exc:
                function_call_content = f"工具调用失败：{exc}"

        return {
            "role": "tool",
            "content": self._sanitize_text(str(function_call_content)),
            "tool_call_id": function_id,
        }

    def _tool_call_message(self, message) -> Dict[str, Any]:
        assistant_message = {
            "role": "assistant",
            "content": self._sanitize_text(message.content or ""),
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ],
        }
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            assistant_message["reasoning_content"] = self._sanitize_text(reasoning_content)
        return assistant_message

    def manage_context(self) -> None:
        # 保留 system prompt 和最近若干条消息，防止上下文无限增长
        if len(self.messages) > self.max_history_messages + 1:
            self.messages = [self.messages[0]] + self.messages[-self.max_history_messages :]

    def get_completion(self, prompt: str) -> str:
        self.messages = self._normalize_messages(self.messages)
        self.messages.append(
            self._sanitize_message({"role": "user", "content": prompt})
        )

        recalled_memories = []
        if self.memory_manager is not None:
            recalled_memories = self.memory_manager.recall(
                user_id=self.user_id,
                query=prompt,
                top_k=5,
            )

        request_messages = self._normalize_messages(list(self.messages))
        if recalled_memories:
            memory_prompt = self.memory_manager.format_for_prompt(recalled_memories)
            if memory_prompt:
                request_messages.append(
                    self._sanitize_message({
                        "role": "system",
                        "content": (
                            "以下是与当前问题相关的历史记忆，请在回答时酌情参考，"
                            "不要机械复述：\n" + memory_prompt
                        ),
                    })
                )

        # 获取模型的完成响应
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._sanitize_messages(request_messages),
            tools=self.get_tool_schema(),
            stream=False,
        )

        while response.choices[0].message.tool_calls:
            assistant_message = self._tool_call_message(response.choices[0].message)
            persisted_assistant_message = dict(assistant_message)

            self.messages.append(persisted_assistant_message)
            request_messages.append(assistant_message)

            # 处理工具调用
            tool_list = []
            for tool_call in response.choices[0].message.tool_calls:
                # 处理工具调用并将结果添加到消息列表中
                tool_result = self.handle_tool_call(tool_call)
                self.messages.append(tool_result)
                request_messages.append(tool_result)
                tool_list.append([tool_call.function.name, tool_call.function.arguments])
            if self.verbose:
                print("调用工具：", response.choices[0].message.content, tool_list)

            # 再次获取模型的完成响应，这次包含工具调用的结果
            request_messages = self._normalize_messages(request_messages)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._sanitize_messages(request_messages),
                tools=self.get_tool_schema(),
                stream=False,
            )

        # 将模型的完成响应添加到消息列表中
        final_content = self._sanitize_text(response.choices[0].message.content or "")
        final_message = {
            "role": "assistant",
            "content": final_content,
        }
        final_reasoning_content = getattr(response.choices[0].message, "reasoning_content", None)
        if final_reasoning_content:
            final_message["reasoning_content"] = self._sanitize_text(final_reasoning_content)
        self.messages.append(self._sanitize_message(final_message))

        if self.memory_manager is not None:
            self.memory_manager.write_back(
                user_id=self.user_id,
                user_message=prompt,
                assistant_message=final_content,
            )

        self.manage_context()
        return final_content
