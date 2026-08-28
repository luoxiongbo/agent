import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import httpx
from google import genai
from google.genai import errors
from google.genai import types

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client


# =========================================================
# 1. MCP Tool -> Gemini Function Declaration
# =========================================================

def convert_mcp_tools_to_gemini(mcp_tools):
    """
    把 MCP Server 暴露出来的 Tool Schema
    转换成 Gemini 能理解的 FunctionDeclaration。
    """

    function_declarations = []

    for tool in mcp_tools:
        declaration = types.FunctionDeclaration(
            name=tool.name,
            description=tool.description or "",
            parameters=tool.input_schema,
        )

        function_declarations.append(declaration)

    return types.Tool(
        function_declarations=function_declarations
    )


# =========================================================
# 2. MCP Result -> 普通 dict
# =========================================================

def normalize_mcp_result(result):
    """
    把 MCP call_tool() 返回值转换成
    可以重新喂给 Gemini 的普通数据。
    """

    # MCP v2 可以提供 structured_content
    if result.structured_content is not None:
        return result.structured_content

    # fallback：
    # 某些 MCP Server 只返回 TextContent
    texts = []

    for item in result.content:
        if hasattr(item, "text"):
            texts.append(item.text)

    return {
        "content": "\n".join(texts)
    }


def create_function_response_part(function_name, normalized_result):
    """
    构造 Gemini SDK 当前版本支持的 FunctionResponse Part。
    """

    return types.Part.from_function_response(
        name=function_name,
        response={
            "result": normalized_result
        },
    )


# =========================================================
# 3. Google API Error -> 可读错误提示
# =========================================================

def _iter_google_error_metadata(error):
    details = getattr(error, "details", {}) or {}
    error_body = details.get("error", {}) if isinstance(details, dict) else {}

    for item in error_body.get("details", []):
        if not isinstance(item, dict):
            continue

        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            yield item.get("reason"), metadata


def format_google_client_error(error):
    """
    把 google-genai 的 ClientError 转成更容易执行的运行提示。
    """

    code = getattr(error, "code", None)
    message = getattr(error, "message", None) or str(error)

    for reason, metadata in _iter_google_error_metadata(error):
        service = metadata.get("service")
        consumer = metadata.get("consumer", "")

        if reason == "SERVICE_DISABLED" and service == "aiplatform.googleapis.com":
            project = consumer.removeprefix("projects/") or os.getenv(
                "GOOGLE_CLOUD_PROJECT",
                "<your-google-cloud-project>",
            )
            activation_url = metadata.get("activationUrl")

            lines = [
                "Gemini/Vertex AI 调用失败：Vertex AI API 未启用。",
                "",
                "当前脚本使用 genai.Client(vertexai=True)，所以需要给 Google Cloud 项目启用 API：",
                f"  project: {project}",
                f"  service: {service}",
                f"  command: gcloud services enable {service} --project {project}",
            ]

            if activation_url:
                lines.append(f"  console: {activation_url}")

            lines.extend(
                [
                    "",
                    "如果你想使用 AI Studio 的 API Key，而不是 Vertex AI，需要把客户端改成",
                    "genai.Client(api_key=...) 并配置 GEMINI_API_KEY。",
                ]
            )

            return "\n".join(lines)

    return f"Google GenAI API 调用失败：{code} {message}"


def format_gemini_call_error(error):
    if isinstance(error, errors.ClientError):
        return format_google_client_error(error)

    if isinstance(error, httpx.ConnectError):
        return (
            "Gemini API 网络连接失败："
            f"{error}\n请检查代理、网络连接，或稍后重试。"
        )

    return f"Gemini API 调用失败：{type(error).__name__}: {error}"


# =========================================================
# 4. Gemini / MCP Client 配置
# =========================================================

def create_gemini_client():
    """
    使用 Google AI Studio 的 Gemini API Key 创建客户端。
    """

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "缺少 GEMINI_API_KEY。请先在 Google AI Studio 创建 API Key，"
            "然后把 GEMINI_API_KEY=你的key 写入 .env。"
        )

    return genai.Client(api_key=api_key)


def create_mcp_server_parameters():
    """
    使用 agent.py 同目录下的 server.py，避免从其它目录运行时找不到文件。
    """

    server_path = Path(__file__).with_name("server.py")

    return StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )


# =========================================================
# 5. Agent 主循环
# =========================================================

async def main():
    # -----------------------------------------------------
    # Gemini / AI Studio Client
    # -----------------------------------------------------

    try:
        gemini = create_gemini_client()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return

    # -----------------------------------------------------
    # MCP Server 配置
    #
    # Agent 会自动：
    # python server.py
    #
    # 然后通过 stdin/stdout 和它通信
    # -----------------------------------------------------

    server_parameters = create_mcp_server_parameters()

    # -----------------------------------------------------
    # 创建 MCP Client
    # -----------------------------------------------------

    try:
        async with Client(
                stdio_client(server_parameters)
        ) as mcp_client:

            # =================================================
            # STEP 1
            #
            # Agent -> MCP Server
            #
            # tools/list
            # =================================================

            tools_result = await mcp_client.list_tools()

            print("\n===== MCP Server 提供的工具 =====")

            for tool in tools_result.tools:
                print(
                    f"""
Tool:
  name: {tool.name}
  description: {tool.description}
  schema: {tool.input_schema}
"""
                )

            # =================================================
            # STEP 2
            #
            # MCP Tool
            #
            #      ↓
            #
            # Gemini FunctionDeclaration
            # =================================================

            gemini_tool = convert_mcp_tools_to_gemini(
                tools_result.tools
            )

            # =================================================
            # 用户输入
            # =================================================

            user_input = input(
                "\nUser: "
            )

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=user_input
                        )
                    ],
                )
            ]

            # =================================================
            # Agent Loop
            # =================================================

            while True:

                # =============================================
                # STEP 3
                #
                # Agent -> LLM
                #
                # 把 MCP Tool Schema 作为 Function
                # 告诉 Gemini
                # =============================================

                try:
                    response = await gemini.aio.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=contents,
                        config=types.GenerateContentConfig(
                            tools=[gemini_tool],

                            # 为了学习原理，我们自己执行 Function Call
                            automatic_function_calling=
                            types.AutomaticFunctionCallingConfig(
                                disable=True
                            ),
                        ),
                    )
                except Exception as exc:
                    print(
                        "\n" + format_gemini_call_error(exc),
                        file=sys.stderr,
                    )
                    return

                model_content = response.candidates[0].content

                # =============================================
                # STEP 4
                #
                # 查看 Gemini 有没有产生 Function Call
                # =============================================

                function_calls = []

                for part in model_content.parts:

                    if part.function_call is not None:
                        function_calls.append(
                            part.function_call
                        )

                # =============================================
                # 没有 Function Call
                #
                # 说明 Gemini 已经得到了最终答案
                # =============================================

                if not function_calls:
                    print(
                        "\nAssistant:",
                        response.text,
                    )

                    break

                # =============================================
                # 保存 Gemini 的 Function Call
                #
                # 很重要！
                #
                # 下一轮需要把这段历史继续发回去
                # =============================================

                contents.append(
                    model_content
                )

                function_response_parts = []

                # =============================================
                # STEP 5
                #
                # 执行所有 Function Calls
                # =============================================

                for function_call in function_calls:
                    function_name = function_call.name

                    arguments = dict(
                        function_call.args
                    )

                    print(
                        f"""
===== Gemini Function Calling =====

LLM 决定调用：

{function_name}({arguments})
"""
                    )

                    # =========================================
                    # 这里才真正进入 MCP
                    #
                    # Function Calling
                    #
                    #         ↓
                    #
                    # Agent Runtime
                    #
                    #         ↓
                    #
                    # MCP tools/call
                    # =========================================

                    result = await mcp_client.call_tool(
                        function_name,
                        arguments,
                    )

                    normalized_result = normalize_mcp_result(
                        result
                    )

                    print(
                        f"""
===== MCP tools/call =====

Tool Result:

{normalized_result}
"""
                    )

                    # =========================================
                    # MCP Result
                    #
                    #     ↓
                    #
                    # Function Response
                    #
                    #     ↓
                    #
                    # Gemini
                    # =========================================

                    function_response_parts.append(
                        create_function_response_part(
                            function_name,
                            normalized_result,
                        )
                    )

                # =============================================
                # STEP 6
                #
                # 把工具执行结果重新喂给 LLM
                # =============================================

                contents.append(
                    types.Content(
                        role="user",
                        parts=function_response_parts,
                    )
                )
    finally:
        await gemini.aio.aclose()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
