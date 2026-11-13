"""Music Store Customer Service Agent

基于 LangGraph 101 / 201 multi_agent.ipynb 实现：
1. Music Catalog Subagent：手动 StateGraph + ToolNode
2. Invoice Information Subagent：LangChain create_agent
3. Supervisor：将两个子 Agent 包装成 Tool
4. Customer verification：interrupt + Command(resume=...)
5. Long-term memory：InMemoryStore

本文件不包含 Evaluation；评估部分可以单独放在下一阶段实现。

运行前确保：
- 项目中存在 utils/models.py，且其中导出已配置好的 chat model：model
- 已安装 langchain、langgraph、langchain-community、sqlalchemy、requests、pydantic
- 模型支持 tool calling 和 structured output
"""

from __future__ import annotations

import re
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import requests
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt


# ---------------------------------------------------------------------------
# Model import
# ---------------------------------------------------------------------------
# 自动向上查找包含 utils/models.py 的项目目录，便于把本文件放在项目子目录中运行。
for _parent in Path(__file__).resolve().parents:
    if (_parent / "utils" / "models.py").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

try:
    from utils.models import model as default_model
except ModuleNotFoundError:
    default_model = None


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
class InputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class State(InputState, total=False):
    customer_id: int
    loaded_memory: str
    remaining_steps: RemainingSteps


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
CHINOOK_SQL_URL = (
    "https://raw.githubusercontent.com/lerocha/chinook-database/"
    "master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
)

_engine: Engine | None = None


def get_engine_for_chinook_db(sql_url: str = CHINOOK_SQL_URL) -> Engine:
    """下载 Chinook SQL 脚本并创建共享的 SQLite 内存数据库。"""
    response = requests.get(sql_url, timeout=30)
    response.raise_for_status()

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(response.text)

    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


def _query_rows(sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """执行参数化 SQL，返回字典列表。"""
    if _engine is None:
        raise RuntimeError(
            "Database is not initialized. Call build_customer_service() first."
        )

    with _engine.connect() as connection:
        result = connection.execute(text(sql), params or {})
        return [dict(row._mapping) for row in result]


def _normalize_phone(phone: str | None) -> str:
    return re.sub(r"\D", "", phone or "")


def _find_customer_id_by_phone(phone: str) -> int | None:
    """按数字归一化后的电话号码匹配 Chinook Customer。"""
    target = _normalize_phone(phone)
    if not target:
        return None

    rows = _query_rows("SELECT CustomerId, Phone FROM Customer")
    for row in rows:
        if _normalize_phone(row.get("Phone")) == target:
            return int(row["CustomerId"])
    return None


# ---------------------------------------------------------------------------
# Music Catalog Subagent tools
# ---------------------------------------------------------------------------
@tool
def get_albums_by_artist(artist: str) -> list[dict[str, Any]]:
    """根据艺术家名称查询专辑。支持艺术家名称的部分匹配。"""
    return _query_rows(
        """
        SELECT Album.Title AS AlbumTitle, Artist.Name AS ArtistName
        FROM Album
        JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Artist.Name LIKE :artist
        ORDER BY Album.Title
        LIMIT 30
        """,
        {"artist": f"%{artist.strip()}%"},
    )


@tool
def get_tracks_by_artist(artist: str) -> list[dict[str, Any]]:
    """根据艺术家名称查询歌曲，并返回歌曲、艺术家和专辑名称。"""
    return _query_rows(
        """
        SELECT
            Track.Name AS SongName,
            Artist.Name AS ArtistName,
            Album.Title AS AlbumTitle
        FROM Track
        JOIN Album ON Track.AlbumId = Album.AlbumId
        JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Artist.Name LIKE :artist
        ORDER BY Album.Title, Track.Name
        LIMIT 50
        """,
        {"artist": f"%{artist.strip()}%"},
    )


@tool
def get_songs_by_genre(genre: str) -> list[dict[str, Any]]:
    """根据音乐流派查询歌曲，返回歌曲名称、艺术家和流派。"""
    return _query_rows(
        """
        SELECT
            Track.Name AS SongName,
            Artist.Name AS ArtistName,
            Genre.Name AS GenreName
        FROM Track
        JOIN Genre ON Track.GenreId = Genre.GenreId
        JOIN Album ON Track.AlbumId = Album.AlbumId
        JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Genre.Name LIKE :genre
        ORDER BY Artist.Name, Track.Name
        LIMIT 30
        """,
        {"genre": f"%{genre.strip()}%"},
    )


@tool
def check_for_songs(song_title: str) -> list[dict[str, Any]]:
    """检查音乐目录中是否存在指定歌曲，支持歌曲名称的部分匹配。"""
    return _query_rows(
        """
        SELECT
            Track.Name AS SongName,
            Artist.Name AS ArtistName,
            Album.Title AS AlbumTitle
        FROM Track
        JOIN Album ON Track.AlbumId = Album.AlbumId
        JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Track.Name LIKE :song_title
        ORDER BY Artist.Name, Track.Name
        LIMIT 30
        """,
        {"song_title": f"%{song_title.strip()}%"},
    )


MUSIC_TOOLS = [
    get_albums_by_artist,
    get_tracks_by_artist,
    get_songs_by_genre,
    check_for_songs,
]


# ---------------------------------------------------------------------------
# Invoice Information Subagent tools
# ToolRuntime is injected by LangChain and is not exposed as an LLM argument.
# ---------------------------------------------------------------------------
@tool
def get_invoices_by_customer_sorted_by_date(
    runtime: ToolRuntime,
) -> list[dict[str, Any]]:
    """查询当前已验证客户的发票，按发票日期从新到旧排序。"""
    customer_id = runtime.state.get("customer_id")
    if customer_id is None:
        return []

    return _query_rows(
        """
        SELECT
            InvoiceId,
            InvoiceDate,
            BillingCity,
            BillingCountry,
            Total
        FROM Invoice
        WHERE CustomerId = :customer_id
        ORDER BY InvoiceDate DESC
        LIMIT 20
        """,
        {"customer_id": customer_id},
    )


@tool
def get_invoices_sorted_by_unit_price(
    runtime: ToolRuntime,
) -> list[dict[str, Any]]:
    """查询当前已验证客户购买的商品明细，按商品单价从高到低排序。"""
    customer_id = runtime.state.get("customer_id")
    if customer_id is None:
        return []

    return _query_rows(
        """
        SELECT
            Invoice.InvoiceId,
            Invoice.InvoiceDate,
            Track.Name AS TrackName,
            InvoiceLine.UnitPrice,
            InvoiceLine.Quantity,
            InvoiceLine.UnitPrice * InvoiceLine.Quantity AS LineTotal
        FROM Invoice
        JOIN InvoiceLine ON Invoice.InvoiceId = InvoiceLine.InvoiceId
        JOIN Track ON InvoiceLine.TrackId = Track.TrackId
        WHERE Invoice.CustomerId = :customer_id
        ORDER BY InvoiceLine.UnitPrice DESC
        LIMIT 30
        """,
        {"customer_id": customer_id},
    )


@tool
def get_employee_by_invoice_and_customer(
    invoice_id: int,
    runtime: ToolRuntime,
) -> list[dict[str, Any]]:
    """根据发票 ID 和当前已验证客户，查询负责该客户的客服员工。"""
    customer_id = runtime.state.get("customer_id")
    if customer_id is None:
        return []

    return _query_rows(
        """
        SELECT
            Employee.EmployeeId,
            Employee.FirstName,
            Employee.LastName,
            Employee.Title,
            Employee.Email
        FROM Invoice
        JOIN Customer ON Invoice.CustomerId = Customer.CustomerId
        JOIN Employee ON Customer.SupportRepId = Employee.EmployeeId
        WHERE Invoice.InvoiceId = :invoice_id
          AND Invoice.CustomerId = :customer_id
        LIMIT 1
        """,
        {"invoice_id": invoice_id, "customer_id": customer_id},
    )


INVOICE_TOOLS = [
    get_invoices_by_customer_sorted_by_date,
    get_invoices_sorted_by_unit_price,
    get_employee_by_invoice_and_customer,
]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def generate_music_assistant_prompt(memory: str = "None") -> str:
    return f"""
You are a specialist agent for the music catalog of a digital music store.

Your responsibilities:
- Search for songs, albums, and artists in the catalog.
- Recommend music based on the customer's explicitly stated interests and saved preferences.
- Include artist and album information when available.
- Search thoroughly before saying that an item is unavailable.
- If an exact match is not found, try partial artist or song-name matches.
- Be honest when the catalog does not contain the requested item.
- You only handle music-catalog questions. Do not answer invoice, billing, or unrelated questions.

Available tools:
- get_albums_by_artist
- get_tracks_by_artist
- get_songs_by_genre
- check_for_songs

Saved customer preferences:
{memory}

Use the conversation history to understand the customer's request. Do not invent catalog results.
""".strip()


INVOICE_SUBAGENT_PROMPT = """
You are a specialist agent for customer invoices and purchase history in a digital music store.

Your responsibilities:
- Answer questions about the customer's invoices, purchase dates, totals, and purchased tracks.
- Find the most recent invoice when the customer asks about their latest purchase.
- Use get_invoices_sorted_by_unit_price when the customer asks about expensive items or unit prices.
- Use get_employee_by_invoice_and_customer when the customer asks which employee helped with an invoice.
- The verified customer_id is available through ToolRuntime state. Never ask the customer to provide a customer ID.
- Only use data returned by the tools. Do not invent invoice, payment, or employee information.
- You only handle invoice and purchase-related questions. Do not answer music-catalog questions directly.
""".strip()


SUPERVISOR_PROMPT = """
You are the supervisor of a digital music store customer-service team.

You have two specialist subagents available as tools:
1. music_catalog_subagent: handles songs, albums, artists, genres, and music recommendations.
2. invoice_information_subagent: handles invoices, purchase history, invoice totals, and support-employee information.

Your responsibilities:
- Understand the customer's request and delegate each part to the appropriate specialist.
- For a request involving both music and invoices, call both relevant subagents.
- You may call more than one subagent when the request has multiple independent parts.
- Combine the subagents' results into one clear answer.
- Do not fabricate database results or claim that an operation succeeded unless a tool result supports it.
- If a request is unrelated to the music store's catalog or invoices, politely explain the scope of this service.
- Do not ask for a customer ID. The customer has already been verified by the outer workflow.
""".strip()


# ---------------------------------------------------------------------------
# Memory models
# ---------------------------------------------------------------------------
class PhoneNumberExtraction(BaseModel):
    phone_number: str | None = Field(
        default=None,
        description="The phone number explicitly provided by the customer, or null if none was provided.",
    )


class UserProfile(BaseModel):
    music_preferences: list[str] = Field(
        default_factory=list,
        description="The customer's explicitly stated music preferences, artists, genres, or styles.",
    )


# ---------------------------------------------------------------------------
# Helpers for subagent calls
# ---------------------------------------------------------------------------
def _child_config(runtime: ToolRuntime, child_name: str) -> dict[str, Any]:
    """Give each child agent a stable, separate checkpoint thread.

    The parent thread ID is reused with a suffix so that:
    - the same customer conversation can resume a child agent's state;
    - different child agents do not collide in the shared checkpointer.
    """
    configurable = runtime.config.get("configurable", {})
    parent_thread_id = configurable.get("thread_id") or str(uuid.uuid4())
    return {
        "configurable": {
            "thread_id": f"{parent_thread_id}:{child_name}"
        }
    }


def _last_ai_content(result: dict[str, Any]) -> str:
    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage) and message.content:
            if isinstance(message.content, str):
                return message.content
            return str(message.content)
    return "The specialist agent did not return a final response."


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
@dataclass
class CustomerServiceApp:
    graph: Any
    supervisor: Any
    music_catalog_subagent: Any
    invoice_information_subagent: Any
    checkpointer: MemorySaver
    store: InMemoryStore


def build_customer_service(
    chat_model: Any | None = None,
    chinook_sql_url: str = CHINOOK_SQL_URL,
) -> CustomerServiceApp:
    """构建完整的音乐商店客服 Agent。"""
    global _engine

    if chat_model is None:
        if default_model is None:
            raise RuntimeError(
                "无法导入 utils.models.model。请将本文件放在能访问项目 utils/models.py 的目录，"
                "或调用 build_customer_service(chat_model=your_chat_model)。"
            )
        chat_model = default_model

    # 初始化数据库。工具函数通过模块级 _engine 访问它。
    _engine = get_engine_for_chinook_db(chinook_sql_url)

    # 短期记忆：保存每个 thread 的执行状态。
    checkpointer = MemorySaver()

    # 长期记忆：跨 thread 保存客户画像。
    store = InMemoryStore()

    # ------------------------------------------------------------------
    # 1. Music Catalog Subagent：手动实现 StateGraph + ToolNode
    # ------------------------------------------------------------------
    llm_with_music_tools = chat_model.bind_tools(MUSIC_TOOLS)
    music_tool_node = ToolNode(MUSIC_TOOLS)

    def music_assistant(state: State) -> dict[str, Any]:
        memory = state.get("loaded_memory") or "None"
        prompt = generate_music_assistant_prompt(memory)

        response = llm_with_music_tools.invoke(
            [SystemMessage(content=prompt), *state["messages"]]
        )
        return {"messages": [response]}

    def should_continue_music(state: State) -> str:
        last_message = state["messages"][-1]
        return "continue" if getattr(last_message, "tool_calls", None) else "end"

    music_workflow = StateGraph(State)
    music_workflow.add_node("music_assistant", music_assistant)
    music_workflow.add_node("music_tool_node", music_tool_node)
    music_workflow.add_edge(START, "music_assistant")
    music_workflow.add_conditional_edges(
        "music_assistant",
        should_continue_music,
        {
            "continue": "music_tool_node",
            "end": END,
        },
    )
    music_workflow.add_edge("music_tool_node", "music_assistant")

    music_catalog_subagent = music_workflow.compile(
        name="music_catalog_subagent",
        checkpointer=checkpointer,
        store=store,
    )

    # ------------------------------------------------------------------
    # 2. Invoice Information Subagent：使用 create_agent
    # ------------------------------------------------------------------
    invoice_information_subagent = create_agent(
        model=chat_model,
        tools=INVOICE_TOOLS,
        name="invoice_information_subagent",
        system_prompt=INVOICE_SUBAGENT_PROMPT,
        state_schema=State,
        checkpointer=checkpointer,
        store=store,
    )

    # ------------------------------------------------------------------
    # 3. Supervisor：将子 Agent 包装成 Tools
    # ------------------------------------------------------------------
    @tool(
        "music_catalog_subagent",
        description=(
            "A specialist agent for music-catalog questions, including songs, albums, "
            "artists, genres, and music recommendations. Input should be the customer's "
            "complete music-related question."
        ),
    )
    def call_music_catalog_subagent(
        query: str,
        runtime: ToolRuntime,
    ) -> str:
        result = music_catalog_subagent.invoke(
            {
                "messages": [HumanMessage(content=query)],
                "loaded_memory": runtime.state.get("loaded_memory", "None"),
            },
            config=_child_config(runtime, "music_catalog_subagent"),
        )
        return _last_ai_content(result)

    @tool(
        "invoice_information_subagent",
        description=(
            "A specialist agent for invoice and purchase-history questions, including "
            "recent purchases, invoice totals, purchased tracks, and support employees. "
            "Input should be the customer's complete invoice-related question."
        ),
    )
    def call_invoice_information_subagent(
        query: str,
        runtime: ToolRuntime,
    ) -> str:
        customer_id = runtime.state.get("customer_id")
        if customer_id is None:
            return "Customer identity has not been verified. Do not access invoice data."

        result = invoice_information_subagent.invoke(
            {
                "messages": [HumanMessage(content=query)],
                "customer_id": customer_id,
                "loaded_memory": runtime.state.get("loaded_memory", "None"),
            },
            config=_child_config(runtime, "invoice_information_subagent"),
        )
        return _last_ai_content(result)

    supervisor = create_agent(
        model=chat_model,
        tools=[
            call_music_catalog_subagent,
            call_invoice_information_subagent,
        ],
        name="supervisor",
        system_prompt=SUPERVISOR_PROMPT,
        state_schema=State,
        checkpointer=checkpointer,
        store=store,
    )

    # ------------------------------------------------------------------
    # 4. Customer identity verification: structured output + interrupt
    # ------------------------------------------------------------------
    phone_extractor = chat_model.with_structured_output(PhoneNumberExtraction)

    def verify_info(state: State) -> dict[str, Any]:
        # 已验证过的客户不需要重复验证。
        if state.get("customer_id") is not None:
            return {}

        latest_user_message = next(
            (
                message
                for message in reversed(state["messages"])
                if isinstance(message, HumanMessage)
            ),
            None,
        )

        extracted_phone: str | None = None
        if latest_user_message is not None:
            extracted = phone_extractor.invoke(
                [
                    SystemMessage(
                        content=(
                            "Extract a phone number only if the user explicitly provided one. "
                            "Do not infer or invent a phone number. If absent, return null."
                        )
                    ),
                    latest_user_message,
                ]
            )
            extracted_phone = extracted.phone_number

        if extracted_phone:
            customer_id = _find_customer_id_by_phone(extracted_phone)
            if customer_id is not None:
                return {
                    "customer_id": customer_id,
                    "messages": [
                        AIMessage(content="Your account has been verified.")
                    ],
                }

            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I could not match that phone number to a customer account. "
                            "Please check the number and provide it again."
                        )
                    )
                ]
            }

        return {
            "messages": [
                AIMessage(
                    content=(
                        "Before I can access your purchase history, please provide "
                        "the phone number associated with your music-store account."
                    )
                )
            ]
        }

    def human_input(state: State) -> dict[str, Any]:
        resumed_value = interrupt(
            {
                "type": "customer_phone_number",
                "message": "Please provide the phone number associated with your account.",
            }
        )

        if isinstance(resumed_value, dict):
            phone_text = (
                resumed_value.get("phone_number")
                or resumed_value.get("input")
                or str(resumed_value)
            )
        else:
            phone_text = str(resumed_value)

        return {"messages": [HumanMessage(content=phone_text)]}

    def route_after_verification(state: State) -> str:
        return "continue" if state.get("customer_id") is not None else "interrupt"

    # ------------------------------------------------------------------
    # 5. Long-term memory: load_memory + create_memory
    # ------------------------------------------------------------------
    def load_memory(state: State, store: BaseStore) -> dict[str, Any]:
        customer_id = state.get("customer_id")
        if customer_id is None:
            return {"loaded_memory": "None"}

        namespace = ("memory_profile", str(customer_id))
        item = store.get(namespace, "user_memory")

        if not item or not item.value:
            return {"loaded_memory": "None"}

        memory = item.value.get("memory", item.value)
        if hasattr(memory, "model_dump"):
            memory = memory.model_dump()

        preferences = memory.get("music_preferences", []) if isinstance(memory, dict) else []
        if not preferences:
            return {"loaded_memory": "None"}

        return {
            "loaded_memory": "Music Preferences: " + ", ".join(preferences)
        }

    def create_memory(state: State, store: BaseStore) -> dict[str, Any]:
        customer_id = state.get("customer_id")
        if customer_id is None:
            return {}

        namespace = ("memory_profile", str(customer_id))
        existing_item = store.get(namespace, "user_memory")
        existing_memory: dict[str, Any] = {}
        if existing_item and existing_item.value:
            existing_memory = existing_item.value.get("memory", existing_item.value)
            if hasattr(existing_memory, "model_dump"):
                existing_memory = existing_memory.model_dump()
            if not isinstance(existing_memory, dict):
                existing_memory = {}

        existing_preferences = existing_memory.get("music_preferences", [])

        conversation_lines: list[str] = []
        for message in state.get("messages", []):
            if isinstance(message, (HumanMessage, AIMessage)):
                content = message.content
                if isinstance(content, str) and content.strip():
                    role = "Customer" if isinstance(message, HumanMessage) else "Assistant"
                    conversation_lines.append(f"{role}: {content}")

        prompt = f"""
You maintain a customer's long-term music preference profile for a music store.

Extract only durable preferences explicitly stated by the customer, such as favorite
artists, genres, or music styles. Do not treat every one-time search request as a stable
preference unless the customer says they like or prefer it.

Preserve all existing preferences unless the customer clearly corrects or changes them.
Do not invent preferences. Return an empty list if no preference is supported.

Existing music preferences:
{existing_preferences}

Conversation:
{chr(10).join(conversation_lines)}

Return the complete updated music preference list.
""".strip()

        updated_profile = chat_model.with_structured_output(UserProfile).invoke(
            [SystemMessage(content=prompt)]
        )

        merged_preferences: list[str] = []
        seen: set[str] = set()
        for preference in [
            *existing_preferences,
            *updated_profile.music_preferences,
        ]:
            cleaned = preference.strip()
            key = cleaned.casefold()
            if cleaned and key not in seen:
                seen.add(key)
                merged_preferences.append(cleaned)

        store.put(
            namespace,
            "user_memory",
            {
                "memory": {
                    "customer_id": customer_id,
                    "music_preferences": merged_preferences,
                }
            },
        )
        return {}

    # ------------------------------------------------------------------
    # 6. Final customer-service graph
    # ------------------------------------------------------------------
    workflow = StateGraph(State, input_schema=InputState)

    workflow.add_node("verify_info", verify_info)
    workflow.add_node("human_input", human_input)
    workflow.add_node("load_memory", load_memory)
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("create_memory", create_memory)

    workflow.add_edge(START, "verify_info")
    workflow.add_conditional_edges(
        "verify_info",
        route_after_verification,
        {
            "continue": "load_memory",
            "interrupt": "human_input",
        },
    )
    workflow.add_edge("human_input", "verify_info")
    workflow.add_edge("load_memory", "supervisor")
    workflow.add_edge("supervisor", "create_memory")
    workflow.add_edge("create_memory", END)

    graph = workflow.compile(
        name="music_store_customer_service",
        checkpointer=checkpointer,
        store=store,
    )

    return CustomerServiceApp(
        graph=graph,
        supervisor=supervisor,
        music_catalog_subagent=music_catalog_subagent,
        invoice_information_subagent=invoice_information_subagent,
        checkpointer=checkpointer,
        store=store,
    )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------
def ask_customer_service(
    app: CustomerServiceApp,
    question: str,
    thread_id: str,
) -> dict[str, Any]:
    """发送一条客户消息。多轮对话请复用同一个 thread_id。"""
    return app.graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )


def resume_customer_service(
    app: CustomerServiceApp,
    user_input: str,
    thread_id: str,
) -> dict[str, Any]:
    """恢复被 interrupt 暂停的 Graph，例如提交客户电话号码。"""
    return app.graph.invoke(
        Command(resume=user_input),
        config={"configurable": {"thread_id": thread_id}},
    )


if __name__ == "__main__":
    app = build_customer_service()
    thread_id = str(uuid.uuid4())

    first_result = ask_customer_service(
        app,
        "How much was my most recent purchase and what albums do you have by the Rolling Stones?",
        thread_id,
    )

    if first_result.get("__interrupt__"):
        print("Graph paused for customer verification:")
        print(first_result["__interrupt__"])
        phone_number = input("Enter the phone number associated with your account: ")
        final_result = resume_customer_service(app, phone_number, thread_id)
    else:
        final_result = first_result

    for message in final_result.get("messages", []):
        if isinstance(message, AIMessage):
            message.pretty_print()
