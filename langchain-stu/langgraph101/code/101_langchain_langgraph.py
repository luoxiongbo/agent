import os
import json
import sys
from pathlib import Path

import requests
from langchain_core.messages import AIMessage, AIMessageChunk
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.models import model


def part1():
    response = model.invoke("LangChain是什么?")
    response.pretty_print()


def part2():
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is an agent?")
    ]

    response = model.invoke(messages)
    response.pretty_print()


@tool
def get_weather(latitude: float, longitude: float) -> str:
    """Get current temperature in Fahrenheit and weather code for given coordinates.

    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate

    Returns:
        JSON string with temperature_fahrenheit and weather_code (do not include the code in your response, translate it to plain English)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,weather_code",
        "temperature_unit": "fahrenheit"
    }

    weather = requests.get(url, params=params).json()["current"]
    temperature = weather["temperature_2m"]
    weather_code = weather["weather_code"]
    result = {
        "temperature_fahrenheit": temperature,
        "weather_code": weather_code
    }

    return json.dumps(result)


@tool
def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。

    Args:
        city:
        weather:

    Returns:
        JSON string with weather and attraction
    """
    # 1. 从环境变量中读取API密钥
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量。"

    # 2. 初始化Tavily客户端
    tavily = TavilyClient(api_key=api_key)

    # 3. 构造一个精确的查询
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"

    try:
        # 4. 调用API，include_answer=True会返回一个综合性的回答
        response = tavily.search(query=query, search_depth="basic", include_answer=True)

        # 5. Tavily返回的结果已经非常干净，可以直接使用
        # response['answer'] 是一个基于所有搜索结果的总结性回答
        if response.get("answer"):
            return response["answer"]

        # 如果没有综合性回答，则格式化原始结果
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
            return "抱歉，没有找到相关的旅游景点推荐。"

        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"


@tool
def get_user_preferences(user_id: str) -> str:
    """Get a user's saved preferences."""
    # Simulate a user database
    preferences = {
        "alice": "Loves sci-fi movies, prefers warm weather destinations",
        "bob": "Enjoys comedy films, likes cold climates for travel"
    }
    return preferences.get(user_id.lower(), "No preferences found")


@tool
def book_recommendation(genre: str, user_preferences: str = "") -> str:
    """Get personalized movie recommendations based on genre and user preferences."""
    recommendations = {
        "sci-fi": "Based on your preferences, try: Arrival, Ex Machina, or The Martian",
        "comedy": "Based on your preferences, try: The Big Lebowski, Anchorman, or Bridesmaids"
    }
    return recommendations.get(genre.lower(), "No recommendations available")


def part3():
    tools = [get_weather]
    model_with_tools = model.bind_tools(tools)

    response = model_with_tools.invoke(
        "What's the weather like in Seattle?"
    )

    print(response.tool_calls)


def part4():
    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="You are a helpful assistant."
    )

    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "Recommend some sci-fi movies."}
        ]
    })


def part5():
    # 这里使用的 MemorySaver() 是内存检查点实现，适合教学和开发；如果需要进程重启后仍能恢复，应该使用数据库等持久化存储。
    checkpointer = MemorySaver()

    agent_with_memory = create_agent(
        model=model,
        tools=[get_weather],
        checkpointer=checkpointer
    )

    config = {"configurable": {"thread_id": "conversation-1"}}

    agent_with_memory.invoke(
        {"messages": [{"role": "user", "content": "My name is Alice."}]},
        config=config
    )


def part6():
    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="You are a helpful assistant."
    )

    for chunk in agent.stream(
            {"messages": [{"role": "user", "content": "Hello"}]},
            stream_mode="messages"
    ):
        print(chunk)


def part7():
    assistant = create_agent(
        model=model,
        tools=[
            get_weather,
            get_user_preferences,
            book_recommendation
        ],
        system_prompt="You are a helpful personal assistant.",
        checkpointer=MemorySaver()
    )


def part8():
    tools = [get_weather, get_user_preferences, book_recommendation]
    model_with_tools = model.bind_tools(tools)

    def call_model(state: MessagesState):
        messages = [
            SystemMessage(content="You are a helpful personal assistant."),
            *state["messages"],
        ]
        return {"messages": [model_with_tools.invoke(messages)]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile(checkpointer=MemorySaver(), name="agent")


def use_part8():
    graph = part8()
    config = {"configurable": {"thread_id": "demo-1"}}

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "请查询 alice 的偏好"}]},
        config=config,
    )
    print(result["messages"][-1].content)

    # 同一个 graph 和 thread_id 会保留上一轮对话
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "根据刚才的偏好推荐电影"}]},
        config=config,
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    # part1()
    # part2()
    # part3()
    # part4()
    # part5()
    # part6()
    # part7()
    # part8()
    use_part8()
