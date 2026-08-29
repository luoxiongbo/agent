from src.core import Agent
from src.memory import MemoryManager
from src.store import MemoryStore
from src.tools import add, count_letter_in_string, compare, get_current_datetime, search_wikipedia
from src.vector_store import VectorStore

from openai import OpenAI
import os


if __name__ == "__main__":
    # DeepSeek 官方 API（默认配置）
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY", "your deepseek api key"),
        base_url="https://api.deepseek.com",
    )

    # 旧版 SiliconFlow 配置，保留作教学对照
    # client = OpenAI(
    #     api_key="your siliconflow api key",
    #     base_url="https://api.siliconflow.cn/v1",
    # )

    store = MemoryStore(db_path="data/memory.db")
    vector_store = VectorStore(persist_directory="data/chroma")
    memory_manager = MemoryManager(
        client=client,
        store=store,
        vector_store=vector_store,
    )

    agent = Agent(
        client=client,
        model="deepseek-v4-flash",
        memory_manager=memory_manager,
        user_id="demo_user_001",
        tools=[get_current_datetime, add, compare, count_letter_in_string, search_wikipedia],
    )

    while True:
        # 使用彩色输出区分用户输入和AI回答
        prompt = input("\033[94mUser: \033[0m")  # 蓝色显示用户输入提示
        if prompt.lower() == "exit":
            break
        response = agent.get_completion(prompt)
        print("\033[92mAssistant: \033[0m", response)  # 绿色显示AI助手回答
