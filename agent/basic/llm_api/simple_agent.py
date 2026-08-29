# -*- encode = utf-8 -*-
# @Time : 2026-08-22 15:12:49
# @Auther : 罗雄波
# @File : simple_agent.py
# @Software : PyCharm
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
)

model = os.getenv("DEEPSEEK_MODEL")


def chat(user_input, system_prompt, assistant_message, json_mode=False):
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_message}
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
        stream=True
    )

    result = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            result += delta

    return result


def agent():
    system_prompt = """
    你是一个 agent 的 LLM，你必须确保每一次回答都是简介且命中要点的，尽可能地节省 token，能一次说完的别说二次
    """
    assistant_message = ""
    retries = 3
    i = 0
    while i < retries:
        user_input = input("\n你: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("\nbyb")
            break

        try:
            result = chat(user_input, system_prompt, assistant_message, False)
        except Exception as e:
            raise RuntimeError(e, "运行出错")
        assistant_message += result
        i += 1


agent()
