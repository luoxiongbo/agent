# -*- encode = utf-8 -*-
# @Time : 2026-08-22 11:03:51
# @Auther : 罗雄波
# @File : demo_simple_05.py
# @Software : PyCharm

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
)

model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

messages = [
    {"role": "system", "content": "你是一个 AI 产品运营导师，回答适合研究生实习生理解。"}
]

while True:
    user_input = input("\n你：")

    if user_input.lower() in ["exit", "quit", "q"]:
        print("已退出。")
        break

    messages.append({"role": "user", "content": user_input})

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.5
    )

    answer = completion.choices[0].message.content
    print("\nAI：", answer)

    messages.append({"role": "assistant", "content": answer})
