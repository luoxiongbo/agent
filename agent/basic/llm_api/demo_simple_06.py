# -*- encode = utf-8 -*-
# @Time : 2026-08-22 11:06:12
# @Auther : 罗雄波
# @File : demo_simple_06.py
# @Software : PyCharm

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
)

stream = client.chat.completions.create(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    messages=[
        {"role": "user", "content": "请用 5 个要点解释 AI 产品运营岗位需要什么能力。"}
    ],
    stream=True
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)

print()
