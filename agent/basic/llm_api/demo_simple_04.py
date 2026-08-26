# -*- encode = utf-8 -*-
# @Time : 2026-08-22 11:02:58
# @Auther : 罗雄波
# @File : demo_simple_04.py
# @Software : PyCharm

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
)

completion = client.chat.completions.create(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    messages=[
        {"role": "system", "content": "你是一名 AI 产品经理，回答要结构化、具体、避免空话。"},
        {"role": "user", "content": "请分析美团 AI 助手对 C 端用户的核心价值。"}
    ],
    temperature=0.4
)

print(completion.choices[0].message.content)