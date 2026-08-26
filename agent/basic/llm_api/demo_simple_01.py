# -*- encode = utf-8 -*-
# @Time : 2026-08-22 10:43:23
# @Auther : 罗雄波
# @File : demo_simple_01.py
# @Software : PyCharm

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
)

# 自动补全
completion = client.chat.completions.create(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    messages=[
        {"role": "user", "content": "请用一句话解释什么是大模型 API。"}
    ]
)

print(completion.choices[0].message.content)