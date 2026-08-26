# -*- encode = utf-8 -*-
# @Time : 2026-08-22 10:54:12
# @Auther : 罗雄波
# @File : demo_simple_02.py
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

completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "请列出 AI 产品运营需要关注的 3 个核心指标。"}
    ]
)

print(completion.choices[0].message.content)