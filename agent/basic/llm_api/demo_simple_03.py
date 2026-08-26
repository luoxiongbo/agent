# -*- encode = utf-8 -*-
# @Time : 2026-08-22 10:58:46
# @Auther : 罗雄波
# @File : demo_simple_03.py
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
        {"role": "system", "content": "你是一个耐心的 Python 老师，回答要适合初学者。"},
        {"role": "user", "content": "请解释 API Key 的作用。"}
    ],
    temperature=0.4
)

print(completion.choices[0].message.content)
