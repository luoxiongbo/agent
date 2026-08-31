# -*- encode = utf-8 -*-
# @Time : 2026-08-30 21:28:48
# @Auther : 罗雄波
# @File : tool_syntax.py
# @Software : PyCharm

import os

from dotenv import load_dotenv
import aisuite as ai
from datetime import datetime

load_dotenv()


def get_current_time():
    """return current time as string"""
    return datetime.now().strftime("%H:%M:%S")


# aisuite 的 deepseek provider 会硬编码 base_url 为官方地址 https://api.deepseek.com,
# 而本项目的 DEEPSEEK_API_KEY 属于中转站(api.chatanywhere.tech),直接用 deepseek 前缀会 401。
# 中转站是 OpenAI 兼容接口,因此改用 openai provider 并显式传入自定义 base_url。
client = ai.Client(
    provider_configs={
        "openai": {
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "base_url": os.getenv("DEEPSEEK_BASE_URL"),
        }
    }
)

response = client.chat.completions.create(
    model=f"openai:{os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-flash')}",
    messages=[
        {"role": "user", "content": "What time is it now?"}
    ],
    tools=[get_current_time],
    max_turns=5
)

print(response.choices[0].message.content)
