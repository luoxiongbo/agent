# -*- encode = utf-8 -*-
# @Time : 2026-08-22 11:12:23
# @Auther : 罗雄波
# @File : llm_client.py
# @Software : PyCharm

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
)

MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")


def chat(
        prompt: str,
        system_prompt: str = "你是一个有帮助的 AI 助手。",
        json_mode: bool = False
) -> str:
    extra_params = {}
    if json_mode:
        extra_params["response_format"] = {"type": "json_object"}

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        **extra_params
    )

    return completion.choices[0].message.content
