# -*- encode = utf-8 -*-
# @Time : 2026-08-22 11:09:50
# @Auther : 罗雄波
# @File : demo_simple_08.py
# @Software : PyCharm

import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
)

MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")


def ask_llm(prompt: str, retries: int = 3) -> str:
    for attempt in range(1, retries + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"第 {attempt} 次调用失败：{e}")

            if attempt == retries:
                raise

            time.sleep(2 * attempt)


answer = ask_llm("请解释什么是 API 重试机制。")
print(answer)
