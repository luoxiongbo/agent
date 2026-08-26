# -*- encode = utf-8 -*-
# @Time : 2026-08-22 11:08:53
# @Auther : 罗雄波
# @File : demo_simple_07.py
# @Software : PyCharm

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
)

prompt = """
请分析下面这条用户反馈，并只输出 JSON，不要输出其他文字。

用户反馈：
“推荐的餐厅太远了，而且优惠券不能用。”

JSON 格式：
{
  "sentiment": "positive/neutral/negative",
  "issues": ["问题1", "问题2"],
  "suggestion": "改进建议"
}
"""

completion = client.chat.completions.create(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    messages=[
        {"role": "system", "content": "你是一个严谨的数据标注助手，只输出合法 JSON。"},
        {"role": "user", "content": prompt}
    ],
    response_format={"type": "json_object"},
    temperature=0.2
)

text = completion.choices[0].message.content
print("原始输出：")
print(text)

data = json.loads(text)
print("解析后：")
print(data["sentiment"], data["issues"])
