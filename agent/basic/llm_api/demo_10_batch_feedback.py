# -*- encode = utf-8 -*-
# @Time : 2026-08-22 11:20:21
# @Auther : 罗雄波
# @File : demo_10_batch_feedback.py
# @Software : PyCharm

import json
from llm_client import chat

feedback_list = [
    "推荐的店离我太远了。",
    "优惠券很好用，希望下次还能自动帮我找。",
    "AI 给的路线不合理，绕路太多。",
    "回答速度很快，但推荐理由不够清楚。"
]

results = []

for feedback in feedback_list:
    prompt = f"""
请把用户反馈分类，只输出 JSON。

可选分类：
- 推荐不准
- 优惠相关
- 路线问题
- 解释不足
- 正向反馈
- 其他

用户反馈：{feedback}

输出格式：
{{
  "feedback": "原文",
  "category": "分类",
  "sentiment": "positive/neutral/negative"
}}
"""

    answer = chat(
        prompt,
        system_prompt="你是一个用户反馈分析助手，只输出合法 JSON。",
        json_mode=True
    )
    data = json.loads(answer)
    results.append(data)

for item in results:
    print(item)
