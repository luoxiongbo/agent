# -*- encode = utf-8 -*-
# @Time : 2026-08-29 23:25:28
# @Auther : 罗雄波
# @File : hello_agents_demo.py
# @Software : PyCharm

import os
from dotenv import load_dotenv

# 加载根目录 .env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# 代理配置（Clash 端口 7897）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

from huggingface_hub import InferenceClient

# 需要免费注册获取 Token：https://huggingface.co/settings/tokens
client = InferenceClient(api_key=os.environ["HF_TOKEN"])

# 文本生成（该模型仅支持 conversational 任务，需用 chat_completion）
response = client.chat_completion(
    messages=[
        {"role": "user", "content": "Explain neural networks in two sentences."}
    ],
    model="Qwen/Qwen2.5-72B-Instruct",
    max_tokens=100
)
print(response.choices[0].message.content)
