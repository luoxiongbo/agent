# -*- encode = utf-8 -*-
# @Time : 2026-08-22 12:14:42
# @Auther : 罗雄波
# @File : demo_simple_11_rag.py
# @Software : PyCharm

from llm_client import chat

docs = [
    "美团 AI 助手可以帮助用户进行本地生活服务搜索，例如找餐厅、找优惠、规划行程。",
    "RAG 是检索增强生成，适合基于外部知识库回答问题。",
    "AI 产品运营需要关注用户激活、留存、转化、满意度和反馈闭环。",
]


def simple_search(query: str, docs: list[str]) -> list[str]:
    # 这是非常简单的关键词匹配，只用于教学演示。
    # 真实项目通常会使用 embedding + 向量数据库做语义检索。
    keywords = ["美团", "AI", "助手", "本地生活", "优惠", "行程"]
    matched = []
    for doc in docs:
        if any(keyword in doc for keyword in keywords if keyword in query):
            matched.append(doc)
    return matched[:3]


question = "美团 AI 助手有什么用？"
contexts = simple_search(question, docs)

prompt = f"""
请只基于下面资料回答问题。
如果资料中没有答案，请回答“资料中没有相关信息”。

资料：
{chr(10).join(contexts)}

问题：
{question}
"""

answer = chat(prompt, system_prompt="你是一个严谨的知识库问答助手。")
print(answer)
