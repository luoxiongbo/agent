# -*- encode = utf-8 -*-
# @Time : 2026-08-13 09:14:45
# @Auther : 罗雄波
# @File : prompt_chaining.py
# @Software : PyCharm

def llm_call(message):
    return ""


prompts = ["", ""]


def prompt_chain(human_input: str):
    result = human_input

    for prompt in prompts:
        result = llm_call(
            prompt + "\nInput: " + result
        )

    return result


"""
搜索 → 阅读 → 提取 → 分析 → 写报告
代码 → 找 bug → 修复 → 写测试
文章 → 提炼观点 → 判断证据 → 写摘要
"""

"""
可以看到每个节点解决一个问题，过滤的方式处理问题，每次处理都是方便下一个节点的处理
把问题变成多个简单的问题，模型每次只需要做好一个简单的事情

chain 最大的问题，错误会向下传播
串行运行，所以延迟会积累
这也是A社文章里的所说的权衡，用延迟和高成本换取任务的更好表现

"""