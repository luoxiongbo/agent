# -*- encode = utf-8 -*-
# @Time : 2026-08-13 06:48:07
# @Auther : 罗雄波
# @File : extension_demo_code_and_run.py
# @Software : PyCharm

"""
agent 工具分类中的 extension 类型的 demo
以 Google 的 Code Interpreter extension 作为例⼦，从⾃然语⾔描述⽣成和运⾏ Python 代码。
"""

import vertexai
import pprint

PROJECT_ID = "YOUR_PROJECT_ID"
REGION = "us-central1"
vertexai.init(project=PROJECT_ID, location=REGION)
from vertexai.preview.extensions import Extension

extension_code_interpreter = Extension.from_hub("code_interpreter")
CODE_QUERY = """Write a python method to invert a binary tree in O(n) time."""
response = extension_code_interpreter.execute(
    operation_id="generate_and_execute",
    operation_params={"query": CODE_QUERY}
)
print("Generated Code:")
pprint.pprint(response['generated_code'])
