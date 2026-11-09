# -*- encode = utf-8 -*-
# @Time : 2026-10-17 13:23:33
# @Auther : 罗雄波
# @File : my_reflection_agent.py
# @Software : PyCharm

from typing import Optional
from hello_agents import HelloAgentsLLM
from hello_agents import ReflectionAgent

DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务:

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间:

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"[无需改进]"。
""",
    "refine": """
请根据反馈意见改进你的回答:

# 原始任务:
{task}

# 当前回答:
{content}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
"""
}


class MyReflectionAgent(ReflectionAgent):
    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            custom_prompts: Optional[dict] = None,
            max_iterations: int = 3
    ):
        super().__init__(name, llm, system_prompt, max_iterations=max_iterations)
        self.prompt_template = custom_prompts if custom_prompts is not None else DEFAULT_PROMPTS

    def run(self, input_text, **kwargs):
        current_iteration = 0
        prompt = self.prompt_template["initial"].format(
            task=input_text
        )
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.llm.invoke(messages, **kwargs).content

        while current_iteration < self.max_iterations:
            current_iteration += 1
            reflect_prompt = self.prompt_template["reflect"].format(
                task=input_text,
                content=response
            )
            messages.clear()
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": reflect_prompt})
            reflect_response = self.llm.invoke(messages, **kwargs).content

            if "[无需改进]" in reflect_response.strip():
                return response

            refine_prompt = self.prompt_template["refine"].format(
                task=input_text,
                content=response,
                feedback=reflect_response
            )
            messages.clear()
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": refine_prompt})
            response = self.llm.invoke(messages, **kwargs).content

        return response
