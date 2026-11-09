# -*- encode = utf-8 -*-
# @Time : 2026-10-17 15:44:11
# @Auther : 罗雄波
# @File : my_plan_solve_agent.py
# @Software : PyCharm

from typing import Optional
from hello_agents import PlanSolveAgent, HelloAgentsLLM, ToolRegistry

# 默认规划器提示词模板
DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# 默认执行器提示词模板
DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""


class MyPlanAndSolveAgent(PlanSolveAgent):
    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompts: Optional[str] = None,
            planner_prompt: Optional[str] = DEFAULT_PLANNER_PROMPT,
            executor_prompt: Optional[str] = DEFAULT_EXECUTOR_PROMPT,
            tool_registry: Optional["ToolRegistry"] = None,
            enable_tool_calling: bool = True,
            max_tool_iterations: int = 3
    ):
        super().__init__(
            name,
            llm,
            system_prompts,
            planner_prompt=planner_prompt,
            executor_prompt=executor_prompt,
            tool_registry=tool_registry,
            enable_tool_calling=enable_tool_calling,
            max_tool_iterations=max_tool_iterations
        )

    def run(self, input_text, **kwargs):
        plan_steps = self.planner.plan(input_text, **kwargs)

        result = self.executor.execute(input_text, plan_steps, **kwargs)

        return result
