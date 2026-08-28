from __future__ import annotations

from company_project.sensors_ai.components.scene_agent import SceneAgent, SceneTask
from company_project.sensors_ai.core.result import ExecutionResult


class CustomerSegmentSceneAgent(SceneAgent):
    name = "customer_segment"
    description = "Handles customer segmentation tasks."

    def handle(self, task: SceneTask) -> ExecutionResult:
        skill = self.skills.get("segment_build")
        if skill is None:
            return self.complete(task, "缺少分群构建技能")
        return skill.execute(goal=task.goal, user_id=task.user_id, context=task.context, trace=task.trace)


class StrategySceneAgent(SceneAgent):
    name = "strategy"
    description = "Handles open-ended operation strategy design tasks."

    def handle(self, task: SceneTask) -> ExecutionResult:
        knowledge = {}
        for term in task.context.knowledge:
            if term in task.goal:
                knowledge.update(task.context.lookup_knowledge(term))
        memory = {}
        for term in task.context.memories:
            if term in task.goal:
                memory[term] = task.context.memories[term]
        output = "运营策略: 基于目标人群、指标口径和企业知识生成策略草案"
        if knowledge:
            output += f"; knowledge={knowledge}"
        if memory:
            output += f"; memory={memory}"
        return self.complete(task, output)
