from __future__ import annotations

from typing import List

from company_project.sensors_ai.core.context import ContextStore
from company_project.sensors_ai.core.result import ExecutionResult, Trace


class Skill:
    name = "skill"
    risk_action = ""

    def __init__(self, tools: List[object] | None = None, atomic_agents: List[object] | None = None) -> None:
        self.tools = {getattr(tool, "name"): tool for tool in tools or []}
        self.atomic_agents = {getattr(agent, "name"): agent for agent in atomic_agents or []}

    def execute(self, goal: str, user_id: str, context: ContextStore, trace: Trace) -> ExecutionResult:
        raise NotImplementedError
