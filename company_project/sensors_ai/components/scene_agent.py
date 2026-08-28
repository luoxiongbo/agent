from __future__ import annotations

from dataclasses import dataclass
from typing import List

from company_project.sensors_ai.core.context import ContextStore
from company_project.sensors_ai.core.result import ExecutionResult, Trace


class SceneAgentBoundaryError(ValueError):
    pass


@dataclass(frozen=True)
class SceneTask:
    goal: str
    user_id: str
    context: ContextStore
    trace: Trace


class SceneAgent:
    name = "scene"
    description = ""

    def __init__(self, skills: List[object] | None = None, tools: List[object] | None = None) -> None:
        self.skills = {getattr(skill, "name"): skill for skill in skills or []}
        self.tools = {getattr(tool, "name"): tool for tool in tools or []}

    def mount_scene(self, scene: "SceneAgent") -> None:
        raise SceneAgentBoundaryError("SceneAgent cannot mount or call another SceneAgent")

    def handle(self, task: SceneTask) -> ExecutionResult:
        raise NotImplementedError

    def complete(self, task: SceneTask, output: str) -> ExecutionResult:
        return ExecutionResult(
            status="completed",
            output=output,
            route=f"scene:{self.name}",
            trace=task.trace,
            assets=[],
        )
