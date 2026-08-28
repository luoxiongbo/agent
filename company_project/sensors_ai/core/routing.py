from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Route:
    kind: str
    name: str
    risk_action: str = ""

    @property
    def label(self) -> str:
        if self.kind == "governance":
            return "governance"
        return f"{self.kind}:{self.name}"


class Router:
    def choose(self, goal: str) -> Route:
        if "知识" in goal:
            return Route(kind="tool", name="knowledge_lookup")
        if "策略" in goal or "运营方案" in goal:
            return Route(kind="scene", name="strategy")
        if "分群" in goal or "圈选" in goal or "客群" in goal:
            return Route(kind="skill", name="segment_build", risk_action="segment:create")
        return Route(kind="tool", name="knowledge_lookup")
