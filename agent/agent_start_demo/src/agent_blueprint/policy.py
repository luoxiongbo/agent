from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .models import AgentState
from .tools import Tool


ApprovalCallback = Callable[[str, dict[str, Any]], bool]


@dataclass(slots=True)
class PolicyDecision:
    allowed: bool
    reason: str


@dataclass(slots=True)
class AgentPolicy:
    allowed_tools: set[str]
    max_steps: int = 12
    max_tool_calls: int = 10
    require_approval_for_side_effects: bool = True

    def check(
        self,
        *,
        tool: Tool,
        arguments: dict[str, Any],
        state: AgentState,
        approval_callback: ApprovalCallback | None,
    ) -> PolicyDecision:
        if tool.name not in self.allowed_tools:
            return PolicyDecision(False, f"Tool is not allowed: {tool.name}")

        if state.step_count >= self.max_steps:
            return PolicyDecision(False, "Maximum step budget reached.")

        if state.tool_call_count >= self.max_tool_calls:
            return PolicyDecision(False, "Maximum tool-call budget reached.")

        needs_approval = tool.requires_approval or (
            self.require_approval_for_side_effects and tool.side_effect
        )
        if needs_approval:
            if approval_callback is None:
                return PolicyDecision(False, f"Tool {tool.name} requires human approval.")
            if not approval_callback(tool.name, arguments):
                return PolicyDecision(False, f"Human rejected tool call: {tool.name}")

        return PolicyDecision(True, "Allowed")
