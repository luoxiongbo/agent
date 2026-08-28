from __future__ import annotations

from company_project.sensors_ai.core.context import ContextStore
from company_project.sensors_ai.core.governance import Governance
from company_project.sensors_ai.core.registry import Registry
from company_project.sensors_ai.core.result import ExecutionResult, Trace
from company_project.sensors_ai.core.routing import Router

from .scene_agent import SceneTask


class MainAgent:
    def __init__(
        self,
        registry: Registry,
        context: ContextStore,
        governance: Governance | None = None,
        router: Router | None = None,
    ) -> None:
        self.registry = registry
        self.context = context
        self.governance = governance or Governance()
        self.router = router or Router()

    def run(self, goal: str, user_id: str) -> ExecutionResult:
        trace = Trace()
        trace.add("main_agent", "receive_goal", {"goal": goal, "user_id": user_id})
        self.context.audit_log.record("main_agent", "receive_goal", {"goal": goal, "user_id": user_id})

        route = self.router.choose(goal)
        decision = self.governance.check(user_id=user_id, action=route.risk_action)
        if not decision.allowed:
            trace.add("governance", "blocked", {"reason": decision.reason})
            self.context.audit_log.record("governance", "blocked", {"reason": decision.reason})
            output = f"blocked by governance: {decision.reason}"
            self.context.record_history(user_id, goal, output)
            return ExecutionResult(status="blocked", output=output, route="governance", trace=trace)

        if route.kind == "skill":
            skill = self.registry.skill(route.name)
            if skill is None:
                return self._missing(route.label, trace, goal, user_id)
            result = skill.execute(goal=goal, user_id=user_id, context=self.context, trace=trace)
            self.context.record_history(user_id, goal, result.output)
            return result

        if route.kind == "scene":
            scene = self.registry.scene(route.name)
            if scene is None:
                return self._missing(route.label, trace, goal, user_id)
            trace.add(f"scene:{route.name}", "delegate", {"goal": goal})
            self.context.audit_log.record(f"scene:{route.name}", "delegate", {"goal": goal})
            result = scene.handle(SceneTask(goal=goal, user_id=user_id, context=self.context, trace=trace))
            self.context.record_history(user_id, goal, result.output)
            return result

        if route.kind == "tool":
            return self._run_base_tool(route.name, goal, user_id, trace)

        return self._missing(route.label, trace, goal, user_id)

    def _run_base_tool(self, name: str, goal: str, user_id: str, trace: Trace) -> ExecutionResult:
        if name == "knowledge_lookup":
            trace.add("tool:knowledge_lookup", "lookup", {"goal": goal})
            self.context.audit_log.record("tool:knowledge_lookup", "lookup", {"goal": goal})
            terms = [term for term in self.context.knowledge if term in goal]
            matches = {}
            for term in terms:
                matches.update(self.context.lookup_knowledge(term))
            output = str(matches or self.context.lookup_knowledge(goal) or {})
            self.context.record_history(user_id, goal, output)
            return ExecutionResult(status="completed", output=output, route="tool:knowledge_lookup", trace=trace)
        return self._missing(f"tool:{name}", trace, goal, user_id)

    def _missing(self, route: str, trace: Trace, goal: str, user_id: str) -> ExecutionResult:
        output = f"missing component: {route}"
        self.context.record_history(user_id, goal, output)
        return ExecutionResult(status="failed", output=output, route=route, trace=trace)
