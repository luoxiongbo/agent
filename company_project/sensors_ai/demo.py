from __future__ import annotations

from company_project.sensors_ai.core.context import AssetStore, AuditLog, ContextStore
from company_project.sensors_ai.components.main_agent import MainAgent, Registry
from company_project.sensors_ai.builtins import BusinessDataTool, SegmentBuildSkill, StrategySceneAgent


def build_demo_agent() -> MainAgent:
    context = ContextStore(
        knowledge={"代发客群": "payroll customer segment"},
        memories={"理财购买率": "use purchase_rate metric"},
        assets=AssetStore(),
        audit_log=AuditLog(),
    )
    registry = Registry()
    segment_skill = SegmentBuildSkill(tools=[BusinessDataTool()])
    registry.register_skill(segment_skill)
    registry.register_scene(StrategySceneAgent(skills=[segment_skill], tools=[]))
    return MainAgent(registry=registry, context=context)


def main() -> None:
    agent = build_demo_agent()
    result = agent.run("针对代发客群设计提升理财购买率的运营策略", user_id="demo-user")
    print(result.output)
    print("route:", result.route)
    print("trace:", " -> ".join(result.trace.component_names()))


if __name__ == "__main__":
    main()
