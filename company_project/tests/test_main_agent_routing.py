import unittest

from company_project import (
    AssetStore,
    AuditLog,
    ContextStore,
    ExecutionContext,
    MainAgent,
    Registry,
)
from company_project.builtins import (
    BusinessDataTool,
    CustomerSegmentSceneAgent,
    SegmentBuildSkill,
    StrategySceneAgent,
)


def build_agent() -> MainAgent:
    context = ContextStore(
        knowledge={"代发客群": "payroll customer segment"},
        memories={"理财购买率": "use purchase_rate metric"},
        assets=AssetStore(),
        audit_log=AuditLog(),
    )
    registry = Registry()
    business_tool = BusinessDataTool()
    segment_skill = SegmentBuildSkill(tools=[business_tool])
    registry.register_skill(segment_skill)
    registry.register_scene(CustomerSegmentSceneAgent(skills=[segment_skill], tools=[]))
    registry.register_scene(StrategySceneAgent(skills=[], tools=[]))
    return MainAgent(registry=registry, context=context)


class MainAgentRoutingTest(unittest.TestCase):
    def test_main_agent_routes_clear_segment_goal_directly_to_skill(self):
        agent = build_agent()

        result = agent.run("帮我创建一个代发客群分群", user_id="u1")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.route, "skill:segment_build")
        self.assertEqual(result.assets[0].asset_type, "segment")
        self.assertEqual(
            result.trace.component_names(),
            ["main_agent", "skill:segment_build", "tool:business_data"],
        )


    def test_main_agent_delegates_open_strategy_goal_to_scene_agent(self):
        agent = build_agent()

        result = agent.run("针对代发客群设计提升理财购买率的运营策略", user_id="u1")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.route, "scene:strategy")
        self.assertIn("运营策略", result.output)
        self.assertEqual(result.trace.component_names(), ["main_agent", "scene:strategy"])


    def test_main_agent_can_directly_use_base_tool_for_knowledge_query(self):
        agent = build_agent()

        result = agent.run("查询代发客群的知识", user_id="u1")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.route, "tool:knowledge_lookup")
        self.assertIn("payroll customer segment", result.output)
        self.assertEqual(result.trace.component_names(), ["main_agent", "tool:knowledge_lookup"])


if __name__ == "__main__":
    unittest.main()
