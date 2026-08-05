from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from .llm import LLM
from .memory import SQLiteMemory
from .models import Action, AgentState, Observation, PlanStep, Reflection
from .policy import AgentPolicy, ApprovalCallback
from .tools import ToolRegistry
from .tracing import JsonlTracer


PLANNER_SYSTEM = """
You are the planning and control component of a tool-using software agent.
Never claim a tool was executed unless an observation says so.
Respect tool schemas, current state, policy limits and the user's goal.
Return exactly one JSON object matching the requested shape.
Keep plans short and executable. Do not expose private chain-of-thought;
provide only concise decision reasons.
""".strip()


class AgentRuntime:
    def __init__(
        self,
        *,
        llm: LLM,
        memory: SQLiteMemory,
        tools: ToolRegistry,
        policy: AgentPolicy,
        tracer: JsonlTracer | None = None,
        approval_callback: ApprovalCallback | None = None,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.policy = policy
        self.tracer = tracer or JsonlTracer()
        self.approval_callback = approval_callback

    def run(
        self,
        *,
        goal: str,
        namespace: str = "default",
        resume_run_id: str | None = None,
    ) -> AgentState:
        if resume_run_id:
            state = self.memory.load_checkpoint(resume_run_id)
            if state is None:
                raise KeyError(f"No checkpoint found for run_id={resume_run_id}")
            if state.status == "completed":
                return state
        else:
            state = AgentState(goal=goal, namespace=namespace)
            memories = self.memory.search(goal, namespace=namespace)
            state.memories_used = [record.content for record in memories]
            state.plan = self._create_plan(state)

        state.status = "running"
        self._save_and_trace(state, "run_started", {"goal": state.goal})

        while state.step_count < self.policy.max_steps:
            state.step_count += 1
            action = self._choose_action(state)
            self.tracer.emit(
                "action_selected",
                run_id=state.run_id,
                payload=asdict(action),
            )

            if action.kind == "finish":
                state.final_answer = action.final_answer or self._make_final_answer(state)
                state.status = "completed"
                self._persist_completion_memories(state)
                self._save_and_trace(
                    state,
                    "run_completed",
                    {"final_answer": state.final_answer},
                )
                return state

            if action.kind == "replan":
                state.plan = self._create_plan(state, reason=action.reason)
                self._save_and_trace(state, "plan_replaced", {"reason": action.reason})
                continue

            observation = self._execute_tool(action, state)
            state.observations.append(observation)
            state.tool_call_count += 1
            self.tracer.emit(
                "observation",
                run_id=state.run_id,
                payload=asdict(observation),
            )

            reflection = self._reflect(state)
            self.tracer.emit(
                "reflection",
                run_id=state.run_id,
                payload=asdict(reflection),
            )

            if reflection.status == "done":
                state.final_answer = reflection.final_answer or self._make_final_answer(state)
                state.status = "completed"
                self._persist_completion_memories(state)
                self._save_and_trace(
                    state,
                    "run_completed",
                    {"final_answer": state.final_answer},
                )
                return state

            if reflection.status == "blocked":
                state.status = "blocked"
                state.errors.append(reflection.reason)
                state.final_answer = reflection.final_answer
                self._save_and_trace(state, "run_blocked", {"reason": reflection.reason})
                return state

            if reflection.status == "replan":
                state.plan = self._create_plan(state, reason=reflection.reason)

            self.memory.save_checkpoint(state)

        state.status = "stopped"
        state.errors.append("Maximum step budget reached.")
        state.final_answer = self._make_final_answer(state, forced=True)
        self._save_and_trace(state, "run_stopped", {"reason": state.errors[-1]})
        return state

    def _create_plan(self, state: AgentState, reason: str = "") -> list[PlanStep]:
        prompt = f"""
MODE: PLAN
GOAL: {state.goal}
REPLAN_REASON: {reason}
RELEVANT_MEMORIES:
{json.dumps(state.memories_used, ensure_ascii=False)}
AVAILABLE_TOOLS:
{json.dumps(self.tools.describe(), ensure_ascii=False)}

Return:
{{
  "steps": [
    {{"id": "short-id", "description": "executable step", "depends_on": []}}
  ]
}}
Use 2-7 high-level steps.
""".strip()
        data = self.llm.complete_json(system=PLANNER_SYSTEM, prompt=prompt)
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise RuntimeError("Planner returned no plan steps.")

        result: list[PlanStep] = []
        for index, item in enumerate(raw_steps, start=1):
            if not isinstance(item, dict):
                continue
            result.append(
                PlanStep(
                    id=str(item.get("id") or f"step-{index}"),
                    description=str(item.get("description") or "").strip(),
                    depends_on=[str(value) for value in item.get("depends_on", [])],
                )
            )
        if not result or any(not step.description for step in result):
            raise RuntimeError("Planner returned invalid plan steps.")
        return result

    def _choose_action(self, state: AgentState) -> Action:
        prompt = f"""
MODE: DECIDE
GOAL: {state.goal}
STEP_COUNT: {state.step_count}
OBSERVATION_COUNT: {len(state.observations)}
PLAN:
{json.dumps([asdict(step) for step in state.plan], ensure_ascii=False)}
LATEST_OBSERVATIONS:
{json.dumps([asdict(item) for item in state.observations[-4:]], ensure_ascii=False, default=str)}
RELEVANT_MEMORIES:
{json.dumps(state.memories_used, ensure_ascii=False)}
AVAILABLE_TOOLS:
{json.dumps(self.tools.describe(), ensure_ascii=False)}

Return one of:
1. {{"kind":"tool","reason":"...","tool_name":"...","arguments":{{...}}}}
2. {{"kind":"finish","reason":"...","final_answer":"..."}}
3. {{"kind":"replan","reason":"..."}}
""".strip()
        data = self.llm.complete_json(system=PLANNER_SYSTEM, prompt=prompt)
        kind = str(data.get("kind", ""))
        if kind not in {"tool", "finish", "replan"}:
            raise RuntimeError(f"Invalid action kind: {kind}")

        action = Action(
            kind=kind,  # type: ignore[arg-type]
            reason=str(data.get("reason", "")).strip() or "No reason provided.",
            tool_name=data.get("tool_name"),
            arguments=data.get("arguments") or {},
            final_answer=data.get("final_answer"),
        )
        if action.kind == "tool" and not action.tool_name:
            raise RuntimeError("Tool action is missing tool_name.")
        return action

    def _execute_tool(self, action: Action, state: AgentState) -> Observation:
        assert action.tool_name is not None
        try:
            tool = self.tools.get(action.tool_name)
            decision = self.policy.check(
                tool=tool,
                arguments=action.arguments,
                state=state,
                approval_callback=self.approval_callback,
            )
            if not decision.allowed:
                return Observation(
                    tool_name=action.tool_name,
                    ok=False,
                    error=f"Policy denied action: {decision.reason}",
                )

            output = self.tools.invoke(action.tool_name, action.arguments)
            return Observation(tool_name=action.tool_name, ok=True, output=output)
        except Exception as exc:
            return Observation(
                tool_name=action.tool_name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _reflect(self, state: AgentState) -> Reflection:
        latest = state.observations[-1]
        prompt = f"""
MODE: REFLECT
GOAL: {state.goal}
LAST_OK: {str(latest.ok).lower()}
LATEST_OBSERVATION:
{json.dumps(asdict(latest), ensure_ascii=False, default=str)}
ALL_OBSERVATIONS:
{json.dumps([asdict(item) for item in state.observations[-6:]], ensure_ascii=False, default=str)}
PLAN:
{json.dumps([asdict(step) for step in state.plan], ensure_ascii=False)}

Check whether the goal is satisfied, more work is needed, the plan should change,
or progress is blocked.

Return:
{{
  "status": "continue|done|replan|blocked",
  "reason": "concise verification reason",
  "final_answer": "optional answer when done or blocked"
}}
""".strip()
        data = self.llm.complete_json(system=PLANNER_SYSTEM, prompt=prompt)
        status = str(data.get("status", ""))
        if status not in {"continue", "done", "replan", "blocked"}:
            raise RuntimeError(f"Invalid reflection status: {status}")
        return Reflection(
            status=status,  # type: ignore[arg-type]
            reason=str(data.get("reason", "")).strip() or "No reflection reason.",
            final_answer=data.get("final_answer"),
        )

    def _make_final_answer(self, state: AgentState, forced: bool = False) -> str:
        latest_output = ""
        if state.observations:
            latest = state.observations[-1]
            latest_output = json.dumps(
                latest.output if latest.ok else latest.error,
                ensure_ascii=False,
                default=str,
            )

        prompt = f"""
MODE: FINAL
GOAL: {state.goal}
STATUS: {state.status}
FORCED_STOP: {str(forced).lower()}
LATEST_OUTPUT: {latest_output}
OBSERVATIONS:
{json.dumps([asdict(item) for item in state.observations], ensure_ascii=False, default=str)}
ERRORS:
{json.dumps(state.errors, ensure_ascii=False)}

Return:
{{
  "answer": "truthful concise final answer based only on observations",
  "memories": [
    {{
      "kind": "semantic|episodic|procedural|preference",
      "content": "future-useful verified information",
      "importance": 0.0,
      "confidence": 0.0
    }}
  ]
}}
Do not invent successful actions.
""".strip()
        data = self.llm.complete_json(system=PLANNER_SYSTEM, prompt=prompt)
        answer = str(data.get("answer", "")).strip()
        return answer or "任务结束，但没有生成可用答案。"

    def _persist_completion_memories(self, state: AgentState) -> None:
        if not state.observations:
            return

        latest = state.observations[-1]
        if latest.ok:
            self.memory.add(
                namespace=state.namespace,
                kind="episodic",
                content=(
                    f"Goal: {state.goal}\n"
                    f"Last successful tool: {latest.tool_name}\n"
                    f"Result: {json.dumps(latest.output, ensure_ascii=False, default=str)}"
                ),
                importance=0.55,
                confidence=0.9,
                metadata={"run_id": state.run_id},
            )

    def _save_and_trace(
        self,
        state: AgentState,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        self.memory.save_checkpoint(state)
        self.tracer.emit(event, run_id=state.run_id, payload=payload)
