from typing import Literal
from dataclasses import dataclass, field


@dataclass
class Observation:
    output: str = ""


@dataclass
class AgentState:
    observations: list[Observation] = field(default_factory=list)
    status: Literal["new", "running", "done", "failed", "stopped"] = "new"


def llm(state: AgentState) -> str:
    if len(state.observations) > 0:
        state.status = "done"
        return state.observations[0].output
    else:
        state.status = "failed"
        return ""


def reflection(state: AgentState):
    data = llm(state)
    if state.status not in {"running", "done", "failed", "stopped"}:
        raise RuntimeError(f"Invalid reflection status: {state.status}")
    else:
        return data


class TinyAgent:
    def __init__(self):
        self.count = 9

    def run(self, user_input: str) -> str:
        num = 0
        state = AgentState()
        state.status = "running"
        while num < self.count:

            if user_input.count("weather") > 0:
                observation = weather()
                state.observations.append(observation)

            data = reflection(state)

            if state.status == "done":
                return data
            elif state.status == "failed":
                return "[ERROR]execute failed"
            elif state.status == "stopped":
                return "[WARRING]execute stopped"

            num += 1

        return "[WARRING]out run num."


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def weather() -> Observation:
    return Observation(output="☀️")


def main():
    agent = TinyAgent()
    user_input = "what the weather?"
    result = agent.run(user_input)
    print(result)


if __name__ == "__main__":
    main()
