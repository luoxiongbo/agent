from __future__ import annotations

from abc import ABC, abstractmethod
import ast
import json
import os
import re
from typing import Any


class LLM(ABC):
    @abstractmethod
    def complete_json(self, *, system: str, prompt: str) -> dict[str, Any]:
        raise NotImplementedError


class OpenAIResponsesLLM(LLM):
    """Small adapter around the OpenAI Responses API."""

    def __init__(self, model: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the project dependencies first: pip install -e .") from exc

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.5")
        self.client = OpenAI()

    def complete_json(self, *, system: str, prompt: str) -> dict[str, Any]:
        response = self.client.responses.create(
            model=self.model,
            instructions=system,
            input=prompt,
            text={"format": {"type": "json_object"}},
        )
        text = response.output_text
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Model did not return valid JSON: {text[:500]}") from exc
        if not isinstance(value, dict):
            raise RuntimeError("Model JSON output must be an object.")
        return value


class DemoLLM(LLM):
    """
    Deterministic offline model for demos/tests.

    It recognizes simple arithmetic goals and demonstrates the complete agent
    lifecycle without an API key. It is intentionally not a general LLM.
    """

    def complete_json(self, *, system: str, prompt: str) -> dict[str, Any]:
        mode = self._extract(prompt, "MODE")
        goal = self._extract(prompt, "GOAL")
        observation_count = int(self._extract(prompt, "OBSERVATION_COUNT") or "0")

        if mode == "PLAN":
            return {
                "steps": [
                    {
                        "id": "step-1",
                        "description": "Use an appropriate registered tool to solve the goal.",
                        "depends_on": [],
                    },
                    {
                        "id": "step-2",
                        "description": "Check the result and return a concise answer.",
                        "depends_on": ["step-1"],
                    },
                ]
            }

        if mode == "DECIDE":
            if observation_count > 0:
                return {
                    "kind": "finish",
                    "reason": "A successful tool observation is available.",
                    "final_answer": "Use the latest observation as the answer.",
                }

            expression = self._find_expression(goal)
            if expression:
                return {
                    "kind": "tool",
                    "reason": "The goal contains an arithmetic expression.",
                    "tool_name": "calculator",
                    "arguments": {"expression": expression},
                }

            return {
                "kind": "finish",
                "reason": "Offline demo has no suitable tool for this goal.",
                "final_answer": (
                    "离线 DemoLLM 只演示基础算术。设置 OPENAI_API_KEY 后可使用真实模型。"
                ),
            }

        if mode == "REFLECT":
            last_ok = self._extract(prompt, "LAST_OK") == "true"
            if last_ok:
                return {
                    "status": "done",
                    "reason": "The tool completed successfully.",
                    "final_answer": None,
                }
            return {
                "status": "replan",
                "reason": "The latest tool call failed.",
                "final_answer": None,
            }

        if mode == "FINAL":
            latest_output = self._extract(prompt, "LATEST_OUTPUT")
            return {
                "answer": f"执行结果：{latest_output}",
                "memories": [],
            }

        raise RuntimeError(f"Unsupported DemoLLM mode: {mode}")

    @staticmethod
    def _extract(prompt: str, key: str) -> str:
        match = re.search(rf"^{re.escape(key)}:\s*(.*)$", prompt, flags=re.MULTILINE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _find_expression(goal: str) -> str | None:
        candidates = re.findall(r"[0-9\.\s\+\-\*\/\(\)%]+", goal)
        for candidate in sorted(candidates, key=len, reverse=True):
            candidate = candidate.strip()
            if not candidate or not any(ch.isdigit() for ch in candidate):
                continue
            try:
                ast.parse(candidate, mode="eval")
            except SyntaxError:
                continue
            return candidate
        return None
