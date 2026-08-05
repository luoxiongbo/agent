from __future__ import annotations

from dataclasses import dataclass
import ast
import json
import operator
from pathlib import Path
from typing import Any, Callable


ToolHandler = Callable[..., Any]


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, dict[str, Any]]
    handler: ToolHandler
    side_effect: bool = False
    requires_approval: bool = False


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def names(self) -> set[str]:
        return set(self._tools)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def describe(self) -> list[dict[str, Any]]:
        result = []
        for tool in self._tools.values():
            result.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "side_effect": tool.side_effect,
                    "requires_approval": tool.requires_approval,
                }
            )
        return result

    def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self.get(name)
        self._validate_arguments(tool, arguments)
        return tool.handler(**arguments)

    @staticmethod
    def _validate_arguments(tool: Tool, arguments: dict[str, Any]) -> None:
        if not isinstance(arguments, dict):
            raise TypeError("Tool arguments must be a JSON object.")

        for param_name, spec in tool.parameters.items():
            required = bool(spec.get("required", False))
            if required and param_name not in arguments:
                raise ValueError(f"Missing required argument: {param_name}")
            if param_name not in arguments:
                continue

            expected = spec.get("type")
            value = arguments[param_name]
            type_map = {
                "string": str,
                "number": (int, float),
                "integer": int,
                "boolean": bool,
                "object": dict,
                "array": list,
            }
            if expected in type_map and not isinstance(value, type_map[expected]):
                raise TypeError(
                    f"Argument {param_name!r} must be {expected}, "
                    f"got {type(value).__name__}."
                )


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_calculate(expression: str) -> int | float:
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent is too large.")
            result = _BINARY_OPERATORS[type(node.op)](left, right)
            if abs(float(result)) > 1e100:
                raise ValueError("Calculation result is too large.")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    return evaluate(tree)


class Workspace:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_path: str) -> Path:
        if not relative_path or relative_path.strip() in {".", ".."}:
            raise ValueError("A concrete relative file path is required.")
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError("Path escapes the configured workspace.") from exc
        return candidate

    def list_files(self, path: str = "") -> list[str]:
        target = self.root if not path else self.resolve(path)
        if not target.exists():
            raise FileNotFoundError(path)
        if not target.is_dir():
            raise NotADirectoryError(path)
        return sorted(
            str(item.relative_to(self.root))
            for item in target.rglob("*")
            if item.is_file()
        )

    def read_file(self, path: str, max_chars: int = 50_000) -> str:
        target = self.resolve(path)
        text = target.read_text(encoding="utf-8")
        if len(text) > max_chars:
            return text[:max_chars] + "\n...[truncated]"
        return text

    def write_file(self, path: str, content: str, overwrite: bool = False) -> dict[str, Any]:
        target = self.resolve(path)
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"{path} already exists. Set overwrite=true only when replacement is intended."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"path": str(target.relative_to(self.root)), "characters": len(content)}


def build_default_tools(workspace_path: str | Path) -> ToolRegistry:
    workspace = Workspace(workspace_path)
    registry = ToolRegistry()

    registry.register(
        Tool(
            name="calculator",
            description="Evaluate a basic arithmetic expression safely.",
            parameters={
                "expression": {
                    "type": "string",
                    "required": True,
                    "description": "Expression using numbers and + - * / // % ** parentheses.",
                }
            },
            handler=safe_calculate,
        )
    )

    registry.register(
        Tool(
            name="list_files",
            description="List files inside the configured workspace.",
            parameters={
                "path": {
                    "type": "string",
                    "required": False,
                    "description": "Optional relative subdirectory.",
                }
            },
            handler=workspace.list_files,
        )
    )

    registry.register(
        Tool(
            name="read_file",
            description="Read a UTF-8 text file from the workspace.",
            parameters={
                "path": {"type": "string", "required": True},
                "max_chars": {"type": "integer", "required": False},
            },
            handler=workspace.read_file,
        )
    )

    registry.register(
        Tool(
            name="write_file",
            description="Write a UTF-8 text file inside the workspace.",
            parameters={
                "path": {"type": "string", "required": True},
                "content": {"type": "string", "required": True},
                "overwrite": {"type": "boolean", "required": False},
            },
            handler=workspace.write_file,
            side_effect=True,
            requires_approval=True,
        )
    )

    return registry
