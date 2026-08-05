from __future__ import annotations

import ast
import io
import keyword
import re
import tokenize
from dataclasses import dataclass
from typing import Sequence

_PYTHON_START_RE = re.compile(
    r"^\s*(?:@|async\s+def\b|def\b|class\b|from\b|import\b|if\b|for\b|while\b|try\b|with\b|"
    r"return\b|yield\b|raise\b|break\b|continue\b|pass\b|elif\b|else\b|except\b|finally\b|"
    r"[A-Za-z_]\w*\s*=)"
)
_BLOCK_HEADER_RE = re.compile(
    r"^\s*(?:async\s+def|def|class|if|elif|else|for|while|try|except|finally|with|match|case)\b.*:\s*(?:#.*)?$"
)
_DEDENT_KEYWORD_RE = re.compile(r"^\s*(?:elif|else|except|finally|case)\b")
_IDENTIFIER_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(slots=True)
class PythonCodeAnalysis:
    is_python: bool
    ast_parseable: bool
    syntax_error: str | None
    indentation_issue_count: int
    split_identifier_candidate_count: int
    function_boundary_violation_count: int
    top_level_definition_count: int

    @property
    def has_issue(self) -> bool:
        return bool(
            self.is_python
            and (
                not self.ast_parseable
                or self.indentation_issue_count
                or self.split_identifier_candidate_count
                or self.function_boundary_violation_count
            )
        )


def looks_like_python_code(text: str) -> bool:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return False
    markers = sum(bool(_PYTHON_START_RE.match(line)) for line in lines)
    punctuation = sum(bool(re.search(r"[:=()\[\]{}]", line)) for line in lines)
    strong_marker = any(
        re.match(r"^\s*(?:async\s+def|def|class|from|import)\b", line)
        for line in lines
    )
    fragment_marker = any(
        re.match(
            r"^\s*(?:return|yield|raise|break|continue|pass|elif|else|except|finally)\b",
            line,
        )
        for line in lines
    )
    return strong_marker or fragment_marker or (markers >= 1 and punctuation >= 1)


def repair_code_physical_lines(
    lines: Sequence[str],
    x0s: Sequence[float] | None = None,
) -> tuple[list[str], list[float]]:
    xs = list(x0s or [0.0] * len(lines))
    pairs = [(str(line), float(xs[index] if index < len(xs) else 0.0)) for index, line in enumerate(lines)]
    output_lines: list[str] = []
    output_xs: list[float] = []
    index = 0
    while index < len(pairs):
        line, x0 = pairs[index]
        current = line.rstrip()
        while index + 1 < len(pairs):
            right, _ = pairs[index + 1]
            if not should_join_code_physical_lines(current, right):
                break
            current = current.rstrip() + right.lstrip()
            index += 1
        output_lines.append(current)
        output_xs.append(x0)
        index += 1
    return output_lines, output_xs


def should_join_code_physical_lines(left: str, right: str) -> bool:
    left_s = left.rstrip()
    right_s = right.lstrip()
    if not left_s or not right_s:
        return False

    if _has_unclosed_quote(left_s) or _has_unclosed_brackets(left_s):
        return bool(re.match(r"^[A-Za-z0-9_.'\"\]\)}]", right_s))

    left_token_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)$", left_s)
    right_token_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", right_s)
    if not left_token_match or not right_token_match:
        return False
    left_token = left_token_match.group(1)
    right_token = right_token_match.group(1)

    if left_token.endswith("_"):
        return True
    if len(left_token) == 1 and left_token + right_token in keyword.kwlist:
        return True
    if len(left_token) >= 4 and len(right_token) <= 2:
        if not re.search(r"[,:;=+\-*/%<>]\s*$", left_s):
            return True
    return False


def analyze_python_code(text: str) -> PythonCodeAnalysis:
    source = (text or "").strip()
    is_python = looks_like_python_code(source)
    if not is_python:
        return PythonCodeAnalysis(False, False, None, 0, 0, 0, 0)

    parseable = False
    syntax_error: str | None = None
    top_defs = 0
    try:
        tree = ast.parse(source)
        compile(tree, "<pdf-code>", "exec")
        parseable = True
        top_defs = sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in tree.body
        )
    except (SyntaxError, IndentationError) as exc:
        syntax_error = f"{exc.__class__.__name__}: {exc.msg} (line {exc.lineno})"

    indentation_issues = _count_indentation_issues(source)
    split_identifiers = _count_split_identifier_candidates(source)
    boundary_violations = _count_function_boundary_violations(source)
    if syntax_error and re.search(
        r"(?:outside function|outside loop|not properly in loop)",
        syntax_error,
        re.IGNORECASE,
    ):
        boundary_violations += 1
    return PythonCodeAnalysis(
        is_python=True,
        ast_parseable=parseable,
        syntax_error=syntax_error,
        indentation_issue_count=indentation_issues,
        split_identifier_candidate_count=split_identifiers,
        function_boundary_violation_count=boundary_violations,
        top_level_definition_count=top_defs,
    )


def split_python_top_level_blocks(text: str) -> list[str]:
    source = (text or "").strip("\n")
    if not source:
        return []
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except (SyntaxError, IndentationError):
        return _heuristic_top_level_blocks(lines)

    spans: list[tuple[int, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno, *[decorator.lineno for decorator in node.decorator_list]])
            end = int(getattr(node, "end_lineno", node.lineno))
            spans.append((start, end))
    if not spans:
        return [source]

    blocks: list[str] = []
    cursor = 1
    for start, end in spans:
        if start > cursor:
            preamble = "\n".join(lines[cursor - 1 : start - 1]).strip("\n")
            if preamble.strip():
                blocks.append(preamble)
        blocks.append("\n".join(lines[start - 1 : end]).strip("\n"))
        cursor = end + 1
    if cursor <= len(lines):
        tail = "\n".join(lines[cursor - 1 :]).strip("\n")
        if tail.strip():
            blocks.append(tail)
    return blocks


def first_definition_signature(text: str) -> str | None:
    for line in (text or "").splitlines():
        if re.match(r"^\s*(?:async\s+def|def|class)\b", line):
            return line.rstrip()
    return None


def _heuristic_top_level_blocks(lines: Sequence[str]) -> list[str]:
    starts = [
        index
        for index, line in enumerate(lines)
        if re.match(r"^(?:@|async\s+def\b|def\b|class\b)", line)
    ]
    if not starts:
        return ["\n".join(lines).strip("\n")]
    if starts[0] != 0:
        starts.insert(0, 0)
    starts.append(len(lines))
    blocks: list[str] = []
    for left, right in zip(starts, starts[1:]):
        block = "\n".join(lines[left:right]).strip("\n")
        if block.strip():
            blocks.append(block)
    return blocks


def _count_indentation_issues(source: str) -> int:
    issues = 0
    lines = source.splitlines()
    for index, line in enumerate(lines[:-1]):
        if not _BLOCK_HEADER_RE.match(line):
            continue
        current_indent = _indent_width(line)
        next_index = index + 1
        while next_index < len(lines) and not lines[next_index].strip():
            next_index += 1
        if next_index >= len(lines):
            issues += 1
            continue
        if _indent_width(lines[next_index]) <= current_indent:
            issues += 1
    try:
        list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (IndentationError, tokenize.TokenError):
        issues += 1
    return issues


def _count_split_identifier_candidates(source: str) -> int:
    count = 0
    lines = source.splitlines()
    for left, right in zip(lines, lines[1:]):
        if should_join_code_physical_lines(left, right):
            count += 1
    return count


def _count_function_boundary_violations(source: str) -> int:
    lines = source.splitlines()
    violations = 0
    active_definition_indent: int | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        indent = _indent_width(line)
        if re.match(r"^(?:async\s+def|def|class)\b", stripped):
            active_definition_indent = indent
            continue
        if active_definition_indent is not None:
            if indent <= active_definition_indent and not re.match(
                r"^(?:@|async\s+def|def|class|#)", stripped
            ):
                if re.match(r"^(?:return|yield|raise|break|continue|pass)\b", stripped):
                    violations += 1
            if indent < active_definition_indent:
                active_definition_indent = None
    return violations


def _indent_width(line: str) -> int:
    prefix = re.match(r"^[ \t]*", line).group(0)
    return len(prefix.expandtabs(4))


def _has_unclosed_quote(value: str) -> bool:
    single = len(re.findall(r"(?<!\\)'", value)) % 2
    double = len(re.findall(r'(?<!\\)"', value)) % 2
    return bool(single or double)


def _has_unclosed_brackets(value: str) -> bool:
    pairs = {"(": ")", "[": "]", "{": "}"}
    return any(value.count(left) > value.count(right) for left, right in pairs.items())
