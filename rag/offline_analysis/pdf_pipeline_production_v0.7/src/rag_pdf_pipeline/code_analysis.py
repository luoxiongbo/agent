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
_IDENTIFIER_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_PYTHON_STATEMENT_START_RE = re.compile(
    r"^(?:@|async\s+def\b|def\b|class\b|if\b|elif\b|else\b|for\b|while\b|try\b|"
    r"except\b|finally\b|with\b|match\b|case\b|return\b|yield\b|raise\b|break\b|"
    r"continue\b|pass\b|import\b|from\b)"
)
_COMPLETE_SIMPLE_STATEMENT_RE = re.compile(
    r"^(?:break|continue|pass|return(?:\s+.+)?|raise(?:\s+.+)?)\s*(?:#.*)?$"
)
_NARRATIVE_PREFIX_RE = re.compile(
    r"^(?:测试|构建|计算|定义|初始化|假设|对查询|对用户|准备|配置|微调|将查询|将候选|"
    r"示例|运行|输出|使用|加载|创建|获取|生成|调用|读取|保存|打印|排序|合并|设置|"
    r"其他格式|这里|注意|说明|可选|@假设|\(可选\)|（可选）)"
)


@dataclass(slots=True)
class PythonCodeAnalysis:
    is_python: bool
    ast_parseable: bool
    syntax_error: str | None
    normalized_ast_parseable: bool
    normalized_syntax_error: str | None
    normalized_source: str | None
    code_type: str
    indentation_issue_count: int
    split_identifier_candidate_count: int
    function_boundary_violation_count: int
    top_level_definition_count: int
    narrative_line_count: int

    @property
    def has_issue(self) -> bool:
        return bool(
            self.is_python
            and (
                not self.normalized_ast_parseable
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
    x1s: Sequence[float] | None = None,
    block_x1: float | None = None,
) -> tuple[list[str], list[float]]:
    """Join only high-confidence visual soft wraps.

    This function deliberately preserves logical newlines and leading spaces. PDF code
    panels in the target corpus already contain correct indentation; the main damage is
    visual wrapping at the panel's right edge.
    """
    xs0 = list(x0s or [0.0] * len(lines))
    xs1 = list(x1s or [0.0] * len(lines))
    pairs = [
        (
            str(line),
            float(xs0[index] if index < len(xs0) else 0.0),
            float(xs1[index] if index < len(xs1) else 0.0),
        )
        for index, line in enumerate(lines)
    ]
    output_lines: list[str] = []
    output_xs: list[float] = []
    index = 0
    while index < len(pairs):
        line, x0, x1 = pairs[index]
        current = line.rstrip()
        current_x1 = x1
        while index + 1 < len(pairs):
            right, _, right_x1 = pairs[index + 1]
            if not should_join_code_physical_lines(
                current,
                right,
                left_x1=current_x1,
                block_x1=block_x1,
            ):
                break
            current = current.rstrip() + right.lstrip()
            current_x1 = right_x1
            index += 1
        output_lines.append(current)
        output_xs.append(x0)
        index += 1
    return output_lines, output_xs


def should_join_code_physical_lines(
    left: str,
    right: str,
    *,
    left_x1: float | None = None,
    block_x1: float | None = None,
) -> bool:
    left_s = left.rstrip()
    right_s = right.lstrip()
    if not left_s or not right_s:
        return False

    right_indent = len(right) - len(right.lstrip(" \t"))
    near_right_edge = bool(
        block_x1 is not None
        and left_x1 is not None
        and left_x1 >= block_x1 - 20.0
    )

    # A quoted string visually wrapped by the browser/PDF renderer. CJK is valid
    # inside the string, so do not restrict the continuation to ASCII.
    if _has_unclosed_quote(left_s):
        return right_indent == 0

    # Common operators split at the right edge: `!` + `=`, `.` + attribute, etc.
    if near_right_edge and right_indent == 0:
        if re.search(r"(?:!|<|>|=|\.|\\|[+\-*/%])$", left_s) and re.match(
            r"^[=A-Za-z_\[\(\"'\u3400-\u9fff]", right_s
        ):
            return True

        left_token_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)$", left_s)
        right_token_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", right_s)
        if left_token_match and right_token_match:
            left_token = left_token_match.group(1)
            right_token = right_token_match.group(1)
            # Preserve genuine split identifiers and split Python keywords before
            # applying statement-boundary guards.
            if left_token.endswith("_"):
                return True
            if len(left_token) == 1 and left_token + right_token in keyword.kwlist:
                return True
            if _COMPLETE_SIMPLE_STATEMENT_RE.match(left_s.strip()):
                return False
            if _PYTHON_STATEMENT_START_RE.match(right_s):
                return False
            # Right-edge word wrapping can split identifiers, but must never glue a
            # complete statement to the next statement (`break` + `if`, comment + if).
            return True

        # Wrapped comments are joined only to prose continuations, never to Python
        # statements that start on the next logical line.
        if left_s.lstrip().startswith("#") and _CJK_RE.match(right_s):
            return not _PYTHON_STATEMENT_START_RE.match(right_s)
        if _CJK_RE.search(left_s[-1:]) and _CJK_RE.match(right_s):
            return True

        # Bracketed expressions should keep real multiline formatting. Join only
        # when the previous physical line actually hit the panel edge and the next
        # line has no indentation, which is the visual-wrap signature.
        if _has_unclosed_brackets(left_s):
            return True

    # When geometry is present and the line did not hit the code panel's right edge,
    # it is a real logical newline. Do not apply the geometry-free suffix heuristic:
    # that heuristic previously produced `breakif` and glued comments to `if`.
    if block_x1 is not None and left_x1 is not None:
        return False

    # Geometry is unavailable in audit mode. Retain only very conservative joins.
    left_token_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)$", left_s)
    right_token_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", right_s)
    if left_token_match and right_token_match:
        left_token = left_token_match.group(1)
        right_token = right_token_match.group(1)
        if left_token.endswith("_"):
            return True
        if len(left_token) == 1 and left_token + right_token in keyword.kwlist:
            return True
        if len(left_token) >= 4 and len(right_token) <= 2:
            if _COMPLETE_SIMPLE_STATEMENT_RE.match(left_s.strip()):
                return False
            if _PYTHON_STATEMENT_START_RE.match(right_s):
                return False
            if left_s.lstrip().startswith("#"):
                return False
            if not re.search(r"[,:;=+\-*/%<>]\s*$", left_s):
                return True
    return False


def normalize_python_for_analysis(text: str) -> str:
    """Create a conservative analysis-only variant.

    The original extracted text remains untouched. Standalone narrative labels that
    appear inside web-rendered code panels (for example `测试`) are converted to
    comments so AST validation measures code structure rather than presentation noise.
    """
    output: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        indent = line[: len(line) - len(line.lstrip(" \t"))]
        if _is_standalone_code_narrative(stripped):
            output.append(f"{indent}# {stripped}")
        else:
            output.append(line.rstrip())
    return "\n".join(output).strip("\n")


def _is_standalone_code_narrative(stripped: str) -> bool:
    if not stripped or stripped.startswith("#") or not _CJK_RE.search(stripped):
        return False
    if re.match(
        r"^(?:async\s+def|def|class|if|elif|else|for|while|try|except|finally|with|"
        r"return|yield|raise|break|continue|pass|import|from|print)\b",
        stripped,
    ):
        return False
    if re.match(r"^[A-Za-z_]\w*(?:\.\w+)*\s*=", stripped):
        return False
    if stripped.startswith(("'", '"')):
        return False
    # The source PDF uses Chinese presentation labels inside code panels, including
    # labels with parentheses/colons. Keep this high-confidence prefix list instead
    # of treating every CJK-containing expression as prose.
    if _NARRATIVE_PREFIX_RE.match(stripped):
        return True
    if len(stripped) <= 32 and not re.search(r"[=\[\]{}]", stripped):
        # Bare Chinese labels such as `测试` or `对用户查询进行分类` are valid Python
        # identifiers, so AST alone cannot detect them as presentation noise.
        return not re.match(r"^[A-Za-z_]\w*(?:\.\w+)*\s*\(", stripped)
    return False


def _parse_python(source: str) -> tuple[bool, str | None, int]:
    try:
        tree = ast.parse(source)
        compile(tree, "<pdf-code>", "exec")
        top_defs = sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in tree.body
        )
        return True, None, top_defs
    except (SyntaxError, IndentationError) as exc:
        return False, f"{exc.__class__.__name__}: {exc.msg} (line {exc.lineno})", 0


def analyze_python_code(text: str) -> PythonCodeAnalysis:
    source = (text or "").strip()
    is_python = looks_like_python_code(source)
    if not is_python:
        return PythonCodeAnalysis(
            False, False, None, False, None, None, "unknown", 0, 0, 0, 0, 0
        )

    parseable, syntax_error, top_defs = _parse_python(source)
    narrative_count = sum(
        _is_standalone_code_narrative(line.strip())
        for line in source.splitlines()
    )
    normalized = normalize_python_for_analysis(source)
    normalized_parseable, normalized_error, normalized_top_defs = _parse_python(normalized)
    top_defs = max(top_defs, normalized_top_defs)

    indentation_issues = _count_indentation_issues(normalized)
    split_identifiers = _count_split_identifier_candidates(source)
    boundary_violations = _count_function_boundary_violations(normalized)
    for error in (syntax_error, normalized_error):
        if error and re.search(
            r"(?:outside function|outside loop|not properly in loop)",
            error,
            re.IGNORECASE,
        ):
            boundary_violations += 1

    if parseable and narrative_count == 0:
        code_type = "python_executable"
    elif normalized_parseable:
        code_type = "python_normalized"
    else:
        code_type = "python_pseudocode"

    return PythonCodeAnalysis(
        is_python=True,
        ast_parseable=parseable,
        syntax_error=syntax_error,
        normalized_ast_parseable=normalized_parseable,
        normalized_syntax_error=normalized_error,
        normalized_source=normalized if normalized != source else None,
        code_type=code_type,
        indentation_issue_count=indentation_issues,
        split_identifier_candidate_count=split_identifiers,
        function_boundary_violation_count=boundary_violations,
        top_level_definition_count=top_defs,
        narrative_line_count=narrative_count,
    )


def split_python_top_level_blocks(text: str) -> list[str]:
    source = (text or "").strip("\n")
    if not source:
        return []
    lines = source.splitlines()
    normalized = normalize_python_for_analysis(source)
    try:
        tree = ast.parse(normalized)
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
    match = re.match(r"^[ \t]*", line)
    prefix = match.group(0) if match else ""
    return len(prefix.expandtabs(4))


def _has_unclosed_quote(value: str) -> bool:
    single = len(re.findall(r"(?<!\\)'", value)) % 2
    double = len(re.findall(r'(?<!\\)"', value)) % 2
    return bool(single or double)


def _has_unclosed_brackets(value: str) -> bool:
    pairs = {"(": ")", "[": "]", "{": "}"}
    return any(value.count(left) > value.count(right) for left, right in pairs.items())
