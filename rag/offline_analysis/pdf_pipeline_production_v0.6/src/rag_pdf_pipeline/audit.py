from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from .code_analysis import analyze_python_code

from .text_utils import (
    has_broken_cjk_line_candidate,
    is_suspicious_mid_sentence_start,
    visible_char_count,
)

_MULTI_LEVEL_NUMBER_RE = re.compile(r"^\d+(?:\.\d+){1,5}\s+")
_SINGLE_NUMBER_RE = re.compile(r"^\d+[、.]\s+")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path} 第 {line_number} 行必须是 JSON object")
        records.append(record)
    return records


def audit_existing_chunks(
    children: Sequence[dict[str, Any]],
    parents: Sequence[dict[str, Any]],
    atomic_units: Sequence[dict[str, Any]] | None = None,
    min_tokens: int = 180,
    target_tokens: int = 500,
    max_tokens: int = 800,
) -> dict[str, Any]:
    indexable_children = [
        item
        for item in children
        if bool((item.get("metadata") or {}).get("index_enabled", True))
    ]
    evaluated_children = indexable_children or list(children)
    child_tokens = [int(item.get("token_estimate", 0) or 0) for item in evaluated_children] or [0]
    child_texts = [str(item.get("text", "")) for item in evaluated_children]

    hashes = Counter(text.strip() for text in child_texts if text.strip())
    duplicate_count = sum(count - 1 for count in hashes.values() if count > 1)

    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for child in evaluated_children:
        by_parent[str(child.get("parent_id", ""))].append(child)
    evaluated_parents = [
        parent for parent in parents if by_parent.get(str(parent.get("parent_id", "")))
    ] or list(parents)
    parent_child_counts = [
        len(by_parent.get(str(parent.get("parent_id", "")), []))
        for parent in evaluated_parents
    ] or [0]

    heading_depths = [len(item.get("heading_path") or []) for item in evaluated_children] or [0]
    suspicious_parents = [
        item for item in evaluated_parents if _is_suspicious_heading(str(item.get("title", "")))
    ]
    broken_lines = [text for text in child_texts if has_broken_cjk_line_candidate(text)]
    multi_level_titles = sum(
        bool(_MULTI_LEVEL_NUMBER_RE.match(str(item.get("title", "")).strip()))
        for item in evaluated_parents
    )

    code_analyses = []
    code_unit_ids: set[str] = set()
    if atomic_units is not None:
        for item in atomic_units:
            if str(item.get("kind", "")) != "code":
                continue
            code_unit_ids.add(str(item.get("unit_id", "")))
            analysis = analyze_python_code(str(item.get("text", "")))
            if analysis.is_python:
                code_analyses.append(analysis)
    code_count = len(code_analyses)
    code_child_ids = {
        str(child.get("child_id", ""))
        for child in evaluated_children
        if any(str(unit_id) in code_unit_ids for unit_id in (child.get("unit_ids") or []))
    }
    mid_sentence = [
        str(child.get("text", ""))
        for child in evaluated_children
        if str(child.get("child_id", "")) not in code_child_ids
        and is_suspicious_mid_sentence_start(str(child.get("text", "")))
    ]

    result = {
        "parent_count": len(parents),
        "child_count": len(children),
        "indexable_parent_count": len(evaluated_parents),
        "indexable_child_count": len(evaluated_children),
        "duplicate_child_ratio": round(duplicate_count / max(len(evaluated_children), 1), 6),
        "short_child_ratio_lt_100": round(
            sum(value < 100 for value in child_tokens) / max(len(evaluated_children), 1), 6
        ),
        "under_min_tokens_ratio": round(
            sum(value < min_tokens for value in child_tokens) / max(len(evaluated_children), 1), 6
        ),
        "over_target_tokens_ratio": round(
            sum(value > target_tokens for value in child_tokens) / max(len(evaluated_children), 1), 6
        ),
        "over_max_tokens_count": sum(value > max_tokens for value in child_tokens),
        "single_child_parent_ratio": round(
            sum(value == 1 for value in parent_child_counts) / max(len(evaluated_parents), 1), 6
        ),
        "flat_heading_path_ratio": round(
            sum(depth <= 1 for depth in heading_depths) / max(len(evaluated_children), 1), 6
        ),
        "multi_level_numbered_parent_count": multi_level_titles,
        "suspicious_heading_ratio": round(
            len(suspicious_parents) / max(len(evaluated_parents), 1), 6
        ),
        "mid_sentence_start_ratio": round(
            len(mid_sentence) / max(len(evaluated_children), 1), 6
        ),
        "broken_line_candidate_ratio": round(
            len(broken_lines) / max(len(evaluated_children), 1), 6
        ),
        "child_token_min": min(child_tokens),
        "child_token_p10": _percentile(child_tokens, 0.10),
        "child_token_p25": _percentile(child_tokens, 0.25),
        "child_token_median": float(statistics.median(child_tokens)),
        "child_token_p75": _percentile(child_tokens, 0.75),
        "child_token_p90": _percentile(child_tokens, 0.90),
        "child_token_max": max(child_tokens),
        "children_per_parent_median": float(statistics.median(parent_child_counts)),
        "python_code_unit_count": code_count,
        "python_ast_parseable_ratio": (
            round(sum(item.ast_parseable for item in code_analyses) / code_count, 6)
            if code_count
            else None
        ),
        "code_indentation_issue_ratio": (
            round(sum(item.indentation_issue_count > 0 for item in code_analyses) / code_count, 6)
            if code_count
            else None
        ),
        "split_identifier_candidate_ratio": (
            round(
                sum(item.split_identifier_candidate_count > 0 for item in code_analyses)
                / code_count,
                6,
            )
            if code_count
            else None
        ),
        "function_boundary_violation_ratio": (
            round(
                sum(item.function_boundary_violation_count > 0 for item in code_analyses)
                / code_count,
                6,
            )
            if code_count
            else None
        ),
        "examples": {
            "short_children": [
                {
                    "child_id": item.get("child_id"),
                    "tokens": item.get("token_estimate"),
                    "heading_path": item.get("heading_path"),
                    "text": str(item.get("text", ""))[:100],
                }
                for item in sorted(evaluated_children, key=lambda row: int(row.get("token_estimate", 0) or 0))[:10]
            ],
            "suspicious_headings": [str(item.get("title", "")) for item in suspicious_parents[:10]],
            "mid_sentence_starts": [text[:120] for text in mid_sentence[:10]],
            "broken_line_candidates": [text[:160] for text in broken_lines[:10]],
            "python_syntax_errors": [
                item.syntax_error for item in code_analyses if item.syntax_error
            ][:10],
        },
    }
    result["verdict"] = _verdict(result)
    return result


def _verdict(result: dict[str, Any]) -> str:
    severe = (
        result["over_max_tokens_count"] > 0
        or result["short_child_ratio_lt_100"] > 0.15
        or result["single_child_parent_ratio"] > 0.7
        or result["mid_sentence_start_ratio"] > 0.08
        or result["broken_line_candidate_ratio"] > 0.25
    )
    code_warning = (
        result.get("python_ast_parseable_ratio") is not None
        and (
            result["python_ast_parseable_ratio"] < 0.65
            or result["code_indentation_issue_ratio"] > 0.20
            or result["split_identifier_candidate_ratio"] > 0.10
            or result["function_boundary_violation_ratio"] > 0.10
        )
    )
    warning = (
        result["short_child_ratio_lt_100"] > 0.10
        or result["under_min_tokens_ratio"] > 0.25
        or result["single_child_parent_ratio"] > 0.60
        or result["suspicious_heading_ratio"] > 0.12
        or result["mid_sentence_start_ratio"] > 0.03
        or result["broken_line_candidate_ratio"] > 0.10
        or (
            result["multi_level_numbered_parent_count"] >= 3
            and result["flat_heading_path_ratio"] > 0.95
        )
        or code_warning
    )
    if severe:
        return "not_ready"
    if warning:
        return "review"
    return "ready"


def _percentile(values: Sequence[int], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = int(position // 1)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 3)


def _is_suspicious_heading(title: str) -> bool:
    text = title.strip()
    compact = visible_char_count(text)
    if not text:
        return True
    if re.search(r"[。！？!?；;]$", text):
        return True
    if _SINGLE_NUMBER_RE.match(text):
        if compact > 38:
            return True
        if (":" in text or "：" in text) and compact > 28:
            return True
    if text.count("，") + text.count(",") >= 2:
        return True
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="审计现有 parents.jsonl / children.jsonl")
    parser.add_argument("--children", required=True)
    parser.add_argument("--parents", required=True)
    parser.add_argument(
        "--atomic-units",
        help="可选：atomic_units.jsonl；提供后增加 Python AST/缩进/标识符质量审计",
    )
    parser.add_argument("--min-tokens", type=int, default=180)
    parser.add_argument("--target-tokens", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("-o", "--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit_existing_chunks(
        children=load_jsonl(args.children),
        parents=load_jsonl(args.parents),
        atomic_units=load_jsonl(args.atomic_units) if args.atomic_units else None,
        min_tokens=args.min_tokens,
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
