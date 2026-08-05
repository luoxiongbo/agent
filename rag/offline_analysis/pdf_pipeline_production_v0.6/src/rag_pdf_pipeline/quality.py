from __future__ import annotations

import hashlib
import re
import statistics
from collections import Counter, defaultdict
from typing import Sequence

from .config import ChunkConfig, QualityConfig
from .code_analysis import analyze_python_code
from .models import (
    AtomicUnit,
    ChildChunk,
    ParentChunk,
    ParsedDocument,
    QualityIssue,
    QualityReport,
)
from .text_utils import (
    has_broken_cjk_line_candidate,
    has_glued_list_paragraph,
    is_suspicious_mid_sentence_start,
    looks_malformed_code,
    visible_char_count,
)

_MULTI_LEVEL_NUMBER_RE = re.compile(r"^\d+(?:\.\d+){1,5}\s+")
_SINGLE_NUMBER_RE = re.compile(r"^\d+[、.]\s+")


class QualityValidator:
    def __init__(self, quality: QualityConfig, chunk: ChunkConfig) -> None:
        self.quality = quality
        self.chunk = chunk
        self.quality.validate()
        self.chunk.validate()

    def validate(
        self,
        document: ParsedDocument,
        units: Sequence[AtomicUnit],
        parents: Sequence[ParentChunk],
        children: Sequence[ChildChunk],
    ) -> QualityReport:
        issues: list[QualityIssue] = []

        if document.page_count == 0:
            issues.append(QualityIssue("error", "EMPTY_DOCUMENT", "PDF 没有页面"))
        if not units:
            issues.append(QualityIssue("error", "NO_ATOMIC_UNITS", "没有生成原子单元"))
        if not children:
            issues.append(QualityIssue("error", "NO_CHILD_CHUNKS", "没有生成 Child Chunk"))

        parent_ids = {parent.parent_id for parent in parents}
        child_ids = {child.child_id for child in children}

        for child in children:
            if child.parent_id not in parent_ids:
                issues.append(
                    QualityIssue(
                        "error",
                        "ORPHAN_CHILD",
                        "Child Chunk 找不到 Parent",
                        {"child_id": child.child_id, "parent_id": child.parent_id},
                    )
                )
            if child.token_estimate > self.chunk.max_tokens:
                issues.append(
                    QualityIssue(
                        "error",
                        "CHILD_TOO_LARGE",
                        "Child Chunk 超过硬上限",
                        {
                            "child_id": child.child_id,
                            "tokens": child.token_estimate,
                            "max_tokens": self.chunk.max_tokens,
                        },
                    )
                )
            if not child.text.strip():
                issues.append(
                    QualityIssue(
                        "error",
                        "EMPTY_CHILD",
                        "Child Chunk 正文为空",
                        {"child_id": child.child_id},
                    )
                )
            if child.previous_child_id and child.previous_child_id not in child_ids:
                issues.append(
                    QualityIssue(
                        "error",
                        "BROKEN_PREVIOUS_LINK",
                        "previous_child_id 不存在",
                        {"child_id": child.child_id},
                    )
                )
            if child.next_child_id and child.next_child_id not in child_ids:
                issues.append(
                    QualityIssue(
                        "error",
                        "BROKEN_NEXT_LINK",
                        "next_child_id 不存在",
                        {"child_id": child.child_id},
                    )
                )

        non_indexable_children = [
            child for child in children if not bool(child.metadata.get("index_enabled", True))
        ]
        indexable_children = [
            child for child in children if bool(child.metadata.get("index_enabled", True))
        ]
        evaluated_children = indexable_children or list(children)
        non_indexable_ratio = len(non_indexable_children) / max(len(children), 1)

        hashes = [
            hashlib.sha256(child.text.strip().encode("utf-8")).hexdigest()
            for child in evaluated_children
            if child.text.strip()
        ]
        counts = Counter(hashes)
        duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
        duplicate_ratio = duplicate_count / max(len(evaluated_children), 1)

        token_values = [child.token_estimate for child in evaluated_children] or [0]
        short_ratio = sum(value < 100 for value in token_values) / max(
            len(evaluated_children), 1
        )
        under_min_ratio = sum(value < self.chunk.min_tokens for value in token_values) / max(
            len(evaluated_children), 1
        )
        over_target_ratio = sum(value > self.chunk.target_tokens for value in token_values) / max(
            len(evaluated_children), 1
        )

        children_by_parent: dict[str, list[ChildChunk]] = defaultdict(list)
        for child in indexable_children:
            children_by_parent[child.parent_id].append(child)
        indexable_parents = [
            parent for parent in parents if children_by_parent.get(parent.parent_id)
        ]
        evaluated_parents = indexable_parents or list(parents)
        child_counts = [
            len(children_by_parent.get(parent.parent_id, [])) for parent in evaluated_parents
        ] or [0]
        single_child_parent_ratio = sum(value == 1 for value in child_counts) / max(
            len(evaluated_parents), 1
        )

        heading_depths = [len(child.heading_path) for child in evaluated_children] or [0]
        flat_heading_path_ratio = sum(depth <= 1 for depth in heading_depths) / max(
            len(evaluated_children), 1
        )
        hierarchical_title_count = sum(
            bool(_MULTI_LEVEL_NUMBER_RE.match(parent.title.strip()))
            for parent in evaluated_parents
        )

        suspicious_parents = [
            parent for parent in evaluated_parents if _is_suspicious_heading(parent.title)
        ]
        suspicious_heading_ratio = len(suspicious_parents) / max(
            len(evaluated_parents), 1
        )

        code_unit_ids = {unit.unit_id for unit in units if unit.kind == "code"}
        code_children = [
            child
            for child in evaluated_children
            if any(unit_id in code_unit_ids for unit_id in child.unit_ids)
        ]
        code_child_ids = {child.child_id for child in code_children}

        mid_sentence_children = [
            child
            for child in evaluated_children
            if child.child_id not in code_child_ids
            and is_suspicious_mid_sentence_start(child.text)
        ]
        non_code_count = max(len(evaluated_children) - len(code_children), 1)
        mid_sentence_start_ratio = len(mid_sentence_children) / non_code_count

        broken_line_children = [
            child for child in evaluated_children if has_broken_cjk_line_candidate(child.text)
        ]
        broken_line_ratio = len(broken_line_children) / max(len(evaluated_children), 1)

        malformed_code_children = [
            child for child in code_children if looks_malformed_code(child.text)
        ]
        malformed_code_ratio = len(malformed_code_children) / max(len(code_children), 1)

        python_code_pairs = []
        for unit in units:
            if unit.kind != "code":
                continue
            analysis = analyze_python_code(unit.text)
            if analysis.is_python:
                python_code_pairs.append((unit, analysis))
        python_code_units = [unit for unit, _ in python_code_pairs]
        python_code_unit_count = len(python_code_units)
        ast_parseable_count = sum(analysis.ast_parseable for _, analysis in python_code_pairs)
        python_ast_parseable_ratio = ast_parseable_count / max(python_code_unit_count, 1)
        indentation_issue_units = [
            unit
            for unit, analysis in python_code_pairs
            if analysis.indentation_issue_count > 0
        ]
        split_identifier_units = [
            unit
            for unit, analysis in python_code_pairs
            if analysis.split_identifier_candidate_count > 0
        ]
        boundary_violation_units = [
            unit
            for unit, analysis in python_code_pairs
            if analysis.function_boundary_violation_count > 0
        ]
        code_indentation_issue_ratio = len(indentation_issue_units) / max(python_code_unit_count, 1)
        split_identifier_candidate_ratio = len(split_identifier_units) / max(
            python_code_unit_count, 1
        )
        function_boundary_violation_ratio = len(boundary_violation_units) / max(
            python_code_unit_count, 1
        )
        function_level_code_piece_count = sum(
            unit.metadata.get("split_strategy") == "code_top_level" for unit in units
        )

        glued_children = [
            child for child in evaluated_children if has_glued_list_paragraph(child.text)
        ]
        glued_ratio = len(glued_children) / max(len(evaluated_children), 1)

        ocr_page_count = sum(page.used_ocr for page in document.pages)
        ocr_ratio = ocr_page_count / max(document.page_count, 1)
        complex_layout_page_count = sum(
            page.layout_kind == "complex" and page.page_role != "toc"
            for page in document.pages
        )
        rejected_table_count = sum(page.rejected_table_count for page in document.pages)
        removed_noise_count = sum(page.removed_noise_block_count for page in document.pages)
        removed_promotional_count = sum(
            page.removed_promotional_block_count for page in document.pages
        )
        reconstructed_code_count = sum(
            page.reconstructed_code_block_count for page in document.pages
        )
        toc_page_count = sum(page.page_role == "toc" for page in document.pages)

        units_per_child = [len(child.unit_ids) for child in children] or [0]

        if len(children) >= 10:
            self._add_ratio_warning(
                issues,
                duplicate_ratio,
                self.quality.max_duplicate_ratio,
                "HIGH_DUPLICATE_RATIO",
                "Child Chunk 完全重复率过高",
            )
        if ocr_ratio >= self.quality.max_ocr_page_ratio_warning:
            issues.append(
                QualityIssue(
                    "warning",
                    "HIGH_OCR_RATIO",
                    "大部分页面使用 OCR，建议抽样检查识别质量",
                    {"value": round(ocr_ratio, 6), "threshold": self.quality.max_ocr_page_ratio_warning},
                )
            )
        if len(children) >= 10 and self.chunk.target_tokens >= 200:
            self._add_ratio_warning(
                issues,
                short_ratio,
                self.quality.max_short_child_ratio_lt_100,
                "HIGH_SHORT_CHILD_RATIO",
                "小于 100 token 的 Child 比例偏高，可能存在碎片化",
            )
        if len(children) >= 10:
            self._add_ratio_warning(
                issues,
                under_min_ratio,
                self.quality.max_under_min_tokens_ratio,
                "HIGH_UNDER_MIN_TOKEN_RATIO",
                "低于 min_tokens 的 Child 比例偏高",
            )
        if len(parents) >= 8:
            self._add_ratio_warning(
                issues,
                single_child_parent_ratio,
                self.quality.max_single_child_parent_ratio,
                "HIGH_SINGLE_CHILD_PARENT_RATIO",
                "大量 Parent 只有一个 Child，Parent 层级可能切得过细",
            )
        if (
            hierarchical_title_count >= 3
            and flat_heading_path_ratio > self.quality.max_flat_heading_path_ratio
        ):
            issues.append(
                QualityIssue(
                    "warning",
                    "FLAT_HEADING_HIERARCHY",
                    "文档存在多级编号标题，但 heading_path 基本为单层",
                    {
                        "value": round(flat_heading_path_ratio, 6),
                        "threshold": self.quality.max_flat_heading_path_ratio,
                        "hierarchical_title_count": hierarchical_title_count,
                    },
                )
            )
        if len(parents) >= 5:
            self._add_ratio_warning(
                issues,
                suspicious_heading_ratio,
                self.quality.max_suspicious_heading_ratio,
                "SUSPICIOUS_HEADING_RATIO",
                "疑似列表项或完整句子被识别为标题的比例偏高",
                examples=[parent.title for parent in suspicious_parents[:5]],
            )
        self._add_ratio_warning(
            issues,
            mid_sentence_start_ratio,
            self.quality.max_mid_sentence_start_ratio,
            "MID_SENTENCE_CHUNK_STARTS",
            "部分 Child 疑似从半句话开始",
            examples=[child.text[:80] for child in mid_sentence_children[:5]],
        )
        self._add_ratio_warning(
            issues,
            broken_line_ratio,
            self.quality.max_broken_line_candidate_ratio,
            "BROKEN_CJK_LINE_RATIO",
            "大量 Child 存在中文字符被空行拆开的候选模式",
            examples=[child.text[:120] for child in broken_line_children[:5]],
        )
        if code_children:
            self._add_ratio_warning(
                issues,
                malformed_code_ratio,
                self.quality.max_malformed_code_chunk_ratio,
                "MALFORMED_CODE_CHUNK_RATIO",
                "部分代码 Chunk 仍疑似丢失换行或缩进",
                examples=[child.text[:160] for child in malformed_code_children[:5]],
            )
        if python_code_units:
            ast_failure_ratio = 1.0 - python_ast_parseable_ratio
            self._add_ratio_warning(
                issues,
                ast_failure_ratio,
                self.quality.max_python_ast_failure_ratio,
                "PYTHON_AST_FAILURE_RATIO",
                "部分 Python 代码原子单元无法通过 AST 语法解析",
                examples=[
                    str(analysis.syntax_error or unit.text[:120])
                    for unit, analysis in python_code_pairs
                    if not analysis.ast_parseable
                ][:5],
            )
            self._add_ratio_warning(
                issues,
                code_indentation_issue_ratio,
                self.quality.max_code_indentation_issue_ratio,
                "CODE_INDENTATION_ISSUE_RATIO",
                "部分 Python 代码块存在控制语句后无缩进等问题",
                examples=[unit.text[:180] for unit in indentation_issue_units[:5]],
            )
            self._add_ratio_warning(
                issues,
                split_identifier_candidate_ratio,
                self.quality.max_split_identifier_candidate_ratio,
                "SPLIT_IDENTIFIER_CANDIDATE_RATIO",
                "部分 Python 标识符疑似被 PDF 换行拆开",
                examples=[unit.text[:180] for unit in split_identifier_units[:5]],
            )
            self._add_ratio_warning(
                issues,
                function_boundary_violation_ratio,
                self.quality.max_function_boundary_violation_ratio,
                "FUNCTION_BOUNDARY_VIOLATION_RATIO",
                "部分代码片段疑似从函数内部开始或把函数体切到顶层",
                examples=[unit.text[:180] for unit in boundary_violation_units[:5]],
            )
        self._add_ratio_warning(
            issues,
            glued_ratio,
            self.quality.max_glued_list_paragraph_ratio,
            "GLUED_LIST_PARAGRAPH_RATIO",
            "部分列表末项与后续说明段落疑似粘连",
            examples=[child.text[:180] for child in glued_children[:5]],
        )
        if toc_page_count and non_indexable_children:
            issues.append(
                QualityIssue(
                    "info",
                    "NON_INDEXABLE_NAVIGATION_CHUNKS",
                    "检测到目录/导航 Chunk，已从 indexable_children.jsonl 排除",
                    {
                        "toc_page_count": toc_page_count,
                        "non_indexable_child_count": len(non_indexable_children),
                    },
                )
            )

        if complex_layout_page_count:
            issues.append(
                QualityIssue(
                    "warning",
                    "COMPLEX_LAYOUT_PAGES",
                    "检测到复杂图解/幻灯片式页面，建议检查 document.md",
                    {"page_count": complex_layout_page_count},
                )
            )

        has_error = any(issue.severity == "error" for issue in issues)
        has_warning = any(issue.severity == "warning" for issue in issues)
        readiness = "blocked" if has_error else "review" if has_warning else "ready"

        report = QualityReport(
            passed=not has_error,
            retrieval_readiness=readiness,
            page_count=document.page_count,
            parent_count=len(parents),
            child_count=len(children),
            atomic_unit_count=len(units),
            ocr_page_count=ocr_page_count,
            complex_layout_page_count=complex_layout_page_count,
            rejected_table_count=rejected_table_count,
            removed_noise_block_count=removed_noise_count,
            removed_promotional_block_count=removed_promotional_count,
            reconstructed_code_block_count=reconstructed_code_count,
            python_code_unit_count=python_code_unit_count,
            function_level_code_piece_count=function_level_code_piece_count,
            toc_page_count=toc_page_count,
            non_indexable_child_count=len(non_indexable_children),
            duplicate_child_ratio=round(duplicate_ratio, 6),
            non_indexable_child_ratio=round(non_indexable_ratio, 6),
            malformed_code_chunk_ratio=round(malformed_code_ratio, 6),
            python_ast_parseable_ratio=round(python_ast_parseable_ratio, 6),
            code_indentation_issue_ratio=round(code_indentation_issue_ratio, 6),
            split_identifier_candidate_ratio=round(split_identifier_candidate_ratio, 6),
            function_boundary_violation_ratio=round(function_boundary_violation_ratio, 6),
            glued_list_paragraph_ratio=round(glued_ratio, 6),
            short_child_ratio_lt_100=round(short_ratio, 6),
            under_min_tokens_ratio=round(under_min_ratio, 6),
            over_target_tokens_ratio=round(over_target_ratio, 6),
            single_child_parent_ratio=round(single_child_parent_ratio, 6),
            flat_heading_path_ratio=round(flat_heading_path_ratio, 6),
            suspicious_heading_ratio=round(suspicious_heading_ratio, 6),
            mid_sentence_start_ratio=round(mid_sentence_start_ratio, 6),
            broken_line_candidate_ratio=round(broken_line_ratio, 6),
            child_token_min=min(token_values),
            child_token_p10=_percentile(token_values, 0.10),
            child_token_p25=_percentile(token_values, 0.25),
            child_token_median=float(statistics.median(token_values)),
            child_token_p75=_percentile(token_values, 0.75),
            child_token_p90=_percentile(token_values, 0.90),
            child_token_max=max(token_values),
            children_per_parent_median=float(statistics.median(child_counts)),
            atomic_units_per_child_median=float(statistics.median(units_per_child)),
            issues=issues,
        )

        if self.quality.strict and has_error:
            messages = "; ".join(issue.message for issue in issues if issue.severity == "error")
            raise RuntimeError(f"质量校验失败：{messages}")

        return report

    @staticmethod
    def _add_ratio_warning(
        issues: list[QualityIssue],
        value: float,
        threshold: float,
        code: str,
        message: str,
        examples: list[str] | None = None,
    ) -> None:
        if value <= threshold:
            return
        context: dict[str, object] = {
            "value": round(value, 6),
            "threshold": threshold,
        }
        if examples:
            context["examples"] = examples
        issues.append(QualityIssue("warning", code, message, context))


def _percentile(values: Sequence[int], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = math_floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 3)


def math_floor(value: float) -> int:
    return int(value // 1)


def _is_suspicious_heading(title: str) -> bool:
    text = title.strip()
    compact_length = visible_char_count(text)
    if not text:
        return True
    if re.search(r"[。！？!?；;]$", text):
        return True
    if _SINGLE_NUMBER_RE.match(text):
        if compact_length > 38:
            return True
        if (":" in text or "：" in text) and compact_length > 28:
            return True
    if text.count("，") + text.count(",") >= 2:
        return True
    return False
