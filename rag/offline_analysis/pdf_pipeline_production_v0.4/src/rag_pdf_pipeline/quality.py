from __future__ import annotations

import hashlib
import re
import statistics
from collections import Counter, defaultdict
from typing import Sequence

from .config import ChunkConfig, QualityConfig
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
    is_suspicious_mid_sentence_start,
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

        hashes = [
            hashlib.sha256(child.text.strip().encode("utf-8")).hexdigest()
            for child in children
            if child.text.strip()
        ]
        counts = Counter(hashes)
        duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
        duplicate_ratio = duplicate_count / max(len(children), 1)

        token_values = [child.token_estimate for child in children] or [0]
        short_ratio = sum(value < 100 for value in token_values) / max(len(children), 1)
        under_min_ratio = sum(value < self.chunk.min_tokens for value in token_values) / max(
            len(children), 1
        )
        over_target_ratio = sum(value > self.chunk.target_tokens for value in token_values) / max(
            len(children), 1
        )

        children_by_parent: dict[str, list[ChildChunk]] = defaultdict(list)
        for child in children:
            children_by_parent[child.parent_id].append(child)
        child_counts = [len(children_by_parent[parent.parent_id]) for parent in parents] or [0]
        single_child_parent_ratio = sum(value == 1 for value in child_counts) / max(len(parents), 1)

        heading_depths = [len(child.heading_path) for child in children] or [0]
        flat_heading_path_ratio = sum(depth <= 1 for depth in heading_depths) / max(
            len(children), 1
        )
        hierarchical_title_count = sum(
            bool(_MULTI_LEVEL_NUMBER_RE.match(parent.title.strip())) for parent in parents
        )

        suspicious_parents = [parent for parent in parents if _is_suspicious_heading(parent.title)]
        suspicious_heading_ratio = len(suspicious_parents) / max(len(parents), 1)

        mid_sentence_children = [
            child for child in children if is_suspicious_mid_sentence_start(child.text)
        ]
        mid_sentence_start_ratio = len(mid_sentence_children) / max(len(children), 1)

        broken_line_children = [
            child for child in children if has_broken_cjk_line_candidate(child.text)
        ]
        broken_line_ratio = len(broken_line_children) / max(len(children), 1)

        ocr_page_count = sum(page.used_ocr for page in document.pages)
        ocr_ratio = ocr_page_count / max(document.page_count, 1)
        complex_layout_page_count = sum(page.layout_kind == "complex" for page in document.pages)
        rejected_table_count = sum(page.rejected_table_count for page in document.pages)
        removed_noise_count = sum(page.removed_noise_block_count for page in document.pages)

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
            duplicate_child_ratio=round(duplicate_ratio, 6),
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
