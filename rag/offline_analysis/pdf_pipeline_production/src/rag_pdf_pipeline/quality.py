from __future__ import annotations

import hashlib
import statistics
from collections import Counter
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

        if duplicate_ratio > self.quality.max_duplicate_ratio:
            issues.append(
                QualityIssue(
                    "warning",
                    "HIGH_DUPLICATE_RATIO",
                    "Child Chunk 重复率过高",
                    {
                        "duplicate_ratio": duplicate_ratio,
                        "threshold": self.quality.max_duplicate_ratio,
                    },
                )
            )

        ocr_page_count = sum(page.used_ocr for page in document.pages)
        ocr_ratio = ocr_page_count / max(document.page_count, 1)
        if ocr_ratio >= self.quality.max_ocr_page_ratio_warning:
            issues.append(
                QualityIssue(
                    "warning",
                    "HIGH_OCR_RATIO",
                    "大部分页面使用 OCR，建议抽样检查识别质量",
                    {"ocr_page_ratio": ocr_ratio},
                )
            )

        token_values = [child.token_estimate for child in children] or [0]
        has_error = any(issue.severity == "error" for issue in issues)

        report = QualityReport(
            passed=not has_error,
            page_count=document.page_count,
            parent_count=len(parents),
            child_count=len(children),
            atomic_unit_count=len(units),
            ocr_page_count=ocr_page_count,
            duplicate_child_ratio=round(duplicate_ratio, 6),
            child_token_min=min(token_values),
            child_token_median=float(statistics.median(token_values)),
            child_token_max=max(token_values),
            issues=issues,
        )

        if self.quality.strict and has_error:
            messages = "; ".join(issue.message for issue in issues if issue.severity == "error")
            raise RuntimeError(f"质量校验失败：{messages}")

        return report
