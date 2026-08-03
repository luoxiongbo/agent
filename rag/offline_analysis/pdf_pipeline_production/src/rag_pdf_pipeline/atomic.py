from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Sequence

from .models import AtomicUnit, ParsedDocument, Section, TableData, TextBlock
from .text_utils import (
    LIST_LINE_RE,
    estimate_tokens,
    is_list_line,
    normalize_multiline_text,
    stable_id,
)

_Q_RE = re.compile(r"^(?:Q(?:uestion)?|问(?:题)?)[：:]\s*", re.IGNORECASE)
_A_RE = re.compile(r"^(?:A(?:nswer)?|答(?:案)?)[：:]\s*", re.IGNORECASE)
_CALLOUT_RE = re.compile(
    r"^(?:注意|警告|提示|说明|重要|备注|Note|Warning|Tip|Important)[：:]\s*",
    re.IGNORECASE,
)


class AtomicUnitBuilder:
    """把版面元素转换成尽量不可拆分的原子信息单元。"""

    def build(self, document: ParsedDocument) -> list[Section]:
        sections: list[Section] = []
        current_heading: list[str] = []
        current_title = "文档正文"
        current_units: list[AtomicUnit] = []
        current_start_page = 1

        def flush_section(end_page: int) -> None:
            nonlocal current_units, current_start_page, current_title
            if not current_units:
                current_start_page = end_page
                return
            units = self._coalesce_units(current_units)
            sections.append(
                Section(
                    section_id=stable_id(
                        "section",
                        document.document_id,
                        current_heading,
                        current_start_page,
                        end_page,
                        len(sections),
                    ),
                    document_id=document.document_id,
                    title=current_title,
                    heading_path=list(current_heading),
                    page_start=min(unit.page_start for unit in units),
                    page_end=max(unit.page_end for unit in units),
                    units=units,
                )
            )
            current_units = []
            current_start_page = end_page

        for page in document.pages:
            ordered: list[tuple[float, str, Any]] = []
            ordered.extend((block.y0, "block", block) for block in page.blocks)
            ordered.extend((table.y0, "table", table) for table in page.tables)
            ordered.sort(key=lambda item: item[0])

            for _, kind, element in ordered:
                if kind == "block" and element.heading_level:
                    flush_section(page.page_number)
                    _update_heading_path(
                        current_heading,
                        element.heading_level,
                        element.text,
                    )
                    current_title = element.text
                    current_start_page = page.page_number
                    continue

                if kind == "table":
                    current_units.append(self._table_unit(document, element, current_heading))
                else:
                    current_units.extend(
                        self._text_units(document, element, current_heading)
                    )

        flush_section(document.page_count)
        return sections

    def _text_units(
        self,
        document: ParsedDocument,
        block: TextBlock,
        heading_path: Sequence[str],
    ) -> list[AtomicUnit]:
        text = normalize_multiline_text(block.text)
        if not text:
            return []

        kind, protected = self._classify(block, text)
        group_id = stable_id(
            "group",
            document.document_id,
            block.page_number,
            block.block_no,
            text,
        )

        metadata: dict[str, Any] = {
            "bbox": list(block.bbox),
            "font_size": block.avg_font_size,
            "bold": block.bold,
            "monospace": block.monospace,
            "source": block.source,
        }

        return [
            AtomicUnit(
                unit_id=stable_id(
                    "unit",
                    document.document_id,
                    block.page_number,
                    block.block_no,
                    text,
                ),
                document_id=document.document_id,
                kind=kind,
                text=text,
                page_start=block.page_number,
                page_end=block.page_number,
                heading_path=list(heading_path),
                protected=protected,
                group_id=group_id,
                metadata=metadata,
            )
        ]

    def _table_unit(
        self,
        document: ParsedDocument,
        table: TableData,
        heading_path: Sequence[str],
    ) -> AtomicUnit:
        group_id = stable_id(
            "table_group",
            document.document_id,
            table.page_number,
            table.table_index,
            table.markdown,
        )
        return AtomicUnit(
            unit_id=stable_id(
                "unit",
                document.document_id,
                table.page_number,
                "table",
                table.table_index,
                table.markdown,
            ),
            document_id=document.document_id,
            kind="table",
            text=table.markdown,
            page_start=table.page_number,
            page_end=table.page_number,
            heading_path=list(heading_path),
            protected=True,
            group_id=group_id,
            metadata={
                "bbox": list(table.bbox),
                "table_index": table.table_index,
                "rows": table.rows,
            },
        )

    def _classify(self, block: TextBlock, text: str) -> tuple[str, bool]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        list_line_count = sum(1 for line in lines if is_list_line(line))

        if _Q_RE.search(text) and _A_RE.search(text):
            return "faq", True
        if _CALLOUT_RE.match(text):
            return "callout", True
        if block.monospace or _looks_like_code(text):
            return "code", True
        if list_line_count >= 2 or (
            len(lines) >= 2
            and lines[0].endswith(("：", ":"))
            and is_list_line(lines[1])
        ):
            return "list", True
        return "paragraph", False

    def _coalesce_units(self, units: list[AtomicUnit]) -> list[AtomicUnit]:
        """
        合并跨文本块的问答，以及“引导语 + 列表”。
        """
        result: list[AtomicUnit] = []
        index = 0

        while index < len(units):
            current = units[index]
            next_unit = units[index + 1] if index + 1 < len(units) else None

            if (
                next_unit
                and current.kind == "paragraph"
                and _Q_RE.match(current.text)
                and _A_RE.match(next_unit.text)
            ):
                result.append(
                    replace(
                        current,
                        unit_id=stable_id("unit", current.unit_id, next_unit.unit_id),
                        kind="faq",
                        text=current.text + "\n" + next_unit.text,
                        page_end=next_unit.page_end,
                        protected=True,
                        group_id=stable_id("faq_group", current.unit_id, next_unit.unit_id),
                        metadata={
                            "merged_unit_ids": [current.unit_id, next_unit.unit_id]
                        },
                    )
                )
                index += 2
                continue

            if (
                next_unit
                and current.kind == "paragraph"
                and current.text.rstrip().endswith(("：", ":"))
                and next_unit.kind == "list"
                and current.page_end == next_unit.page_start
            ):
                result.append(
                    replace(
                        next_unit,
                        unit_id=stable_id("unit", current.unit_id, next_unit.unit_id),
                        text=current.text + "\n" + next_unit.text,
                        page_start=current.page_start,
                        metadata={
                            **next_unit.metadata,
                            "lead_in_unit_id": current.unit_id,
                        },
                    )
                )
                index += 2
                continue

            result.append(current)
            index += 1

        return result


def _update_heading_path(path: list[str], level: int, title: str) -> None:
    level = max(1, min(level, 6))
    while len(path) >= level:
        path.pop()
    while len(path) < level - 1:
        path.append("未命名章节")
    path.append(title.strip())


def _looks_like_code(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False

    code_markers = 0
    for line in lines:
        stripped = line.strip()
        if re.search(r"[{}[\]();=<>]|^\s*(?:def|class|import|from|SELECT|INSERT|UPDATE)\b", stripped):
            code_markers += 1
        if stripped.startswith(("```", "$ ", ">>> ")):
            code_markers += 1

    punctuation_ratio = sum(text.count(char) for char in "{}[]();=<>") / max(len(text), 1)
    return code_markers / len(lines) >= 0.45 or punctuation_ratio >= 0.08
