from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Sequence

from .config import ParserConfig
from .models import AtomicUnit, ParsedDocument, Section, TableData, TextBlock
from .text_utils import (
    clean_extracted_text,
    ends_with_sentence_terminal,
    estimate_tokens,
    is_list_line,
    is_noise_line,
    looks_like_sentence_or_list_item,
    normalize_inline_text,
    smart_join_text,
    stable_id,
    starts_like_continuation,
    visible_char_count,
)

_Q_RE = re.compile(r"^(?:Q(?:uestion)?|问(?:题)?)[：:]\s*", re.IGNORECASE)
_A_RE = re.compile(r"^(?:A(?:nswer)?|答(?:案)?)[：:]\s*", re.IGNORECASE)
_CALLOUT_RE = re.compile(
    r"^(?:注意|警告|提示|说明|重要|备注|Note|Warning|Tip|Important)[：:]\s*",
    re.IGNORECASE,
)


class AtomicUnitBuilder:
    """把版面元素转换成尽量不可拆分的原子信息单元。"""

    def __init__(self, config: ParserConfig | None = None) -> None:
        self.config = config or ParserConfig()
        self.config.validate()

    def build(self, document: ParsedDocument) -> list[Section]:
        sections: list[Section] = []
        heading_stack: list[tuple[int, str]] = []
        current_title = "文档正文"
        current_units: list[AtomicUnit] = []
        current_start_page = 1

        def current_heading_path() -> list[str]:
            return [title for _, title in heading_stack]

        def flush_section(end_page: int) -> None:
            nonlocal current_units, current_start_page, current_title
            if not current_units:
                current_start_page = end_page
                return
            units = self._coalesce_units(current_units)
            if not units:
                current_units = []
                current_start_page = end_page
                return
            heading_path = current_heading_path()
            sections.append(
                Section(
                    section_id=stable_id(
                        "section",
                        document.document_id,
                        heading_path,
                        min(unit.page_start for unit in units),
                        max(unit.page_end for unit in units),
                        len(sections),
                    ),
                    document_id=document.document_id,
                    title=current_title,
                    heading_path=heading_path,
                    page_start=min(unit.page_start for unit in units),
                    page_end=max(unit.page_end for unit in units),
                    units=units,
                )
            )
            current_units = []
            current_start_page = end_page

        previous_page: PageData | None = None
        for page in document.pages:
            # 封面/流程图页的视觉标题不能成为后续正文的章节祖先。
            if (
                previous_page is not None
                and previous_page.page_number <= 2
                and previous_page.layout_kind in {"complex", "sparse"}
            ):
                flush_section(previous_page.page_number)
                heading_stack.clear()
                current_title = "文档正文"
                current_start_page = page.page_number

            if page.layout_kind == "complex" and self.config.complex_layout_strategy == "skip":
                previous_page = page
                continue

            ordered: list[tuple[float, float, str, Any]] = []
            ordered.extend((block.y0, block.bbox[0], "block", block) for block in page.blocks)
            ordered.extend((table.y0, table.bbox[0], "table", table) for table in page.tables)
            ordered.sort(key=lambda item: (item[0], item[1]))

            for _, _, kind, element in ordered:
                if kind == "block" and element.heading_level:
                    if self._accept_heading(element, page.layout_kind):
                        normalized_title = normalize_inline_text(element.text)
                        if heading_stack and _same_heading(
                            heading_stack[-1][1], normalized_title
                        ):
                            # 翻页重复出现的章节标题不应创建新 Parent。
                            continue

                        flush_section(page.page_number)
                        _update_heading_stack(
                            heading_stack,
                            element.heading_level,
                            normalized_title,
                        )
                        current_title = normalized_title
                        current_start_page = page.page_number
                        continue

                    # 结构层再做一次标题复核。即使版面层误判，也不能把一整句
                    # “1. 问题说明……”创建成 Parent；降级后作为普通内容继续处理。
                    element = replace(
                        element,
                        heading_level=None,
                        heading_source=f"demoted:{element.heading_source or 'unknown'}",
                    )

                heading_path = current_heading_path()
                if kind == "table":
                    current_units.append(self._table_unit(document, element, heading_path))
                else:
                    current_units.extend(
                        self._text_units(
                            document=document,
                            block=element,
                            heading_path=heading_path,
                            page_layout=page.layout_kind,
                            page_width=page.width,
                            page_height=page.height,
                        )
                    )

            previous_page = page

        flush_section(document.page_count)
        return self._merge_adjacent_same_heading_sections(sections)

    def _accept_heading(self, block: TextBlock, page_layout: str) -> bool:
        if not self.config.revalidate_headings_in_structure:
            return True
        text = normalize_inline_text(block.text)
        source = block.heading_source or ""
        length = visible_char_count(text)
        if not text or "\n" in text or is_noise_line(text):
            return False
        if length > self.config.heading_max_chars:
            return False
        if page_layout == "complex":
            return source == "complex_layout_dominant_title"
        if looks_like_sentence_or_list_item(text):
            return False
        if source == "single_numeric_styled":
            return length <= min(self.config.single_level_heading_max_chars, 32)
        if source == "numeric_hierarchy":
            return length <= 60
        if source in {"font_ratio", "bold_style", "bold_fallback"}:
            return length <= 48
        return True

    def _text_units(
        self,
        document: ParsedDocument,
        block: TextBlock,
        heading_path: Sequence[str],
        page_layout: str,
        page_width: float,
        page_height: float,
    ) -> list[AtomicUnit]:
        text = clean_extracted_text(block.text) if not block.monospace else block.text.strip()
        if not text or is_noise_line(text):
            return []

        kind, protected = self._classify(block, text, page_layout)
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
            "page_layout": page_layout,
            "page_width": page_width,
            "page_height": page_height,
            "heading_source": block.heading_source,
            "demoted_heading": bool(
                block.heading_source and block.heading_source.startswith("demoted:")
            ),
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
                "table_quality_score": table.quality_score,
                "table_quality_reasons": table.quality_reasons,
            },
        )

    def _classify(
        self,
        block: TextBlock,
        text: str,
        page_layout: str,
    ) -> tuple[str, bool]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        list_line_count = sum(1 for line in lines if is_list_line(line))

        if page_layout in {"complex", "sparse"} and visible_char_count(text) <= 80:
            return "layout", True
        if _Q_RE.search(text) and _A_RE.search(text):
            return "faq", True
        if _CALLOUT_RE.match(text):
            return "callout", True
        if block.monospace or _looks_like_code(text):
            return "code", True
        if (
            list_line_count >= 1
            or (
                len(lines) >= 2
                and lines[0].endswith(("：", ":"))
                and is_list_line(lines[1])
            )
        ):
            return "list", True
        return "paragraph", False

    def _coalesce_units(self, units: list[AtomicUnit]) -> list[AtomicUnit]:
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
                result.append(self._merge_units(current, next_unit, kind="faq", separator="\n"))
                index += 2
                continue

            if (
                next_unit
                and current.kind == "paragraph"
                and current.text.rstrip().endswith(("：", ":"))
                and next_unit.kind == "list"
                and current.page_end <= next_unit.page_start <= current.page_end + 1
            ):
                result.append(self._merge_units(current, next_unit, kind="list", separator="\n"))
                index += 2
                continue

            if next_unit and current.kind == "list" and next_unit.kind == "list":
                if current.heading_path == next_unit.heading_path:
                    result.append(self._merge_units(current, next_unit, kind="list", separator="\n"))
                    index += 2
                    continue

            if (
                next_unit
                and current.kind == "list"
                and next_unit.kind == "paragraph"
                and current.metadata.get("demoted_heading")
                and current.heading_path == next_unit.heading_path
                and not ends_with_sentence_terminal(current.text)
                and starts_like_continuation(next_unit.text)
            ):
                result.append(self._merge_units(current, next_unit, kind="list", separator=""))
                index += 2
                continue

            if next_unit and self._should_merge_paragraph_units(current, next_unit):
                result.append(self._merge_units(current, next_unit, kind="paragraph", separator=""))
                index += 2
                continue

            result.append(current)
            index += 1

        # 上面的相邻合并可能形成新的可合并关系，迭代到稳定。
        if len(result) < len(units):
            return self._coalesce_units(result)
        return result

    def _should_merge_paragraph_units(
        self,
        current: AtomicUnit,
        next_unit: AtomicUnit,
    ) -> bool:
        if current.kind != "paragraph" or next_unit.kind != "paragraph":
            return False
        if current.heading_path != next_unit.heading_path:
            return False
        if ends_with_sentence_terminal(current.text) or current.text.rstrip().endswith(("：", ":")):
            return False
        if is_list_line(next_unit.text):
            return False
        if not starts_like_continuation(next_unit.text):
            return False
        if next_unit.page_start < current.page_end or next_unit.page_start > current.page_end + 1:
            return False

        current_font = float(current.metadata.get("font_size") or 0.0)
        next_font = float(next_unit.metadata.get("font_size") or 0.0)
        if current_font and next_font:
            if abs(current_font - next_font) / max(current_font, next_font) > 0.16:
                return False

        current_bbox = current.metadata.get("bbox")
        next_bbox = next_unit.metadata.get("bbox")
        if not (
            isinstance(current_bbox, list)
            and len(current_bbox) == 4
            and isinstance(next_bbox, list)
            and len(next_bbox) == 4
        ):
            return visible_char_count(current.text) <= 160

        if current.page_end == next_unit.page_start:
            vertical_gap = float(next_bbox[1]) - float(current_bbox[3])
            x_offset = abs(float(current_bbox[0]) - float(next_bbox[0]))
            max_gap = max(current_font, next_font, 10.0) * 2.2
            return -3 <= vertical_gap <= max_gap and x_offset <= self.config.max_block_x_offset

        if not self.config.merge_cross_page_continuations:
            return False
        current_page_height = float(current.metadata.get("page_height") or 0.0)
        next_page_height = float(next_unit.metadata.get("page_height") or 0.0)
        if current_page_height <= 0 or next_page_height <= 0:
            return False
        current_near_bottom = float(current_bbox[3]) >= current_page_height * 0.72
        next_near_top = float(next_bbox[1]) <= next_page_height * 0.30
        x_offset = abs(float(current_bbox[0]) - float(next_bbox[0]))
        return current_near_bottom and next_near_top and x_offset <= self.config.max_block_x_offset

    def _merge_units(
        self,
        current: AtomicUnit,
        next_unit: AtomicUnit,
        kind: str,
        separator: str,
    ) -> AtomicUnit:
        if separator:
            merged_text = current.text.rstrip() + separator + next_unit.text.lstrip()
        else:
            merged_text = smart_join_text(current.text, next_unit.text)
        current_bbox = current.metadata.get("bbox")
        next_bbox = next_unit.metadata.get("bbox")
        merged_bbox = next_bbox
        if (
            current.page_end == next_unit.page_start
            and isinstance(current_bbox, list)
            and len(current_bbox) == 4
            and isinstance(next_bbox, list)
            and len(next_bbox) == 4
        ):
            merged_bbox = [
                min(float(current_bbox[0]), float(next_bbox[0])),
                min(float(current_bbox[1]), float(next_bbox[1])),
                max(float(current_bbox[2]), float(next_bbox[2])),
                max(float(current_bbox[3]), float(next_bbox[3])),
            ]
        return replace(
            current,
            unit_id=stable_id("unit", current.unit_id, next_unit.unit_id, merged_text),
            kind=kind,  # type: ignore[arg-type]
            text=merged_text,
            page_end=max(current.page_end, next_unit.page_end),
            protected=current.protected or next_unit.protected or kind in {"faq", "list"},
            group_id=stable_id("merged_group", current.unit_id, next_unit.unit_id),
            metadata={
                **current.metadata,
                "bbox": merged_bbox,
                "page_height": next_unit.metadata.get("page_height", current.metadata.get("page_height")),
                "page_width": next_unit.metadata.get("page_width", current.metadata.get("page_width")),
                "merged_unit_ids": _merged_ids(current) + _merged_ids(next_unit),
                "end_bbox": next_bbox,
                "end_page_height": next_unit.metadata.get("page_height"),
            },
        )

    def _merge_adjacent_same_heading_sections(
        self,
        sections: list[Section],
    ) -> list[Section]:
        if not sections:
            return []
        result: list[Section] = [sections[0]]
        for section in sections[1:]:
            previous = result[-1]
            if (
                previous.heading_path == section.heading_path
                and section.page_start <= previous.page_end + 1
            ):
                units = self._coalesce_units(previous.units + section.units)
                result[-1] = replace(
                    previous,
                    section_id=stable_id(
                        "section",
                        previous.section_id,
                        section.section_id,
                    ),
                    page_end=max(previous.page_end, section.page_end),
                    units=units,
                )
            else:
                result.append(section)
        return result


def _merged_ids(unit: AtomicUnit) -> list[str]:
    merged = unit.metadata.get("merged_unit_ids")
    if isinstance(merged, list):
        return [str(item) for item in merged]
    return [unit.unit_id]


def _same_heading(left: str, right: str) -> bool:
    canonical = lambda value: re.sub(r"\s+", "", normalize_inline_text(value)).lower()
    return canonical(left) == canonical(right)


def _update_heading_stack(
    stack: list[tuple[int, str]],
    level: int,
    title: str,
) -> None:
    level = max(1, min(level, 6))
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title.strip()))


def _looks_like_code(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False

    code_markers = 0
    for line in lines:
        stripped = line.strip()
        if re.search(
            r"[{}[\]();=<>]|^\s*(?:def|class|import|from|SELECT|INSERT|UPDATE|CREATE)\b",
            stripped,
            re.IGNORECASE,
        ):
            code_markers += 1
        if stripped.startswith(("```", "$ ", ">>> ")):
            code_markers += 1

    punctuation_ratio = sum(text.count(char) for char in "{}[]();=<>") / max(len(text), 1)
    return code_markers / len(lines) >= 0.45 or punctuation_ratio >= 0.08
