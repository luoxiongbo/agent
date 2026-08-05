from __future__ import annotations

import logging
import math
import re
import statistics
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

try:
    import pymupdf
except ImportError as exc:
    raise SystemExit("缺少 PyMuPDF，请先安装项目依赖") from exc

from .config import ParserConfig
from .models import PageData, ParsedDocument, TableData, TextBlock
from .text_utils import (
    clean_code_text,
    clean_extracted_text,
    ends_with_sentence_terminal,
    is_list_line,
    is_noise_line,
    is_paragraph_starter,
    is_promotional_contact_line,
    looks_like_sentence_or_list_item,
    looks_like_code_lines,
    merge_extracted_lines,
    normalize_code_line,
    normalize_inline_text,
    normalize_rows,
    reconstruct_code_from_lines,
    safe_json_value,
    sha256_file,
    smart_join_text,
    table_to_markdown,
    toc_entry_count,
    visible_char_count,
)

LOGGER = logging.getLogger(__name__)


class PDFParser:
    def __init__(self, config: ParserConfig) -> None:
        self.config = config
        self.config.validate()

    def parse(self, pdf_path: str | Path, password: str | None = None) -> ParsedDocument:
        path = Path(pdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF 不存在：{path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"只接受 PDF 文件：{path}")

        file_hash = sha256_file(path)
        document_id = f"pdf_{file_hash[:20]}"

        with pymupdf.open(path) as document:
            if document.needs_pass:
                if not password or not document.authenticate(password):
                    raise PermissionError("PDF 已加密，请提供正确密码")

            metadata = {
                key: safe_json_value(value)
                for key, value in (document.metadata or {}).items()
                if value not in (None, "")
            }
            toc = self._extract_toc(document)
            pages = self._extract_pages(document)

        repeated = self._find_repeated_header_footer_patterns(pages)
        self._remove_headers_and_footers(pages, repeated)
        body_font_size = self._infer_body_font_size(pages)
        self._classify_page_layouts(pages)
        if self.config.merge_adjacent_text_blocks:
            # 先修复 PDF 把一句话拆成多个 TextBlock 的情况，再判断标题。
            self._merge_adjacent_blocks(pages, body_font_size, respect_headings=False)
        self._detect_headings(pages, body_font_size)
        if self.config.merge_adjacent_text_blocks:
            # 标题识别后再做一次保守合并，防止越过真实标题。
            self._merge_adjacent_blocks(pages, body_font_size, respect_headings=True)
        self._build_page_text(pages)

        return ParsedDocument(
            document_id=document_id,
            source_path=str(path),
            file_name=path.name,
            sha256=file_hash,
            page_count=len(pages),
            metadata=metadata,
            toc=toc,
            body_font_size=round(body_font_size, 3),
            removed_header_footer_patterns=sorted(repeated),
            pages=pages,
        )

    def _extract_toc(self, document: pymupdf.Document) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        try:
            for item in document.get_toc(simple=True):
                if len(item) < 3:
                    continue
                level, title, page_number = item[:3]
                result.append(
                    {
                        "level": int(level),
                        "title": normalize_inline_text(str(title)),
                        "page_number": int(page_number),
                    }
                )
        except Exception as exc:
            LOGGER.warning("PDF 目录读取失败：%s", exc)
        return result

    def _extract_pages(self, document: pymupdf.Document) -> list[PageData]:
        pages: list[PageData] = []

        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            page_number = page_index + 1
            LOGGER.info("解析第 %d/%d 页", page_number, document.page_count)

            native_text = page.get_text("text", sort=True)
            native_char_count = visible_char_count(native_text)
            should_ocr = self._should_ocr(native_char_count)
            text_page = None
            used_ocr = False

            if should_ocr:
                try:
                    text_page = page.get_textpage_ocr(
                        language=self.config.ocr_language,
                        dpi=self.config.ocr_dpi,
                        full=True,
                        tessdata=self.config.tessdata,
                    )
                    used_ocr = True
                except Exception as exc:
                    message = (
                        f"第 {page_number} 页需要 OCR，但执行失败：{exc}。"
                        "请检查 Tesseract 和语言包，或使用 ocr_mode=never。"
                    )
                    if self.config.fail_on_ocr_error:
                        raise RuntimeError(message) from exc
                    LOGGER.warning(message)

            (
                blocks,
                removed_noise_count,
                removed_promotional_count,
                reconstructed_code_count,
            ) = self._extract_text_blocks(
                page=page,
                page_number=page_number,
                text_page=text_page,
                source="ocr" if used_ocr else "native",
            )
            if self.config.extract_tables:
                tables, rejected_table_count = self._extract_tables(page, page_number)
            else:
                tables, rejected_table_count = [], 0

            if tables:
                blocks = [
                    block
                    for block in blocks
                    if not any(
                        _rect_intersection_ratio(block.bbox, table.bbox) >= 0.72
                        for table in tables
                    )
                ]

            pages.append(
                PageData(
                    page_number=page_number,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    rotation=int(page.rotation),
                    used_ocr=used_ocr,
                    native_text_char_count=native_char_count,
                    blocks=blocks,
                    tables=tables,
                    rejected_table_count=rejected_table_count,
                    removed_noise_block_count=removed_noise_count,
                    removed_promotional_block_count=removed_promotional_count,
                    reconstructed_code_block_count=reconstructed_code_count,
                )
            )

        return pages

    def _should_ocr(self, native_char_count: int) -> bool:
        if self.config.ocr_mode == "always":
            return True
        if self.config.ocr_mode == "never":
            return False
        return native_char_count < self.config.min_native_text_chars

    def _extract_text_blocks(
        self,
        page: pymupdf.Page,
        page_number: int,
        text_page: pymupdf.TextPage | None,
        source: str,
    ) -> tuple[list[TextBlock], int, int, int]:
        kwargs: dict[str, Any] = {"sort": True}
        if text_page is not None:
            kwargs["textpage"] = text_page

        page_dict = page.get_text("dict", **kwargs)
        blocks: list[TextBlock] = []
        removed_noise_count = 0
        removed_promotional_count = 0
        reconstructed_code_count = 0

        for raw_block in page_dict.get("blocks", []):
            if raw_block.get("type") != 0:
                continue

            normalized_lines: list[str] = []
            code_lines: list[str] = []
            line_x0s: list[float] = []
            font_samples: list[tuple[float, int]] = []
            is_bold = False
            mono_chars = 0
            total_chars = 0

            for line in raw_block.get("lines", []):
                spans = line.get("spans", [])
                raw_line = "".join(str(span.get("text", "")) for span in spans)
                normalized_line = normalize_inline_text(raw_line)
                code_line = normalize_code_line(raw_line)
                if normalized_line:
                    if (
                        self.config.remove_promotional_contacts
                        and is_promotional_contact_line(normalized_line)
                    ):
                        removed_promotional_count += 1
                        continue
                    normalized_lines.append(normalized_line)
                    code_lines.append(code_line)
                    line_bbox = line.get("bbox", (0, 0, 0, 0))
                    line_x0s.append(float(line_bbox[0]) if line_bbox else 0.0)

                for span in spans:
                    span_text = normalize_inline_text(str(span.get("text", "")))
                    if not span_text:
                        continue
                    size = float(span.get("size", 0.0) or 0.0)
                    weight = max(visible_char_count(span_text), 1)
                    font_samples.append((size, weight))
                    flags = int(span.get("flags", 0) or 0)
                    font_name = str(span.get("font", "")).lower()
                    is_bold = is_bold or bool(flags & 16) or "bold" in font_name
                    total_chars += weight
                    if any(token in font_name for token in ("mono", "courier", "consolas", "code")):
                        mono_chars += weight

            monospace = total_chars > 0 and mono_chars / total_chars >= 0.6
            code_like = self.config.reconstruct_code_blocks and looks_like_code_lines(
                code_lines,
                min_markers=self.config.code_min_marker_lines,
            )
            if monospace or code_like:
                text = reconstruct_code_from_lines(
                    code_lines,
                    line_x0s,
                    indent_spaces=self.config.code_indent_spaces,
                )
                reconstructed_code_count += int(bool(text))
            else:
                text = merge_extracted_lines(normalized_lines)
                text = clean_extracted_text(text)
            monospace = monospace or code_like
            if not text or (self.config.remove_ui_noise and is_noise_line(text)):
                removed_noise_count += 1
                continue

            bbox_raw = raw_block.get("bbox", (0, 0, 0, 0))
            bbox = tuple(round(float(value), 3) for value in bbox_raw[:4])
            max_size = max((size for size, _ in font_samples), default=0.0)
            total_weight = sum(weight for _, weight in font_samples)
            avg_size = (
                sum(size * weight for size, weight in font_samples) / total_weight
                if total_weight
                else 0.0
            )

            blocks.append(
                TextBlock(
                    page_number=page_number,
                    bbox=bbox,  # type: ignore[arg-type]
                    text=text,
                    block_no=int(raw_block.get("number", len(blocks))),
                    source="ocr" if source == "ocr" else "native",
                    max_font_size=round(max_size, 3),
                    avg_font_size=round(avg_size, 3),
                    bold=is_bold,
                    monospace=monospace,
                    line_texts=list(code_lines),
                    line_x0s=list(line_x0s),
                )
            )

        return (
            _sort_blocks(blocks),
            removed_noise_count,
            removed_promotional_count,
            reconstructed_code_count,
        )

    def _extract_tables(
        self,
        page: pymupdf.Page,
        page_number: int,
    ) -> tuple[list[TableData], int]:
        result: list[TableData] = []
        rejected = 0
        try:
            finder = page.find_tables(strategy=self.config.table_strategy)
        except TypeError:
            finder = page.find_tables()
        except Exception as exc:
            LOGGER.warning("第 %d 页表格检测失败：%s", page_number, exc)
            return result, rejected

        for index, table in enumerate(getattr(finder, "tables", []), start=1):
            try:
                rows = normalize_rows(table.extract() or [])
            except Exception as exc:
                LOGGER.warning("第 %d 页第 %d 个表格提取失败：%s", page_number, index, exc)
                rejected += 1
                continue
            if not rows:
                rejected += 1
                continue

            bbox_raw = getattr(table, "bbox", (0, 0, 0, 0))
            bbox = tuple(round(float(value), 3) for value in bbox_raw[:4])
            score, reasons = self._score_table(rows, bbox, page.rect)
            if score < self.config.min_table_quality_score:
                LOGGER.info(
                    "第 %d 页第 %d 个候选表格被拒绝，score=%.3f，原因=%s",
                    page_number,
                    index,
                    score,
                    reasons,
                )
                rejected += 1
                continue

            result.append(
                TableData(
                    page_number=page_number,
                    table_index=index,
                    bbox=bbox,  # type: ignore[arg-type]
                    rows=rows,
                    markdown=table_to_markdown(rows),
                    quality_score=round(score, 4),
                    quality_reasons=reasons,
                )
            )
        return result, rejected

    def _score_table(
        self,
        rows: Sequence[Sequence[str]],
        bbox: tuple[float, float, float, float],
        page_rect: pymupdf.Rect,
    ) -> tuple[float, list[str]]:
        reasons: list[str] = []
        row_count = len(rows)
        column_count = max((len(row) for row in rows), default=0)
        if row_count < self.config.min_table_rows or column_count < self.config.min_table_columns:
            return 0.0, ["rows_or_columns_too_few"]

        padded = [list(row) + [""] * (column_count - len(row)) for row in rows]
        cells = [cell for row in padded for cell in row]
        nonempty = [cell for cell in cells if cell.strip()]
        nonempty_ratio = len(nonempty) / max(len(cells), 1)
        if nonempty_ratio < self.config.min_table_nonempty_ratio:
            return 0.0, ["nonempty_ratio_too_low"]

        nonempty_per_row = [sum(bool(cell.strip()) for cell in row) for row in padded]
        single_cell_row_ratio = sum(value <= 1 for value in nonempty_per_row) / row_count
        long_cell_ratio = sum(
            visible_char_count(cell) > self.config.max_narrative_cell_chars
            for cell in nonempty
        ) / max(len(nonempty), 1)
        avg_cell_chars = statistics.mean(visible_char_count(cell) for cell in nonempty)

        widths = [len(row) for row in rows]
        modal_width, modal_count = Counter(widths).most_common(1)[0]
        width_consistency = modal_count / row_count

        x0, y0, x1, y1 = bbox
        table_area = max((x1 - x0) * (y1 - y0), 0)
        page_area = max(float(page_rect.width * page_rect.height), 1.0)
        area_ratio = table_area / page_area

        score = 0.42
        score += min(0.18, nonempty_ratio * 0.18)
        score += min(0.16, width_consistency * 0.16)
        score += 0.12 if row_count >= 3 else 0.05
        score += 0.12 if column_count >= 3 else 0.06

        if single_cell_row_ratio > 0.5:
            score -= 0.32
            reasons.append("too_many_single_cell_rows")
        if long_cell_ratio > 0.35:
            score -= 0.28
            reasons.append("too_many_narrative_cells")
        if row_count <= 3 and avg_cell_chars > 65:
            score -= 0.25
            reasons.append("small_narrative_layout")
        if area_ratio > 0.72 and row_count <= 3:
            score -= 0.25
            reasons.append("large_page_textbox_like")
        if nonempty_ratio < 0.6:
            score -= 0.08
            reasons.append("sparse_cells")
        if width_consistency < 0.7:
            score -= 0.08
            reasons.append("inconsistent_columns")

        return max(0.0, min(score, 1.0)), reasons

    def _find_repeated_header_footer_patterns(self, pages: Sequence[PageData]) -> set[str]:
        if len(pages) < self.config.min_repeated_header_footer_pages:
            return set()

        occurrences: dict[str, set[int]] = {}
        for page in pages:
            top_limit = page.height * self.config.header_footer_zone_ratio
            bottom_limit = page.height * (1 - self.config.header_footer_zone_ratio)
            for block in page.blocks:
                if block.bbox[1] > top_limit and block.bbox[3] < bottom_limit:
                    continue
                pattern = _canonical_header_footer(block.text)
                if pattern:
                    occurrences.setdefault(pattern, set()).add(page.page_number)

        threshold = max(
            self.config.min_repeated_header_footer_pages,
            math.ceil(len(pages) * self.config.repeated_header_footer_ratio),
        )
        return {
            pattern
            for pattern, page_numbers in occurrences.items()
            if len(page_numbers) >= threshold
        }

    def _remove_headers_and_footers(
        self,
        pages: Sequence[PageData],
        repeated_patterns: set[str],
    ) -> None:
        if not repeated_patterns:
            return

        for page in pages:
            top_limit = page.height * self.config.header_footer_zone_ratio
            bottom_limit = page.height * (1 - self.config.header_footer_zone_ratio)
            page.blocks = [
                block
                for block in page.blocks
                if not (
                    (block.bbox[1] <= top_limit or block.bbox[3] >= bottom_limit)
                    and _canonical_header_footer(block.text) in repeated_patterns
                )
            ]

    def _infer_body_font_size(self, pages: Sequence[PageData]) -> float:
        samples: list[tuple[float, int]] = []
        for page in pages:
            for block in page.blocks:
                if block.avg_font_size <= 0 or block.monospace:
                    continue
                weight = min(max(visible_char_count(block.text), 1), 200)
                samples.append((block.avg_font_size, weight))
        return _weighted_median(samples) if samples else 0.0

    def _classify_page_layouts(self, pages: Sequence[PageData]) -> None:
        for page in pages:
            if not self.config.detect_complex_layout:
                page.layout_kind = "normal"
                self._classify_page_role(page)
                continue
            blocks = page.blocks
            total_chars = sum(visible_char_count(block.text) for block in blocks)
            if len(blocks) <= 3 and total_chars < 100:
                page.layout_kind = "sparse"
                self._classify_page_role(page)
                continue
            if len(blocks) < self.config.complex_layout_min_blocks:
                page.layout_kind = "normal"
                self._classify_page_role(page)
                continue

            char_counts = [visible_char_count(block.text) for block in blocks]
            short_count = sum(
                count <= self.config.complex_layout_short_block_chars
                for count in char_counts
            )
            short_ratio = short_count / len(char_counts)
            centers = [(block.bbox[0] + block.bbox[2]) / 2 for block in blocks]
            x_spread = (max(centers) - min(centers)) / max(page.width, 1.0)
            median_chars = statistics.median(char_counts)
            distinct_x_bands = len(
                {round(center / max(page.width, 1.0) * 6) for center in centers}
            )
            short_numbered = sum(
                bool(_MULTI_NUMERIC_RE.match(normalize_inline_text(block.text)))
                and visible_char_count(block.text) <= 55
                for block in blocks
            )

            generic_complex = (
                short_ratio >= self.config.complex_layout_short_block_ratio
                and x_spread >= self.config.complex_layout_min_x_spread_ratio
                and median_chars <= self.config.complex_layout_short_block_chars * 1.5
                and distinct_x_bands >= 3
            )
            # 封面、目录和流程图页常由多个短标签横向散布组成。
            first_page_slide = (
                page.page_number <= 2
                and short_count >= 4
                and distinct_x_bands >= 3
                and (x_spread >= 0.22 or short_numbered >= 3)
            )
            page.layout_kind = "complex" if generic_complex or first_page_slide else "normal"
            self._classify_page_role(page)

    def _classify_page_role(self, page: PageData) -> None:
        page.page_role = "body"
        page.index_enabled = True
        lines: list[str] = []
        for block in page.blocks:
            lines.extend(line for line in block.text.splitlines() if normalize_inline_text(line))

        numbered = toc_entry_count(lines)
        entry_lengths = [
            visible_char_count(line)
            for line in lines
            if _MULTI_NUMERIC_RE.match(normalize_inline_text(line))
        ]
        average_entry = statistics.mean(entry_lengths) if entry_lengths else 0.0
        if (
            self.config.detect_toc_pages
            and page.page_number <= 5
            and numbered >= self.config.toc_min_numbered_entries
            and average_entry <= self.config.toc_max_average_entry_chars
        ):
            page.page_role = "toc"
            page.index_enabled = self.config.index_toc_pages
            return

        if page.page_number == 1 and page.layout_kind == "sparse" and len(lines) <= 4:
            page.page_role = "cover"
            page.index_enabled = False
        elif page.layout_kind == "complex":
            page.page_role = "complex"

    def _detect_headings(self, pages: Sequence[PageData], body_font_size: float) -> None:
        for page in pages:
            largest_font = max((block.max_font_size for block in page.blocks), default=0.0)
            for block in page.blocks:
                level, source = _infer_heading(
                    block=block,
                    body_font_size=body_font_size,
                    page=page,
                    largest_font=largest_font,
                    config=self.config,
                )
                block.heading_level = level
                block.heading_source = source

    def _merge_adjacent_blocks(
        self,
        pages: Sequence[PageData],
        body_font_size: float,
        respect_headings: bool,
    ) -> None:
        for page in pages:
            if page.layout_kind == "complex":
                continue
            merged: list[TextBlock] = []
            for block in _sort_blocks(page.blocks):
                if merged and self._should_merge_code_blocks(merged[-1], block):
                    merged[-1] = _merge_code_blocks(
                        merged[-1],
                        block,
                        indent_spaces=self.config.code_indent_spaces,
                    )
                elif merged and self._should_merge_blocks(
                    merged[-1], block, body_font_size, respect_headings
                ):
                    merged[-1] = _merge_blocks(merged[-1], block)
                else:
                    merged.append(block)
            page.blocks = merged

    def _should_merge_code_blocks(self, left: TextBlock, right: TextBlock) -> bool:
        if not (left.monospace and right.monospace):
            return False
        if left.heading_level or right.heading_level:
            return False
        left_font = max(left.avg_font_size, 1.0)
        right_font = max(right.avg_font_size, 1.0)
        if abs(left_font - right_font) / max(left_font, right_font) > 0.18:
            return False
        vertical_gap = right.bbox[1] - left.bbox[3]
        max_gap = max(left_font, right_font) * 2.4
        if vertical_gap < -3 or vertical_gap > max_gap:
            return False
        return abs(left.bbox[0] - right.bbox[0]) <= 140

    def _should_merge_blocks(
        self,
        left: TextBlock,
        right: TextBlock,
        body_font_size: float,
        respect_headings: bool,
    ) -> bool:
        if respect_headings and (left.heading_level or right.heading_level):
            return False
        if left.monospace or right.monospace:
            return False
        if is_list_line(right.text):
            return False
        left_lines = [line for line in left.text.splitlines() if line.strip()]
        if any(is_list_line(line) for line in left_lines) and is_paragraph_starter(right.text):
            return False
        if ends_with_sentence_terminal(left.text) or left.text.endswith(("：", ":")):
            return False

        # 预合并阶段也不能吞掉视觉上非常明确的短标题。长编号说明句不算标题，
        # 允许与后续 TextBlock 合并，修复“这里我 / 们直接使用”一类断裂。
        if not respect_headings:
            for candidate in (left, right):
                text = normalize_inline_text(candidate.text)
                ratio = candidate.max_font_size / body_font_size if body_font_size > 0 else 0.0
                strong_style = ratio >= 1.22 or (candidate.bold and ratio >= 1.08)
                strong_shape = (
                    visible_char_count(text) <= 52
                    and not looks_like_sentence_or_list_item(text)
                    and (strong_style or bool(_MULTI_NUMERIC_RE.match(text)))
                )
                if strong_shape:
                    return False

        left_font = max(left.avg_font_size, 1.0)
        right_font = max(right.avg_font_size, 1.0)
        if abs(left_font - right_font) / max(left_font, right_font) > 0.14:
            return False

        vertical_gap = right.bbox[1] - left.bbox[3]
        max_gap = max(left_font, right_font) * self.config.max_block_vertical_gap_ratio
        if vertical_gap < -3 or vertical_gap > max_gap:
            return False

        x_offset = abs(left.bbox[0] - right.bbox[0])
        if x_offset > self.config.max_block_x_offset:
            return False

        overlap = _horizontal_overlap_ratio(left.bbox, right.bbox)
        return overlap >= 0.55

    def _build_page_text(self, pages: Sequence[PageData]) -> None:
        for page in pages:
            elements: list[tuple[float, float, str]] = []
            for block in page.blocks:
                text = block.text
                if block.heading_level:
                    text = f"{'#' * block.heading_level} {text}"
                elements.append((block.y0, block.bbox[0], text))
            elements.extend((table.y0, table.bbox[0], table.markdown) for table in page.tables)
            page.text = "\n\n".join(
                text for _, _, text in sorted(elements, key=lambda item: (item[0], item[1]))
            )


def _sort_blocks(blocks: Sequence[TextBlock]) -> list[TextBlock]:
    return sorted(blocks, key=lambda item: (round(item.bbox[1], 1), item.bbox[0]))


def _merge_blocks(left: TextBlock, right: TextBlock) -> TextBlock:
    left_chars = max(visible_char_count(left.text), 1)
    right_chars = max(visible_char_count(right.text), 1)
    total = left_chars + right_chars
    return replace(
        left,
        bbox=(
            min(left.bbox[0], right.bbox[0]),
            min(left.bbox[1], right.bbox[1]),
            max(left.bbox[2], right.bbox[2]),
            max(left.bbox[3], right.bbox[3]),
        ),
        text=smart_join_text(left.text, right.text),
        max_font_size=max(left.max_font_size, right.max_font_size),
        avg_font_size=round(
            (left.avg_font_size * left_chars + right.avg_font_size * right_chars) / total,
            3,
        ),
        bold=left.bold or right.bold,
        block_no=min(left.block_no, right.block_no),
        line_texts=[*left.line_texts, *right.line_texts],
        line_x0s=[*left.line_x0s, *right.line_x0s],
    )


def _merge_code_blocks(
    left: TextBlock,
    right: TextBlock,
    indent_spaces: int,
) -> TextBlock:
    line_texts = [*left.line_texts, *right.line_texts]
    line_x0s = [*left.line_x0s, *right.line_x0s]
    if not line_texts:
        left_lines = left.text.splitlines()
        right_lines = right.text.splitlines()
        line_texts = [*left_lines, *right_lines]
        line_x0s = [left.bbox[0]] * len(left_lines) + [right.bbox[0]] * len(right_lines)
    text = reconstruct_code_from_lines(
        line_texts,
        line_x0s,
        indent_spaces=indent_spaces,
    )
    return replace(
        left,
        bbox=(
            min(left.bbox[0], right.bbox[0]),
            min(left.bbox[1], right.bbox[1]),
            max(left.bbox[2], right.bbox[2]),
            max(left.bbox[3], right.bbox[3]),
        ),
        text=text,
        max_font_size=max(left.max_font_size, right.max_font_size),
        avg_font_size=round((left.avg_font_size + right.avg_font_size) / 2, 3),
        bold=left.bold or right.bold,
        block_no=min(left.block_no, right.block_no),
        line_texts=line_texts,
        line_x0s=line_x0s,
    )


def _rect_intersection_ratio(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = inner
    bx0, by0, bx1, by1 = outer
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    area = max((ax1 - ax0) * (ay1 - ay0), 1e-9)
    return intersection / area


def _horizontal_overlap_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    smaller = max(min(left[2] - left[0], right[2] - right[0]), 1e-9)
    return overlap / smaller


def _canonical_header_footer(value: str) -> str:
    value = normalize_inline_text(value)
    if not value or visible_char_count(value) > 120:
        return ""
    value = re.sub(r"\d+", "#", value.lower())
    value = re.sub(r"\s+", "", value)
    return re.sub(r"[-–—_/\\|·•]+", "", value)


def _weighted_median(samples: Sequence[tuple[float, int]]) -> float:
    ordered = sorted((value, max(weight, 1)) for value, weight in samples)
    total = sum(weight for _, weight in ordered)
    midpoint = total / 2
    cumulative = 0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= midpoint:
            return float(value)
    return float(ordered[-1][0])


_MULTI_NUMERIC_RE = re.compile(r"^(?P<num>\d+(?:\.\d+){1,5})[、.]?\s+(?P<title>\S.*)$")
_SINGLE_NUMERIC_RE = re.compile(r"^(?P<num>\d+)[、.]\s+(?P<title>\S.*)$")
_CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百千万0-9]+[篇编章]\s*")
_SECTION_RE = re.compile(r"^第[一二三四五六七八九十百千万0-9]+节\s*")
_CHINESE_LIST_RE = re.compile(r"^[一二三四五六七八九十]+、\s*\S+")
_PAREN_LIST_RE = re.compile(r"^[（(][一二三四五六七八九十0-9]+[）)]\s*\S+")


def _infer_heading(
    block: TextBlock,
    body_font_size: float,
    page: PageData,
    largest_font: float,
    config: ParserConfig,
) -> tuple[int | None, str | None]:
    text = normalize_inline_text(block.text)
    compact_length = visible_char_count(text)
    if (
        not text
        or "\n" in text
        or compact_length > config.heading_max_chars
        or block.monospace
        or is_noise_line(text)
    ):
        return None, None

    font_ratio = block.max_font_size / body_font_size if body_font_size > 0 else 0.0
    style_signal = (
        font_ratio >= config.heading_min_font_ratio
        or (block.bold and font_ratio >= config.heading_bold_min_font_ratio)
    )
    near_page_top = block.bbox[1] <= page.height * 0.28
    sentence_like = looks_like_sentence_or_list_item(text)

    if page.layout_kind == "complex" or (page.layout_kind == "sparse" and page.page_number <= 2):
        if config.complex_layout_strategy == "skip":
            return None, None
        # 封面、图解和幻灯片式稀疏页面只允许一个顶部主标题。
        is_dominant = largest_font > 0 and block.max_font_size >= largest_font * 0.94
        if is_dominant and near_page_top and compact_length <= 50:
            return 1, "complex_layout_dominant_title"
        return None, None

    if _CHAPTER_RE.match(text):
        return (1, "chapter_pattern") if style_signal or compact_length <= 28 else (None, None)
    if _SECTION_RE.match(text):
        return (2, "section_pattern") if style_signal or compact_length <= 28 else (None, None)

    multi = _MULTI_NUMERIC_RE.match(text)
    if multi:
        level = min(multi.group("num").count(".") + 1, 6)
        # 多级编号本身是较强信号，但长句、解释句仍不能作为标题。
        if not sentence_like and (style_signal or compact_length <= 52):
            return level, "numeric_hierarchy"
        return None, None

    single = _SINGLE_NUMERIC_RE.match(text)
    if single:
        # 单级编号最容易与步骤列表混淆，必须同时满足样式和短文本条件。
        if (
            style_signal
            and compact_length <= config.single_level_heading_max_chars
            and not sentence_like
        ):
            return 1, "single_numeric_styled"
        return None, None

    if _CHINESE_LIST_RE.match(text) or _PAREN_LIST_RE.match(text):
        if style_signal and compact_length <= 32 and not sentence_like:
            return 2, "styled_chinese_number"
        return None, None

    if body_font_size <= 0:
        return (3, "bold_fallback") if block.bold and compact_length <= 40 else (None, None)

    if font_ratio >= 1.60 and compact_length <= 60:
        return 1, "font_ratio"
    if font_ratio >= 1.35 and compact_length <= 60:
        return 2, "font_ratio"
    if font_ratio >= 1.16 and compact_length <= 52 and not sentence_like:
        return 3, "font_ratio"
    if block.bold and font_ratio >= 1.03 and compact_length <= 42 and not sentence_like:
        return 3, "bold_style"
    return None, None
