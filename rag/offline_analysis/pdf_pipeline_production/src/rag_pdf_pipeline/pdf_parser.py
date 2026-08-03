from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Sequence

try:
    import pymupdf
except ImportError as exc:
    raise SystemExit("缺少 PyMuPDF，请先安装项目依赖") from exc

from .config import ParserConfig
from .models import PageData, ParsedDocument, TableData, TextBlock
from .text_utils import (
    merge_extracted_lines,
    normalize_inline_text,
    normalize_multiline_text,
    normalize_rows,
    safe_json_value,
    sha256_file,
    table_to_markdown,
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
        self._detect_headings(pages, body_font_size)
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

            blocks = self._extract_text_blocks(
                page=page,
                page_number=page_number,
                text_page=text_page,
                source="ocr" if used_ocr else "native",
            )
            tables = self._extract_tables(page, page_number) if self.config.extract_tables else []

            if tables:
                blocks = [
                    block
                    for block in blocks
                    if not any(_rect_intersection_ratio(block.bbox, table.bbox) >= 0.55 for table in tables)
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
    ) -> list[TextBlock]:
        kwargs: dict[str, Any] = {"sort": True}
        if text_page is not None:
            kwargs["textpage"] = text_page

        page_dict = page.get_text("dict", **kwargs)
        blocks: list[TextBlock] = []

        for raw_block in page_dict.get("blocks", []):
            if raw_block.get("type") != 0:
                continue

            line_texts: list[str] = []
            font_samples: list[tuple[float, int]] = []
            is_bold = False
            mono_chars = 0
            total_chars = 0

            for line in raw_block.get("lines", []):
                spans = line.get("spans", [])
                line_text = normalize_inline_text(
                    "".join(str(span.get("text", "")) for span in spans)
                )
                if line_text:
                    line_texts.append(line_text)

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

            text = merge_extracted_lines(line_texts)
            if not text:
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
                    monospace=total_chars > 0 and mono_chars / total_chars >= 0.6,
                )
            )

        return sorted(blocks, key=lambda item: (round(item.bbox[1], 1), item.bbox[0]))

    def _extract_tables(self, page: pymupdf.Page, page_number: int) -> list[TableData]:
        result: list[TableData] = []
        try:
            finder = page.find_tables(strategy=self.config.table_strategy)
        except TypeError:
            finder = page.find_tables()
        except Exception as exc:
            LOGGER.warning("第 %d 页表格检测失败：%s", page_number, exc)
            return result

        for index, table in enumerate(getattr(finder, "tables", []), start=1):
            try:
                rows = normalize_rows(table.extract() or [])
            except Exception as exc:
                LOGGER.warning("第 %d 页第 %d 个表格提取失败：%s", page_number, index, exc)
                continue
            if not rows:
                continue

            bbox_raw = getattr(table, "bbox", (0, 0, 0, 0))
            bbox = tuple(round(float(value), 3) for value in bbox_raw[:4])
            result.append(
                TableData(
                    page_number=page_number,
                    table_index=index,
                    bbox=bbox,  # type: ignore[arg-type]
                    rows=rows,
                    markdown=table_to_markdown(rows),
                )
            )
        return result

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
                if block.avg_font_size <= 0:
                    continue
                weight = min(max(visible_char_count(block.text), 1), 200)
                samples.append((block.avg_font_size, weight))
        return _weighted_median(samples) if samples else 0.0

    def _detect_headings(self, pages: Sequence[PageData], body_font_size: float) -> None:
        for page in pages:
            for block in page.blocks:
                block.heading_level = _infer_heading_level(block, body_font_size)

    def _build_page_text(self, pages: Sequence[PageData]) -> None:
        for page in pages:
            elements: list[tuple[float, str]] = []
            for block in page.blocks:
                text = block.text
                if block.heading_level:
                    text = f"{'#' * block.heading_level} {text}"
                elements.append((block.y0, text))
            elements.extend((table.y0, table.markdown) for table in page.tables)
            page.text = "\n\n".join(text for _, text in sorted(elements, key=lambda item: item[0]))


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


_HEADING_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"^第[一二三四五六七八九十百千万0-9]+[篇编章]\s*"), 1),
    (re.compile(r"^第[一二三四五六七八九十百千万0-9]+节\s*"), 2),
    (re.compile(r"^\d+\s*[、.]\s*\S+"), 1),
    (re.compile(r"^\d+\.\d+\s+\S+"), 2),
    (re.compile(r"^\d+\.\d+\.\d+\s+\S+"), 3),
    (re.compile(r"^[一二三四五六七八九十]+、\s*\S+"), 2),
    (re.compile(r"^[（(][一二三四五六七八九十0-9]+[）)]\s*\S+"), 3),
]


def _infer_heading_level(block: TextBlock, body_font_size: float) -> int | None:
    text = block.text.strip()
    compact_length = visible_char_count(text)
    if not text or compact_length > 100 or ("\n" in text and compact_length > 60):
        return None

    for pattern, level in _HEADING_PATTERNS:
        if pattern.match(text):
            return level

    if body_font_size <= 0:
        return 3 if block.bold and compact_length <= 40 else None

    ratio = block.max_font_size / body_font_size if block.max_font_size else 0.0
    if ratio >= 1.60:
        return 1
    if ratio >= 1.35:
        return 2
    if ratio >= 1.16:
        return 3
    if block.bold and ratio >= 1.03 and compact_length <= 50:
        return 3
    return None
