from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import statistics
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

try:
    import pymupdf
except ImportError as exc:
    raise SystemExit(
        "缺少 PyMuPDF。请执行：python -m pip install -r requirements.txt"
    ) from exc


LOGGER = logging.getLogger("rag_pdf_parser")

OCRMode = Literal["auto", "always", "never"]
OCRSource = Literal["native", "ocr"]


@dataclass(slots=True)
class ParserConfig:
    """PDF 解析配置。"""

    ocr_mode: OCRMode = "auto"
    ocr_language: str = "chi_sim+eng"
    ocr_dpi: int = 300
    tessdata: str | None = None
    min_native_text_chars: int = 30
    fail_on_ocr_error: bool = True

    extract_tables: bool = True
    table_strategy: str = "lines"

    header_footer_zone_ratio: float = 0.08
    repeated_header_footer_ratio: float = 0.35
    min_repeated_header_footer_pages: int = 2

    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 80
    min_chunk_tokens: int = 40

    include_source_prefix: bool = True

    def validate(self) -> None:
        if self.ocr_mode not in {"auto", "always", "never"}:
            raise ValueError(f"不支持的 OCR 模式：{self.ocr_mode}")
        if self.ocr_dpi < 72:
            raise ValueError("ocr_dpi 不能小于 72")
        if self.min_native_text_chars < 0:
            raise ValueError("min_native_text_chars 不能小于 0")
        if not 0 < self.header_footer_zone_ratio < 0.5:
            raise ValueError("header_footer_zone_ratio 必须在 0 和 0.5 之间")
        if not 0 < self.repeated_header_footer_ratio <= 1:
            raise ValueError("repeated_header_footer_ratio 必须在 0 和 1 之间")
        if self.chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens 必须大于 0")
        if not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens 必须大于等于 0，且小于 chunk_size_tokens")


@dataclass(slots=True)
class TextBlock:
    page_number: int
    bbox: tuple[float, float, float, float]
    text: str
    block_no: int
    source: OCRSource
    max_font_size: float = 0.0
    avg_font_size: float = 0.0
    bold: bool = False
    heading_level: int | None = None

    @property
    def y0(self) -> float:
        return self.bbox[1]


@dataclass(slots=True)
class TableData:
    page_number: int
    table_index: int
    bbox: tuple[float, float, float, float]
    rows: list[list[str]]
    markdown: str

    @property
    def y0(self) -> float:
        return self.bbox[1]


@dataclass(slots=True)
class PageData:
    page_number: int
    width: float
    height: float
    rotation: int
    used_ocr: bool
    native_text_char_count: int
    blocks: list[TextBlock] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)
    text: str = ""


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    text: str
    token_estimate: int
    page_start: int
    page_end: int
    heading_path: list[str]
    metadata: dict[str, Any]


@dataclass(slots=True)
class ParsedDocument:
    document_id: str
    source_path: str
    file_name: str
    sha256: str
    page_count: int
    metadata: dict[str, Any]
    toc: list[dict[str, Any]]
    body_font_size: float
    removed_header_footer_patterns: list[str]
    pages: list[PageData]


@dataclass(slots=True)
class _Segment:
    page_number: int
    text: str
    kind: Literal["text", "table"]
    heading_path: tuple[str, ...]


def normalize_inline_text(value: str) -> str:
    """清理单行文本，不主动删除中文标点。"""
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("\u00ad", "")      # soft hyphen
    value = value.replace("\u200b", "")      # zero width space
    value = value.replace("\ufeff", "")
    value = re.sub(r"[\t\v\f\r ]+", " ", value)
    return value.strip()


def normalize_multiline_text(value: str) -> str:
    lines = [normalize_inline_text(line) for line in (value or "").splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def visible_char_count(value: str) -> int:
    return len(re.sub(r"\s+", "", value or ""))


def contains_cjk(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", value or ""))


def is_list_line(value: str) -> bool:
    value = value.strip()
    return bool(
        re.match(
            r"^(?:[-*•·]\s+|"
            r"\(?\d{1,3}[.)、]\s*|"
            r"[一二三四五六七八九十百]+[、.]\s*|"
            r"\([一二三四五六七八九十百]+\)\s*)",
            value,
        )
    )


def merge_extracted_lines(lines: Sequence[str]) -> str:
    """
    将 PDF 中被视觉换行拆开的文本重新合并。

    - 列表项保留换行；
    - 英文断词 "exam-\nple" 合并；
    - 中文行默认直接连接；
    - 英文行之间补一个空格。
    """
    cleaned = [normalize_inline_text(line) for line in lines]
    cleaned = [line for line in cleaned if line]
    if not cleaned:
        return ""

    result = cleaned[0]
    for current in cleaned[1:]:
        previous = result.rstrip()

        if is_list_line(current):
            result += "\n" + current
            continue

        if previous.endswith("-") and re.match(r"^[a-z]", current):
            result = previous[:-1] + current
            continue

        if re.search(r"[。！？!?；;：:]$", previous):
            result += "\n" + current
            continue

        if contains_cjk(previous[-8:]) or contains_cjk(current[:8]):
            result += current
        else:
            result += " " + current

    return normalize_multiline_text(result)


def estimate_tokens(text: str) -> int:
    """
    无需绑定特定模型的轻量 Token 估算。

    中文/日文/韩文字符按约 1 token 估计；
    其余非空白字符按约 4 字符 1 token 估计。
    真正写入向量库前，可替换成目标模型 tokenizer。
    """
    text = text or ""
    cjk_count = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))
    other = re.sub(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\s]", "", text)
    return cjk_count + math.ceil(len(other) / 4)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while data := file.read(chunk_size):
            digest.update(data)
    return digest.hexdigest()


def safe_json_value(value: Any) -> Any:
    """把 PyMuPDF 返回值转成可 JSON 序列化的普通类型。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json_value(v) for v in value]
    return str(value)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def rect_intersection_ratio(
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


def block_inside_any_table(block: TextBlock, tables: Sequence[TableData]) -> bool:
    return any(rect_intersection_ratio(block.bbox, table.bbox) >= 0.55 for table in tables)


class PDFParser:
    def __init__(self, config: ParserConfig | None = None) -> None:
        self.config = config or ParserConfig()
        self.config.validate()

    def parse(self, pdf_path: str | Path, password: str | None = None) -> ParsedDocument:
        path = Path(pdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF 文件不存在：{path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"只接受 .pdf 文件：{path}")

        file_hash = sha256_file(path)
        document_id = f"pdf_{file_hash[:20]}"

        LOGGER.info("打开 PDF：%s", path)
        with pymupdf.open(path) as document:
            if document.needs_pass:
                if not password or not document.authenticate(password):
                    raise PermissionError("PDF 已加密，请通过 --password 提供正确密码")

            metadata = {
                key: safe_json_value(value)
                for key, value in (document.metadata or {}).items()
                if value not in (None, "")
            }
            toc = self._extract_toc(document)
            pages = self._extract_pages(document)

        repeated_patterns = self._find_repeated_header_footer_patterns(pages)
        self._remove_headers_and_footers(pages, repeated_patterns)

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
            removed_header_footer_patterns=sorted(repeated_patterns),
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
        except Exception as exc:  # 某些损坏 PDF 的目录对象会报错
            LOGGER.warning("读取 PDF 目录失败：%s", exc)
        return result

    def _extract_pages(self, document: pymupdf.Document) -> list[PageData]:
        pages: list[PageData] = []
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            page_number = page_index + 1
            LOGGER.info("解析第 %s/%s 页", page_number, document.page_count)

            native_text = page.get_text("text", sort=True)
            native_char_count = visible_char_count(native_text)
            should_ocr = self._should_ocr(native_char_count)

            used_ocr = False
            text_page = None

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
                        f"第 {page_number} 页需要 OCR，但 OCR 执行失败：{exc}\n"
                        "请确认已安装 Tesseract 和对应语言数据；"
                        "也可以使用 --ocr never 禁用 OCR。"
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

            tables: list[TableData] = []
            if self.config.extract_tables:
                tables = self._extract_tables(page, page_number)

            # 表格正文通常也会出现在普通文本块中。删除大部分落在表格内部的块，避免重复。
            if tables:
                blocks = [block for block in blocks if not block_inside_any_table(block, tables)]

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
        source: OCRSource,
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

            for line in raw_block.get("lines", []):
                spans = line.get("spans", [])
                line_text = "".join(str(span.get("text", "")) for span in spans)
                line_text = normalize_inline_text(line_text)
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
                    is_bold = is_bold or bool(flags & 16)

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
                    source=source,
                    max_font_size=round(max_size, 3),
                    avg_font_size=round(avg_size, 3),
                    bold=is_bold,
                )
            )

        return sorted(blocks, key=lambda item: (round(item.bbox[1], 1), item.bbox[0]))

    def _extract_tables(self, page: pymupdf.Page, page_number: int) -> list[TableData]:
        result: list[TableData] = []
        try:
            finder = page.find_tables(strategy=self.config.table_strategy)
        except TypeError:
            # 兼容不接受 strategy 参数的较旧 PyMuPDF。
            finder = page.find_tables()
        except Exception as exc:
            LOGGER.warning("第 %s 页表格检测失败：%s", page_number, exc)
            return result

        for table_index, table in enumerate(getattr(finder, "tables", []), start=1):
            try:
                raw_rows = table.extract()
            except Exception as exc:
                LOGGER.warning(
                    "第 %s 页第 %s 个表格提取失败：%s",
                    page_number,
                    table_index,
                    exc,
                )
                continue

            rows: list[list[str]] = []
            for raw_row in raw_rows or []:
                row = [normalize_multiline_text("" if cell is None else str(cell)) for cell in raw_row]
                if any(cell for cell in row):
                    rows.append(row)

            if not rows:
                continue

            bbox_raw = getattr(table, "bbox", (0, 0, 0, 0))
            bbox = tuple(round(float(value), 3) for value in bbox_raw[:4])
            result.append(
                TableData(
                    page_number=page_number,
                    table_index=table_index,
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
                in_header = block.bbox[1] <= top_limit
                in_footer = block.bbox[3] >= bottom_limit
                if not (in_header or in_footer):
                    continue

                pattern = canonical_header_footer_pattern(block.text)
                if not pattern:
                    continue
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
            kept: list[TextBlock] = []
            for block in page.blocks:
                top_limit = page.height * self.config.header_footer_zone_ratio
                bottom_limit = page.height * (1 - self.config.header_footer_zone_ratio)
                in_zone = block.bbox[1] <= top_limit or block.bbox[3] >= bottom_limit
                pattern = canonical_header_footer_pattern(block.text)
                if in_zone and pattern in repeated_patterns:
                    continue
                kept.append(block)
            page.blocks = kept

    def _infer_body_font_size(self, pages: Sequence[PageData]) -> float:
        samples: list[tuple[float, int]] = []
        for page in pages:
            for block in page.blocks:
                if block.avg_font_size <= 0:
                    continue
                # 过短文字更可能是标题、页码或图注，不作为主要正文样本。
                weight = min(max(visible_char_count(block.text), 1), 200)
                samples.append((block.avg_font_size, weight))

        if not samples:
            return 0.0
        return weighted_median(samples)

    def _detect_headings(self, pages: Sequence[PageData], body_font_size: float) -> None:
        for page in pages:
            for block in page.blocks:
                block.heading_level = infer_heading_level(block, body_font_size)

    def _build_page_text(self, pages: Sequence[PageData]) -> None:
        for page in pages:
            elements: list[tuple[float, str]] = []
            for block in page.blocks:
                text = block.text
                if block.heading_level:
                    text = f"{'#' * block.heading_level} {text}"
                elements.append((block.y0, text))
            for table in page.tables:
                elements.append((table.y0, table.markdown))

            page.text = "\n\n".join(text for _, text in sorted(elements, key=lambda item: item[0]))


def canonical_header_footer_pattern(value: str) -> str:
    value = normalize_inline_text(value)
    if not value or visible_char_count(value) > 120:
        return ""
    value = value.lower()
    value = re.sub(r"\d+", "#", value)
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[-–—_/\\|·•]+", "", value)
    return value


def weighted_median(samples: Sequence[tuple[float, int]]) -> float:
    ordered = sorted((value, max(weight, 1)) for value, weight in samples)
    total = sum(weight for _, weight in ordered)
    midpoint = total / 2
    cumulative = 0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= midpoint:
            return float(value)
    return float(ordered[-1][0])


HEADING_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"^第[一二三四五六七八九十百千万0-9]+[篇编章]\s*"), 1),
    (re.compile(r"^第[一二三四五六七八九十百千万0-9]+节\s*"), 2),
    (re.compile(r"^\d+\s*[、.]\s*\S+"), 1),
    (re.compile(r"^\d+\.\d+\s+\S+"), 2),
    (re.compile(r"^\d+\.\d+\.\d+\s+\S+"), 3),
    (re.compile(r"^[一二三四五六七八九十]+、\s*\S+"), 2),
    (re.compile(r"^[（(][一二三四五六七八九十0-9]+[）)]\s*\S+"), 3),
]


def infer_heading_level(block: TextBlock, body_font_size: float) -> int | None:
    text = block.text.strip()
    compact_length = visible_char_count(text)

    if not text or compact_length > 100 or "\n" in text and compact_length > 60:
        return None

    for pattern, level in HEADING_PATTERNS:
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


def table_to_markdown(rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    normalized = [
        [markdown_escape(cell) for cell in list(row) + [""] * (width - len(row))]
        for row in rows
    ]

    header = normalized[0]
    separator = ["---"] * width
    body = normalized[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def markdown_escape(value: str) -> str:
    value = normalize_multiline_text(value)
    value = value.replace("|", r"\|")
    value = value.replace("\n", "<br>")
    return value


class RAGChunker:
    def __init__(self, config: ParserConfig | None = None) -> None:
        self.config = config or ParserConfig()
        self.config.validate()

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        segments_buffer: list[_Segment] = []
        current_heading: list[str] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal segments_buffer, current_tokens
            if not segments_buffer:
                return

            body = "\n\n".join(segment.text for segment in segments_buffer).strip()
            if not body:
                segments_buffer = []
                current_tokens = 0
                return

            heading_path = list(segments_buffer[-1].heading_path)
            pages = [segment.page_number for segment in segments_buffer]
            page_start, page_end = min(pages), max(pages)
            chunk_index = len(chunks)

            prefix_parts: list[str] = []
            if self.config.include_source_prefix:
                prefix_parts.append(f"文档：{document.file_name}")
                if heading_path:
                    prefix_parts.append("章节：" + " > ".join(heading_path))
                prefix_parts.append(
                    f"页码：{page_start}"
                    if page_start == page_end
                    else f"页码：{page_start}-{page_end}"
                )

            content = (
                "\n".join(prefix_parts) + "\n\n" + body
                if prefix_parts
                else body
            )
            token_count = estimate_tokens(content)
            chunk_id = f"{document.document_id}_chunk_{chunk_index:06d}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    content=content,
                    text=body,
                    token_estimate=token_count,
                    page_start=page_start,
                    page_end=page_end,
                    heading_path=heading_path,
                    metadata={
                        "source": document.source_path,
                        "file_name": document.file_name,
                        "sha256": document.sha256,
                        "page_start": page_start,
                        "page_end": page_end,
                        "heading_path": heading_path,
                    },
                )
            )

            segments_buffer = self._overlap_tail(segments_buffer)
            current_tokens = sum(estimate_tokens(item.text) for item in segments_buffer)

        for page in document.pages:
            ordered: list[tuple[float, Literal["block", "table"], Any]] = []
            for block in page.blocks:
                ordered.append((block.y0, "block", block))
            for table in page.tables:
                ordered.append((table.y0, "table", table))
            ordered.sort(key=lambda item: item[0])

            for _, kind, element in ordered:
                if kind == "block" and element.heading_level:
                    flush()
                    update_heading_path(current_heading, element.heading_level, element.text)
                    continue

                raw_text = element.text if kind == "block" else element.markdown
                segment_kind: Literal["text", "table"] = "text" if kind == "block" else "table"

                for piece in split_oversized_text(
                    raw_text,
                    max_tokens=self.config.chunk_size_tokens,
                ):
                    piece_tokens = estimate_tokens(piece)

                    if (
                        segments_buffer
                        and current_tokens + piece_tokens > self.config.chunk_size_tokens
                    ):
                        flush()

                    segment = _Segment(
                        page_number=page.page_number,
                        text=piece,
                        kind=segment_kind,
                        heading_path=tuple(current_heading),
                    )
                    segments_buffer.append(segment)
                    current_tokens += piece_tokens

                    if current_tokens >= self.config.chunk_size_tokens:
                        flush()

        flush()

        # 合并过小的末尾 Chunk，避免只留下一个短尾巴。
        return self._merge_tiny_tail(chunks)

    def _overlap_tail(self, segments: Sequence[_Segment]) -> list[_Segment]:
        if self.config.chunk_overlap_tokens <= 0:
            return []

        result: list[_Segment] = []
        token_count = 0
        last_heading = segments[-1].heading_path if segments else ()

        for segment in reversed(segments):
            if segment.heading_path != last_heading:
                break
            segment_tokens = estimate_tokens(segment.text)
            if result and token_count + segment_tokens > self.config.chunk_overlap_tokens:
                break
            result.append(segment)
            token_count += segment_tokens
            if token_count >= self.config.chunk_overlap_tokens:
                break

        return list(reversed(result))

    def _merge_tiny_tail(self, chunks: list[Chunk]) -> list[Chunk]:
        if len(chunks) < 2:
            return chunks

        tail = chunks[-1]
        previous = chunks[-2]
        if (
            tail.token_estimate >= self.config.min_chunk_tokens
            or tail.heading_path != previous.heading_path
        ):
            return chunks

        merged_text = previous.text.rstrip() + "\n\n" + tail.text.lstrip()
        previous.text = merged_text
        previous.page_end = max(previous.page_end, tail.page_end)
        previous.content = rebuild_chunk_content(previous, previous.metadata["file_name"])
        previous.token_estimate = estimate_tokens(previous.content)
        previous.metadata["page_end"] = previous.page_end
        chunks.pop()
        return chunks


def rebuild_chunk_content(chunk: Chunk, file_name: str) -> str:
    prefix = [f"文档：{file_name}"]
    if chunk.heading_path:
        prefix.append("章节：" + " > ".join(chunk.heading_path))
    prefix.append(
        f"页码：{chunk.page_start}"
        if chunk.page_start == chunk.page_end
        else f"页码：{chunk.page_start}-{chunk.page_end}"
    )
    return "\n".join(prefix) + "\n\n" + chunk.text


def update_heading_path(path: list[str], level: int, title: str) -> None:
    level = max(1, min(level, 6))
    while len(path) >= level:
        path.pop()
    while len(path) < level - 1:
        path.append("未命名章节")
    path.append(normalize_inline_text(title))


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+(?=[A-Z0-9])")


def split_oversized_text(text: str, max_tokens: int) -> list[str]:
    text = normalize_multiline_text(text)
    if not text:
        return []
    if estimate_tokens(text) <= max_tokens:
        return [text]

    paragraphs = [item.strip() for item in re.split(r"\n{1,}", text) if item.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if estimate_tokens(paragraph) <= max_tokens:
            units.append(paragraph)
            continue

        sentences = [item.strip() for item in SENTENCE_BOUNDARY_RE.split(paragraph) if item.strip()]
        if len(sentences) <= 1:
            units.extend(hard_split_text(paragraph, max_tokens))
        else:
            for sentence in sentences:
                if estimate_tokens(sentence) <= max_tokens:
                    units.append(sentence)
                else:
                    units.extend(hard_split_text(sentence, max_tokens))

    result: list[str] = []
    buffer: list[str] = []
    token_count = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if buffer and token_count + unit_tokens > max_tokens:
            result.append("\n".join(buffer))
            buffer = []
            token_count = 0
        buffer.append(unit)
        token_count += unit_tokens

    if buffer:
        result.append("\n".join(buffer))
    return result


def hard_split_text(text: str, max_tokens: int) -> list[str]:
    """
    最后的保底切分。按估算比例切字符，而不是在这里依赖特定 tokenizer。
    """
    if not text:
        return []
    estimated = max(estimate_tokens(text), 1)
    char_limit = max(50, int(len(text) * max_tokens / estimated))
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + char_limit, len(text))
        if end < len(text):
            # 尽量退到标点处。
            candidate = max(
                text.rfind("。", start, end),
                text.rfind("；", start, end),
                text.rfind("\n", start, end),
                text.rfind(" ", start, end),
            )
            if candidate > start + char_limit // 2:
                end = candidate + 1
        pieces.append(text[start:end].strip())
        start = end
    return [piece for piece in pieces if piece]


def parsed_document_to_dict(document: ParsedDocument) -> dict[str, Any]:
    return safe_json_value(asdict(document))


def chunks_to_jsonl(chunks: Iterable[Chunk]) -> str:
    return "\n".join(
        json.dumps(safe_json_value(asdict(chunk)), ensure_ascii=False)
        for chunk in chunks
    ) + "\n"


def document_to_markdown(document: ParsedDocument) -> str:
    lines = [f"# {document.file_name}", ""]
    for page in document.pages:
        lines.extend([f"<!-- page: {page.page_number} -->", "", page.text, ""])
    return "\n".join(lines).rstrip() + "\n"


def save_outputs(
    document: ParsedDocument,
    chunks: Sequence[Chunk],
    output_dir: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    document_json = output / "document.json"
    chunks_jsonl = output / "chunks.jsonl"
    markdown_file = output / "document.md"

    atomic_write_text(
        document_json,
        json.dumps(parsed_document_to_dict(document), ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write_text(chunks_jsonl, chunks_to_jsonl(chunks))
    atomic_write_text(markdown_file, document_to_markdown(document))

    return {
        "document_json": document_json,
        "chunks_jsonl": chunks_jsonl,
        "markdown": markdown_file,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="面向 RAG 离线知识库的 PDF 解析、清洗、表格提取与 Chunk 切分工具"
    )
    parser.add_argument("pdf", help="输入 PDF 路径")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="./output",
        help="输出目录，默认 ./output",
    )
    parser.add_argument("--password", help="加密 PDF 密码")

    parser.add_argument(
        "--ocr",
        choices=["auto", "always", "never"],
        default="auto",
        help="OCR 模式：auto/always/never，默认 auto",
    )
    parser.add_argument(
        "--ocr-language",
        default="chi_sim+eng",
        help="Tesseract 语言，默认 chi_sim+eng",
    )
    parser.add_argument("--ocr-dpi", type=int, default=300, help="OCR DPI，默认 300")
    parser.add_argument("--tessdata", help="tessdata 目录")
    parser.add_argument(
        "--min-native-text-chars",
        type=int,
        default=30,
        help="原生文字少于该字符数时，auto 模式触发 OCR",
    )
    parser.add_argument(
        "--skip-ocr-errors",
        action="store_true",
        help="OCR 失败时跳过该页，而不是终止任务",
    )

    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="禁用表格提取",
    )
    parser.add_argument(
        "--table-strategy",
        choices=["lines", "lines_strict", "text"],
        default="lines",
        help="PyMuPDF 表格检测策略，默认 lines",
    )

    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk 估算 token 上限")
    parser.add_argument("--chunk-overlap", type=int, default=80, help="Chunk 重叠估算 token")
    parser.add_argument("--min-chunk-size", type=int, default=40, help="最小 Chunk 估算 token")

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="输出详细日志",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = ParserConfig(
        ocr_mode=args.ocr,
        ocr_language=args.ocr_language,
        ocr_dpi=args.ocr_dpi,
        tessdata=args.tessdata,
        min_native_text_chars=args.min_native_text_chars,
        fail_on_ocr_error=not args.skip_ocr_errors,
        extract_tables=not args.no_tables,
        table_strategy=args.table_strategy,
        chunk_size_tokens=args.chunk_size,
        chunk_overlap_tokens=args.chunk_overlap,
        min_chunk_tokens=args.min_chunk_size,
    )

    try:
        document = PDFParser(config).parse(args.pdf, password=args.password)
        chunks = RAGChunker(config).chunk(document)
        paths = save_outputs(document, chunks, args.output_dir)
    except Exception as exc:
        LOGGER.error("%s", exc)
        if args.verbose:
            LOGGER.exception("完整异常")
        return 1

    summary = {
        "document_id": document.document_id,
        "page_count": document.page_count,
        "ocr_pages": [page.page_number for page in document.pages if page.used_ocr],
        "table_count": sum(len(page.tables) for page in document.pages),
        "chunk_count": len(chunks),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
