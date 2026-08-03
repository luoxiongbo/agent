from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

OCRSource = Literal["native", "ocr"]
AtomicKind = Literal["paragraph", "list", "table", "faq", "code", "callout"]


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
    monospace: bool = False
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
class AtomicUnit:
    unit_id: str
    document_id: str
    kind: AtomicKind
    text: str
    page_start: int
    page_end: int
    heading_path: list[str]
    protected: bool
    group_id: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Section:
    section_id: str
    document_id: str
    title: str
    heading_path: list[str]
    page_start: int
    page_end: int
    units: list[AtomicUnit]


@dataclass(slots=True)
class ParentChunk:
    parent_id: str
    document_id: str
    title: str
    heading_path: list[str]
    content: str
    text: str
    token_estimate: int
    page_start: int
    page_end: int
    child_ids: list[str]
    metadata: dict[str, Any]


@dataclass(slots=True)
class ChildChunk:
    child_id: str
    parent_id: str
    document_id: str
    chunk_index: int
    content: str
    text: str
    token_estimate: int
    page_start: int
    page_end: int
    heading_path: list[str]
    unit_ids: list[str]
    overlap_unit_ids: list[str]
    previous_child_id: str | None
    next_child_id: str | None
    metadata: dict[str, Any]


@dataclass(slots=True)
class QualityIssue:
    severity: Literal["warning", "error"]
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QualityReport:
    passed: bool
    page_count: int
    parent_count: int
    child_count: int
    atomic_unit_count: int
    ocr_page_count: int
    duplicate_child_ratio: float
    child_token_min: int
    child_token_median: float
    child_token_max: int
    issues: list[QualityIssue]
