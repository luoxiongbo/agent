from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

OCRSource = Literal["native", "ocr"]
AtomicKind = Literal["paragraph", "list", "table", "faq", "code", "callout", "layout", "toc"]
PageLayoutKind = Literal["normal", "complex", "sparse"]
PageRoleKind = Literal["body", "toc", "cover", "complex"]
RetrievalReadiness = Literal["ready", "review", "blocked"]


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
    heading_source: str | None = None
    line_texts: list[str] = field(default_factory=list)
    line_x0s: list[float] = field(default_factory=list)

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
    quality_score: float = 1.0
    quality_reasons: list[str] = field(default_factory=list)

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
    layout_kind: PageLayoutKind = "normal"
    rejected_table_count: int = 0
    removed_noise_block_count: int = 0
    removed_promotional_block_count: int = 0
    reconstructed_code_block_count: int = 0
    page_role: PageRoleKind = "body"
    index_enabled: bool = True


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
    metadata: dict[str, Any] = field(default_factory=dict)


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
    severity: Literal["info", "warning", "error"]
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QualityReport:
    passed: bool
    retrieval_readiness: RetrievalReadiness
    page_count: int
    parent_count: int
    child_count: int
    atomic_unit_count: int
    ocr_page_count: int
    complex_layout_page_count: int
    rejected_table_count: int
    removed_noise_block_count: int
    removed_promotional_block_count: int
    reconstructed_code_block_count: int
    toc_page_count: int
    non_indexable_child_count: int

    duplicate_child_ratio: float
    non_indexable_child_ratio: float
    malformed_code_chunk_ratio: float
    glued_list_paragraph_ratio: float
    short_child_ratio_lt_100: float
    under_min_tokens_ratio: float
    over_target_tokens_ratio: float
    single_child_parent_ratio: float
    flat_heading_path_ratio: float
    suspicious_heading_ratio: float
    mid_sentence_start_ratio: float
    broken_line_candidate_ratio: float

    child_token_min: int
    child_token_p10: float
    child_token_p25: float
    child_token_median: float
    child_token_p75: float
    child_token_p90: float
    child_token_max: int
    children_per_parent_median: float
    atomic_units_per_child_median: float

    issues: list[QualityIssue]
