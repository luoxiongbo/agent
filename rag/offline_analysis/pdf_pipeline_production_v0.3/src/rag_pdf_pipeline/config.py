from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

OCRMode = Literal["auto", "always", "never"]
SemanticBackend = Literal["none", "hashing", "sentence_transformers"]
ComplexLayoutStrategy = Literal["conservative", "keep", "skip"]


@dataclass(slots=True)
class ParserConfig:
    ocr_mode: OCRMode = "auto"
    ocr_language: str = "chi_sim+eng"
    ocr_dpi: int = 300
    tessdata: str | None = None
    min_native_text_chars: int = 30
    fail_on_ocr_error: bool = True

    extract_tables: bool = True
    table_strategy: str = "lines"
    min_table_rows: int = 2
    min_table_columns: int = 2
    min_table_nonempty_ratio: float = 0.45
    min_table_quality_score: float = 0.52
    max_narrative_cell_chars: int = 160

    header_footer_zone_ratio: float = 0.08
    repeated_header_footer_ratio: float = 0.35
    min_repeated_header_footer_pages: int = 2

    merge_adjacent_text_blocks: bool = True
    merge_cross_page_continuations: bool = True
    max_block_vertical_gap_ratio: float = 1.8
    max_block_x_offset: float = 18.0
    remove_ui_noise: bool = True

    detect_complex_layout: bool = True
    complex_layout_strategy: ComplexLayoutStrategy = "conservative"
    complex_layout_min_blocks: int = 7
    complex_layout_short_block_chars: int = 24
    complex_layout_short_block_ratio: float = 0.65
    complex_layout_min_x_spread_ratio: float = 0.45

    heading_max_chars: int = 80
    single_level_heading_max_chars: int = 38
    heading_min_font_ratio: float = 1.08
    heading_bold_min_font_ratio: float = 1.0

    def validate(self) -> None:
        if self.ocr_mode not in {"auto", "always", "never"}:
            raise ValueError(f"不支持的 OCR 模式：{self.ocr_mode}")
        if self.ocr_dpi < 72:
            raise ValueError("ocr_dpi 不能小于 72")
        if self.min_native_text_chars < 0:
            raise ValueError("min_native_text_chars 不能小于 0")
        if not 0 < self.header_footer_zone_ratio < 0.5:
            raise ValueError("header_footer_zone_ratio 必须位于 (0, 0.5)")
        if not 0 < self.repeated_header_footer_ratio <= 1:
            raise ValueError("repeated_header_footer_ratio 必须位于 (0, 1]")
        if self.min_repeated_header_footer_pages < 2:
            raise ValueError("min_repeated_header_footer_pages 不能小于 2")
        if self.table_strategy not in {"lines", "lines_strict", "text"}:
            raise ValueError(f"不支持的表格策略：{self.table_strategy}")
        if self.min_table_rows < 2 or self.min_table_columns < 2:
            raise ValueError("表格至少需要 2 行 2 列")
        if not 0 <= self.min_table_nonempty_ratio <= 1:
            raise ValueError("min_table_nonempty_ratio 必须位于 [0, 1]")
        if not 0 <= self.min_table_quality_score <= 1:
            raise ValueError("min_table_quality_score 必须位于 [0, 1]")
        if self.max_narrative_cell_chars <= 0:
            raise ValueError("max_narrative_cell_chars 必须大于 0")
        if self.max_block_vertical_gap_ratio <= 0:
            raise ValueError("max_block_vertical_gap_ratio 必须大于 0")
        if self.max_block_x_offset < 0:
            raise ValueError("max_block_x_offset 不能为负数")
        if self.complex_layout_strategy not in {"conservative", "keep", "skip"}:
            raise ValueError(f"不支持的复杂版面策略：{self.complex_layout_strategy}")
        if self.complex_layout_min_blocks < 3:
            raise ValueError("complex_layout_min_blocks 不能小于 3")
        if not 0 <= self.complex_layout_short_block_ratio <= 1:
            raise ValueError("complex_layout_short_block_ratio 必须位于 [0, 1]")
        if not 0 <= self.complex_layout_min_x_spread_ratio <= 1:
            raise ValueError("complex_layout_min_x_spread_ratio 必须位于 [0, 1]")
        if self.heading_max_chars < 10:
            raise ValueError("heading_max_chars 过小")
        if self.single_level_heading_max_chars < 8:
            raise ValueError("single_level_heading_max_chars 过小")


@dataclass(slots=True)
class ChunkConfig:
    """
    min_tokens:
        小于该值时，优先继续合并。
    target_tokens:
        期望大小，不是硬切点。
    max_tokens:
        最终 Child Chunk 的硬上限。
    """

    min_tokens: int = 180
    target_tokens: int = 500
    max_tokens: int = 800
    overlap_tokens: int = 80

    boundary_lookback_units: int = 24
    semantic_weight: float = 1.0
    length_weight: float = 0.35
    same_group_break_penalty: float = 1.25

    include_source_prefix: bool = True

    def validate(self) -> None:
        if not 0 < self.min_tokens <= self.target_tokens <= self.max_tokens:
            raise ValueError("必须满足 0 < min_tokens <= target_tokens <= max_tokens")
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("overlap_tokens 必须大于等于 0 且小于 max_tokens")
        if self.boundary_lookback_units < 2:
            raise ValueError("boundary_lookback_units 不能小于 2")
        if self.semantic_weight < 0 or self.length_weight < 0:
            raise ValueError("semantic_weight 和 length_weight 不能为负数")
        if self.same_group_break_penalty < 0:
            raise ValueError("same_group_break_penalty 不能为负数")


@dataclass(slots=True)
class SemanticConfig:
    backend: SemanticBackend = "hashing"
    model_name_or_path: str | None = None
    local_files_only: bool = True
    device: str | None = None
    batch_size: int = 32
    hashing_dimensions: int = 1024

    def validate(self) -> None:
        if self.backend not in {"none", "hashing", "sentence_transformers"}:
            raise ValueError(f"不支持的语义后端：{self.backend}")
        if self.backend == "sentence_transformers" and not self.model_name_or_path:
            raise ValueError(
                "使用 sentence_transformers 时必须配置 model_name_or_path；"
                "生产环境建议填写已经下载好的本地模型目录"
            )
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        if self.hashing_dimensions < 128:
            raise ValueError("hashing_dimensions 不能小于 128")


@dataclass(slots=True)
class QualityConfig:
    strict: bool = True
    max_duplicate_ratio: float = 0.05
    max_ocr_page_ratio_warning: float = 0.8

    max_short_child_ratio_lt_100: float = 0.10
    max_under_min_tokens_ratio: float = 0.25
    max_single_child_parent_ratio: float = 0.60
    max_mid_sentence_start_ratio: float = 0.03
    max_broken_line_candidate_ratio: float = 0.10
    max_flat_heading_path_ratio: float = 0.95
    max_suspicious_heading_ratio: float = 0.12

    def validate(self) -> None:
        ratio_fields = {
            "max_duplicate_ratio": self.max_duplicate_ratio,
            "max_ocr_page_ratio_warning": self.max_ocr_page_ratio_warning,
            "max_short_child_ratio_lt_100": self.max_short_child_ratio_lt_100,
            "max_under_min_tokens_ratio": self.max_under_min_tokens_ratio,
            "max_single_child_parent_ratio": self.max_single_child_parent_ratio,
            "max_mid_sentence_start_ratio": self.max_mid_sentence_start_ratio,
            "max_broken_line_candidate_ratio": self.max_broken_line_candidate_ratio,
            "max_flat_heading_path_ratio": self.max_flat_heading_path_ratio,
            "max_suspicious_heading_ratio": self.max_suspicious_heading_ratio,
        }
        for name, value in ratio_fields.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} 必须位于 [0, 1]")


@dataclass(slots=True)
class PipelineConfig:
    parser: ParserConfig = field(default_factory=ParserConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)

    def validate(self) -> None:
        self.parser.validate()
        self.chunk.validate()
        self.semantic.validate()
        self.quality.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        config = cls(
            parser=ParserConfig(**data.get("parser", {})),
            chunk=ChunkConfig(**data.get("chunk", {})),
            semantic=SemanticConfig(**data.get("semantic", {})),
            quality=QualityConfig(**data.get("quality", {})),
        )
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("配置文件根节点必须是 JSON object")
        return cls.from_dict(payload)
