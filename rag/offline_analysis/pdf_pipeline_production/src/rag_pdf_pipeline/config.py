from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

OCRMode = Literal["auto", "always", "never"]
SemanticBackend = Literal["none", "hashing", "sentence_transformers"]


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

    header_footer_zone_ratio: float = 0.08
    repeated_header_footer_ratio: float = 0.35
    min_repeated_header_footer_pages: int = 2

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

    def validate(self) -> None:
        if not 0 <= self.max_duplicate_ratio <= 1:
            raise ValueError("max_duplicate_ratio 必须位于 [0, 1]")
        if not 0 <= self.max_ocr_page_ratio_warning <= 1:
            raise ValueError("max_ocr_page_ratio_warning 必须位于 [0, 1]")


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
