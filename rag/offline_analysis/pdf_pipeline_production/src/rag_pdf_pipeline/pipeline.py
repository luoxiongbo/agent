from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .atomic import AtomicUnitBuilder
from .chunker import SmartParentChildChunker
from .config import PipelineConfig
from .io import save_pipeline_outputs
from .pdf_parser import PDFParser
from .quality import QualityValidator
from .semantic import build_semantic_encoder

LOGGER = logging.getLogger(__name__)


class RAGPDFPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.config.validate()

    def run(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        password: str | None = None,
    ) -> dict[str, Any]:
        LOGGER.info("阶段 1/5：PDF 解析")
        document = PDFParser(self.config.parser).parse(pdf_path, password=password)

        LOGGER.info("阶段 2/5：原子信息单元识别")
        sections = AtomicUnitBuilder().build(document)

        LOGGER.info("阶段 3/5：加载语义后端")
        encoder = build_semantic_encoder(self.config.semantic)

        LOGGER.info("阶段 4/5：父子智能 Chunk")
        parents, children, units = SmartParentChildChunker(
            self.config.chunk,
            encoder,
        ).chunk(document, sections)

        LOGGER.info("阶段 5/5：质量校验和原子写盘")
        report = QualityValidator(
            self.config.quality,
            self.config.chunk,
        ).validate(document, units, parents, children)

        paths = save_pipeline_outputs(
            output_dir=output_dir,
            document=document,
            units=units,
            parents=parents,
            children=children,
            quality=report,
            config=self.config.to_dict(),
        )

        return {
            "document_id": document.document_id,
            "file_name": document.file_name,
            "page_count": document.page_count,
            "ocr_pages": [
                page.page_number for page in document.pages if page.used_ocr
            ],
            "table_count": sum(len(page.tables) for page in document.pages),
            "section_count": len(sections),
            "atomic_unit_count": len(units),
            "parent_count": len(parents),
            "child_count": len(children),
            "quality_passed": report.passed,
            "semantic_backend": encoder.name,
            "outputs": {key: str(path) for key, path in paths.items()},
        }
