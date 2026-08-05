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
from .runtime import runtime_info
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
        runtime = runtime_info()
        LOGGER.info(
            "运行版本=%s，代码指纹=%s，源码=%s",
            runtime["pipeline_version"],
            runtime["code_fingerprint"],
            runtime["package_root"],
        )
        LOGGER.info("阶段 1/5：PDF 解析")
        document = PDFParser(self.config.parser).parse(pdf_path, password=password)
        document.metadata = {**document.metadata, "_pipeline": runtime}

        LOGGER.info("阶段 2/5：原子信息单元识别")
        sections = AtomicUnitBuilder(self.config.parser).build(document)

        LOGGER.info("阶段 3/5：加载语义后端")
        encoder = build_semantic_encoder(self.config.semantic)

        LOGGER.info("阶段 4/5：父子智能 Chunk")
        parents, children, units = SmartParentChildChunker(
            self.config.chunk,
            encoder,
        ).chunk(document, sections)
        provenance = {
            "pipeline_version": runtime["pipeline_version"],
            "code_fingerprint": runtime["code_fingerprint"],
            "package_root": runtime["package_root"],
        }
        for record in [*parents, *children]:
            record.metadata.update(provenance)
        for unit in units:
            unit.metadata.update(provenance)

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
            runtime=runtime,
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
            "indexable_child_count": sum(
                bool(child.metadata.get("index_enabled", True)) for child in children
            ),
            "toc_page_count": report.toc_page_count,
            "removed_promotional_block_count": report.removed_promotional_block_count,
            "reconstructed_code_block_count": report.reconstructed_code_block_count,
            "quality_passed": report.passed,
            "retrieval_readiness": report.retrieval_readiness,
            "semantic_backend": encoder.name,
            "runtime": runtime,
            "outputs": {key: str(path) for key, path in paths.items()},
        }
