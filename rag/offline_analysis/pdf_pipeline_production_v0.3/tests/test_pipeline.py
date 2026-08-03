from __future__ import annotations

import json
from pathlib import Path

import pymupdf

from rag_pdf_pipeline.config import (
    ChunkConfig,
    ParserConfig,
    PipelineConfig,
    QualityConfig,
    SemanticConfig,
)
from rag_pdf_pipeline.pipeline import RAGPDFPipeline


def test_end_to_end_native_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    document = pymupdf.open()

    for page_number in range(1, 4):
        page = document.new_page(width=595, height=842)
        page.insert_text((72, 35), "Repeated Header", fontsize=9)
        page.insert_text((72, 100), f"{page_number}. User Management", fontsize=20)
        page.insert_text(
            (72, 150),
            (
                "Administrators can create users and reset passwords. "
                "Each operation must be audited and recorded."
            ),
            fontsize=12,
        )
        page.insert_text((72, 810), f"Page {page_number}", fontsize=9)

    document.save(pdf_path)
    document.close()

    config = PipelineConfig(
        parser=ParserConfig(ocr_mode="never", extract_tables=False),
        chunk=ChunkConfig(
            min_tokens=20,
            target_tokens=60,
            max_tokens=110,
            overlap_tokens=10,
        ),
        semantic=SemanticConfig(backend="hashing", hashing_dimensions=256),
        quality=QualityConfig(strict=True),
    )

    output = tmp_path / "output"
    summary = RAGPDFPipeline(config).run(pdf_path, output)

    assert summary["quality_passed"] is True
    assert summary["parent_count"] == 3
    assert summary["child_count"] >= 3

    parsed = json.loads((output / "document.json").read_text(encoding="utf-8"))
    assert "Repeated Header" not in parsed["pages"][0]["text"]

    children = [
        json.loads(line)
        for line in (output / "children.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(item["token_estimate"] <= config.chunk.max_tokens for item in children)
