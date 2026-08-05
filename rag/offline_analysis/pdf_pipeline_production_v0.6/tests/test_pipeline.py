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
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime"]["pipeline_version"] == "0.6.0"
    assert manifest["runtime"]["code_fingerprint"]
    assert all(item["metadata"]["pipeline_version"] == "0.6.0" for item in children)
    assert (output / "indexable_children.jsonl").exists()


def test_toc_children_are_excluded_from_indexable_output(tmp_path: Path) -> None:
    pdf_path = tmp_path / "toc.pdf"
    document = pymupdf.open()

    page = document.new_page(width=595, height=842)
    page.insert_text((180, 50), "RAG Course Contents", fontsize=24)
    entries = [
        "1.1 Overview",
        "1.2 Offline Flow",
        "1.3 Online Flow",
        "2.1 Query Module",
        "2.2 Query Rewrite",
        "2.3 Query Expansion",
    ]
    for index, entry in enumerate(entries):
        page.insert_text((72 + (index % 2) * 250, 120 + (index // 2) * 70), entry, fontsize=14)

    page = document.new_page(width=595, height=842)
    page.insert_text((72, 80), "1.1 Overview", fontsize=20)
    page.insert_text((72, 140), "RAG combines retrieval with generation.", fontsize=12)
    document.save(pdf_path)
    document.close()

    config = PipelineConfig(
        parser=ParserConfig(
            ocr_mode="never",
            extract_tables=False,
            toc_min_numbered_entries=5,
        ),
        chunk=ChunkConfig(min_tokens=20, target_tokens=80, max_tokens=140, overlap_tokens=0),
        semantic=SemanticConfig(backend="hashing", hashing_dimensions=256),
        quality=QualityConfig(strict=True),
    )
    output = tmp_path / "out"
    RAGPDFPipeline(config).run(pdf_path, output)

    all_children = [
        json.loads(line)
        for line in (output / "children.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    indexable = [
        json.loads(line)
        for line in (output / "indexable_children.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(all_children) == 2
    assert len(indexable) == 1
    assert any(item["metadata"]["content_type"] == "toc" for item in all_children)
    assert all(item["metadata"]["index_enabled"] is True for item in indexable)
