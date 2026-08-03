from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

from .models import (
    AtomicUnit,
    ChildChunk,
    ParentChunk,
    ParsedDocument,
    QualityReport,
)
from .text_utils import safe_json_value


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def save_pipeline_outputs(
    output_dir: str | Path,
    document: ParsedDocument,
    units: Sequence[AtomicUnit],
    parents: Sequence[ParentChunk],
    children: Sequence[ChildChunk],
    quality: QualityReport,
    config: dict,
) -> dict[str, Path]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    paths = {
        "document": output / "document.json",
        "markdown": output / "document.md",
        "atomic_units": output / "atomic_units.jsonl",
        "parents": output / "parents.jsonl",
        "children": output / "children.jsonl",
        "quality": output / "quality_report.json",
        "manifest": output / "manifest.json",
    }

    atomic_write_text(
        paths["document"],
        json.dumps(safe_json_value(asdict(document)), ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write_text(paths["markdown"], document_to_markdown(document))
    atomic_write_text(paths["atomic_units"], to_jsonl(units))
    atomic_write_text(paths["parents"], to_jsonl(parents))
    atomic_write_text(paths["children"], to_jsonl(children))
    atomic_write_text(
        paths["quality"],
        json.dumps(safe_json_value(asdict(quality)), ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write_text(
        paths["manifest"],
        json.dumps(
            {
                "document_id": document.document_id,
                "file_name": document.file_name,
                "sha256": document.sha256,
                "page_count": document.page_count,
                "parent_count": len(parents),
                "child_count": len(children),
                "quality_passed": quality.passed,
                "retrieval_readiness": quality.retrieval_readiness,
                "config": config,
                "files": {key: path.name for key, path in paths.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    return paths


def to_jsonl(records: Iterable[object]) -> str:
    lines = [
        json.dumps(safe_json_value(asdict(record)), ensure_ascii=False)
        for record in records
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def document_to_markdown(document: ParsedDocument) -> str:
    lines = [f"# {document.file_name}", ""]
    for page in document.pages:
        lines.extend([f"<!-- page: {page.page_number} -->", "", page.text, ""])
    return "\n".join(lines).rstrip() + "\n"
