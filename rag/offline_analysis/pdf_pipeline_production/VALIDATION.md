# Validation

- Python syntax compilation: passed
- Automated tests: 6 passed
- CLI smoke test return code: 0

## CLI output

```json
{
  "document_id": "pdf_5eeee6e3706d977a30e4",
  "file_name": "demo.pdf",
  "page_count": 3,
  "ocr_pages": [],
  "table_count": 0,
  "section_count": 3,
  "atomic_unit_count": 3,
  "parent_count": 3,
  "child_count": 3,
  "quality_passed": true,
  "semantic_backend": "hashing",
  "outputs": {
    "document": "/mnt/data/rag_pdf_pipeline_production/sample/output/document.json",
    "markdown": "/mnt/data/rag_pdf_pipeline_production/sample/output/document.md",
    "atomic_units": "/mnt/data/rag_pdf_pipeline_production/sample/output/atomic_units.jsonl",
    "parents": "/mnt/data/rag_pdf_pipeline_production/sample/output/parents.jsonl",
    "children": "/mnt/data/rag_pdf_pipeline_production/sample/output/children.jsonl",
    "quality": "/mnt/data/rag_pdf_pipeline_production/sample/output/quality_report.json",
    "manifest": "/mnt/data/rag_pdf_pipeline_production/sample/output/manifest.json"
  }
}
```
