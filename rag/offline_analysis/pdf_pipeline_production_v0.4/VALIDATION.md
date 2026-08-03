# Validation v0.4.0

- Python syntax compilation: passed
- Automated tests: 15 passed
- End-to-end CLI smoke test: passed
- Runtime provenance: passed
- Existing new-output audit verdict: not_ready
- Old/new uploaded output comparison: behaviorally identical

## Test output

```text
...............                                                          [100%]
```

## Runtime diagnostics

```json
{
  "pipeline_version": "0.4.0",
  "code_fingerprint": "4b6031b6156501692490",
  "package_root": "/mnt/data/work_v04/rag_pdf_pipeline_production_v0.4/src/rag_pdf_pipeline",
  "module_file": "/mnt/data/work_v04/rag_pdf_pipeline_production_v0.4/src/rag_pdf_pipeline/runtime.py",
  "python_executable": "/opt/pyvenv/bin/python",
  "python_version": "3.13.5"
}
```

## CLI smoke output

```json
{
  "document_id": "pdf_2adf81e837d3f3492a4d",
  "file_name": "v04_regression.pdf",
  "page_count": 2,
  "ocr_pages": [],
  "table_count": 0,
  "section_count": 2,
  "atomic_unit_count": 4,
  "parent_count": 2,
  "child_count": 2,
  "quality_passed": true,
  "retrieval_readiness": "ready",
  "semantic_backend": "hashing",
  "runtime": {
    "pipeline_version": "0.4.0",
    "code_fingerprint": "4b6031b6156501692490",
    "package_root": "/mnt/data/work_v04/rag_pdf_pipeline_production_v0.4/src/rag_pdf_pipeline",
    "module_file": "/mnt/data/work_v04/rag_pdf_pipeline_production_v0.4/src/rag_pdf_pipeline/runtime.py",
    "python_executable": "/opt/pyvenv/bin/python",
    "python_version": "3.13.5"
  },
  "outputs": {
    "document": "/mnt/data/work_v04/rag_pdf_pipeline_production_v0.4/sample/output_v04_regression_final/document.json",
    "markdown": "/mnt/data/work_v04/rag_pdf_pipeline_production_v0.4/sample/output_v04_regression_final/document.md",
    "atomic_units": "/mnt/data/work_v04/rag_pdf_pipeline_production_v0.4/sample/output_v04_regression_final/atomic_units.jsonl",
    "parents": "/mnt/data/work_v04/rag_pdf_pipeline_production_v0.4/sample/output_v04_regression_final/parents.jsonl",
    "children": "/mnt/data/work_v04/rag_pdf_pipeline_production_v0.4/sample/output_v04_regression_final/children.jsonl",
    "quality": "/mnt/data/work_v04/rag_pdf_pipeline_production_v0.4/sample/output_v04_regression_final/quality_report.json",
    "manifest": "/mnt/data/work_v04/rag_pdf_pipeline_production_v0.4/sample/output_v04_regression_final/manifest.json"
  }
}
```
