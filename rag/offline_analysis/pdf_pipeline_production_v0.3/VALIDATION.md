# Validation

- Version: 0.3.0
- Python syntax compilation: passed
- Automated tests: **13 passed**
- End-to-end CLI smoke test: passed
- Sample CLI retrieval readiness: **ready**
- Existing user output audit: **not_ready**

## Tests

```text
.............                                                            [100%]
```

## CLI smoke output

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
  "retrieval_readiness": "ready",
  "semantic_backend": "hashing",
  "outputs": {
    "document": "/tmp/tmp.8cLIctRuxv/output/document.json",
    "markdown": "/tmp/tmp.8cLIctRuxv/output/document.md",
    "atomic_units": "/tmp/tmp.8cLIctRuxv/output/atomic_units.jsonl",
    "parents": "/tmp/tmp.8cLIctRuxv/output/parents.jsonl",
    "children": "/tmp/tmp.8cLIctRuxv/output/children.jsonl",
    "quality": "/tmp/tmp.8cLIctRuxv/output/quality_report.json",
    "manifest": "/tmp/tmp.8cLIctRuxv/output/manifest.json"
  }
}
```

## Existing output audit highlights

- short_child_ratio_lt_100: 0.153846
- under_min_tokens_ratio: 0.282051
- single_child_parent_ratio: 0.682927
- flat_heading_path_ratio: 1.0
- suspicious_heading_ratio: 0.243902
- mid_sentence_start_ratio: 0.076923
- broken_line_candidate_ratio: 0.794872
- verdict: not_ready

The original PDF was not available in this environment, so the optimized parser could not be rerun against that exact PDF. The supplied old Parent/Child JSONL files were audited directly, and all parser changes are covered by synthetic and end-to-end regression tests.
