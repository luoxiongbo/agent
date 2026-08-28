# RAG PDF Pipeline v0.7

Production-oriented PDF parsing, structure recovery, Parent/Child chunking, and explainable quality auditing for RAG.

```text
PDF
→ native text / OCR fallback
→ headers, footers, UI noise, and promotional-contact cleanup
→ TOC / body / complex-page classification
→ heading hierarchy recovery
→ tables, prose, lists, and code routed through separate logic
→ faithful code extraction + analysis-only normalization
→ atomic information units
→ rule + semantic-aware Parent/Child chunking
→ indexable_children.jsonl
→ quality_report.json
```

## What changed in v0.7

v0.7 was validated directly against the original 42-page target PDF rather than only synthetic fixtures.

### Preserve real PDF code indentation

The target PDF stores indentation as leading spaces even though all visual code lines have almost the same `x0`. v0.7 no longer strips those spaces or guesses indentation from coordinates.

### Conservative visual-wrap repair

High-confidence code wraps are repaired:

```text
fine_tuned_i + ntent_model → fine_tuned_intent_model
file_ + path               → file_path
doc_id + s)                → doc_ids)
```

Real newlines are protected, including regressions that previously produced `breakif` or glued a comment to the following `if` statement.

### Faithful source and normalized analysis are separate

For each Python code unit:

- `text` preserves the PDF source;
- `normalized_code_text` is optional and used only for AST analysis;
- Chinese presentation labels inside code panels are comment-normalized without changing source text;
- `code_type` is `python_executable`, `python_normalized`, or `python_pseudocode`.

### Cross-page code listing recovery

Line-number gutters are recorded. Consecutive code panels such as pages 18-21, whose gutter continues from line 1 to line 178, are merged before analysis.

### Correct code-quality denominator

Quality metrics are calculated once per original code listing, not once per retrieval piece created by function-aware splitting.

## Original-PDF validation

The original 42-page PDF produced:

```json
{
  "page_count": 42,
  "parent_count": 27,
  "child_count": 66,
  "indexable_child_count": 65,
  "python_code_unit_count": 12,
  "python_ast_parseable_ratio": 0.416667,
  "normalized_python_ast_parseable_ratio": 0.916667,
  "python_pseudocode_ratio": 0.083333,
  "code_indentation_issue_ratio": 0.083333,
  "split_identifier_candidate_ratio": 0.0,
  "function_boundary_violation_ratio": 0.0,
  "retrieval_readiness": "review"
}
```

`review` is intentional: one source listing is genuinely presentation-oriented/incomplete, and a few prose boundary warnings remain. The pipeline does not fabricate executable code to force a green score.

See:

- `ROOT_CAUSE_ANALYSIS.md`
- `TARGET_PDF_VALIDATION.json`
- `V06_V07_COMPARISON.json`
- `VALIDATION.md`

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Or install the wheel:

```bash
python -m pip install dist/rag_pdf_pipeline-0.7.0-py3-none-any.whl
```

Verify the loaded source:

```bash
rag-pdf --version
rag-pdf-diagnose
```

## Parse a PDF

```bash
rag-pdf input.pdf \
  -o output \
  --semantic-backend hashing \
  --min-tokens 180 \
  --target-tokens 500 \
  --max-tokens 800 \
  --overlap-tokens 80 \
  -v
```

Use this file for vector ingestion:

```text
output/indexable_children.jsonl
```

The full output also includes:

```text
manifest.json
document.json
document.md
atomic_units.jsonl
parents.jsonl
children.jsonl
quality_report.json
```

## Code metadata

Example:

```json
{
  "code_language": "python",
  "code_type": "python_normalized",
  "python_ast_parseable": true,
  "normalized_python_ast_parseable": true,
  "code_narrative_line_count": 1,
  "normalized_code_text": "# 对用户查询进行分类\nquery = ...",
  "answer_mode": "explanation_only"
}
```

The exact extracted `text` remains available for quotation and traceability. Only code marked executable should be treated as copy-and-run source.

## Audit existing output

```bash
rag-pdf-audit \
  --children output/children.jsonl \
  --parents output/parents.jsonl \
  --atomic-units output/atomic_units.jsonl \
  --min-tokens 180 \
  --target-tokens 500 \
  --max-tokens 800 \
  -o audit.json
```

## Semantic backends

Deterministic hashing backend:

```bash
rag-pdf input.pdf -o output --semantic-backend hashing
```

Local SentenceTransformer model:

```bash
python -m pip install -e '.[semantic]'
rag-pdf input.pdf \
  -o output \
  --semantic-backend sentence_transformers \
  --semantic-model /absolute/path/to/local/model
```

Remote model downloads remain opt-in.

## Scope

This is a production-oriented parsing and chunking component. It does not include queues, object storage, vector-database deployment, job state, or distributed retry orchestration.
