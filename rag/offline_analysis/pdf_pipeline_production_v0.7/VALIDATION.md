# Validation v0.7.0

## Validation basis

Unlike earlier iterations, v0.7 was tested directly against the uploaded original PDF:

```text
【RAG实战-第1天】RAG流程详解与优化方案：案例+代码+图解.pdf
42 pages
```

The validation included rendered page inspection, PyMuPDF block/span data, fonts, line coordinates, leading whitespace, and code-gutter line numbers.

Final validated runtime:

```json
{
  "pipeline_version": "0.7.0",
  "code_fingerprint": "2b7176587210e0e452e4"
}
```

## Automated checks

```text
Python compile checks: passed
Pytest: 36 passed
Original-PDF end-to-end CLI run: passed
Wheel build: passed
Wheel clean-install smoke test: passed
```

The six new regression tests cover:

- preserving indentation when every code line has the same `x0`;
- preventing `breakif` over-joining;
- preventing comment + `if` over-joining;
- joining a right-edge split identifier;
- separating faithful source from comment-normalized presentation labels;
- normalizing Chinese labels containing parentheses.

## Original-PDF result

```json
{
  "page_count": 42,
  "section_count": 27,
  "atomic_unit_count": 263,
  "parent_count": 27,
  "child_count": 66,
  "indexable_child_count": 65,
  "toc_page_count": 1,
  "removed_promotional_block_count": 1,
  "reconstructed_code_block_count": 17,
  "python_code_unit_count": 12,
  "python_ast_parseable_ratio": 0.416667,
  "normalized_python_ast_parseable_ratio": 0.916667,
  "python_pseudocode_ratio": 0.083333,
  "code_indentation_issue_ratio": 0.083333,
  "split_identifier_candidate_ratio": 0.0,
  "function_boundary_violation_ratio": 0.0,
  "quality_passed": true,
  "retrieval_readiness": "review"
}
```

## Interpreting the code metrics

- Source AST parseability is intentionally lower because `text` faithfully preserves presentation labels from the PDF.
- Normalized AST parseability comments only high-confidence display labels; it does not rewrite algorithms or silently invent indentation.
- One original listing remains `python_pseudocode`. The source PDF itself contains incomplete/presentation-oriented code, so the correct behavior is `explanation_only`, not fabricated executable code.
- Metrics are calculated once for each original listing. Function-aware retrieval pieces no longer inflate the denominator.

## Chunk distribution

```json
{
  "short_child_ratio_lt_100": 0.0,
  "under_min_tokens_ratio": 0.076923,
  "child_token_min": 112,
  "child_token_p10": 204.0,
  "child_token_median": 437.0,
  "child_token_p90": 592.6,
  "child_token_max": 777,
  "duplicate_child_ratio": 0.0
}
```

## Remaining review items

`retrieval_readiness` remains `review`, not `ready`, because:

- one complex/TOC-style page is preserved but excluded from the index;
- one code listing is genuine teaching pseudocode;
- several prose chunks still trigger conservative sentence-boundary warnings.

These do not block ordinary RAG retrieval testing. Code marked `python_pseudocode` should only be used for explanation, not copy-and-run answers.
