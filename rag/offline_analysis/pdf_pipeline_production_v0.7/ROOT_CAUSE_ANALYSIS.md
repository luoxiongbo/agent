# Root-cause analysis based on the original 42-page PDF

Target document: `【RAG实战-第1天】RAG流程详解与优化方案：案例+代码+图解.pdf`

This version was not tuned from Parent/Child symptoms alone. The original PDF was rendered and inspected together with PyMuPDF `dict/rawdict` blocks, line coordinates, fonts, leading whitespace, and code-gutter line numbers.

## Verified causes

### 1. Correct indentation already exists in the PDF text layer

On code pages such as pages 6, 9, and 18-21, the Menlo code lines contain their original leading spaces. Their visual `x0` coordinates are nearly identical because indentation is encoded in text, not geometry.

The v0.6 reconstruction path called `strip()` and deleted that evidence. It then tried to infer indentation from `x0`, which cannot work reliably for this PDF.

v0.7 treats extracted leading whitespace as authoritative. Coordinate inference is only a fallback for PDFs that genuinely omit indentation.

### 2. Code boxes contain visual soft wraps

The browser-generated PDF visually wraps long source lines. Examples found in the original document include:

```text
fine_tuned_i + ntent_model
 doc_id + s)
query_v + ec
file_ + path
```

v0.7 joins only high-confidence wraps near the code panel's right edge, inside unclosed strings/brackets, or after explicit continuation operators. It preserves real logical newlines.

Regression guards explicitly prevent:

```text
break + if rank != 0  -> breakif rank != 0
comment + if ...      -> one invalid line
```

### 3. Display labels and source code are different data

The PDF places labels such as `测试`, `对用户查询进行分类`, and `初始化句向量模型` inside code panels. Some are legal Python bare names, so `ast.parse()` alone can incorrectly classify the panel as executable.

v0.7 stores two layers:

- `text`: faithful source extraction; no silent rewrite;
- `metadata.normalized_code_text`: analysis-only variant where high-confidence presentation labels are comments.

The metadata distinguishes:

```text
python_executable
python_normalized
python_pseudocode
```

### 4. Cross-page code listings must be merged before analysis

Pages 18-21 contain one listing whose left gutter continues from line 1 through line 178. v0.7 records gutter ranges and merges consecutive pages before code analysis and function-aware splitting.

### 5. Extraction quality must be measured per original listing

A long source listing may be divided into several retrieval pieces. v0.6 re-parsed every piece and counted each one as a separate code unit, which penalized valid continuation fragments and distorted the denominator.

v0.7 evaluates each original code listing once. Retrieval pieces inherit the source listing's analysis metadata.

## Remaining limitation

One original listing is still classified as `python_pseudocode`. The source PDF itself contains an incomplete final example (`for ch in chunks:` followed by an unindented `print(...)`) and presentation-oriented code. v0.7 preserves it faithfully and marks it `explanation_only` rather than inventing a correction.
