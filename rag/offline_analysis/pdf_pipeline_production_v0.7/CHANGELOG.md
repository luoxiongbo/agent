# Changelog

## 0.7.0

### Root-cause fixes validated on the original PDF

- Preserve code leading whitespace from the PDF text layer; stop stripping valid indentation.
- Use x-coordinate indentation inference only as a fallback for PDFs that omit whitespace.
- Repair only high-confidence visual soft wraps at the code-panel boundary.
- Prevent `break + if` from becoming `breakif` and prevent comments from gluing to statements.
- Keep faithful source code separate from analysis-only `normalized_code_text`.
- Detect Chinese presentation labels inside code panels and comment-normalize them for AST analysis.
- Add `python_executable`, `python_normalized`, and `python_pseudocode` classifications.
- Record code gutter line ranges and merge continuous cross-page listings.
- Evaluate code quality once per original listing rather than per split retrieval piece.
- Add `normalized_python_ast_parseable_ratio` and `python_pseudocode_ratio`.
- Add six regression tests derived from pages 6, 18-21, and 36 of the target PDF.

### Validation

- Original 42-page PDF processed end to end.
- 36 tests passed.
- Normalized AST parseability: 91.67% across 12 original Python listings.
- Split identifier candidate ratio: 0%.
- Function boundary violation ratio: 0%.

## 0.6.0

### 新增

- `code_analysis.py`：Python AST、缩进、标识符断裂和函数边界分析。
- Python 最终代码单元重新执行 `ast.parse()`，避免沿用切分前的旧诊断结果。
- 函数、异步函数和类级安全切分。
- 超长函数续段重复函数签名。
- 代码片段强制 Child 边界，避免不同函数重新合并。
- 高置信度英文标识符断行修复。
- 高置信度中文空行断词二次修复。
- `rag-pdf-audit --atomic-units` 代码质量审计。
- 新质量指标：
  - `python_code_unit_count`
  - `function_level_code_piece_count`
  - `python_ast_parseable_ratio`
  - `code_indentation_issue_ratio`
  - `split_identifier_candidate_ratio`
  - `function_boundary_violation_ratio`

### 修复

- 复杂版面中的等宽代码被误判为 `layout`。
- 复杂页面跳过代码 TextBlock 合并。
- `fine_tuned_i / ntent_model`、`relevant_ / docs`、`losse / s` 等标识符断裂。
- 有坐标缩进时，`return` 被错误维持在更深的 `if` 层级。
- 代码开头被现有输出审计器误判为半句话。
- 审计器把目录 Child 计入短块和标题层级比例。

### 行为变化

- 完整短函数优先保持整体，即使略高于 `target_tokens`。
- 两个独立函数默认进入不同 Child。
- Python AST 失败属于 warning，不会直接将 `passed` 设为 false。
- 表格候选逐条拒绝日志降为 `DEBUG`，`INFO` 按页汇总。
