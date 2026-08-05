# Validation v0.6.0

## 验证结论

- Python 语法编译：通过
- Pytest：30 个测试全部通过
- Wheel 构建：通过
- Wheel 独立安装与运行时诊断：通过
- 通用功能 PDF 冒烟测试：通过
- 函数级代码 PDF 冒烟测试：通过
- `rag-pdf-audit --atomic-units`：通过
- Ruff：当前执行环境未安装，因此未运行

运行版本与源码指纹：

```json
{
  "pipeline_version": "0.6.0",
  "code_fingerprint": "2b0f8b781884626aeb07"
}
```

## 自动化测试

```text
30 passed
```

新增测试覆盖：

- `fine_tuned_i / ntent_model` 标识符拼接；
- `relevant_ / docs` 标识符拼接；
- `losse / s` 标识符拼接；
- `i / f` Python 关键字拼接；
- 代码坐标缩进恢复后可通过 AST；
- 坏缩进检测；
- 顶层 `return` 函数边界异常检测；
- 顶层函数/类边界提取；
- 函数级代码切分；
- 超长函数续段重复函数签名；
- 中文空行断词修复；
- 代码主导页面不误判为复杂幻灯片页。

## 通用功能冒烟测试

测试文件：`sample/features_demo.pdf`

包含：

1. 目录页；
2. 列表与后续说明段落；
3. 带缩进的 Python 代码。

结果：

```json
{
  "page_count": 3,
  "parent_count": 3,
  "child_count": 3,
  "indexable_child_count": 2,
  "toc_page_count": 1,
  "python_code_unit_count": 1,
  "python_ast_parseable_ratio": 1.0,
  "code_indentation_issue_ratio": 0.0,
  "split_identifier_candidate_ratio": 0.0,
  "function_boundary_violation_ratio": 0.0,
  "quality_passed": true,
  "retrieval_readiness": "ready"
}
```

目录 Child 保存在 `children.jsonl`，但未写入 `indexable_children.jsonl`。

## 函数级代码冒烟测试

测试文件：`sample/v06_code_quality.pdf`

包含：

- 两个独立 Python 函数；
- 多层缩进；
- 被 PDF 分成两行的 `fine_tuned_intent_model` 标识符。

结果：

```json
{
  "page_count": 1,
  "atomic_unit_count": 3,
  "child_count": 3,
  "python_code_unit_count": 3,
  "function_level_code_piece_count": 2,
  "python_ast_parseable_ratio": 1.0,
  "code_indentation_issue_ratio": 0.0,
  "split_identifier_candidate_ratio": 0.0,
  "function_boundary_violation_ratio": 0.0,
  "complex_layout_page_count": 0,
  "quality_passed": true,
  "retrieval_readiness": "ready"
}
```

切分后的代码单元保持完整函数：

```python
import os

def first(query):
    normalized = query.strip().lower()
    values = []
    for index in range(20):
        values.append(f"first-{index}-{normalized}")
    return values
```

```python
def second(query):
    normalized = query.strip().lower()
    values = []
    for index in range(20):
        values.append(f"second-{index}-{normalized}")
    return values
```

标识符恢复结果：

```python
model_path = "./fine_tuned_intent_model"
```

## Wheel

```text
dist/rag_pdf_pipeline-0.6.0-py3-none-any.whl
```

使用以下命令构建：

```bash
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

Wheel 安装到独立目录后，运行时诊断正确返回版本、代码指纹和安装路径。

## 尚未完成的真实文档验证

当前环境没有原始 42 页 `input.pdf`，只有此前生成的日志和 JSONL 输出，因此无法直接对同一原始 PDF 执行 v0.6。

请在本地重新生成后重点检查：

```text
quality_report.json
atomic_units.jsonl
indexable_children.jsonl
parents.jsonl
children.jsonl
manifest.json
```

重点指标：

```text
python_ast_parseable_ratio
code_indentation_issue_ratio
split_identifier_candidate_ratio
function_boundary_violation_ratio
broken_line_candidate_ratio
retrieval_readiness
```
