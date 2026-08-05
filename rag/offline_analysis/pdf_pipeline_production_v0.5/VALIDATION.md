# Validation v0.5.0

## 结果

- Python 语法编译：通过
- Pytest：21 个测试全部通过
- Wheel 构建：通过（使用 `--no-build-isolation`）
- CLI `--version`：`0.5.0`
- 运行时代码指纹：`c180c9878d9d268d6e84`
- 功能冒烟测试：通过
- Ruff：当前执行环境未安装，因此未运行

## 功能冒烟测试

测试 PDF 包含：

1. 一页目录；
2. 一页列表及后续说明段落；
3. 一页带缩进的 Python 代码。

运行结果：

```json
{
  "page_count": 3,
  "parent_count": 3,
  "child_count": 3,
  "indexable_child_count": 2,
  "toc_page_count": 1,
  "quality_passed": true,
  "retrieval_readiness": "ready",
  "pipeline_version": "0.5.0",
  "code_fingerprint": "c180c9878d9d268d6e84"
}
```

关键质量指标：

```json
{
  "non_indexable_child_count": 1,
  "malformed_code_chunk_ratio": 0.0,
  "glued_list_paragraph_ratio": 0.0,
  "mid_sentence_start_ratio": 0.0,
  "broken_line_candidate_ratio": 0.0
}
```

目录 Child 保存在 `children.jsonl`，但未出现在 `indexable_children.jsonl`。

代码恢复结果：

```python
import openai
from sentence_transformers import SentenceTransformer
def retrieve(query):
    embedding = model.encode(query)
    return embedding
```

列表末项和后续说明段落之间保留空行，没有发生文本粘连。

## Wheel

```text
dist/rag_pdf_pipeline-0.5.0-py3-none-any.whl
```

首次使用隔离构建时，执行环境无法从内部离线索引取得 `setuptools>=75`。改用当前环境已安装的构建依赖：

```bash
python -m pip wheel . --no-deps --no-build-isolation
```

构建成功。该失败属于执行环境的离线依赖解析限制，不是项目源码错误。

## 测试范围

- 标题层级识别；
- 长编号说明句降级；
- Parent/Child 关系；
- max token 硬限制；
- 页眉页脚清除；
- 跨页中文续句；
- 短中文断词恢复；
- 目录页检测和索引排除；
- 推广联系方式识别；
- 表格文本框误报过滤；
- UI 噪声清除；
- 列表末项与说明段落边界；
- 代码行坐标和缩进恢复；
- 输出运行版本及代码指纹；
- `indexable_children.jsonl` 输出。

## 未完成的真实文档验证

本轮没有收到原始 `input.pdf`，只有上一版生成的 Parent/Child 文本。因此无法在当前环境中直接对同一份 42 页 PDF 重新运行 v0.5。

请在本地对原 PDF 重新生成后检查：

```text
quality_report.json
indexable_children.jsonl
parents.jsonl
children.jsonl
manifest.json
```
