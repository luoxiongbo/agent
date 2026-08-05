# RAG PDF Pipeline v0.5

生产导向的 PDF 离线解析、结构恢复、智能 Parent/Child Chunk 和质量审计工具。

```text
PDF
→ 原生文本 / OCR
→ 页眉页脚、界面噪声、推广联系方式清洗
→ 表格候选质量过滤
→ 目录、封面、复杂图解页识别
→ 标题层级恢复
→ 跨 TextBlock / 跨页续句修复
→ 代码块坐标级重建
→ 原子信息单元
→ 规则 + 语义融合切分
→ Parent / Child Chunk
→ indexable_children.jsonl
→ 可解释质量报告
```

## v0.5 重点改进

### 目录页不再污染正文检索

目录、课程导航和流程总览页会保留在完整输出中，但默认设置：

```json
{
  "content_type": "toc",
  "index_enabled": false
}
```

所有 Child 仍写入 `children.jsonl`，真正建议写入向量数据库的记录单独写入：

```text
indexable_children.jsonl
```

需要索引目录页时，可显式设置：

```bash
--index-toc-pages
```

或在配置中设置：

```json
{
  "parser": {
    "index_toc_pages": true
  }
}
```

### 清理推广联系方式

默认清除以下宣传性内容：

```text
更多课程联系微信:xxxx
扫码关注公众号 xxxx
添加微信获取资料
```

普通业务正文中的“请联系管理员”不会被宽泛删除。需要保留宣传联系方式时使用：

```bash
--keep-promotional-contacts
```

### 修复列表末项与后续正文粘连

例如旧结果：

```text
3. 向量数据库:建立检索索引离线流程只需执行一次……
```

新版恢复为：

```text
3. 向量数据库:建立检索索引

离线流程只需执行一次……
```

系统会识别“离线流程、在线流程、上述、因此、这两个流程、在实际工程中”等新段落起始语。

### 补强中文断词修复

针对：

```text
能翻 + 阅自己的参考书
销售技 + 巧
因此候 + 选列表
```

新版会结合中文短尾、短头、字体、坐标和章节路径恢复为完整词语，同时避免把两个完整段落无条件拼接。

### 代码块坐标级重建

代码区域不再使用普通段落合并规则。系统会保留每行文字和横坐标，根据缩进位置恢复代码：

```python
import openai
from sentence_transformers import SentenceTransformer

def retrieve(query):
    embedding = model.encode(query)
    return embedding
```

支持：

- 相邻等宽代码 TextBlock 合并；
- 根据 `x0` 推断缩进层级；
- 在 `from`、`import`、`def`、`class` 等高置信度声明前恢复换行；
- 清除 `Plain Text`、`Copy code`、独立项目符号和连续代码行号。

关闭代码重建：

```bash
--no-code-reconstruction
```

## 1. 安装

建议 Python 3.11 或 3.12。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

语义模型可选依赖：

```bash
python -m pip install -e '.[semantic]'
```

生产环境建议提前下载模型，并使用本地路径：

```json
{
  "semantic": {
    "backend": "sentence_transformers",
    "model_name_or_path": "/models/embedding-model",
    "local_files_only": true
  }
}
```

## 2. OCR

扫描 PDF 需要本地 Tesseract。

macOS：

```bash
brew install tesseract
brew install tesseract-lang
tesseract --list-langs | grep -E 'chi_sim|eng'
```

Ubuntu / Debian：

```bash
sudo apt-get update
sudo apt-get install -y \
  tesseract-ocr \
  tesseract-ocr-chi-sim \
  tesseract-ocr-eng
```

## 3. 运行

推荐先使用完全离线的 hashing 语义后端验证解析：

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

使用配置文件：

```bash
rag-pdf input.pdf -o output --config config.example.json -v
```

为避免 shell 中残留旧入口，也可以直接调用当前 Python 模块：

```bash
python -m rag_pdf_pipeline.cli input.pdf -o output --config config.example.json -v
```

确认运行版本：

```bash
rag-pdf --version
rag-pdf-diagnose
```

输出应包含：

```json
{
  "pipeline_version": "0.5.0",
  "code_fingerprint": "...",
  "package_root": ".../rag_pdf_pipeline_production_v0.5/src/rag_pdf_pipeline"
}
```

## 4. 标题和结构恢复

标题联合判断：

```text
编号层级
+ 字体相对正文大小
+ 粗体样式
+ 文本长度
+ 是否像完整解释句
+ 页面布局角色
```

编号层级：

```text
2.1.1 → level 3
2.1   → level 2
2.     → 只有短文本且具有明显标题样式时才作为 level 1
```

例如：

```json
[
  "Query模块",
  "2.1 意图识别(Intent Recognition)",
  "2.1.1 基于规则的方法"
]
```

完整步骤说明不会轻易创建 Parent：

```text
3. 向量数据库:存储所有文档的向量表示，并建立高效检索索引……
```

## 5. 规则 + 语义融合 Chunk

默认推荐：

```text
min_tokens    = 180
target_tokens = 500
max_tokens    = 800
overlap        = 80
```

含义：

- 小于 `min_tokens`：优先继续合并；
- 接近 `target_tokens`：在多个安全边界中选择语义低谷；
- `target～max`：允许完整信息保持整体；
- 超过 `max_tokens`：按照类型专用规则强制拆分。

类型专用策略：

- 列表：按完整列表项；
- 表格：按行并重复表头；
- FAQ：每个片段保留问题；
- 代码：按函数、类、代码行组；
- 普通文本：先分句，再做语义边界优化；
- 单句自身超限：最后才按字符兜底。

## 6. 输出文件

```text
output/
├── manifest.json
├── document.json
├── document.md
├── atomic_units.jsonl
├── parents.jsonl
├── children.jsonl
├── indexable_children.jsonl
└── quality_report.json
```

用途：

- `document.md`：人工检查阅读顺序；
- `atomic_units.jsonl`：检查列表、代码、表格、FAQ 等原子单元；
- `parents.jsonl`：命中 Child 后扩展完整章节；
- `children.jsonl`：包含目录、导航和正文的完整 Child；
- `indexable_children.jsonl`：默认可直接写入向量数据库；
- `quality_report.json`：工程和结构质量报告。

## 7. quality_report.json

v0.5 新增或强化：

```json
{
  "retrieval_readiness": "ready",
  "toc_page_count": 1,
  "non_indexable_child_count": 1,
  "non_indexable_child_ratio": 0.02,
  "removed_promotional_block_count": 1,
  "reconstructed_code_block_count": 12,
  "malformed_code_chunk_ratio": 0.0,
  "glued_list_paragraph_ratio": 0.0,
  "mid_sentence_start_ratio": 0.0,
  "broken_line_candidate_ratio": 0.0
}
```

### retrieval_readiness

- `ready`：没有 error 或 warning，可进入检索评测；
- `review`：程序成功，但存在结构或解析警告；
- `blocked`：存在超限、空块、孤儿关系等工程错误。

`info` 类型问题不会把状态降为 `review`。例如：

```json
{
  "severity": "info",
  "code": "NON_INDEXABLE_NAVIGATION_CHUNKS"
}
```

表示目录已被正常排除，并不是失败。

质量统计默认以 `index_enabled=true` 的 Child 为主体，因此目录短块不会污染正文的 token 分布。

## 8. 推荐在线检索流程

```text
用户问题
→ 检索 indexable_children.jsonl 对应的向量记录
→ 可选 Rerank
→ 根据 parent_id 读取 Parent
→ 必要时补充 previous_child_id / next_child_id
→ 交给 LLM
```

向量记录建议保留：

```json
{
  "child_id": "...",
  "parent_id": "...",
  "document_id": "...",
  "page_start": 10,
  "page_end": 12,
  "heading_path": ["Query模块", "Query重写"],
  "content_type": "body",
  "index_enabled": true
}
```

## 9. 审计已有输出

```bash
rag-pdf-audit \
  --children output/children.jsonl \
  --parents output/parents.jsonl \
  --min-tokens 180 \
  --target-tokens 500 \
  --max-tokens 800 \
  -o audit.json
```

比较两次结果：

```bash
rag-pdf-compare \
  --old-parents old/parents.jsonl \
  --old-children old/children.jsonl \
  --new-parents new/parents.jsonl \
  --new-children new/children.jsonl \
  -o comparison.json
```

## 10. 测试

```bash
python -m pip install -e '.[dev]'
pytest
```

v0.5 回归测试覆盖：

- 标题层级与长编号句降级；
- 页眉页脚清除；
- 跨 Block、跨页中文断词；
- 目录页识别和索引排除；
- 推广联系方式检测；
- 列表与后续正文边界；
- 代码坐标缩进恢复；
- 表格文本框误报过滤；
- Parent/Child 关系和硬上限；
- 运行版本与代码指纹。

## 11. 生产部署边界

该项目负责确定性的离线解析和 Chunk。正式上线仍建议接入：

- 对象存储；
- 队列、重试和死信；
- 文档版本与幂等控制；
- 租户和权限隔离；
- 指标、日志和告警；
- 向量数据库批量 upsert；
- 人工抽样和 Recall@K 评测集。
