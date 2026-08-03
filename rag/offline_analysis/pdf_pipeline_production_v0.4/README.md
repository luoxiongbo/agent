# RAG PDF Pipeline v0.3

生产导向的 PDF 离线解析、结构恢复、智能 Chunk 和质量审计工具。

```text
PDF
→ 原生文本 / OCR
→ 页眉页脚与界面噪声清洗
→ 表格候选质量过滤
→ 复杂版面识别
→ 标题层级恢复
→ 跨 Block / 跨页续句合并
→ 原子信息单元
→ 规则 + 语义融合切分
→ Parent / Child Chunk
→ 可解释质量报告
```

## v0.3 解决的问题

本版本针对真实 PDF 解析结果中常见的以下问题进行了修复：

- `2.1.1` 被宽泛正则提前匹配成一级标题；
- `3. 向量数据库：……` 这类列表步骤被误识别成 Parent 标题；
- 所有 `heading_path` 都只有一层；
- 图解/幻灯片页面的多个短标签被错误配对成独立 Parent；
- `增\n\n强`、`我\n\n们` 等跨 TextBlock 或跨页断词；
- 普通文本框被 `find_tables()` 误识别为表格；
- `Plain Text`、独立 `●`、代码框行号混入知识库；
- `quality_report.json` 只有 min/median/max，无法判断结构质量。

## 1. 安装

建议 Python 3.11 或 3.12。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

使用 Sentence Transformers：

```bash
python -m pip install -e '.[semantic]'
```

生产环境建议提前下载模型，运行时使用本地目录并保持：

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

推荐使用配置文件：

```bash
rag-pdf input.pdf -o output --config config.example.json -v
```

复杂流程图或幻灯片式页面有三种策略：

```bash
# 推荐默认值：保留页面内容，但只允许明显的顶部大标题创建章节
--complex-layout-strategy conservative

# 不特殊处理
--complex-layout-strategy keep

# 完全跳过复杂版面页
--complex-layout-strategy skip
```

## 4. 标题识别策略

标题不再只靠正则，而是联合判断：

```text
编号层级
+ 字体相对正文的大小
+ 是否粗体
+ 文本长度
+ 是否像完整解释句
+ 是否位于复杂图解页面
```

编号规则从具体到宽泛：

```text
2.1.1 → level 3
2.1   → level 2
2.     → 只有短文本且有明显标题样式时才判为 level 1
```

标题栈不再填充“未命名章节”。例如文档从 `2.1` 开始时：

```json
["2.1 Query模块"]
```

其子标题为：

```json
["2.1 Query模块", "2.1.1 基于规则的方法"]
```

翻页重复出现的同名章节标题不会重复创建 Parent。

## 5. 断行和断词修复

同页 TextBlock 合并要求：

- 字号接近；
- 横向位置接近；
- 垂直间距接近正常行距；
- 两块都不是标题、列表或代码；
- 前一块没有完整句末标点。

跨页合并还要求：

- 前一块靠近页底；
- 后一块靠近下一页顶部；
- heading_path 相同；
- 后一块表现为续句。

例如：

```text
我们直 + 接使用本地模型
```

会恢复为：

```text
我们直接使用本地模型
```

## 6. 表格误报过滤

`Page.find_tables()` 的候选不会直接入库。系统会检查：

- 行数和列数；
- 非空单元格比例；
- 行列结构一致性；
- 单行单元格比例；
- 超长叙述单元格比例；
- 表格占页面面积；
- 是否更像覆盖整页的文本框。

低于 `min_table_quality_score` 的候选会被拒绝，数量写入：

```json
"rejected_table_count": 5
```

## 7. 双阈值 Chunk

```text
min_tokens    = 180
目标 target   = 500
硬上限 max    = 800
```

- 小于 min：优先继续合并；
- 接近 target：优先在语义低谷切分；
- target～max：允许完整信息保留；
- 超过 max：使用类型专用策略强制拆分。

类型专用策略：

- 列表：按完整列表项切；
- 表格：按行切并重复表头；
- FAQ：每个片段重复问题；
- 代码：优先函数、类、SQL 或代码行边界；
- 普通文本：分句后使用相邻语义边界；
- 单句自身超限：最后才使用字符级兜底。

## 8. 输出

```text
output/
├── manifest.json
├── document.json
├── document.md
├── atomic_units.jsonl
├── parents.jsonl
├── children.jsonl
└── quality_report.json
```

`children.jsonl` 用于向量检索，`parents.jsonl` 用于命中后扩展完整章节。

## 9. 新版 quality_report.json

除了工程合法性，还包含结构质量指标：

```json
{
  "passed": true,
  "retrieval_readiness": "review",
  "short_child_ratio_lt_100": 0.08,
  "under_min_tokens_ratio": 0.15,
  "single_child_parent_ratio": 0.35,
  "flat_heading_path_ratio": 0.40,
  "suspicious_heading_ratio": 0.02,
  "mid_sentence_start_ratio": 0.0,
  "broken_line_candidate_ratio": 0.01,
  "child_token_p10": 190,
  "child_token_p90": 650,
  "issues": []
}
```

### retrieval_readiness

- `ready`：没有结构告警，可以进入检索评测；
- `review`：程序成功，但存在碎片化、复杂版面或标题层级等告警；
- `blocked`：存在空块、超限、孤儿 Child 或断裂链接等错误。

`passed=true` 仅表示没有工程错误。是否适合检索，以 `retrieval_readiness` 和真实 Recall@K 评测为准。

### 建议观察值

| 指标 | 推荐目标 |
|---|---:|
| `short_child_ratio_lt_100` | `< 10%` |
| `under_min_tokens_ratio` | `< 25%` |
| `single_child_parent_ratio` | `< 60%`，最好 `< 40%` |
| `suspicious_heading_ratio` | `< 12%`，最好接近 0 |
| `mid_sentence_start_ratio` | `< 3%`，最好为 0 |
| `broken_line_candidate_ratio` | `< 10%`，最好接近 0 |
| `child_token_max` | `<= max_tokens` |

## 10. 审计已有 JSONL

无需重新解析 PDF，也可以检查旧版输出：

```bash
rag-pdf-audit \
  --children output/children.jsonl \
  --parents output/parents.jsonl \
  --min-tokens 180 \
  --target-tokens 500 \
  --max-tokens 800 \
  -o audit.json
```

结果包含：

- 最短 Child 样例；
- 疑似错误标题；
- 半句话开头；
- 中文断行候选；
- `ready / review / not_ready` 结论。

仓库中的 `OLD_OUTPUT_AUDIT.json` 是对本次旧结果的实际审计示例。

## 11. 测试

```bash
python -m pip install -e '.[dev]'
pytest
```

当前测试覆盖：

- 标题正则层级和列表项误判；
- heading_path 层级恢复；
- 复杂图解页标题抑制；
- 同页和跨页续句合并；
- 文本框式假表格拒绝；
- UI 噪声清洗；
- 长列表和长表格类型专用切分；
- Child 硬上限；
- Parent/Child 链接；
- 旧 JSONL 审计结论；
- 端到端 CLI 输出。

## 12. 最终判断仍需检索评测

质量报告只能判断解析和切分健康度。真正上线前应准备 30～100 个问题，评估：

```text
Recall@1
Recall@3
Recall@5
MRR
正确章节命中率
答案完整率
```

推荐至少达到：

```text
Recall@5 >= 90%
Recall@3 >= 80%
MRR >= 0.70
```


## v0.4 运行版本自检

先确认当前虚拟环境实际加载的是 v0.4：

```bash
rag-pdf --version
rag-pdf-diagnose
```

输出中必须包含：

```json
{
  "pipeline_version": "0.4.0",
  "code_fingerprint": "...",
  "package_root": ".../rag_pdf_pipeline_production_v0.4/src/rag_pdf_pipeline"
}
```

每次解析生成的 `manifest.json`、`parents.jsonl` 和 `children.jsonl` 也会携带相同信息。若 `package_root` 指向旧目录，应删除旧虚拟环境并重新安装。

推荐使用当前 Python 直接运行，避免命令入口残留：

```bash
python -m rag_pdf_pipeline.cli input.pdf -o output_v04 -v
```

## v0.4 重点修复

结构层不会只相信 PDF 字号判断。类似下面的内容即使被误标为标题：

```text
1. 多轮对话的语义连贯性不足：在用户多轮提问时……例如用户
问：“这个怎么申请？”……
```

也会被降级为正文并合并，避免生成一个长句 Parent 和一个半句 Child。流程图页面中的 `Query模块`、`离线解析模块` 等短标签会保留在同一复杂页面语境，不再各自创建 Parent。

## 比较两次解析结果

```bash
rag-pdf-compare \
  --old-parents old/parents.jsonl \
  --old-children old/children.jsonl \
  --new-parents new/parents.jsonl \
  --new-children new/children.jsonl \
  -o comparison.json
```

当 `behaviorally_identical=true` 时，表示忽略文件来源路径和运行版本元数据后，两次输出内容完全相同。
