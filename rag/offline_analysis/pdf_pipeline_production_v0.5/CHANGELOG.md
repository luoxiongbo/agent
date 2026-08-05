# Changelog

## 0.5.0

### 新增

- 页面角色识别：`body`、`toc`、`cover`、`complex`。
- 目录/导航页默认 `index_enabled=false`。
- 新输出 `indexable_children.jsonl`，只包含建议写入向量库的 Child。
- CLI 参数：
  - `--index-toc-pages`
  - `--keep-promotional-contacts`
  - `--no-code-reconstruction`
- 推广联系方式清理计数。
- 代码行文本与横坐标保留。
- 基于代码行坐标的缩进恢复。
- 相邻等宽代码 TextBlock 合并。
- 压缩代码声明的安全换行恢复。
- 新质量指标：
  - `toc_page_count`
  - `non_indexable_child_count`
  - `non_indexable_child_ratio`
  - `removed_promotional_block_count`
  - `reconstructed_code_block_count`
  - `malformed_code_chunk_ratio`
  - `glued_list_paragraph_ratio`

### 修复

- 目录页标题不再成为后续正文的错误祖先。
- 目录短 Chunk 不再污染正文 token 分布。
- “向量数据库……离线流程……”等列表末项与正文粘连。
- “能翻 / 阅”“销售技 / 巧”“因此候 / 选”等剩余中文断词。
- 正常英文代码 Chunk 不再被半句开头检测误报。
- 正常的 `import x` 换行 `from y import z` 不再被判为坏代码。
- 封面识别限制为第一页，避免把第二页短正文错误排除。

### 行为变化

- `children.jsonl` 仍保存完整 Child，包括目录和导航。
- 向量入库应优先使用 `indexable_children.jsonl`。
- `NON_INDEXABLE_NAVIGATION_CHUNKS` 为 `info`，不会将 `retrieval_readiness` 降为 `review`。
- 质量比例默认基于可索引 Child 计算。
