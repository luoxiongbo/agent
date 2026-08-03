# Changelog

## 0.3.0

- 修正多级编号标题匹配优先级；
- 标题检测改为编号、字体、粗体、长度、句式和版面联合判断；
- 单级编号长解释句不再默认创建 Parent；
- 使用真实层级栈生成 heading_path，不再插入“未命名章节”；
- 重复章节标题不再跨页重复建 Parent；
- 新增复杂图解/幻灯片页面识别与三种处理策略；
- 新增同页 TextBlock 和跨页续句合并；
- 新增 Plain Text、独立项目符号、代码框行号清洗；
- 代码块保留换行；
- 表格候选新增质量评分，过滤文本框式误报；
- quality_report 新增分位数、短块率、单 Child Parent 比例、标题结构、半句开头和断行指标；
- 新增 retrieval_readiness：ready / review / blocked；
- 新增 rag-pdf-audit，可审计既有 parents.jsonl / children.jsonl；
- 新增 7 项结构与质量回归测试，测试总数 13。
