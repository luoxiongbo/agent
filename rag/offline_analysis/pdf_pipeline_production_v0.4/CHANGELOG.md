# Changelog

## 0.4.0

- 在标题识别前执行保守 TextBlock 预合并，修复同句被拆成多个块。
- 结构层再次复核标题，长编号说明句即使被版面层误判，也不会创建 Parent。
- 修复“例如用户 / 问……”与“这里我 / 们……”等续句合并。
- 加强封面、流程图和幻灯片式复杂页面检测。
- 收紧单级编号标题判断，保留真正的 2.1 / 2.1.1 层级。
- 清理两项以上连续代码行号、Plain Text 和独立项目符号。
- Parent、Child、Atomic Unit 与 manifest 写入 pipeline_version、package_root 和 code_fingerprint。
- 新增 rag-pdf-diagnose，用于确认当前 Python 实际加载的源码。
- 新增相关端到端和结构回归测试。

## 0.3.0

- 初步增加标题层级、复杂版面、噪声和质量审计。
