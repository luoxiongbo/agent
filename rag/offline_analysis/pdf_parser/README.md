# RAG PDF 离线解析器

这是一套不依赖 LangChain 的可运行实现，主要完成：

- 文本型 PDF 提取
- 扫描 PDF 自动 OCR
- PDF 目录和元数据提取
- 基于坐标的阅读顺序
- 重复页眉、页脚和页码清洗
- 标题层级识别
- 原生表格提取并转 Markdown
- 面向 RAG 的 Chunk 切分
- 输出 `document.json`、`document.md` 和 `chunks.jsonl`

## 1. 环境

建议 Python 3.11 或 3.12。普通文字 PDF 只安装 Python 依赖即可。

```bash
cd rag_pdf_parser
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## 2. OCR 安装

只有扫描版 PDF 才需要 Tesseract。

### macOS

```bash
brew install tesseract
brew install tesseract-lang
```

验证中文和英文语言包：

```bash
tesseract --list-langs | grep -E 'chi_sim|eng'
```

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng
```

验证：

```bash
tesseract --version
tesseract --list-langs
```

如果 PyMuPDF 找不到语言数据，可显式传入：

```bash
python rag_pdf_parser.py input.pdf \
  --tessdata /你的/tessdata/目录
```

也可设置：

```bash
export TESSDATA_PREFIX=/你的/tessdata/目录
```

## 3. 最简单运行

```bash
python rag_pdf_parser.py ./input.pdf -o ./output -v
```

默认行为：

- 页面原生文字少于 30 个字符时自动 OCR；
- OCR 语言为简体中文加英文；
- 自动尝试提取有边框的原生 PDF 表格；
- Chunk 上限约 500 token，重叠约 80 token。

## 4. 常用命令

只解析原生文字，不使用 OCR：

```bash
python rag_pdf_parser.py input.pdf -o output --ocr never
```

强制所有页 OCR：

```bash
python rag_pdf_parser.py input.pdf -o output \
  --ocr always \
  --ocr-language chi_sim+eng \
  --ocr-dpi 300
```

无边框表格可尝试文本策略：

```bash
python rag_pdf_parser.py input.pdf -o output \
  --table-strategy text
```

关闭表格检测以提高速度：

```bash
python rag_pdf_parser.py input.pdf -o output --no-tables
```

修改 Chunk 参数：

```bash
python rag_pdf_parser.py input.pdf -o output \
  --chunk-size 700 \
  --chunk-overlap 100
```

加密 PDF：

```bash
python rag_pdf_parser.py input.pdf -o output \
  --password 'your-password'
```

## 5. 输出

### `document.json`

保存完整文档结构：

- 文档 ID、SHA-256
- PDF 元数据与目录
- 每页尺寸、OCR 状态
- 文本块坐标、字体大小、标题层级
- 表格数据
- 清洗后的页面文本

### `document.md`

适合人工检查解析质量，包含页码注释、标题和 Markdown 表格。

### `chunks.jsonl`

每一行是一个可写入向量数据库的 Chunk：

```json
{
  "chunk_id": "pdf_xxx_chunk_000000",
  "document_id": "pdf_xxx",
  "content": "文档：手册.pdf\n章节：用户管理\n页码：3\n\n...",
  "text": "...",
  "token_estimate": 438,
  "page_start": 3,
  "page_end": 4,
  "heading_path": ["用户管理"],
  "metadata": {
    "source": "/path/手册.pdf",
    "file_name": "手册.pdf"
  }
}
```

其中 `content` 适合拿去计算 Embedding，`text` 是不含来源前缀的正文。

## 6. Python 中直接调用

```python
from rag_pdf_parser import (
    PDFParser,
    ParserConfig,
    RAGChunker,
    save_outputs,
)

config = ParserConfig(
    ocr_mode="auto",
    ocr_language="chi_sim+eng",
    chunk_size_tokens=500,
    chunk_overlap_tokens=80,
)

document = PDFParser(config).parse("input.pdf")
chunks = RAGChunker(config).chunk(document)
save_outputs(document, chunks, "output")

for chunk in chunks[:3]:
    print(chunk.chunk_id)
    print(chunk.content)
    print("=" * 80)
```

## 7. 重要限制

1. `Page.find_tables()` 主要适合带线框或原生文字的 PDF 表格。
2. 扫描图片里的复杂表格，Tesseract 能识别文字，但不会稳定还原单元格结构；此类场景应换用专门的版面/表格模型。
3. 双栏论文的阅读顺序依赖 PDF 内部坐标，建议检查 `document.md`。
4. `estimate_tokens()` 只是模型无关的估算。确定 Embedding 模型后，建议替换为对应 tokenizer。
5. 页眉页脚删除采用跨页重复统计，极短文档可能无法可靠识别。
