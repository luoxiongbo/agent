from __future__ import annotations

import pymupdf

from rag_pdf_pipeline.atomic import AtomicUnitBuilder
from rag_pdf_pipeline.config import ParserConfig
from rag_pdf_pipeline.models import PageData, ParsedDocument, TextBlock
from rag_pdf_pipeline.pdf_parser import PDFParser, _infer_heading
from rag_pdf_pipeline.text_utils import clean_code_text, clean_extracted_text


def _block(
    text: str,
    page: int = 1,
    y0: float = 100,
    y1: float = 120,
    font: float = 12,
    bold: bool = False,
    level: int | None = None,
    x0: float = 72,
    x1: float = 500,
    no: int = 0,
) -> TextBlock:
    return TextBlock(
        page_number=page,
        bbox=(x0, y0, x1, y1),
        text=text,
        block_no=no,
        source="native",
        max_font_size=font,
        avg_font_size=font,
        bold=bold,
        heading_level=level,
    )


def _document(pages: list[PageData]) -> ParsedDocument:
    return ParsedDocument(
        document_id="pdf_test",
        source_path="/tmp/test.pdf",
        file_name="test.pdf",
        sha256="abc",
        page_count=len(pages),
        metadata={},
        toc=[],
        body_font_size=12,
        removed_header_footer_patterns=[],
        pages=pages,
    )


def test_numeric_heading_hierarchy_and_list_item_rejection() -> None:
    config = ParserConfig()
    page = PageData(
        page_number=1,
        width=595,
        height=842,
        rotation=0,
        used_ocr=False,
        native_text_char_count=200,
    )

    level, _ = _infer_heading(
        _block("2.1 Query模块", font=14, bold=True),
        body_font_size=12,
        page=page,
        largest_font=14,
        config=config,
    )
    assert level == 2

    level, _ = _infer_heading(
        _block("2.1.1 基于规则的方法", font=14, bold=True),
        body_font_size=12,
        page=page,
        largest_font=14,
        config=config,
    )
    assert level == 3

    level, _ = _infer_heading(
        _block(
            "3. 向量数据库:存储所有文档的向量表示,并建立高效检索索引",
            font=12,
            bold=False,
        ),
        body_font_size=12,
        page=page,
        largest_font=14,
        config=config,
    )
    assert level is None


def test_atomic_builder_preserves_real_heading_stack_without_placeholders() -> None:
    page = PageData(
        page_number=1,
        width=595,
        height=842,
        rotation=0,
        used_ocr=False,
        native_text_char_count=200,
        blocks=[
            _block("2.1 Query模块", font=16, bold=True, level=2, no=1),
            _block("Query 模块介绍。", y0=140, y1=170, no=2),
            _block("2.1.1 基于规则的方法", y0=200, y1=225, font=15, bold=True, level=3, no=3),
            _block("规则方法通过关键词匹配实现。", y0=240, y1=280, no=4),
        ],
    )
    sections = AtomicUnitBuilder(ParserConfig()).build(_document([page]))
    assert len(sections) == 2
    assert sections[0].heading_path == ["2.1 Query模块"]
    assert sections[1].heading_path == ["2.1 Query模块", "2.1.1 基于规则的方法"]
    assert "未命名章节" not in " ".join(sections[1].heading_path)


def test_cross_page_broken_sentence_is_merged() -> None:
    config = ParserConfig(merge_cross_page_continuations=True)
    page1 = PageData(
        page_number=1,
        width=595,
        height=842,
        rotation=0,
        used_ocr=False,
        native_text_char_count=100,
        blocks=[
            _block("1.1 示例", level=2, font=16, bold=True, no=1),
            _block("我们直", page=1, y0=760, y1=810, no=2),
        ],
    )
    page2 = PageData(
        page_number=2,
        width=595,
        height=842,
        rotation=0,
        used_ocr=False,
        native_text_char_count=100,
        blocks=[
            _block("接使用本地模型进行向量化。", page=2, y0=40, y1=80, no=1),
        ],
    )
    sections = AtomicUnitBuilder(config).build(_document([page1, page2]))
    assert len(sections) == 1
    assert len(sections[0].units) == 1
    assert sections[0].units[0].text == "我们直接使用本地模型进行向量化。"
    assert sections[0].units[0].page_end == 2


def test_complex_layout_only_accepts_dominant_top_title() -> None:
    config = ParserConfig(complex_layout_strategy="conservative")
    page = PageData(
        page_number=1,
        width=595,
        height=842,
        rotation=0,
        used_ocr=False,
        native_text_char_count=100,
        layout_kind="complex",
    )
    title = _block("RAG流程图解", y0=30, y1=70, font=24, bold=True)
    label = _block("1.3 在线查询处理流程", y0=250, y1=275, font=15, bold=True)

    title_level, _ = _infer_heading(title, 12, page, 24, config)
    label_level, _ = _infer_heading(label, 12, page, 24, config)
    assert title_level == 1
    assert label_level is None


def test_table_quality_rejects_textbox_like_layout() -> None:
    parser = PDFParser(ParserConfig())
    rows = [
        ["RAG就像AI的开卷考试模式，这是一整段很长的叙述文字。" * 5, "用户query"],
        ["翻阅自己的参考书找到资料，然后用自己的语言组织回答。" * 5, "知识库"],
    ]
    score, reasons = parser._score_table(rows, (10, 10, 580, 800), pymupdf.Rect(0, 0, 595, 842))
    assert score < parser.config.min_table_quality_score
    assert reasons


def test_ui_noise_cleanup() -> None:
    text = clean_extracted_text("Plain Text\n●\n正文内容\n1 2 3 4 5 6")
    assert text == "正文内容"
    code = clean_code_text(["Plain Text", "1 2 3 4 5", "def f():", "    return 1"])
    assert "Plain Text" not in code
    assert "1 2 3" not in code
    assert "def f():" in code
