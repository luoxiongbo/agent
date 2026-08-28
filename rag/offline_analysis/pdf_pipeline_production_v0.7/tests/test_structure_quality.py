from __future__ import annotations

import pymupdf

from rag_pdf_pipeline.atomic import AtomicUnitBuilder
from rag_pdf_pipeline.config import ParserConfig
from rag_pdf_pipeline.models import PageData, ParsedDocument, TextBlock
from rag_pdf_pipeline.pdf_parser import PDFParser, _infer_heading
from rag_pdf_pipeline.text_utils import (
    clean_code_text,
    clean_extracted_text,
    is_promotional_contact_line,
    reconstruct_code_from_lines,
    smart_join_text,
)


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


def test_structure_layer_demotes_long_numbered_sentence_and_repairs_continuation() -> None:
    page = PageData(
        page_number=1,
        width=595,
        height=842,
        rotation=0,
        used_ocr=False,
        native_text_char_count=200,
        blocks=[
            TextBlock(
                page_number=1,
                bbox=(72, 100, 520, 125),
                text="1. 多轮对话的语义连贯性不足:在用户多轮提问时,如果新问题是对上一轮回答的跟进(例如用户",
                block_no=1,
                source="native",
                max_font_size=14,
                avg_font_size=12,
                bold=True,
                heading_level=1,
                heading_source="single_numeric_styled",
            ),
            _block(
                "问:“这个怎么申请?”),系统需要理解“这个”指代什么。",
                y0=126,
                y1=150,
                no=2,
            ),
        ],
    )
    sections = AtomicUnitBuilder(ParserConfig()).build(_document([page]))
    assert len(sections) == 1
    assert sections[0].heading_path == []
    assert len(sections[0].units) == 1
    text = sections[0].units[0].text
    assert "例如用户问" in text
    assert sections[0].units[0].metadata.get("demoted_heading") is True


def test_first_page_slide_layout_is_detected_with_five_short_labels() -> None:
    parser = PDFParser(ParserConfig())
    page = PageData(
        page_number=1,
        width=595,
        height=842,
        rotation=0,
        used_ocr=False,
        native_text_char_count=100,
        blocks=[
            _block("RAG流程图解", x0=220, x1=380, y0=30, y1=60, font=24, no=1),
            _block("Query模块", x0=40, x1=150, y0=200, y1=225, no=2),
            _block("离线解析模块", x0=220, x1=340, y0=200, y1=225, no=3),
            _block("检索召回模块", x0=400, x1=540, y0=200, y1=225, no=4),
            _block("生成阶段优化", x0=220, x1=350, y0=400, y1=425, no=5),
        ],
    )
    parser._classify_page_layouts([page])
    assert page.layout_kind == "complex"



def test_toc_page_is_non_indexable() -> None:
    parser = PDFParser(ParserConfig(toc_min_numbered_entries=5))
    blocks = [
        _block("RAG课程目录", x0=200, x1=400, y0=30, y1=65, font=24, bold=True, no=1),
    ]
    for index, title in enumerate(
        [
            "1.1 整体流程",
            "1.2 离线处理流程",
            "1.3 在线处理流程",
            "2.1 Query模块",
            "2.2 Query重写",
            "2.3 Query扩写",
            "3.1 离线解析",
        ],
        start=2,
    ):
        blocks.append(
            _block(
                title,
                x0=50 + (index % 3) * 150,
                x1=190 + (index % 3) * 150,
                y0=90 + index * 35,
                y1=112 + index * 35,
                no=index,
            )
        )
    page = PageData(
        page_number=1,
        width=595,
        height=842,
        rotation=0,
        used_ocr=False,
        native_text_char_count=200,
        blocks=blocks,
    )
    parser._classify_page_layouts([page])
    assert page.page_role == "toc"
    assert page.index_enabled is False


def test_promotional_contact_detection_is_conservative() -> None:
    assert is_promotional_contact_line("更多课程联系微信:1978201801")
    assert is_promotional_contact_line("扫码关注公众号 rag-course")
    assert not is_promotional_contact_line("如需技术支持，请联系管理员。")


def test_code_reconstruction_uses_line_coordinates() -> None:
    text = reconstruct_code_from_lines(
        [
            "import openai",
            "from sentence_transformers import SentenceTransformer",
            "def retrieve(query):",
            "embedding = model.encode(query)",
            "return embedding",
        ],
        [72, 72, 72, 96, 96],
        indent_spaces=4,
    )
    assert "import openai\nfrom sentence_transformers" in text
    assert "def retrieve(query):\n    embedding" in text
    assert "\n    return embedding" in text


def test_list_tail_and_following_paragraph_keep_boundary() -> None:
    left = (
        "离线处理主要包括:\n"
        "1. 知识文档库:建立知识基础\n"
        "2. 文档向量化:转换为向量\n"
        "3. 向量数据库:建立检索索引"
    )
    joined = smart_join_text(left, "离线流程只需执行一次或定期更新。")
    assert "检索索引\n\n离线流程" in joined
    assert "检索索引离线流程" not in joined


def test_short_cjk_word_fragment_is_repaired() -> None:
    assert smart_join_text("用户可以翻", "阅自己的参考书。") == "用户可以翻阅自己的参考书。"
    assert smart_join_text("提高销售技", "巧。") == "提高销售技巧。"


def test_code_dominant_page_is_not_classified_as_complex_layout() -> None:
    parser = PDFParser(ParserConfig())
    blocks = [_block("6.1 Code", y0=30, y1=60, font=20, bold=True, no=1)]
    for index, text in enumerate(
        [
            "import os",
            "def first(query):",
            "normalized = query.strip()",
            "for item in range(10):",
            "values.append(item)",
            "return values",
        ],
        start=2,
    ):
        block = _block(text, x0=72 + (index % 3) * 18, y0=80 + index * 22, y1=98 + index * 22, no=index)
        block.monospace = True
        blocks.append(block)
    page = PageData(
        page_number=1,
        width=595,
        height=842,
        rotation=0,
        used_ocr=False,
        native_text_char_count=200,
        blocks=blocks,
    )
    parser._classify_page_layouts([page])
    assert page.layout_kind == "normal"
