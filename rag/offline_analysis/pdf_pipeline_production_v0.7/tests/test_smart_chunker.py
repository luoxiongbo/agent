from __future__ import annotations

from rag_pdf_pipeline.atomic import AtomicUnitBuilder
from rag_pdf_pipeline.chunker import SmartParentChildChunker
from rag_pdf_pipeline.config import ChunkConfig
from rag_pdf_pipeline.models import PageData, ParsedDocument, TableData, TextBlock
from rag_pdf_pipeline.semantic import HashingSemanticEncoder
from rag_pdf_pipeline.text_utils import table_to_markdown


def make_document(blocks: list[TextBlock], tables: list[TableData] | None = None) -> ParsedDocument:
    return ParsedDocument(
        document_id="pdf_test",
        source_path="/tmp/test.pdf",
        file_name="test.pdf",
        sha256="abc",
        page_count=1,
        metadata={},
        toc=[],
        body_font_size=12,
        removed_header_footer_patterns=[],
        pages=[
            PageData(
                page_number=1,
                width=595,
                height=842,
                rotation=0,
                used_ocr=False,
                native_text_char_count=100,
                blocks=blocks,
                tables=tables or [],
            )
        ],
    )


def test_long_list_is_split_by_items_and_keeps_parent_relation() -> None:
    heading = TextBlock(
        page_number=1,
        bbox=(72, 80, 500, 110),
        text="1. 修改密码",
        block_no=0,
        source="native",
        max_font_size=20,
        avg_font_size=20,
        bold=True,
        heading_level=1,
    )
    items = "\n".join(
        f"{index}. 这是第 {index} 个完整操作步骤，需要保留步骤含义和上下文。"
        for index in range(1, 36)
    )
    body = TextBlock(
        page_number=1,
        bbox=(72, 130, 500, 700),
        text="请依次完成以下步骤：\n" + items,
        block_no=1,
        source="native",
        max_font_size=12,
        avg_font_size=12,
    )
    document = make_document([heading, body])
    sections = AtomicUnitBuilder().build(document)

    config = ChunkConfig(
        min_tokens=70,
        target_tokens=120,
        max_tokens=180,
        overlap_tokens=20,
    )
    parents, children, units = SmartParentChildChunker(
        config,
        HashingSemanticEncoder(512),
    ).chunk(document, sections)

    assert len(parents) == 1
    assert len(children) > 1
    assert all(child.parent_id == parents[0].parent_id for child in children)
    assert all(child.token_estimate <= config.max_tokens for child in children)
    assert all("修改密码" in child.content for child in children)
    assert any(unit.metadata.get("split_strategy") == "list_items" for unit in units)


def test_large_table_repeats_header() -> None:
    heading = TextBlock(
        page_number=1,
        bbox=(72, 80, 500, 110),
        text="2. 产品价格",
        block_no=0,
        source="native",
        max_font_size=20,
        avg_font_size=20,
        heading_level=1,
    )
    rows = [["产品", "版本", "说明"]] + [
        [f"产品-{index}", f"v{index}", "这是较长的产品说明，用于验证表格按行拆分并重复表头。"]
        for index in range(1, 31)
    ]
    table = TableData(
        page_number=1,
        table_index=1,
        bbox=(72, 140, 520, 750),
        rows=rows,
        markdown=table_to_markdown(rows),
    )
    document = make_document([heading], [table])
    sections = AtomicUnitBuilder().build(document)

    config = ChunkConfig(
        min_tokens=70,
        target_tokens=130,
        max_tokens=190,
        overlap_tokens=0,
    )
    _, children, units = SmartParentChildChunker(
        config,
        HashingSemanticEncoder(512),
    ).chunk(document, sections)

    table_pieces = [unit for unit in units if unit.kind == "table"]
    assert len(table_pieces) > 1
    assert all("| 产品 | 版本 | 说明 |" in piece.text for piece in table_pieces)
    assert all(child.token_estimate <= config.max_tokens for child in children)



def test_many_tiny_units_are_not_forced_into_single_item_chunks() -> None:
    heading = TextBlock(
        page_number=1,
        bbox=(72, 80, 500, 110),
        text="3. 短段落集合",
        block_no=0,
        source="native",
        max_font_size=20,
        avg_font_size=20,
        heading_level=1,
    )
    blocks = [heading]
    for index in range(1, 61):
        blocks.append(
            TextBlock(
                page_number=1,
                bbox=(72, 110 + index * 5, 500, 115 + index * 5),
                text=f"用户管理说明 {index}。",
                block_no=index,
                source="native",
                max_font_size=12,
                avg_font_size=12,
            )
        )

    document = make_document(blocks)
    sections = AtomicUnitBuilder().build(document)
    config = ChunkConfig(
        min_tokens=80,
        target_tokens=120,
        max_tokens=170,
        overlap_tokens=0,
        boundary_lookback_units=8,
    )
    _, children, _ = SmartParentChildChunker(
        config,
        HashingSemanticEncoder(256),
    ).chunk(document, sections)

    assert len(children) < 20
    assert all(child.token_estimate <= config.max_tokens for child in children)
    assert any(len(child.unit_ids) > 3 for child in children)


def test_single_oversized_table_row_still_obeys_hard_limit() -> None:
    heading = TextBlock(
        page_number=1,
        bbox=(72, 80, 500, 110),
        text="4. 超长表格",
        block_no=0,
        source="native",
        max_font_size=20,
        avg_font_size=20,
        heading_level=1,
    )
    rows = [
        ["字段", "说明"],
        ["payload", "这是一个非常长的字段说明。" * 120],
    ]
    table = TableData(
        page_number=1,
        table_index=1,
        bbox=(72, 140, 520, 750),
        rows=rows,
        markdown=table_to_markdown(rows),
    )
    document = make_document([heading], [table])
    sections = AtomicUnitBuilder().build(document)
    config = ChunkConfig(
        min_tokens=60,
        target_tokens=110,
        max_tokens=160,
        overlap_tokens=0,
    )
    _, children, units = SmartParentChildChunker(
        config,
        HashingSemanticEncoder(256),
    ).chunk(document, sections)

    assert len(children) > 1
    assert all(child.token_estimate <= config.max_tokens for child in children)
    assert all(unit.text.strip() for unit in units)



def test_semantic_boundary_can_split_before_hard_max() -> None:
    heading = TextBlock(
        page_number=1,
        bbox=(72, 80, 500, 110),
        text="5. 混合主题",
        block_no=0,
        source="native",
        max_font_size=20,
        avg_font_size=20,
        heading_level=1,
    )
    user_text = "用户账号密码角色权限登录安全设置。" * 12
    server_text = "服务器磁盘网络端口容器部署监控告警。" * 12
    blocks = [
        heading,
        TextBlock(
            page_number=1,
            bbox=(72, 130, 500, 300),
            text=user_text,
            block_no=1,
            source="native",
            max_font_size=12,
            avg_font_size=12,
        ),
        TextBlock(
            page_number=1,
            bbox=(72, 320, 500, 500),
            text=server_text,
            block_no=2,
            source="native",
            max_font_size=12,
            avg_font_size=12,
        ),
    ]
    document = make_document(blocks)
    sections = AtomicUnitBuilder().build(document)
    config = ChunkConfig(
        min_tokens=60,
        target_tokens=180,
        max_tokens=360,
        overlap_tokens=0,
    )
    _, children, _ = SmartParentChildChunker(
        config,
        HashingSemanticEncoder(512),
    ).chunk(document, sections)

    assert len(children) >= 2
    assert all(child.token_estimate <= config.max_tokens for child in children)
