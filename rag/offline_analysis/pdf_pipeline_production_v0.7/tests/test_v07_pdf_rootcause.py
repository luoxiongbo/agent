from __future__ import annotations

from rag_pdf_pipeline.code_analysis import (
    analyze_python_code,
    normalize_python_for_analysis,
    should_join_code_physical_lines,
)
from rag_pdf_pipeline.text_utils import reconstruct_code_from_lines


def test_preserves_pdf_leading_spaces_instead_of_guessing_from_x0() -> None:
    lines = [
        "def classify(query):",
        "    if query:",
        "        return query",
        "    return None",
    ]
    # Chromium/Skia PDF code panels frequently give every line the same x0.
    text = reconstruct_code_from_lines(
        lines,
        [91.0, 91.0, 91.0, 91.0],
        x1s=[210.0, 190.0, 220.0, 180.0],
        block_x1=560.0,
    )
    assert text == "\n".join(lines)
    assert analyze_python_code(text).ast_parseable


def test_real_newline_does_not_become_breakif() -> None:
    assert not should_join_code_physical_lines(
        "                break",
        "        if rank != 0:",
        left_x1=224.0,
        block_x1=559.0,
    )


def test_comment_does_not_glue_to_following_if_statement() -> None:
    assert not should_join_code_physical_lines(
        "            # 遇到新标题块，先结束上一chunk",
        '            if current_chunk["text"]:',
        left_x1=344.0,
        block_x1=540.0,
    )


def test_right_edge_visual_wrap_still_joins_identifier() -> None:
    assert should_join_code_physical_lines(
        'model = load("./fine_tuned_i',
        'ntent_model")',
        left_x1=558.0,
        block_x1=559.0,
    )


def test_presentation_labels_are_normalized_without_mutating_source() -> None:
    source = '''from transformers import AutoTokenizer
对用户查询进行分类
query = "怎么报销保险费用？"
'''.strip()
    analysis = analyze_python_code(source)
    assert analysis.ast_parseable  # Bare CJK labels are legal Python names.
    assert analysis.code_type == "python_normalized"
    assert analysis.narrative_line_count == 1
    assert analysis.normalized_ast_parseable
    assert analysis.normalized_source is not None
    assert "# 对用户查询进行分类" in analysis.normalized_source
    assert "对用户查询进行分类" in source


def test_parenthesized_chinese_presentation_label_becomes_comment() -> None:
    normalized = normalize_python_for_analysis(
        "初始化句向量模型（例如中文多语言模型）\nembed_model = build_model()"
    )
    assert normalized.startswith("# 初始化句向量模型")
