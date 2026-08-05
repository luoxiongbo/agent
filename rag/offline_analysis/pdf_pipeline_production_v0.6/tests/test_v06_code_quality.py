from __future__ import annotations

from rag_pdf_pipeline.chunker import SmartParentChildChunker
from rag_pdf_pipeline.code_analysis import (
    analyze_python_code,
    repair_code_physical_lines,
    split_python_top_level_blocks,
)
from rag_pdf_pipeline.config import ChunkConfig
from rag_pdf_pipeline.models import AtomicUnit
from rag_pdf_pipeline.semantic import HashingSemanticEncoder
from rag_pdf_pipeline.text_utils import (
    reconstruct_code_from_lines,
    repair_broken_cjk_blank_lines,
)


def _code_unit(text: str) -> AtomicUnit:
    return AtomicUnit(
        unit_id="unit_code",
        document_id="pdf_test",
        kind="code",
        text=text,
        page_start=1,
        page_end=1,
        heading_path=["代码示例"],
        protected=True,
        group_id="group_code",
        metadata={"code_language": "python"},
    )


def test_repair_split_python_identifiers() -> None:
    lines, _ = repair_code_physical_lines(
        [
            'model = load("./fine_tuned_i',
            'ntent_model")',
            "losses = losse",
            "s",
            "relevant_",
            "docs = []",
            "i",
            "f answer:",
        ],
        [72] * 8,
    )
    assert lines == [
        'model = load("./fine_tuned_intent_model")',
        "losses = losses",
        "relevant_docs = []",
        "if answer:",
    ]


def test_coordinate_reconstruction_produces_parseable_python() -> None:
    lines = [
        "def classify(query: str) -> str:",
        "query = query.lower()",
        "intent_keywords = {",
        '"报销": ["费用"],',
        "}",
        "for intent, keywords in intent_keywords.items():",
        "if any(keyword in query for keyword in keywords):",
        "return intent",
        'return "未知意图"',
    ]
    x0s = [72, 90, 90, 108, 90, 90, 108, 126, 90]
    code = reconstruct_code_from_lines(lines, x0s)
    analysis = analyze_python_code(code)
    assert analysis.ast_parseable is True
    assert '            return intent' in code
    assert '    return "未知意图"' in code


def test_python_analysis_detects_bad_indentation() -> None:
    analysis = analyze_python_code("def f():\nreturn 1")
    assert analysis.is_python is True
    assert analysis.ast_parseable is False
    assert analysis.indentation_issue_count > 0


def test_python_top_level_blocks_preserve_functions() -> None:
    code = "import os\n\ndef first():\n    return 1\n\ndef second():\n    return 2"
    blocks = split_python_top_level_blocks(code)
    assert blocks == [
        "import os",
        "def first():\n    return 1",
        "def second():\n    return 2",
    ]


def test_code_chunker_splits_at_function_boundaries() -> None:
    code = (
        "import os\n\n"
        "def first():\n"
        "    value = '账号权限登录安全' * 20\n"
        "    return value\n\n"
        "def second():\n"
        "    value = '服务器网络容器部署' * 20\n"
        "    return value\n"
    )
    chunker = SmartParentChildChunker(
        ChunkConfig(min_tokens=40, target_tokens=90, max_tokens=150, overlap_tokens=0),
        HashingSemanticEncoder(256),
    )
    pieces = chunker._split_code(_code_unit(code), target_tokens=90, max_tokens=150)
    assert len(pieces) >= 2
    assert any("def first" in piece.text for piece in pieces)
    assert any("def second" in piece.text for piece in pieces)
    assert all(piece.metadata["split_strategy"] == "code_top_level" for piece in pieces)
    assert any(piece.metadata["force_child_boundary_before"] for piece in pieces[1:])


def test_oversized_function_repeats_signature() -> None:
    body = "\n".join(f"    value_{index} = '{'知识' * 20}'" for index in range(30))
    code = "def build_context(query):\n" + body + "\n    return query"
    chunker = SmartParentChildChunker(
        ChunkConfig(min_tokens=40, target_tokens=90, max_tokens=130, overlap_tokens=0),
        HashingSemanticEncoder(256),
    )
    pieces = chunker._split_code(_code_unit(code), target_tokens=90, max_tokens=130)
    assert len(pieces) > 1
    assert all("def build_context(query):" in piece.text for piece in pieces)
    assert any("PDF 代码续段" in piece.text for piece in pieces[1:])


def test_high_confidence_cjk_blank_break_is_repaired() -> None:
    text = "在需要回答问题时能翻\n\n阅自己的参考书。\n\n这是新的完整段落。"
    repaired = repair_broken_cjk_blank_lines(text)
    assert "能翻阅自己的参考书" in repaired
    assert "参考书。\n\n这是新的完整段落" in repaired


def test_top_level_return_is_detected_as_function_boundary_violation() -> None:
    analysis = analyze_python_code("return blocks")
    assert analysis.is_python is True
    assert analysis.ast_parseable is False
    assert analysis.function_boundary_violation_count > 0
