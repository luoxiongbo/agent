from __future__ import annotations

from rag_pdf_pipeline.audit import audit_existing_chunks


def test_audit_marks_fragmented_flat_output_not_ready() -> None:
    parents = [
        {
            "parent_id": f"p{i}",
            "title": (
                f"{i}. 这是一个很长的步骤说明:用于验证列表项被误识别为标题并创建了Parent"
                if i % 2 == 0
                else f"2.1.{i} 子章节"
            ),
        }
        for i in range(1, 11)
    ]
    children = [
        {
            "child_id": f"c{i}",
            "parent_id": f"p{i}",
            "token_estimate": 40 if i < 5 else 220,
            "heading_path": [parents[i - 1]["title"]],
            "text": "们直接使用模型。" if i == 1 else "这是正文。",
        }
        for i in range(1, 11)
    ]
    result = audit_existing_chunks(children, parents)
    assert result["verdict"] == "not_ready"
    assert result["single_child_parent_ratio"] == 1.0
    assert result["flat_heading_path_ratio"] == 1.0
    assert result["short_child_ratio_lt_100"] == 0.4
