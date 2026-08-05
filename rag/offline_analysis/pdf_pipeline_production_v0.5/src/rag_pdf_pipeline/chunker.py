from __future__ import annotations

import logging
import re
import statistics
from dataclasses import replace
from typing import Sequence

import numpy as np

from .config import ChunkConfig
from .models import AtomicUnit, ChildChunk, ParentChunk, ParsedDocument, Section
from .semantic import SemanticEncoder, adjacent_similarities
from .text_utils import (
    estimate_tokens,
    hard_split_text,
    split_sentences,
    stable_id,
    table_to_markdown,
)

LOGGER = logging.getLogger(__name__)

_Q_SPLIT_RE = re.compile(
    r"^(?P<question>(?:Q(?:uestion)?|问(?:题)?)[：:].*?)(?P<answer>(?:A(?:nswer)?|答(?:案)?)[：:].*)$",
    re.IGNORECASE | re.DOTALL,
)


class SmartParentChildChunker:
    """
    规则 + 语义融合的父子 Chunker。

    规则负责：
    - 标题层级和 Parent Section；
    - 列表、FAQ、表格、代码块的完整性；
    - max_tokens 硬限制；
    - 同组拆分片段尽量不在内部再次断开。

    语义负责：
    - 在多个安全边界中选择相邻语义最低的切点；
    - 兼顾 target_tokens，而不是机械地按固定长度切。
    """

    def __init__(
        self,
        config: ChunkConfig,
        semantic_encoder: SemanticEncoder,
    ) -> None:
        self.config = config
        self.encoder = semantic_encoder
        self.config.validate()

    def chunk(
        self,
        document: ParsedDocument,
        sections: Sequence[Section],
    ) -> tuple[list[ParentChunk], list[ChildChunk], list[AtomicUnit]]:
        all_expanded_units: list[AtomicUnit] = []
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []

        for section in sections:
            expanded = self._expand_oversized_units(document, section)
            if not expanded:
                continue

            parent_id = stable_id("parent", document.document_id, section.section_id)
            section_children = self._chunk_section(document, section, parent_id, expanded)
            all_expanded_units.extend(expanded)

            parent_text = "\n\n".join(unit.text for unit in expanded)
            parent_content = self._with_prefix(
                document=document,
                heading_path=section.heading_path,
                page_start=section.page_start,
                page_end=section.page_end,
                body=parent_text,
                label="Parent",
            )

            parents.append(
                ParentChunk(
                    parent_id=parent_id,
                    document_id=document.document_id,
                    title=section.title,
                    heading_path=list(section.heading_path),
                    content=parent_content,
                    text=parent_text,
                    token_estimate=estimate_tokens(parent_content),
                    page_start=section.page_start,
                    page_end=section.page_end,
                    child_ids=[child.child_id for child in section_children],
                    metadata={
                        "source": document.source_path,
                        "file_name": document.file_name,
                        "sha256": document.sha256,
                        "section_id": section.section_id,
                        "record_type": "parent",
                        "content_type": section.metadata.get("record_type", "body"),
                        "page_roles": section.metadata.get("page_roles", ["body"]),
                        "index_enabled": bool(section.metadata.get("index_enabled", True)),
                    },
                )
            )
            children.extend(section_children)

        self._link_children(children)
        return parents, children, all_expanded_units

    def _expand_oversized_units(
        self,
        document: ParsedDocument,
        section: Section,
    ) -> list[AtomicUnit]:
        prefix_tokens = self._prefix_token_budget(document, section.heading_path)
        body_max = max(40, self.config.max_tokens - prefix_tokens)
        body_target = max(30, self.config.target_tokens - prefix_tokens)

        result: list[AtomicUnit] = []
        for unit in section.units:
            if estimate_tokens(unit.text) <= body_max:
                result.append(unit)
                continue

            if unit.kind == "table":
                pieces = self._split_table(unit, body_target, body_max)
            elif unit.kind == "list":
                pieces = self._split_list(unit, body_target, body_max)
            elif unit.kind == "faq":
                pieces = self._split_faq(unit, body_target, body_max)
            elif unit.kind == "code":
                pieces = self._split_code(unit, body_target, body_max)
            else:
                pieces = self._split_prose(unit, body_target, body_max)

            # 类型专用切分仍可能遇到“单个表格行 / 单个列表项本身超长”。
            # 最终防线必须确保每个原子片段都能装入 Child 的正文预算。
            for piece in pieces:
                if estimate_tokens(piece.text) <= body_max:
                    result.append(piece)
                    continue
                fallback = self._split_prose(piece, body_target, body_max)
                result.extend(fallback)

        return result

    def _split_table(
        self,
        unit: AtomicUnit,
        target_tokens: int,
        max_tokens: int,
    ) -> list[AtomicUnit]:
        rows = unit.metadata.get("rows")
        if not isinstance(rows, list) or len(rows) <= 1:
            return self._split_prose(unit, target_tokens, max_tokens)

        header = rows[0]
        data_rows = rows[1:]
        chunks: list[list[list[str]]] = []
        current = [header]

        for row in data_rows:
            candidate = current + [row]
            candidate_text = table_to_markdown(candidate)
            if len(current) > 1 and estimate_tokens(candidate_text) > target_tokens:
                chunks.append(current)
                current = [header, row]
            else:
                current = candidate

            if estimate_tokens(table_to_markdown(current)) > max_tokens:
                if len(current) > 2:
                    last = current.pop()
                    chunks.append(current)
                    current = [header, last]
                else:
                    # 单行自身过大，保底切分单元格文本。
                    chunks.append(current)
                    current = [header]

        if len(current) > 1:
            chunks.append(current)

        return [
            self._piece(
                unit,
                text=table_to_markdown(rows_piece),
                index=index,
                total=len(chunks),
                extra_metadata={"rows": rows_piece, "split_strategy": "table_rows"},
            )
            for index, rows_piece in enumerate(chunks)
        ]

    def _split_list(
        self,
        unit: AtomicUnit,
        target_tokens: int,
        max_tokens: int,
    ) -> list[AtomicUnit]:
        lines = [line.strip() for line in unit.text.splitlines() if line.strip()]
        first_item = next(
            (index for index, line in enumerate(lines) if _is_list_item(line)),
            0,
        )
        lead = lines[:first_item]
        items = lines[first_item:]

        if len(items) <= 1:
            return self._split_prose(unit, target_tokens, max_tokens)

        pieces: list[list[str]] = []
        current = list(lead)

        for item in items:
            candidate = current + [item]
            if len(current) > len(lead) and estimate_tokens("\n".join(candidate)) > target_tokens:
                pieces.append(current)
                current = list(lead) + [item]
            else:
                current = candidate

            if estimate_tokens("\n".join(current)) > max_tokens:
                if len(current) > len(lead) + 1:
                    last = current.pop()
                    pieces.append(current)
                    current = list(lead) + [last]
                else:
                    hard = hard_split_text(item, max(20, max_tokens - estimate_tokens("\n".join(lead))))
                    current = []
                    for hard_piece in hard:
                        pieces.append(list(lead) + [hard_piece])

        if current and len(current) > len(lead):
            pieces.append(current)

        total = len(pieces)
        return [
            self._piece(
                unit,
                text=(
                    ("\n".join(piece) if index == 0 else _continuation_label(unit) + "\n" + "\n".join(piece))
                ),
                index=index,
                total=total,
                extra_metadata={"split_strategy": "list_items"},
            )
            for index, piece in enumerate(pieces)
        ]

    def _split_faq(
        self,
        unit: AtomicUnit,
        target_tokens: int,
        max_tokens: int,
    ) -> list[AtomicUnit]:
        match = _Q_SPLIT_RE.match(unit.text)
        if not match:
            return self._split_prose(unit, target_tokens, max_tokens)

        question = match.group("question").strip()
        answer = match.group("answer").strip()
        answer_sentences = split_sentences(answer)
        groups = self._semantic_sentence_groups(
            answer_sentences,
            max(20, target_tokens - estimate_tokens(question)),
            max(30, max_tokens - estimate_tokens(question)),
        )

        return [
            self._piece(
                unit,
                text=question + "\n" + "\n".join(group),
                index=index,
                total=len(groups),
                extra_metadata={
                    "question": question,
                    "split_strategy": "faq_answer_sentences",
                },
            )
            for index, group in enumerate(groups)
        ]

    def _split_code(
        self,
        unit: AtomicUnit,
        target_tokens: int,
        max_tokens: int,
    ) -> list[AtomicUnit]:
        lines = unit.text.splitlines()
        groups: list[list[str]] = []
        current: list[str] = []

        for line in lines:
            candidate = current + [line]
            begins_definition = bool(
                re.match(
                    r"^\s*(?:def|class|async\s+def|function|SELECT|INSERT|UPDATE|CREATE)\b",
                    line,
                    re.IGNORECASE,
                )
            )
            if (
                current
                and begins_definition
                and estimate_tokens("\n".join(current)) >= self.config.min_tokens
            ):
                groups.append(current)
                current = [line]
            elif current and estimate_tokens("\n".join(candidate)) > target_tokens:
                groups.append(current)
                current = [line]
            else:
                current = candidate

            if estimate_tokens("\n".join(current)) > max_tokens:
                raw = "\n".join(current)
                hard = hard_split_text(raw, max_tokens)
                groups.extend([[piece] for piece in hard])
                current = []

        if current:
            groups.append(current)

        return [
            self._piece(
                unit,
                text=(
                    "\n".join(group)
                    if index == 0
                    else "代码片段（续）：\n" + "\n".join(group)
                ),
                index=index,
                total=len(groups),
                extra_metadata={"split_strategy": "code_lines"},
            )
            for index, group in enumerate(groups)
        ]

    def _split_prose(
        self,
        unit: AtomicUnit,
        target_tokens: int,
        max_tokens: int,
    ) -> list[AtomicUnit]:
        sentences = split_sentences(unit.text)
        if len(sentences) <= 1:
            raw_pieces = hard_split_text(unit.text, max_tokens)
            return [
                self._piece(
                    unit,
                    text=piece,
                    index=index,
                    total=len(raw_pieces),
                    extra_metadata={"split_strategy": "hard_text"},
                )
                for index, piece in enumerate(raw_pieces)
            ]

        groups = self._semantic_sentence_groups(sentences, target_tokens, max_tokens)
        return [
            self._piece(
                unit,
                text="\n".join(group),
                index=index,
                total=len(groups),
                extra_metadata={"split_strategy": "semantic_sentences"},
            )
            for index, group in enumerate(groups)
        ]

    def _semantic_sentence_groups(
        self,
        sentences: Sequence[str],
        target_tokens: int,
        max_tokens: int,
    ) -> list[list[str]]:
        expanded: list[str] = []
        for sentence in sentences:
            if estimate_tokens(sentence) <= max_tokens:
                expanded.append(sentence)
            else:
                expanded.extend(hard_split_text(sentence, max_tokens))

        vectors = self.encoder.encode(expanded)
        similarities = adjacent_similarities(vectors)
        token_counts = [estimate_tokens(sentence) for sentence in expanded]
        return _semantic_partition(
            items=expanded,
            token_counts=token_counts,
            similarities=similarities,
            min_tokens=min(self.config.min_tokens, target_tokens),
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            lookback=self.config.boundary_lookback_units,
            semantic_weight=self.config.semantic_weight,
            length_weight=self.config.length_weight,
            group_ids=[None] * len(expanded),
            same_group_break_penalty=0.0,
        )

    def _piece(
        self,
        unit: AtomicUnit,
        text: str,
        index: int,
        total: int,
        extra_metadata: dict[str, object],
    ) -> AtomicUnit:
        return replace(
            unit,
            unit_id=stable_id("unit_piece", unit.unit_id, index, text),
            text=text,
            protected=True,
            metadata={
                **unit.metadata,
                **extra_metadata,
                "original_unit_id": unit.unit_id,
                "piece_index": index,
                "piece_count": total,
            },
        )

    def _chunk_section(
        self,
        document: ParsedDocument,
        section: Section,
        parent_id: str,
        units: Sequence[AtomicUnit],
    ) -> list[ChildChunk]:
        prefix_tokens = self._prefix_token_budget(document, section.heading_path)
        body_min = max(1, self.config.min_tokens - prefix_tokens)
        body_target = max(body_min, self.config.target_tokens - prefix_tokens)
        body_max = max(body_target, self.config.max_tokens - prefix_tokens)

        vectors = self.encoder.encode([unit.text for unit in units])
        similarities = adjacent_similarities(vectors)
        token_counts = [estimate_tokens(unit.text) for unit in units]
        primary_groups = _semantic_partition(
            items=list(units),
            token_counts=token_counts,
            similarities=similarities,
            min_tokens=body_min,
            target_tokens=body_target,
            max_tokens=body_max,
            lookback=self.config.boundary_lookback_units,
            semantic_weight=self.config.semantic_weight,
            length_weight=self.config.length_weight,
            group_ids=[unit.group_id for unit in units],
            same_group_break_penalty=self.config.same_group_break_penalty,
        )

        children: list[ChildChunk] = []
        previous_primary: list[AtomicUnit] = []

        for index, primary in enumerate(primary_groups):
            overlap = self._overlap_tail(
                previous_primary,
                max_tokens=self.config.overlap_tokens,
                body_budget=max(0, body_max - sum(estimate_tokens(unit.text) for unit in primary)),
            )
            body_units = overlap + primary
            body = "\n\n".join(unit.text for unit in body_units)
            page_start = min(unit.page_start for unit in body_units)
            page_end = max(unit.page_end for unit in body_units)
            child_id = stable_id(
                "child",
                parent_id,
                index,
                [unit.unit_id for unit in primary],
            )
            content = self._with_prefix(
                document=document,
                heading_path=section.heading_path,
                page_start=page_start,
                page_end=page_end,
                body=body,
                label="Child",
            )
            token_estimate = estimate_tokens(content)

            if token_estimate > self.config.max_tokens:
                # 理论上不应出现；保留明确失败而不是静默写入超限 Chunk。
                raise RuntimeError(
                    f"Child Chunk 超过 max_tokens：{token_estimate} > "
                    f"{self.config.max_tokens}，child_id={child_id}"
                )

            children.append(
                ChildChunk(
                    child_id=child_id,
                    parent_id=parent_id,
                    document_id=document.document_id,
                    chunk_index=index,
                    content=content,
                    text=body,
                    token_estimate=token_estimate,
                    page_start=page_start,
                    page_end=page_end,
                    heading_path=list(section.heading_path),
                    unit_ids=[unit.unit_id for unit in primary],
                    overlap_unit_ids=[unit.unit_id for unit in overlap],
                    previous_child_id=None,
                    next_child_id=None,
                    metadata={
                        "source": document.source_path,
                        "file_name": document.file_name,
                        "sha256": document.sha256,
                        "section_id": section.section_id,
                        "record_type": "child",
                        "content_type": section.metadata.get("record_type", "body"),
                        "page_roles": section.metadata.get("page_roles", ["body"]),
                        "index_enabled": bool(section.metadata.get("index_enabled", True)),
                        "semantic_backend": self.encoder.name,
                    },
                )
            )
            previous_primary = list(primary)

        return children

    def _overlap_tail(
        self,
        previous_units: Sequence[AtomicUnit],
        max_tokens: int,
        body_budget: int,
    ) -> list[AtomicUnit]:
        if max_tokens <= 0 or body_budget <= 0:
            return []

        result: list[AtomicUnit] = []
        used = 0
        for unit in reversed(previous_units):
            unit_tokens = estimate_tokens(unit.text)
            if used + unit_tokens > max_tokens or used + unit_tokens > body_budget:
                break
            result.append(unit)
            used += unit_tokens
        return list(reversed(result))

    def _prefix_token_budget(
        self,
        document: ParsedDocument,
        heading_path: Sequence[str],
    ) -> int:
        sample = self._with_prefix(
            document=document,
            heading_path=heading_path,
            page_start=1,
            page_end=max(1, document.page_count),
            body="",
            label="Child",
        )
        return estimate_tokens(sample) + 4

    def _with_prefix(
        self,
        document: ParsedDocument,
        heading_path: Sequence[str],
        page_start: int,
        page_end: int,
        body: str,
        label: str,
    ) -> str:
        if not self.config.include_source_prefix:
            return body.strip()

        prefix = [
            f"文档：{document.file_name}",
            f"记录类型：{label}",
        ]
        if heading_path:
            prefix.append("章节：" + " > ".join(heading_path))
        prefix.append(
            f"页码：{page_start}"
            if page_start == page_end
            else f"页码：{page_start}-{page_end}"
        )
        return "\n".join(prefix) + "\n\n" + body.strip()

    @staticmethod
    def _link_children(children: list[ChildChunk]) -> None:
        by_parent: dict[str, list[ChildChunk]] = {}
        for child in children:
            by_parent.setdefault(child.parent_id, []).append(child)

        for siblings in by_parent.values():
            siblings.sort(key=lambda child: child.chunk_index)
            for index, child in enumerate(siblings):
                child.previous_child_id = siblings[index - 1].child_id if index > 0 else None
                child.next_child_id = (
                    siblings[index + 1].child_id
                    if index + 1 < len(siblings)
                    else None
                )


def _semantic_partition(
    items: Sequence,
    token_counts: Sequence[int],
    similarities: Sequence[float],
    min_tokens: int,
    target_tokens: int,
    max_tokens: int,
    lookback: int,
    semantic_weight: float,
    length_weight: float,
    group_ids: Sequence[str | None],
    same_group_break_penalty: float,
) -> list[list]:
    """
    在所有满足 max_tokens 的安全切点里，联合优化：

    cut_score =
        边界语义相似度 * semantic_weight
        + 与 target_tokens 的相对偏差 * length_weight
        + 打断同一 group 的惩罚

    相邻语义越低，越适合切；长度越接近 target，代价越低。
    """
    if not items:
        return []
    if len(items) != len(token_counts):
        raise ValueError("items 和 token_counts 长度不一致")
    if len(similarities) != max(0, len(items) - 1):
        raise ValueError("similarities 长度错误")

    groups: list[list] = []
    start = 0
    count = len(items)

    while start < count:
        remaining_tokens = sum(token_counts[start:])
        if remaining_tokens <= target_tokens or count - start == 1:
            groups.append(list(items[start:]))
            break

        cumulative = 0
        candidate_cuts: list[tuple[float, int, int]] = []

        for cut in range(start + 1, count + 1):
            cumulative += token_counts[cut - 1]
            if cumulative > max_tokens:
                break
            if cumulative < min_tokens:
                continue

            # “不再切分”没有真实语义边界，给一个轻微基线成本。
            # 这样：内部边界语义很弱时允许提前切；语义连续时则保留整体。
            boundary_similarity = similarities[cut - 1] if cut < count else 0.15
            length_penalty = abs(cumulative - target_tokens) / max(target_tokens, 1)
            group_penalty = 0.0
            if (
                cut < count
                and group_ids[cut - 1]
                and group_ids[cut - 1] == group_ids[cut]
            ):
                group_penalty = same_group_break_penalty

            score = (
                boundary_similarity * semantic_weight
                + length_penalty * length_weight
                + group_penalty
            )
            candidate_cuts.append((score, cut, cumulative))

        # 只限制参与最终比较的候选数量，不限制向前扫描距离。
        # 这样大量极短原子单元仍能累积到 min_tokens。
        if len(candidate_cuts) > lookback:
            closest_to_target = sorted(
                candidate_cuts,
                key=lambda item: abs(item[2] - target_tokens),
            )[: max(2, lookback // 2)]
            lowest_semantic = sorted(
                candidate_cuts,
                key=lambda item: item[0],
            )[: max(2, lookback - len(closest_to_target))]
            deduplicated = {
                cut: (score, cut, tokens)
                for score, cut, tokens in closest_to_target + lowest_semantic
            }
            candidate_cuts = list(deduplicated.values())

        if not candidate_cuts:
            # 单个 item 应已在上游拆到 max 以内；这里做防御性处理。
            if token_counts[start] > max_tokens:
                raise RuntimeError(
                    f"存在无法装入 max_tokens 的原子单元：{token_counts[start]} > {max_tokens}"
                )
            groups.append([items[start]])
            start += 1
            continue

        _, best_cut, _ = min(candidate_cuts, key=lambda item: (item[0], abs(item[2] - target_tokens)))
        groups.append(list(items[start:best_cut]))
        start = best_cut

    return groups


def _is_list_item(line: str) -> bool:
    return bool(
        re.match(
            r"^(?:[-*•·]\s+|\(?\d{1,3}[.)、]\s*|"
            r"[一二三四五六七八九十百]+[、.]\s*|"
            r"[（(][一二三四五六七八九十百0-9]+[）)]\s*)",
            line.strip(),
        )
    )


def _continuation_label(unit: AtomicUnit) -> str:
    title = unit.heading_path[-1] if unit.heading_path else "列表"
    return f"{title}（续）："
