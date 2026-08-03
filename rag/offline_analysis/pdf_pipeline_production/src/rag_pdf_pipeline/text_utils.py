from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def normalize_inline_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("\u00ad", "")
    value = value.replace("\u200b", "")
    value = value.replace("\ufeff", "")
    value = re.sub(r"[\t\v\f\r ]+", " ", value)
    return value.strip()


def normalize_multiline_text(value: str) -> str:
    lines = [normalize_inline_text(line) for line in (value or "").splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def visible_char_count(value: str) -> int:
    return len(re.sub(r"\s+", "", value or ""))


def contains_cjk(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", value or ""))


LIST_LINE_RE = re.compile(
    r"^(?:[-*•·]\s+|"
    r"\(?\d{1,3}[.)、]\s*|"
    r"[一二三四五六七八九十百]+[、.]\s*|"
    r"[（(][一二三四五六七八九十百0-9]+[）)]\s*)"
)


def is_list_line(value: str) -> bool:
    return bool(LIST_LINE_RE.match(value.strip()))


def merge_extracted_lines(lines: Sequence[str]) -> str:
    cleaned = [normalize_inline_text(line) for line in lines]
    cleaned = [line for line in cleaned if line]
    if not cleaned:
        return ""

    result = cleaned[0]
    for current in cleaned[1:]:
        previous = result.rstrip()

        if is_list_line(current):
            result += "\n" + current
            continue

        if previous.endswith("-") and re.match(r"^[a-z]", current):
            result = previous[:-1] + current
            continue

        if re.search(r"[。！？!?；;：:]$", previous):
            result += "\n" + current
            continue

        if contains_cjk(previous[-8:]) or contains_cjk(current[:8]):
            result += current
        else:
            result += " " + current

    return normalize_multiline_text(result)


def estimate_tokens(text: str) -> int:
    """
    模型无关的保守估算。

    生产环境若已确定最终 Embedding / LLM 模型，可替换为对应 tokenizer。
    """
    text = text or ""
    cjk_count = len(
        re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text)
    )
    other = re.sub(
        r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\s]",
        "",
        text,
    )
    return cjk_count + math.ceil(len(other) / 4)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while data := file.read(chunk_size):
            digest.update(data)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: object, length: int = 24) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:length]
    return f"{prefix}_{digest}"


def safe_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json_value(item) for item in value]
    return str(value)


SENTENCE_RE = re.compile(
    r"(?<=[。！？!?；;])\s*|"
    r"(?<=\.)\s+(?=[A-Z0-9])|"
    r"\n{2,}"
)


def split_sentences(text: str) -> list[str]:
    text = normalize_multiline_text(text)
    if not text:
        return []
    sentences = [item.strip() for item in SENTENCE_RE.split(text) if item.strip()]
    return sentences or [text]


def hard_split_text(text: str, max_tokens: int) -> list[str]:
    if not text:
        return []
    if estimate_tokens(text) <= max_tokens:
        return [text]

    estimated = max(estimate_tokens(text), 1)
    char_limit = max(40, int(len(text) * max_tokens / estimated))
    pieces: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + char_limit, len(text))
        if end < len(text):
            candidates = [
                text.rfind("。", start, end),
                text.rfind("；", start, end),
                text.rfind("\n", start, end),
                text.rfind(" ", start, end),
            ]
            candidate = max(candidates)
            if candidate > start + char_limit // 2:
                end = candidate + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        start = end
    return pieces


def normalize_rows(rows: Iterable[Iterable[str]]) -> list[list[str]]:
    result: list[list[str]] = []
    for raw_row in rows:
        row = [normalize_multiline_text(str(cell or "")) for cell in raw_row]
        if any(row):
            result.append(row)
    return result


def markdown_escape(value: str) -> str:
    return normalize_multiline_text(value).replace("|", r"\|").replace("\n", "<br>")


def table_to_markdown(rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [
        [markdown_escape(cell) for cell in list(row) + [""] * (width - len(row))]
        for row in rows
    ]
    header = normalized[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
    return "\n".join(lines)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))
