from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_TERMINAL_RE = re.compile(r"[。！？!?；;：:]$")
_STANDALONE_BULLET_RE = re.compile(r"^[●•·▪■◆◇○◦‣⁃]+$")
_CODE_UI_LABEL_RE = re.compile(
    r"^(?:Plain\s*Text|Copy\s*code|复制代码|Code|Text)$",
    re.IGNORECASE,
)
_LINE_NUMBER_STRIP_RE = re.compile(r"^(?:\d{1,3}\s+){3,}\d{1,3}$")
_PAGE_NUMBER_RE = re.compile(r"^(?:page\s*)?\d{1,4}(?:\s*/\s*\d{1,4})?$", re.IGNORECASE)


def normalize_inline_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("\u00ad", "")
    value = value.replace("\u200b", "")
    value = value.replace("\ufeff", "")
    value = re.sub(r"[\t\v\f\r ]+", " ", value)
    return value.strip()


def normalize_code_line(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("\u00ad", "").replace("\u200b", "").replace("\ufeff", "")
    value = value.replace("\t", "    ").rstrip()
    return value


def normalize_multiline_text(value: str) -> str:
    lines = [normalize_inline_text(line) for line in (value or "").splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def visible_char_count(value: str) -> int:
    return len(re.sub(r"\s+", "", value or ""))


def contains_cjk(value: str) -> bool:
    return bool(_CJK_RE.search(value or ""))


LIST_LINE_RE = re.compile(
    r"^(?:[-*•·]\s+|"
    r"\(?\d{1,3}[.)、]\s*|"
    r"[一二三四五六七八九十百]+[、.]\s*|"
    r"[（(][一二三四五六七八九十百0-9]+[）)]\s*)"
)


def is_list_line(value: str) -> bool:
    return bool(LIST_LINE_RE.match(value.strip()))


def is_line_number_sequence(value: str) -> bool:
    value = normalize_inline_text(value)
    if not re.fullmatch(r"(?:\d{1,4}\s+)+\d{1,4}", value):
        return False
    numbers = [int(item) for item in value.split()]
    if len(numbers) < 2:
        return False
    return all(right - left == 1 for left, right in zip(numbers, numbers[1:]))


def is_noise_line(value: str) -> bool:
    value = normalize_inline_text(value)
    if not value:
        return True
    if _STANDALONE_BULLET_RE.fullmatch(value):
        return True
    if _CODE_UI_LABEL_RE.fullmatch(value):
        return True
    if _LINE_NUMBER_STRIP_RE.fullmatch(value) or is_line_number_sequence(value):
        return True
    return False


def clean_extracted_text(value: str) -> str:
    lines = [normalize_inline_text(line) for line in (value or "").splitlines()]
    lines = [line for line in lines if line and not is_noise_line(line)]
    return "\n".join(lines).strip()


def clean_code_text(lines: Sequence[str]) -> str:
    cleaned: list[str] = []
    for line in lines:
        normalized = normalize_code_line(line)
        if not normalized.strip() or is_noise_line(normalized.strip()):
            continue
        cleaned.append(normalized)
    return "\n".join(cleaned).strip()


def merge_extracted_lines(lines: Sequence[str]) -> str:
    cleaned = [normalize_inline_text(line) for line in lines]
    cleaned = [line for line in cleaned if line and not is_noise_line(line)]
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

        if _TERMINAL_RE.search(previous):
            result += "\n" + current
            continue

        if contains_cjk(previous[-8:]) or contains_cjk(current[:8]):
            result += current
        else:
            result += " " + current

    return clean_extracted_text(result)


def smart_join_text(left: str, right: str) -> str:
    left = clean_extracted_text(left).rstrip()
    right = clean_extracted_text(right).lstrip()
    if not left:
        return right
    if not right:
        return left
    if left.endswith("-") and re.match(r"^[a-z]", right):
        return left[:-1] + right
    if contains_cjk(left[-8:]) or contains_cjk(right[:8]):
        return left + right
    return left + " " + right


def ends_with_sentence_terminal(value: str) -> bool:
    return bool(re.search(r"[。！？!?；;]$", value.strip()))


def looks_like_sentence_or_list_item(value: str) -> bool:
    text = normalize_inline_text(value)
    compact_length = visible_char_count(text)
    if not text:
        return False
    if re.search(r"[。！？!?；;，,]$", text):
        return True
    if text.count("，") + text.count(",") >= 2:
        return True
    if ":" in text or "：" in text:
        tail = re.split(r"[：:]", text, maxsplit=1)[-1]
        if visible_char_count(tail) >= 12:
            return True
    if compact_length > 36 and re.search(
        r"(?:用于|负责|通过|需要|包括|可以|进行|完成|如果|导致|使其|采用|使用|结合|提供|希望|包含)",
        text,
    ):
        return True
    if compact_length > 52:
        return True
    return False


def starts_like_continuation(value: str) -> bool:
    value = normalize_inline_text(value)
    if not value:
        return False
    if is_list_line(value):
        return False
    if re.match(r"^(?:但是|而且|以及|并且|因此|所以|其中|同时|此外|或者|与|和|及|的|了|们|种|序|限|案|码|拆开|强)", value):
        return True
    if re.match(r"^[a-z,.;:)\]]", value):
        return True
    return contains_cjk(value[:1])


def is_suspicious_mid_sentence_start(value: str) -> bool:
    value = normalize_inline_text(value)
    if not value:
        return False
    suspicious = (
        "们", "种", "序", "限", "案", "码", "拆开", "强", "达", "合", "询", "问",
        "而", "且", "以及", "并且", "其中", "同时", "此外", "因此", "所以",
    )
    if value.startswith(suspicious):
        return True
    return bool(re.match(r"^[a-z,.;:)\]]", value))


def has_broken_cjk_line_candidate(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]\n{2,}[\u3400-\u9fff]", value or ""))


def is_page_number_text(value: str) -> bool:
    return bool(_PAGE_NUMBER_RE.fullmatch(normalize_inline_text(value)))


def estimate_tokens(text: str) -> int:
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
    text = clean_extracted_text(text)
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
        row = [clean_extracted_text(str(cell or "")) for cell in raw_row]
        if any(row):
            result.append(row)
    return result


def markdown_escape(value: str) -> str:
    return clean_extracted_text(value).replace("|", r"\|").replace("\n", "<br>")


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
