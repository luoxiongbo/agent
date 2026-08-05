from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .code_analysis import repair_code_physical_lines

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_TERMINAL_RE = re.compile(r"[。！？!?；;：:]$")
_STANDALONE_BULLET_RE = re.compile(r"^[●•·▪■◆◇○◦‣⁃]+$")
_CODE_UI_LABEL_RE = re.compile(
    r"^(?:Plain\s*Text|Copy\s*code|复制代码|Code|Text)$",
    re.IGNORECASE,
)
_LINE_NUMBER_STRIP_RE = re.compile(r"^(?:\d{1,3}\s+){3,}\d{1,3}$")
_PAGE_NUMBER_RE = re.compile(r"^(?:page\s*)?\d{1,4}(?:\s*/\s*\d{1,4})?$", re.IGNORECASE)
_PROMOTIONAL_CONTACT_RE = re.compile(
    r"(?:更多(?:课程|资料|内容).{0,8}(?:联系|添加)|扫码(?:关注|加入)|(?:关注|添加).{0,6}(?:微信|公众号)|课程咨询)"
    r".*?(?:微信|vx|wechat|公众号|二维码)?\s*[:：]?\s*[A-Za-z0-9_-]{4,}",
    re.IGNORECASE,
)
_PARAGRAPH_START_RE = re.compile(
    r"^(?:离线流程|在线流程|上述|因此|所以|这两个流程|在实际工程中|接下来|例如|其中|同时|此外|"
    r"对于|通过|为了|需要注意|需要说明|总体而言|综上|最后|用户查询|系统会|该方法|该流程)"
)
_TOC_ENTRY_RE = re.compile(r"^\d+(?:\.\d+){1,5}[:：、.]?\s*\S+")
_CODE_MARKER_RE = re.compile(
    r"^(?:from\s+\S+\s+import\s+|import\s+\S+|def\s+\w+\s*\(|class\s+\w+|async\s+def\s+|"
    r"if\s+.+:|elif\s+.+:|else\s*:|for\s+.+\s+in\s+.+:|while\s+.+:|try\s*:|except\b|"
    r"return\b|with\s+.+:|@\w+|SELECT\b|INSERT\b|UPDATE\b|CREATE\b)",
    re.IGNORECASE,
)


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


def repair_broken_cjk_blank_lines(value: str) -> str:
    """修复高置信度的中文词语被 PDF TextBlock 空行拆开。

    只处理常见的单词内部断裂，避免把两个真实段落广泛拼接。
    """
    parts = [part.strip() for part in re.split(r"\n{2,}", value or "") if part.strip()]
    if not parts:
        return ""
    output = [parts[0]]
    for right in parts[1:]:
        left = output[-1]
        if _high_confidence_cjk_blank_break(left, right):
            output[-1] = left.rstrip() + right.lstrip()
        else:
            output.append(right)
    return "\n\n".join(output).strip()


def _high_confidence_cjk_blank_break(left: str, right: str) -> bool:
    if not left or not right or ends_with_sentence_terminal(left):
        return False
    if left.rstrip().endswith(("：", ":", "，", ",")):
        return False
    if is_list_line(right) or is_paragraph_starter(right):
        return False
    left_match = re.search(r"([\u3400-\u9fff]{1,4})$", left.rstrip())
    right_match = re.match(r"([\u3400-\u9fff]{1,4})", right.lstrip())
    if not left_match or not right_match:
        return False
    left_tail = left_match.group(1)
    right_head = right_match.group(1)
    strong_prefixes = {
        "能翻", "混", "销售技", "因此候", "候", "语义检", "向量", "模型",
        "数", "函", "参", "返", "结", "识", "优", "查", "处", "分",
    }
    strong_suffixes = {
        "阅", "杂", "巧", "选", "索", "型", "据", "法", "数", "行",
        "果", "息", "理", "类", "量", "化", "库", "问", "答",
    }
    return left_tail in strong_prefixes or right_head[:1] in strong_suffixes


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
    if ends_with_sentence_terminal(left):
        return left + "\n" + right
    if is_list_line(right):
        return left + "\n" + right
    left_lines = [line for line in left.splitlines() if line.strip()]
    list_count = sum(is_list_line(line) for line in left_lines)
    if list_count and is_paragraph_starter(right):
        return left + "\n\n" + right
    if is_probable_cjk_word_break(left, right):
        return left + right
    if is_paragraph_starter(right):
        return left + "\n\n" + right
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



def is_promotional_contact_line(value: str) -> bool:
    text = normalize_inline_text(value)
    if not text:
        return False
    return bool(_PROMOTIONAL_CONTACT_RE.search(text))


def is_paragraph_starter(value: str) -> bool:
    return bool(_PARAGRAPH_START_RE.match(normalize_inline_text(value)))


def toc_entry_count(lines: Sequence[str]) -> int:
    return sum(bool(_TOC_ENTRY_RE.match(normalize_inline_text(line))) for line in lines)


def looks_like_code_lines(lines: Sequence[str], min_markers: int = 2) -> bool:
    cleaned = [normalize_code_line(line) for line in lines if normalize_inline_text(line)]
    if not cleaned:
        return False
    marker_count = sum(bool(_CODE_MARKER_RE.match(line.strip())) for line in cleaned)
    punctuation_count = sum(
        1 for line in cleaned if re.search(r"[{}[\]();=<>]|\w+\s*=", line)
    )
    return marker_count >= min_markers or (marker_count >= 1 and punctuation_count >= 2)


def split_compacted_code_statements(value: str) -> list[str]:
    text = normalize_code_line(value).strip()
    if not text:
        return []
    # 只在高置信度的顶级 Python 声明前拆分，避免破坏普通表达式和字符串。
    parts = re.split(
        r"\s+(?=(?:from\s+[A-Za-z_]|def\s+[A-Za-z_]|class\s+[A-Za-z_]|async\s+def\s+))",
        text,
    )
    result: list[str] = []
    for part in parts:
        # `import x` 可以独立成行，但 `from x import y` 中的 import 不能拆。
        if not part.lstrip().startswith("from "):
            subparts = re.split(r"\s+(?=import\s+[A-Za-z_])", part)
        else:
            subparts = [part]
        result.extend(item.strip() for item in subparts if item.strip())
    return result


def reconstruct_code_from_lines(
    lines: Sequence[str],
    x0s: Sequence[float],
    indent_spaces: int = 4,
) -> str:
    repaired_lines, repaired_x0s = repair_code_physical_lines(lines, x0s)
    pairs = [
        (float(x0), normalize_code_line(text))
        for text, x0 in zip(repaired_lines, repaired_x0s)
        if normalize_inline_text(text) and not is_noise_line(text.strip())
    ]
    if not pairs:
        return ""

    min_x = min(x for x, _ in pairs)
    positive_diffs = sorted(
        {round(x - min_x, 2) for x, _ in pairs if x - min_x >= 2.0}
    )
    indent_width = positive_diffs[0] if positive_diffs else 18.0
    indent_width = min(max(indent_width, 8.0), 36.0)

    output: list[str] = []
    syntax_stack: list[tuple[int, str]] = []
    previous_level = 0
    previous_fragment = ""
    for x0, raw in pairs:
        stripped = raw.strip()
        if not stripped:
            continue
        fragments = split_compacted_code_statements(stripped)
        for fragment_index, fragment in enumerate(fragments):
            physical_level = max(0, round((x0 - min_x) / indent_width))
            level = physical_level

            if re.match(r"^(?:elif|else|except|finally|case)\b", fragment):
                if syntax_stack:
                    level = syntax_stack[-1][0]
                else:
                    level = max(0, previous_level - 1)
            elif previous_fragment.rstrip().endswith(":"):
                level = max(level, previous_level + 1)
            elif syntax_stack and physical_level == 0:
                # 某些 PDF 丢失全部 x 坐标缩进；保守地维持函数/类作用域。
                nearest_def = next(
                    (item for item in reversed(syntax_stack) if item[1] in {"def", "class"}),
                    None,
                )
                if nearest_def and not re.match(
                    r"^(?:@|async\s+def|def|class|from|import)\b", fragment
                ):
                    level = max(level, nearest_def[0] + 1)

            if physical_level > 0:
                while syntax_stack and level <= syntax_stack[-1][0] and not re.match(
                    r"^(?:elif|else|except|finally|case)\b", fragment
                ):
                    syntax_stack.pop()

            if re.match(r"^(?:return|yield|raise|break|continue|pass)\b", fragment):
                if physical_level == 0 and syntax_stack:
                    level = max(level, syntax_stack[-1][0] + 1)

            output.append(" " * (level * indent_spaces) + fragment)
            if fragment.rstrip().endswith(":"):
                marker = re.match(
                    r"^(?:async\s+)?(def|class|if|elif|else|for|while|try|except|finally|with|match|case)\b",
                    fragment,
                )
                syntax_stack.append((level, marker.group(1) if marker else "block"))
            previous_level = level
            previous_fragment = fragment
    return "\n".join(output).strip()

def is_probable_cjk_word_break(left: str, right: str) -> bool:
    left = normalize_inline_text(left)
    right = normalize_inline_text(right)
    if not left or not right or ends_with_sentence_terminal(left):
        return False
    if is_list_line(right) or is_paragraph_starter(right):
        return False
    left_match = re.search(r"([\u3400-\u9fff]{1,4})$", left)
    right_match = re.match(r"([\u3400-\u9fff]{1,4})", right)
    if not left_match or not right_match:
        return False
    left_tail = left_match.group(1)
    right_head = right_match.group(1)
    # 短尾或短头通常是 PDF 把一个中文词切成两个 TextBlock。
    return len(left_tail) <= 3 or len(right_head) <= 2


def has_glued_list_paragraph(value: str) -> bool:
    text = value or ""
    return bool(
        re.search(
            r"(?:^|\n)\s*\d+[.)、]\s*[^\n。！？!?]{18,}"
            r"(?:离线流程|在线流程|上述|因此|这两个流程|在实际工程中|接下来)",
            text,
        )
    )


def looks_malformed_code(value: str) -> bool:
    text = value or ""
    if not re.search(r"(?:^|\n)\s*(?:def|class|import|from)\b", text):
        return False
    if re.search(r"\bimport[ \t]+\w+[ \t]+from[ \t]+", text):
        return True
    if re.search(r"\)\s*(?:def|class|import|from)\s+", text):
        return True
    if re.search(r"^def\s+.+:\n(?!\s)", text, re.MULTILINE):
        return True
    if re.search(r"\breturn\b[^\n]{0,80}\breturn\b", text):
        return True
    return False


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
