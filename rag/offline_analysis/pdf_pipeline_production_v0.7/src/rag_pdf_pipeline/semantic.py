from __future__ import annotations

import hashlib
import logging
import re
from typing import Protocol, Sequence

import numpy as np

from .config import SemanticConfig

LOGGER = logging.getLogger(__name__)


class SemanticEncoder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """返回形状为 [N, D] 的 L2 归一化向量。"""


class NoopSemanticEncoder:
    name = "none"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), 1), dtype=np.float32)
        return vectors


class HashingSemanticEncoder:
    """
    无模型下载的确定性后备方案。

    它使用中文字符 n-gram、英文词和英文 bigram 的 signed hashing。
    这不是神经语义模型，但比纯长度规则更能识别局部主题连续性，
    适合测试、离线兜底和无模型环境。
    """

    name = "hashing"

    def __init__(self, dimensions: int = 1024) -> None:
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)

        for row, text in enumerate(texts):
            features = _features(text)
            for feature, weight in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little", signed=False)
                index = value % self.dimensions
                sign = 1.0 if (value >> 63) == 0 else -1.0
                matrix[row, index] += sign * weight

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms


class SentenceTransformerSemanticEncoder:
    name = "sentence_transformers"

    def __init__(self, config: SemanticConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "未安装 sentence-transformers。请执行："
                "python -m pip install -e '.[semantic]'"
            ) from exc

        assert config.model_name_or_path
        LOGGER.info("加载语义模型：%s", config.model_name_or_path)
        self.batch_size = config.batch_size
        self.model = SentenceTransformer(
            config.model_name_or_path,
            device=config.device,
            local_files_only=config.local_files_only,
            trust_remote_code=False,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            dimension = int(self.model.get_sentence_embedding_dimension() or 1)
            return np.zeros((0, dimension), dtype=np.float32)

        encode_document = getattr(self.model, "encode_document", None)
        method = encode_document if callable(encode_document) else self.model.encode
        vectors = method(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


def build_semantic_encoder(config: SemanticConfig) -> SemanticEncoder:
    config.validate()
    if config.backend == "none":
        return NoopSemanticEncoder()
    if config.backend == "hashing":
        return HashingSemanticEncoder(config.hashing_dimensions)
    return SentenceTransformerSemanticEncoder(config)


def adjacent_similarities(vectors: np.ndarray) -> list[float]:
    if len(vectors) <= 1:
        return []
    return [
        float(np.dot(vectors[index], vectors[index + 1]))
        for index in range(len(vectors) - 1)
    ]


def _features(text: str) -> list[tuple[str, float]]:
    normalized = re.sub(r"\s+", " ", text.lower())
    features: list[tuple[str, float]] = []

    cjk = "".join(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", normalized))
    features.extend((f"c1:{char}", 0.35) for char in cjk)
    features.extend((f"c2:{cjk[i:i+2]}", 1.0) for i in range(max(0, len(cjk) - 1)))
    features.extend((f"c3:{cjk[i:i+3]}", 0.65) for i in range(max(0, len(cjk) - 2)))

    words = re.findall(r"[a-z0-9_./-]+", normalized)
    features.extend((f"w1:{word}", 1.0) for word in words)
    features.extend(
        (f"w2:{words[i]}_{words[i+1]}", 0.7)
        for i in range(max(0, len(words) - 1))
    )

    if not features and normalized:
        features.append((f"raw:{normalized[:64]}", 1.0))
    return features
