from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from .config import PipelineConfig
from .pipeline import RAGPDFPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生产导向的 RAG PDF 离线解析与智能父子 Chunk 工具"
    )
    parser.add_argument("pdf", help="输入 PDF")
    parser.add_argument("-o", "--output-dir", default="./output", help="输出目录")
    parser.add_argument("--config", help="JSON 配置文件")
    parser.add_argument("--password", help="加密 PDF 密码")

    parser.add_argument("--ocr", choices=["auto", "always", "never"])
    parser.add_argument("--ocr-language")
    parser.add_argument("--tessdata")
    parser.add_argument("--no-tables", action="store_true")
    parser.add_argument(
        "--complex-layout-strategy",
        choices=["conservative", "keep", "skip"],
        help="复杂图解/幻灯片页面处理策略",
    )

    parser.add_argument("--min-tokens", type=int)
    parser.add_argument("--target-tokens", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--overlap-tokens", type=int)

    parser.add_argument(
        "--semantic-backend",
        choices=["none", "hashing", "sentence_transformers"],
    )
    parser.add_argument("--semantic-model")
    parser.add_argument("--device")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="允许 Sentence Transformers 从远端下载模型；默认只读取本地文件",
    )

    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="质量检查存在错误时仍写出结果",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def load_config(args: argparse.Namespace) -> PipelineConfig:
    config = (
        PipelineConfig.from_json(args.config)
        if args.config
        else PipelineConfig()
    )

    if args.ocr:
        config.parser.ocr_mode = args.ocr
    if args.ocr_language:
        config.parser.ocr_language = args.ocr_language
    if args.tessdata:
        config.parser.tessdata = args.tessdata
    if args.no_tables:
        config.parser.extract_tables = False
    if args.complex_layout_strategy:
        config.parser.complex_layout_strategy = args.complex_layout_strategy

    if args.min_tokens is not None:
        config.chunk.min_tokens = args.min_tokens
    if args.target_tokens is not None:
        config.chunk.target_tokens = args.target_tokens
    if args.max_tokens is not None:
        config.chunk.max_tokens = args.max_tokens
    if args.overlap_tokens is not None:
        config.chunk.overlap_tokens = args.overlap_tokens

    if args.semantic_backend:
        config.semantic.backend = args.semantic_backend
    if args.semantic_model:
        config.semantic.model_name_or_path = args.semantic_model
    if args.device:
        config.semantic.device = args.device
    if args.allow_model_download:
        config.semantic.local_files_only = False

    if args.non_strict:
        config.quality.strict = False

    config.validate()
    return config


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    try:
        config = load_config(args)
        summary = RAGPDFPipeline(config).run(
            pdf_path=args.pdf,
            output_dir=args.output_dir,
            password=args.password,
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("执行失败：%s", exc)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
