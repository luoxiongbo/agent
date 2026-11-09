# -*- encode = utf-8 -*-
# @Time : 2026-10-03 22:52:21
# @Auther : 罗雄波
# @File : light_rag_practice.py
# @Software : PyCharm

import asyncio
import json
import os
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_embed

import httpx
from lightrag.utils import EmbeddingFunc
from ground_truth import GROUND_TRUTH

load_dotenv()

WORKING_DIR = "./rag_storage"
DOCS_DIR = Path("./docs")
RESULT_DIR = Path("./results")

RESULT_DIR.mkdir(exist_ok=True)


# ============================================================
# GLM-5.3
# ============================================================

async def llm_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
):
    input_messages = []

    if system_prompt:
        input_messages.append({
            "role": "system",
            "content": system_prompt,
        })

    if history_messages:
        input_messages.extend(history_messages)

    input_messages.append({
        "role": "user",
        "content": prompt,
    })

    payload = {
        "model": os.getenv("LLM_MODEL", "glm-5.3"),
        "input": input_messages,
        "reasoning": {
            "effort": "high"
        },
    }

    base_url = os.getenv(
        "LLM_BINDING_HOST"
    ).rstrip("/")

    url = base_url + "/responses"

    key = os.getenv(
        "LLM_BINDING_API_KEY"
    ).strip()

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(
            timeout=float(
                os.getenv("LLM_TIMEOUT", "240")
            )
    ) as client:

        response = await client.post(
            url,
            headers=headers,
            json=payload,
        )

        print("status:", response.status_code)
        print("body:", response.text[:2000])

        response.raise_for_status()

        data = response.json()

    # Responses API
    for item in data.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text", "")

    raise RuntimeError(
        "GLM Responses API 未返回 output_text:\n"
        + json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        )
    )


# ============================================================
# SiliconFlow: BAAI/bge-m3
#
# 注意：
# openai_embed 本身已经是 EmbeddingFunc，
# 所以这里必须使用 openai_embed.func，
# 避免重复包装。
# ============================================================

embedding_func = EmbeddingFunc(
    embedding_dim=int(
        os.getenv("EMBEDDING_DIM", "1024")
    ),
    max_token_size=int(
        os.getenv("MAX_EMBED_TOKENS", "8192")
    ),
    model_name=os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-m3",
    ),
    func=partial(
        openai_embed.func,
        model=os.getenv(
            "EMBEDDING_MODEL",
            "BAAI/bge-m3",
        ),
        base_url=os.getenv(
            "EMBEDDING_BINDING_HOST",
            "https://api.siliconflow.cn/v1",
        ),
        api_key=os.getenv(
            "EMBEDDING_BINDING_API_KEY"
        ),
    ),
)


# ============================================================
# 初始化 LightRAG
# ============================================================

async def create_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,

        llm_model_func=llm_model_func,
        llm_model_name=os.getenv(
            "LLM_MODEL",
            "glm-5.3",
        ),

        embedding_func=embedding_func,

        # 方便观察中文 Entity / Relation
        addon_params={
            "language": "Simplified Chinese"
        },
    )

    await rag.initialize_storages()

    return rag


# ============================================================
# 检查 API 是否正常
#
# 先测 Embedding，避免跑到 Indexing 一半才发现配置错误
# ============================================================

async def check_embedding(rag: LightRAG):
    print("\n========== TEST EMBEDDING ==========")

    texts = [
        "Flink TaskManager uses network memory."
    ]

    embeddings = await rag.embedding_func(texts)

    print("Embedding shape:", embeddings.shape)

    expected_dim = int(
        os.getenv("EMBEDDING_DIM", "1024")
    )

    assert embeddings.shape == (
        1,
        expected_dim,
    )

    print(
        f"BGE-M3 OK, dimension={expected_dim}"
    )


# ============================================================
# 测试 GLM
# ============================================================

async def check_llm():
    print("\n========== TEST GLM ==========")

    result = await llm_model_func(
        "只回答 OK"
    )

    print("GLM response:", result)


# ============================================================
# 加载文档
# ============================================================

def load_documents():
    documents = []
    file_paths = []

    for file in sorted(
            DOCS_DIR.glob("*.md")
    ):
        content = file.read_text(
            encoding="utf-8"
        )

        documents.append(content)

        file_paths.append(
            str(file)
        )

        print(
            f"[DOC] {file.name} "
            f"chars={len(content)}"
        )

    if not documents:
        raise RuntimeError(
            "docs 目录下没有 .md 文件"
        )

    return documents, file_paths


# ============================================================
# Indexing
#
# Document
#     ↓
# Chunk
#     ↓
# ┌─────────────────┐
# │ chunks_vdb      │
# └─────────────────┘
#
# Chunk
#     ↓
# GLM-5.3
#     ↓
# Entity / Relation Extraction
#     ↓
# Knowledge Graph
#     +
# entities_vdb / relationships_vdb
# ============================================================

async def build_index(
        rag: LightRAG,
):
    print(
        "\n========== INDEXING =========="
    )

    documents, file_paths = (
        load_documents()
    )

    track_id = await rag.ainsert(
        documents,
        file_paths=file_paths,
    )

    print(
        "\nIndexing finished."
    )

    print(
        "track_id:",
        track_id,
    )


# ============================================================
# 查看 Knowledge Graph Entity
# ============================================================

async def inspect_graph(
        rag: LightRAG,
):
    print(
        "\n========== KNOWLEDGE GRAPH =========="
    )

    labels = (
        await rag.get_graph_labels()
    )

    print(
        f"Entity count: {len(labels)}"
    )

    for index, label in enumerate(
            labels[:100],
            1,
    ):
        print(
            f"{index}. {label}"
        )

    await rag.aexport_data(
        "./results/kg.md",
        file_format="md",
    )

    print(
        "\nKG exported -> "
        "results/kg.md"
    )


# ============================================================
# 直接查询 entities_vdb
#
# 验证：
#
# Query
#   ↓
# BGE-M3
#   ↓
# entities_vdb
#   ↓
# 哪些 Entity 被召回
# ============================================================

async def inspect_entity_vdb(
        rag: LightRAG,
):
    query = (
        "TaskManager 网络内存"
    )

    print(
        "\n========== entities_vdb =========="
    )

    print(
        "query:",
        query,
    )

    results = (
        await rag.entities_vdb.query(
            query,
            top_k=10,
        )
    )

    for index, item in enumerate(
            results,
            1,
    ):
        print(
            f"\n#{index}"
        )

        print(
            "entity_name:",
            item.get(
                "entity_name"
            ),
        )

        print(
            "distance:",
            item.get(
                "distance"
            ),
        )

        print(
            "source_id:",
            item.get(
                "source_id"
            ),
        )

        print(
            "file_path:",
            item.get(
                "file_path"
            ),
        )

        print(
            "content:",
            item.get(
                "content"
            ),
        )


# ============================================================
# 直接查询 relationships_vdb
# ============================================================

async def inspect_relationship_vdb(
        rag: LightRAG,
):
    query = (
        "TaskManager Direct Memory "
        "和 Network Memory 的关系"
    )

    print(
        "\n========== relationships_vdb =========="
    )

    print(
        "query:",
        query,
    )

    results = (
        await rag.relationships_vdb.query(
            query,
            top_k=10,
        )
    )

    for index, item in enumerate(
            results,
            1,
    ):
        print(
            f"\n#{index}"
        )

        print(
            "src_id:",
            item.get(
                "src_id"
            ),
        )

        print(
            "tgt_id:",
            item.get(
                "tgt_id"
            ),
        )

        print(
            "distance:",
            item.get(
                "distance"
            ),
        )

        print(
            "source_id:",
            item.get(
                "source_id"
            ),
        )

        print(
            "file_path:",
            item.get(
                "file_path"
            ),
        )

        print(
            "content:",
            item.get(
                "content"
            ),
        )


# ============================================================
# 直接查询 chunks_vdb
#
# 这就是传统 Vector RAG
# ============================================================

async def inspect_chunk_vdb(
        rag: LightRAG,
):
    query = (
        "TaskManager Direct Memory OOM "
        "Network Memory Off Heap"
    )

    print(
        "\n========== chunks_vdb =========="
    )

    print(
        "query:",
        query,
    )

    results = (
        await rag.chunks_vdb.query(
            query,
            top_k=5,
        )
    )

    for index, item in enumerate(
            results,
            1,
    ):
        print(
            f"\n#{index}"
        )

        print(
            "distance:",
            item.get(
                "distance"
            ),
        )

        print(
            "file:",
            item.get(
                "file_path"
            ),
        )

        content = (
            item.get(
                "content",
                "",
            )
        )

        print(
            "content:",
            content[:800]
            .replace(
                "\n",
                " ",
            ),
        )


# ============================================================
# 查看完整 Retrieval Pipeline
#
# 不生成最终答案
#
# Query
#    ↓
# Keyword Extraction
#    ↓
# Entity Search
# Relation Search
# Chunk Search
#    ↓
# Graph
#    ↓
# Merge
#    ↓
# 最终 Context
# ============================================================

async def inspect_query_data(
        rag: LightRAG,
        question: str,
        mode: str,
):
    print(
        f"\n========== QUERY DATA [{mode}] =========="
    )

    print(
        "\nQuestion:"
    )

    print(
        question
    )

    result = (
        await rag.aquery_data(
            question,
            param=QueryParam(
                mode=mode,
                top_k=10,
                chunk_top_k=10,

                # 第一次测试先关闭 reranker
                enable_rerank=False,
            ),
        )
    )

    output_file = (
            RESULT_DIR
            / f"{mode}_query_data.json"
    )

    output_file.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    data = result.get(
        "data",
        {},
    )

    metadata = result.get(
        "metadata",
        {},
    )

    # ------------------------------
    # Keywords
    # ------------------------------

    print(
        "\n---------- Keywords ----------"
    )

    print(
        json.dumps(
            metadata.get(
                "keywords",
                {},
            ),
            ensure_ascii=False,
            indent=2,
        )
    )

    # ------------------------------
    # Entities
    # ------------------------------

    print(
        "\n---------- Entities ----------"
    )

    for entity in data.get(
            "entities",
            [],
    ):
        print(
            entity.get(
                "entity_name"
            ),
            "|",
            entity.get(
                "entity_type"
            ),
        )

        print(
            "description:",
            entity.get(
                "description"
            ),
        )

        print()

    # ------------------------------
    # Relationships
    # ------------------------------

    print(
        "\n---------- Relationships ----------"
    )

    for relation in data.get(
            "relationships",
            [],
    ):
        print(
            relation.get(
                "src_id"
            ),
            " -> ",
            relation.get(
                "tgt_id"
            ),
        )

        print(
            "keywords:",
            relation.get(
                "keywords"
            ),
        )

        print(
            "weight:",
            relation.get(
                "weight"
            ),
        )

        print(
            "description:",
            relation.get(
                "description"
            ),
        )

        print()

    # ------------------------------
    # Chunks
    # ------------------------------

    print(
        "\n---------- Chunks ----------"
    )

    for index, chunk in enumerate(
            data.get(
                "chunks",
                [],
            ),
            1,
    ):
        print(
            f"\nCHUNK #{index}"
        )

        print(
            "file:",
            chunk.get(
                "file_path"
            ),
        )

        print(
            "chunk_id:",
            chunk.get(
                "chunk_id"
            ),
        )

        print(
            chunk.get(
                "content",
                "",
            )[:1000]
        )

    # ------------------------------
    # Processing info
    # ------------------------------

    print(
        "\n---------- Processing Info ----------"
    )

    print(
        json.dumps(
            metadata.get(
                "processing_info",
                {},
            ),
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "\nSaved ->",
        output_file,
    )

    return result


# ============================================================
# naive VS mix
#
# naive:
# Query -> chunks_vdb
#
# mix:
# Query
#   ├─ entities_vdb
#   ├─ relationships_vdb
#   ├─ Knowledge Graph
#   └─ chunks_vdb
# ============================================================

async def compare_answers(
        rag: LightRAG,
        question: str,
):
    print(
        "\n======================================"
    )

    print(
        "NAIVE VS MIX"
    )

    print(
        "======================================"
    )

    naive_answer = (
        await rag.aquery(
            question,
            param=QueryParam(
                mode="naive",
                chunk_top_k=10,
                enable_rerank=False,
            ),
        )
    )

    mix_answer = (
        await rag.aquery(
            question,
            param=QueryParam(
                mode="mix",
                top_k=10,
                chunk_top_k=10,
                enable_rerank=False,
            ),
        )
    )

    print(
        "\n========== NAIVE ==========\n"
    )

    print(
        naive_answer
    )

    print(
        "\n========== MIX ==========\n"
    )

    print(
        mix_answer
    )


# ============================================================
# 一个简单的 Entity Recall 测试
# ============================================================

EXPECTED_ENTITIES = {
    "TASKMANAGER",
    "网络内存",
    "任务堆外内存",
    "框架堆外内存",
}


def evaluate_entities(
        result,
):
    entities = (
        result
        .get(
            "data",
            {},
        )
        .get(
            "entities",
            [],
        )
    )

    retrieved = {
        str(
            entity.get(
                "entity_name",
                "",
            )
        ).upper()
        for entity in entities
    }

    found = (
            EXPECTED_ENTITIES
            & retrieved
    )

    missing = (
            EXPECTED_ENTITIES
            - retrieved
    )

    recall = (
            len(found)
            / len(
        EXPECTED_ENTITIES
    )
    )

    print(
        "\n========== ENTITY EVALUATION =========="
    )

    print(
        "Expected:",
        EXPECTED_ENTITIES,
    )

    print(
        "Found:",
        found,
    )

    print(
        "Missing:",
        missing,
    )

    print(
        f"Entity Recall = "
        f"{len(found)}"
        f"/"
        f"{len(EXPECTED_ENTITIES)}"
        f" = "
        f"{recall:.2%}"
    )


def evaluate_entity_recall(query_data):
    entities = query_data.get(
        "data", {}
    ).get(
        "entities", []
    )

    retrieved = {
        str(x.get("entity_name", "")).strip().lower()
        for x in entities
    }

    expected = {
        x.lower()
        for x in GROUND_TRUTH["expected_entities"]
    }

    found = expected & retrieved
    missing = expected - retrieved

    recall = (
        len(found) / len(expected)
        if expected else 0
    )

    return {
        "entity_recall": recall,
        "found": found,
        "missing": missing,
    }


def build_retrieval_context(query_data):
    data = query_data.get("data", {})

    parts = []

    for entity in data.get("entities", []):
        parts.append(
            f"""
ENTITY:
name={entity.get("entity_name")}
description={entity.get("description")}
"""
        )

    for relation in data.get("relationships", []):
        parts.append(
            f"""
RELATION:
{relation.get("src_id")}
->
{relation.get("tgt_id")}

description={relation.get("description")}
"""
        )

    for chunk in data.get("chunks", []):
        parts.append(
            f"""
CHUNK:
{chunk.get("content")}
"""
        )

    return "\n".join(parts)


async def judge_facts(text: str):
    facts = GROUND_TRUTH["must_have_facts"]

    prompt = f"""
你是 RAG Retrieval Evaluation Judge。

下面有一组标准事实，以及 Retriever 召回的 Context。

你只需要判断：
每条标准事实能否从 Context 中得到支持。

不要使用你自己的知识。
只能根据 Context 判断。

标准事实：

{json.dumps(
        facts,
        ensure_ascii=False,
        indent=2
    )}

Context:

{text}

必须严格返回 JSON：

{{
  "results": [
    {{
      "fact": "...",
      "supported": true
    }}
  ]
}}
"""

    result = await llm_model_func(prompt)

    result = result.strip()

    # 防止模型返回 ```json
    if result.startswith("```"):
        result = result.strip("`")
        if result.startswith("json"):
            result = result[4:].strip()

    data = json.loads(result)

    supported = [
        x
        for x in data["results"]
        if x["supported"]
    ]

    recall = (
            len(supported)
            / len(facts)
    )

    return {
        "fact_recall": recall,
        "details": data["results"],
    }


async def judge_answer(answer: str):
    must_have = GROUND_TRUTH[
        "must_have_facts"
    ]

    incorrect = GROUND_TRUTH[
        "incorrect_claims"
    ]

    prompt = f"""
你是 RAG Answer Evaluation Judge。

只能评价下面这个答案。

必须判断：

1. 每个正确事实是否在答案中得到表达
2. 每个错误说法是否出现在答案中

正确事实：

{json.dumps(
        must_have,
        ensure_ascii=False,
        indent=2
    )}

错误事实：

{json.dumps(
        incorrect,
        ensure_ascii=False,
        indent=2
    )}

待评价答案：

{answer}

严格返回 JSON：

{{
  "correct_facts": [
    {{
      "fact": "...",
      "present": true
    }}
  ],

  "incorrect_claims": [
    {{
      "claim": "...",
      "present": false
    }}
  ]
}}
"""

    result = await llm_model_func(prompt)

    result = result.strip()

    if result.startswith("```"):
        result = result.strip("`")
        if result.startswith("json"):
            result = result[4:].strip()

    data = json.loads(result)

    correct_count = sum(
        1
        for x in data["correct_facts"]
        if x["present"]
    )

    incorrect_count = sum(
        1
        for x in data["incorrect_claims"]
        if x["present"]
    )

    return {
        "fact_coverage":
            correct_count / len(must_have),

        "incorrect_count":
            incorrect_count,

        "details":
            data,
    }


async def evaluate_mode(
        rag,
        mode,
):
    question = GROUND_TRUTH["question"]

    print(
        f"\n========== EVAL {mode} =========="
    )

    query_data = await rag.aquery_data(
        question,
        param=QueryParam(
            mode=mode,
            top_k=10,
            chunk_top_k=10,
            enable_rerank=False,
        ),
    )

    answer = await rag.aquery(
        question,
        param=QueryParam(
            mode=mode,
            top_k=10,
            chunk_top_k=10,
            enable_rerank=False,
        ),
    )

    context = build_retrieval_context(
        query_data
    )

    retrieval_eval = await judge_facts(
        context
    )

    answer_eval = await judge_answer(
        answer
    )

    result = {
        "mode": mode,

        "retrieval_fact_recall":
            retrieval_eval["fact_recall"],

        "answer_fact_coverage":
            answer_eval["fact_coverage"],

        "incorrect_claims":
            answer_eval["incorrect_count"],
    }

    # graph 模式才计算 Entity Recall
    if mode != "naive":
        entity_eval = (
            evaluate_entity_recall(
                query_data
            )
        )

        result["entity_recall"] = (
            entity_eval["entity_recall"]
        )

    return result


async def run_evaluation(rag):
    modes = [
        "naive",
        "local",
        "global",
        "hybrid",
        "mix",
    ]

    results = []

    for mode in modes:
        result = await evaluate_mode(
            rag,
            mode,
        )

        results.append(result)

    print(
        "\n========== FINAL RESULT =========="
    )

    for r in results:
        print(r)


# ============================================================
# main
# ============================================================

async def main():
    # ------------------------------
    # 检查配置
    # ------------------------------

    required_envs = [
        "LLM_BINDING_API_KEY",
        "LLM_BINDING_HOST",
        "EMBEDDING_BINDING_API_KEY",
        "EMBEDDING_BINDING_HOST",
    ]

    for name in required_envs:
        if not os.getenv(name):
            raise RuntimeError(
                f".env 缺少 {name}"
            )

    rag = await create_rag()

    try:

        # 1.
        # 先单独测试两个远程 API
        await check_embedding(rag)

        await check_llm()

        # 2.
        # 建索引
        await build_index(
            rag
        )

        # 3.
        # 查看 Index 结果
        await inspect_graph(
            rag
        )

        # 4.
        # 单独看三个 VDB
        await inspect_entity_vdb(
            rag
        )

        await inspect_relationship_vdb(
            rag
        )

        await inspect_chunk_vdb(
            rag
        )

        # 5.
        # 正式测试问题
        question = (
            "Flink TaskManager 的框架堆外内存、任务堆外内存和网络内存分别如何管理？它们是否存在物理隔离？当直接内存溢出时，为什么调整其中一个配置可能影响其他区域可用的直接内存？"
        )

        # ---------------------------------
        # 普通 Vector RAG
        # ---------------------------------

        await inspect_query_data(
            rag,
            question,
            "naive",
        )

        # ---------------------------------
        # LightRAG Graph + Vector
        # ---------------------------------

        mix_data = (
            await inspect_query_data(
                rag,
                question,
                "mix",
            )
        )

        # 6.
        # 简单 Retriever Evaluation
        evaluate_entities(
            mix_data
        )

        # 7.
        # 最终答案对比
        await compare_answers(
            rag,
            question,
        )

        # 8.
        # 评估所有模式
        await run_evaluation(rag)

    finally:

        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(
        main()
    )
