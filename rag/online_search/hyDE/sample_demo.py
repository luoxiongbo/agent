import os

from sentence_transformers import SentenceTransformer
try:
    from zhipuai import ZhipuAI
except ImportError:
    ZhipuAI = None


DEFAULT_ZHIPU_MODEL = "glm-4.7-flash"
SAMPLE_DOCS = [
    "保险销售前需要先理解客户家庭结构、风险承受能力和保障缺口，再推荐合适产品。",
    "有效的保险销售沟通应重点说明保障责任、免责条款、缴费压力和理赔流程。",
    "客户异议处理可以围绕预算、信任、产品复杂度和长期收益进行耐心解释。",
    "RAG 检索系统通常包含文档切分、向量化、相似度召回和生成回答等步骤。",
]


def get_api_key() -> str:
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置 ZHIPUAI_API_KEY。")

    return api_key


def get_model() -> str:
    return os.getenv("ZHIPUAI_MODEL") or DEFAULT_ZHIPU_MODEL


def create_client():
    if ZhipuAI is None:
        raise RuntimeError("当前环境未安装 zhipuai，请先安装：pip install zhipuai")

    return ZhipuAI(api_key=get_api_key())


def create_chat_completion(query: str, client, model: str) -> str:
    prompt = f"请针对以下问题给出详细的回答：{query}"
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


def generate_hypothetical_doc(query: str, client, model: str = None) -> str:
    model = model or get_model()
    return create_chat_completion(query, client, model)


def hyde_retrieval(query: str, embed_model, doc_embeddings, doc_ids, client=None):
    client = client or create_client()
    hypo_doc = generate_hypothetical_doc(query, client)

    query_vector = embed_model.encode(hypo_doc)

    import numpy as np

    sims = np.dot(doc_embeddings, query_vector)
    top_idx = sims.argsort()[-5:][::-1]
    results = [(doc_ids[i], sims[i]) for i in top_idx]
    return results


def build_sample_index(embed_model):
    import numpy as np

    doc_embeddings = np.asarray(embed_model.encode(SAMPLE_DOCS))
    doc_ids = [f"doc_{i + 1}: {doc}" for i, doc in enumerate(SAMPLE_DOCS)]

    return doc_embeddings, doc_ids


def main():
    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    doc_embeddings, doc_ids = build_sample_index(embed_model)
    query = "保险销售技巧"
    retrieved_docs = hyde_retrieval(query, embed_model, doc_embeddings, doc_ids)
    print("HyDE 检索结果文档ID及分数：", retrieved_docs)


if __name__ == "__main__":
    main()
