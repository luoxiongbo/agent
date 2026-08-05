# features_demo.pdf

<!-- page: 1 -->

# RAG Course Contents

1.1 Overview
1.2 Offline Flow

1.3 Online Flow
2.1 Query Module

2.2 Query Rewrite
2.3 Query Expansion

3.1 Parsing
4.1 Retrieval

<!-- page: 2 -->

## 1.2 Offline Flow

········:1978201801

Offline processing includes:

1. Document store: keep source files

2. Embedding: create vectors

3. Vector DB: build an index

Offline flow runs once or updates periodically.

<!-- page: 3 -->

## 3.4 Code Example

import openai
from sentence_transformers import SentenceTransformer
def retrieve(query):
    embedding = model.encode(query)
    return embedding
