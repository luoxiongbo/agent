import os
from typing import Dict, List, Optional

import chromadb


class VectorStore:
    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "tinyagent_memory",
    ):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_text(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        self.collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {}],
        )

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[str]:
        document_count = self.collection.count()
        if document_count == 0:
            return []

        result_count = min(top_k, document_count)
        query_kwargs = {
            "query_texts": [query_text],
            "n_results": result_count,
        }
        if where:
            query_kwargs["where"] = where

        result = self.collection.query(**query_kwargs)
        ids = result.get("ids", [])
        return ids[0] if ids else []

    def delete(self, doc_id: str) -> None:
        self.collection.delete(ids=[doc_id])

    def update(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        self.add_text(doc_id=doc_id, text=text, metadata=metadata)
