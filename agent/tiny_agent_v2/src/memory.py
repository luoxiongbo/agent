from typing import Dict, List, Optional, Tuple
from uuid import uuid4


class MemoryManager:
    def __init__(self, client, store, vector_store):
        self.client = client
        self.store = store
        self.vector_store = vector_store

    def recall(self, user_id: str, query: str, top_k: int = 5) -> List[Dict]:
        chroma_ids = self.vector_store.search(
            query_text=query,
            top_k=top_k,
            where={"user_id": user_id},
        )
        memories = self.store.get_memories_by_chroma_ids(chroma_ids)
        if memories:
            self.store.update_access_time([memory["id"] for memory in memories])
        return memories

    def write_back(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        candidates = self._extract_candidates(
            user_message=user_message,
            assistant_message=assistant_message,
        )

        for memory_type, content, importance in candidates:
            if self.store.memory_exists(
                user_id=user_id,
                memory_type=memory_type,
                content=content,
            ):
                continue

            chroma_id = str(uuid4())
            self.vector_store.add_text(
                doc_id=chroma_id,
                text=content,
                metadata={
                    "user_id": user_id,
                    "memory_type": memory_type,
                    "importance": importance,
                },
            )
            self.store.add_memory(
                user_id=user_id,
                memory_type=memory_type,
                content=content,
                importance=importance,
                chroma_id=chroma_id,
            )

    def format_for_prompt(self, memories: List[Dict]) -> str:
        if not memories:
            return ""

        lines = []
        for memory in memories:
            lines.append(
                f"- [{memory['memory_type']}] {memory['content']}"
            )
        return "\n".join(lines)

    def _extract_candidates(
        self,
        user_message: str,
        assistant_message: str,
    ) -> List[Tuple[str, str, float]]:
        del assistant_message

        text = user_message.strip()
        if len(text) < 6:
            return []

        preference_markers = [
            "请用",
            "以后",
            "希望",
            "喜欢",
            "尽量",
            "优先",
            "不要",
            "最好",
            "回答时",
            "用中文",
            "用英文",
        ]
        task_markers = [
            "正在",
            "在做",
            "项目",
            "开发",
            "实现",
            "学习",
            "准备",
            "计划",
            "想做",
            "需要",
            "打算",
        ]
        fact_markers = [
            "我是",
            "我叫",
            "我的名字",
            "我在",
            "我来自",
            "我使用",
            "我用",
            "我负责",
            "我的方向",
        ]

        candidates: List[Tuple[str, str, float]] = []
        if any(marker in text for marker in preference_markers):
            candidates.append(("preference", text, 0.9))
        elif any(marker in text for marker in task_markers):
            candidates.append(("task", text, 0.75))
        elif any(marker in text for marker in fact_markers):
            candidates.append(("fact", text, 0.7))

        return candidates
