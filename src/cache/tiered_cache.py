import hashlib
import json
import os
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from src.utils.config import CACHE_DIR, CACHE_THRESHOLD
from src.utils.llm import cosine_similarity


@dataclass
class CacheEntry:
    query_hash: str
    query: str
    query_embedding: List[float]
    answer: str
    thinking: str
    sources: List[str]
    retrieval_results: Optional[List[str]] = None
    timestamp: float = 0.0


class TieredCache:
    def __init__(self, persist_dir: str = CACHE_DIR, threshold: float = CACHE_THRESHOLD):
        self.persist_dir = persist_dir
        self.threshold = threshold
        self.answer_cache: List[CacheEntry] = []
        self.embedding_cache: Dict[str, List[float]] = {}
        self.retrieval_cache: Dict[str, List[str]] = {}
        os.makedirs(persist_dir, exist_ok=True)
        self._load()

    def _cache_path(self) -> str:
        return os.path.join(self.persist_dir, "semantic_cache.json")

    def _load(self):
        path = self._cache_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.answer_cache = [CacheEntry(**e) for e in data.get("answers", [])]
                self.embedding_cache = data.get("embeddings", {})
                self.retrieval_cache = data.get("retrievals", {})
            except Exception:
                pass

    def _save(self):
        try:
            path = self._cache_path()
            data = {
                "answers": [asdict(e) for e in self.answer_cache[-100:]],
                "embeddings": dict(list(self.embedding_cache.items())[-500:]),
                "retrievals": dict(list(self.retrieval_cache.items())[-200:]),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception:
            pass

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get_embedding(self, text: str) -> Optional[List[float]]:
        return self.embedding_cache.get(self._hash(text))

    def put_embedding(self, text: str, embedding: List[float]):
        self.embedding_cache[self._hash(text)] = embedding

    def get_retrieval(self, query: str) -> Optional[List[str]]:
        return self.retrieval_cache.get(self._hash(query))

    def put_retrieval(self, query: str, doc_texts: List[str]):
        self.retrieval_cache[self._hash(query)] = doc_texts

    def lookup_answer(self, query_embedding: List[float]) -> Optional[CacheEntry]:
        best, best_sim = None, 0.0
        for entry in self.answer_cache:
            sim = cosine_similarity(query_embedding, entry.query_embedding)
            if sim > best_sim:
                best, best_sim = entry, sim
        if best and best_sim >= self.threshold:
            best.similarity = best_sim
            return best
        return None

    def store_answer(
        self,
        query: str,
        query_embedding: List[float],
        answer: str,
        thinking: str,
        sources: List[str],
        retrieval_results: Optional[List[str]] = None,
    ):
        entry = CacheEntry(
            query_hash=self._hash(query),
            query=query,
            query_embedding=query_embedding,
            answer=answer,
            thinking=thinking,
            sources=sources,
            retrieval_results=retrieval_results,
        )
        self.answer_cache.append(entry)
        if len(self.answer_cache) > 100:
            self.answer_cache.pop(0)
        self._save()

    def clear(self):
        self.answer_cache = []
        self.embedding_cache = {}
        self.retrieval_cache = {}
        self._save()

    def stats(self) -> Dict[str, int]:
        return {
            "answers": len(self.answer_cache),
            "embeddings": len(self.embedding_cache),
            "retrievals": len(self.retrieval_cache),
        }
