from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever

from src.ingestion.document_processor import DocumentProcessor
from src.retrieval.knowledge_graph import retrieve_from_graph
from src.retrieval.advanced_rag import (
    generate_query_variants,
    reciprocal_rank_fusion,
    expand_query_hyde,
    grade_document_relevance,
)


class RetrieverPipeline:
    def __init__(
        self,
        doc_processor: DocumentProcessor,
        reranker=None,
        ollama_uri: str = "",
        model: str = "",
    ):
        self.proc = doc_processor
        self.reranker = reranker
        self.ollama_uri = ollama_uri
        self.model = model

    def _ensemble_retrieve(self, query: str, k: int = 6) -> List[Document]:
        retrievers = []
        weights = []

        if self.proc.bm25_retriever is not None:
            retrievers.append(self.proc.bm25_retriever)
            weights.append(0.35)

        if self.proc.vector_store is not None:
            retrievers.append(
                self.proc.vector_store.as_retriever(search_kwargs={"k": k})
            )
            weights.append(0.65)

        if not retrievers:
            return []

        if len(retrievers) == 1:
            return retrievers[0].invoke(query)

        ensemble = EnsembleRetriever(retrievers=retrievers, weights=weights)
        return ensemble.invoke(query)

    def retrieve(
        self,
        query: str,
        enable_hyde: bool = True,
        enable_fusion: bool = False,
        enable_graph_rag: bool = True,
        enable_reranking: bool = True,
        enable_crag: bool = False,
        max_contexts: int = 4,
        is_retry: bool = False,
    ) -> Tuple[List[Document], dict]:
        debug_info = {
            "hyde_used": False,
            "fusion_used": False,
            "graph_docs_added": 0,
            "reranked": False,
            "crag_results": None,
            "total_retrieved": 0,
        }

        if enable_fusion and not is_retry:
            debug_info["fusion_used"] = True
            variants = generate_query_variants(
                query, self.ollama_uri, self.model, n=3
            )
            ranked_lists = []
            for v in variants:
                try:
                    ranked_lists.append(self._ensemble_retrieve(v, k=6))
                except Exception:
                    continue
            docs = reciprocal_rank_fusion(ranked_lists) if ranked_lists else []
        else:
            search_query = query
            if enable_hyde and not is_retry:
                debug_info["hyde_used"] = True
                search_query = expand_query_hyde(
                    query, self.ollama_uri, self.model
                )
            k = 8 if is_retry else 6
            docs = self._ensemble_retrieve(search_query, k=k)

        if enable_graph_rag and self.proc.knowledge_graph:
            graph_docs = retrieve_from_graph(
                query,
                self.proc.knowledge_graph,
                self.proc.doc_chunks,
                top_k=3,
            )
            if graph_docs:
                existing = {d.page_content[:200] for d in docs}
                added = 0
                for gdoc in graph_docs:
                    if gdoc.page_content[:200] not in existing:
                        docs.append(gdoc)
                        existing.add(gdoc.page_content[:200])
                        added += 1
                debug_info["graph_docs_added"] = added

        debug_info["total_retrieved"] = len(docs)

        if enable_reranking and self.reranker and docs:
            try:
                pairs = [[query, d.page_content] for d in docs]
                scores = self.reranker.predict(pairs)
                doc_scores = list(zip(scores, docs))
                doc_scores.sort(key=lambda x: x[0], reverse=True)
                docs = [d for _, d in doc_scores]
                debug_info["reranked"] = True
            except Exception:
                pass

        candidates = docs[:max_contexts]

        if enable_crag and candidates:
            graded = []
            relevant = []
            for d in candidates:
                ok, reason = grade_document_relevance(
                    query, d.page_content, self.ollama_uri, self.model
                )
                graded.append({"doc": d, "relevant": ok, "reason": reason})
                if ok:
                    relevant.append(d)
            if relevant:
                debug_info["crag_results"] = {
                    "status": "ok",
                    "kept": len(relevant),
                    "total": len(candidates),
                }
                candidates = relevant
            else:
                debug_info["crag_results"] = {
                    "status": "low",
                    "kept": 0,
                    "total": len(candidates),
                }
                candidates = candidates[:1]

        return candidates, debug_info
