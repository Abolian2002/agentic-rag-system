from typing import TypedDict, Annotated, Optional, List, Dict, Any
from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class QueryRoute(TypedDict):
    route: str
    reason: str


class RAGState(TypedDict, total=False):
    query: str
    chat_history: List[Dict[str, str]]
    route: str
    transformed_queries: List[str]
    retrieval_attempt: int
    documents: List[Document]
    relevant_documents: List[Document]
    confidence: float
    generation: str
    thinking: str
    sources: List[str]
    cache_hit: bool
    steps: List[Dict[str, Any]]
    error: Optional[str]
    needs_retry: bool
    final_answer: str
    citation_verified: bool
