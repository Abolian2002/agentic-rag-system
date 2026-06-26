from langgraph.graph import END
from src.graph.state import RAGState


def after_route(state: RAGState) -> str:
    return "check_cache"


def after_cache(state: RAGState) -> str:
    if state.get("cache_hit"):
        return "finalize"
    route = state.get("route", "rag")
    if route == "chat":
        return "generate"
    return "transform"


def after_retrieve(state: RAGState) -> str:
    if state.get("needs_retry"):
        return "transform"
    return "generate"


def after_generate(state: RAGState) -> str:
    return "verify"


def after_verify(state: RAGState) -> str:
    return "finalize"


def after_finalize(state: RAGState) -> str:
    return END
