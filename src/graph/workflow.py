from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import RAGState
from src.graph.nodes import (
    node_route_query,
    node_check_cache,
    node_transform_query,
    node_retrieve,
    node_generate_answer,
    node_verify_citations,
    node_finalize,
)
from src.graph.edges import (
    after_route,
    after_cache,
    after_retrieve,
    after_generate,
    after_verify,
)


def build_rag_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("route", node_route_query)
    workflow.add_node("check_cache", node_check_cache)
    workflow.add_node("transform", node_transform_query)
    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("generate", node_generate_answer)
    workflow.add_node("verify", node_verify_citations)
    workflow.add_node("finalize", node_finalize)

    workflow.add_edge(START, "route")

    workflow.add_conditional_edges(
        "route",
        after_route,
        {
            "check_cache": "check_cache",
        },
    )

    workflow.add_conditional_edges(
        "check_cache",
        after_cache,
        {
            "finalize": "finalize",
            "transform": "transform",
            "generate": "generate",
        },
    )

    workflow.add_edge("transform", "retrieve")

    workflow.add_conditional_edges(
        "retrieve",
        after_retrieve,
        {
            "transform": "transform",
            "generate": "generate",
        },
    )

    workflow.add_conditional_edges(
        "generate",
        after_generate,
        {
            "verify": "verify",
        },
    )

    workflow.add_conditional_edges(
        "verify",
        after_verify,
        {
            "finalize": "finalize",
        },
    )

    workflow.add_edge("finalize", END)

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    return app


_global_graph = None


def get_graph():
    global _global_graph
    if _global_graph is None:
        _global_graph = build_rag_graph()
    return _global_graph


def run_rag_query(
    query: str,
    chat_history: list,
    config: dict,
    thread_id: str = "default",
):
    graph = get_graph()
    initial_state: RAGState = {
        "query": query,
        "chat_history": chat_history,
        "retrieval_attempt": 0,
        "steps": [],
        "documents": [],
        "relevant_documents": [],
        "sources": [],
        "cache_hit": False,
        "needs_retry": False,
        "citation_verified": False,
    }
    run_config = {
        "configurable": {
            "thread_id": thread_id,
            **config,
        }
    }
    result = graph.invoke(initial_state, run_config)
    return result
