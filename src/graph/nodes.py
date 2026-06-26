import time
from typing import Dict, Any
from langchain_core.documents import Document
from langgraph.graph import END
from src.graph.state import RAGState
from src.retrieval.advanced_rag import route_query, verify_citations
from src.utils.config import MAX_RETRIEVAL_RETRIES


def _add_step(state: RAGState, name: str, info: Dict[str, Any] = None) -> list:
    steps = state.get("steps", [])
    steps.append({"step": name, "time": time.time(), "info": info or {}})
    return steps


def node_route_query(state: RAGState, config) -> Dict[str, Any]:
    query = state["query"]
    llm_uri = config["configurable"]["ollama_uri"]
    model = config["configurable"]["model"]
    has_docs = config["configurable"].get("has_documents", False)

    if not has_docs:
        return {
            "route": "chat",
            "steps": _add_step(state, "route", {"route": "chat", "reason": "no documents loaded"}),
        }

    try:
        route_result = route_query(query, llm_uri, model)
        route = route_result["route"]
        reason = route_result["reason"]
    except Exception:
        route = "rag"
        reason = "default to rag"

    return {
        "route": route,
        "steps": _add_step(state, "route", {"route": route, "reason": reason}),
    }


def node_check_cache(state: RAGState, config) -> Dict[str, Any]:
    from src.utils.config import CACHE_THRESHOLD

    cache = config["configurable"].get("cache")
    embeddings = config["configurable"].get("embeddings")
    query = state["query"]

    if not config["configurable"].get("enable_cache", False) or cache is None:
        return {
            "cache_hit": False,
            "steps": _add_step(state, "cache_check", {"skipped": True}),
        }

    try:
        q_emb = cache.get_embedding(query)
        if q_emb is None and embeddings is not None:
            q_emb = embeddings.embed_query(query)
            cache.put_embedding(query, q_emb)
    except Exception:
        q_emb = None

    if q_emb is None:
        return {
            "cache_hit": False,
            "steps": _add_step(state, "cache_check", {"error": "embedding failed"}),
        }

    hit = cache.lookup_answer(q_emb)
    if hit:
        return {
            "cache_hit": True,
            "final_answer": hit.answer,
            "thinking": hit.thinking,
            "sources": hit.sources,
            "confidence": 1.0,
            "steps": _add_step(
                state, "cache_check", {"hit": True, "similarity": f"{hit.similarity:.1%}"}
            ),
        }

    return {
        "cache_hit": False,
        "steps": _add_step(state, "cache_check", {"hit": False}),
    }


def node_transform_query(state: RAGState, config) -> Dict[str, Any]:
    from src.retrieval.advanced_rag import generate_query_variants, expand_query_hyde

    query = state["query"]
    llm_uri = config["configurable"]["ollama_uri"]
    model = config["configurable"]["model"]
    enable_hyde = config["configurable"].get("enable_hyde", True)
    enable_fusion = config["configurable"].get("enable_fusion", False)
    is_retry = state.get("retrieval_attempt", 0) > 0

    transformed = [query]

    if enable_fusion and not is_retry:
        try:
            variants = generate_query_variants(query, llm_uri, model, n=3)
            transformed = variants if variants else [query]
        except Exception:
            transformed = [query]
    elif enable_hyde and not is_retry:
        try:
            expanded = expand_query_hyde(query, llm_uri, model)
            transformed = [expanded]
        except Exception:
            transformed = [query]
    elif is_retry:
        transformed = [f"Tell me about: {query}"]

    return {
        "transformed_queries": transformed,
        "steps": _add_step(
            state,
            "transform",
            {"queries": transformed[:3], "is_retry": is_retry},
        ),
    }


def node_retrieve(state: RAGState, config) -> Dict[str, Any]:
    retriever = config["configurable"].get("retriever")
    query = state["query"]
    attempt = state.get("retrieval_attempt", 0)
    is_retry = attempt > 0

    enable_hyde = config["configurable"].get("enable_hyde", True)
    enable_fusion = config["configurable"].get("enable_fusion", False)
    enable_graph = config["configurable"].get("enable_graph_rag", True)
    enable_rerank = config["configurable"].get("enable_reranking", True)
    enable_crag = config["configurable"].get("enable_crag", False)
    max_contexts = config["configurable"].get("max_contexts", 4)

    if retriever is None:
        return {
            "documents": [],
            "relevant_documents": [],
            "confidence": 0.0,
            "needs_retry": False,
            "steps": _add_step(state, "retrieve", {"error": "no retriever"}),
        }

    docs, debug = retriever.retrieve(
        query=query,
        enable_hyde=enable_hyde,
        enable_fusion=enable_fusion,
        enable_graph_rag=enable_graph,
        enable_reranking=enable_rerank,
        enable_crag=False,
        max_contexts=max_contexts + (2 if is_retry else 0),
        is_retry=is_retry,
    )

    crag_status = None
    if enable_crag and docs:
        from src.retrieval.advanced_rag import grade_document_relevance

        llm_uri = config["configurable"]["ollama_uri"]
        model = config["configurable"]["model"]
        relevant = []
        for d in docs:
            ok, reason = grade_document_relevance(query, d.page_content, llm_uri, model)
            if ok:
                relevant.append(d)
        if relevant:
            crag_status = {"status": "ok", "kept": len(relevant), "total": len(docs)}
            docs = relevant
            confidence = 0.8
        else:
            crag_status = {"status": "low", "kept": 0, "total": len(docs)}
            confidence = 0.2
            docs = docs[:1]
    else:
        confidence = 0.7 if docs else 0.0

    needs_retry = (
        enable_crag
        and crag_status
        and crag_status["status"] == "low"
        and attempt < MAX_RETRIEVAL_RETRIES
    )

    return {
        "documents": docs,
        "relevant_documents": docs,
        "confidence": confidence,
        "needs_retry": needs_retry,
        "retrieval_attempt": attempt + 1,
        "steps": _add_step(
            state,
            "retrieve",
            {
                "found": len(docs),
                "crag": crag_status,
                "attempt": attempt + 1,
                "needs_retry": needs_retry,
                **debug,
            },
        ),
    }


def node_generate_answer(state: RAGState, config) -> Dict[str, Any]:
    import requests
    import json

    query = state["query"]
    llm_uri = config["configurable"]["ollama_uri"]
    model = config["configurable"]["model"]
    chat_history = state.get("chat_history", [])
    route = state.get("route", "rag")
    docs = state.get("relevant_documents", [])
    temperature = config["configurable"].get("temperature", 0.3)
    show_thinking = config["configurable"].get("enable_thinking", True)
    confidence = state.get("confidence", 0.5)

    chat_hist_str = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in chat_history[-6:]
    )

    if route == "summary" and docs:
        context = "\n\n".join(
            f"[Source {i+1}]:\n{d.page_content}" for i, d in enumerate(docs)
        )
        system_prompt = (
            "You are a helpful assistant. Summarize the key points from the provided documents.\n\n"
            f"Chat History:\n{chat_hist_str}\n\n"
            f"Documents:\n{context}\n\n"
            "Provide a clear, well-structured summary."
        )
    elif route == "rag" and docs:
        context = "\n\n".join(
            f"[Source {i+1}]:\n{d.page_content}" for i, d in enumerate(docs)
        )
        think_instruction = ""
        if show_thinking:
            think_instruction = (
                "Before answering, reason through the problem step by step inside <think>...</think> tags. "
                "Then give your final answer outside those tags.\n"
            )
        system_prompt = (
            f"{think_instruction}"
            f"You are a helpful, thorough AI assistant.\n\n"
            f"Chat History:\n{chat_hist_str}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Instructions:\n"
            f"- Answer based on the provided context. Cite [Source N] when referencing specific information.\n"
            f"- Be concise and well-structured.\n"
            f"- If you don't know from the context, say so clearly.\n"
        )
        if confidence < 0.3:
            system_prompt += "- Note: The retrieved documents may not be fully relevant; state uncertainty if needed.\n"
        if show_thinking:
            system_prompt += "- Put ALL reasoning inside <think>...</think>; the final answer goes after.\n"
    else:
        think_instruction = ""
        if show_thinking:
            think_instruction = (
                "Before answering, reason through the problem inside <think>...</think> tags.\n"
            )
        system_prompt = (
            f"{think_instruction}"
            f"You are a helpful AI assistant.\n\n"
            f"Chat History:\n{chat_hist_str}\n\n"
            f"Question: {query}\n\n"
            "Answer the question helpfully."
        )
        if show_thinking:
            system_prompt += "\nPut ALL reasoning inside <think>...</think>."

    thinking = ""
    answer = ""
    buffer = ""
    in_think = False
    think_done = False

    try:
        resp = requests.post(
            llm_uri,
            json={
                "model": model,
                "prompt": system_prompt,
                "stream": True,
                "options": {"temperature": temperature, "num_ctx": 4096},
            },
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode())
            token = data.get("response", "")
            buffer += token

            changed = True
            while changed:
                changed = False
                if not in_think and not think_done:
                    idx = buffer.find("<think>")
                    if idx != -1:
                        pre = buffer[:idx].strip()
                        if pre:
                            answer += pre
                        buffer = buffer[idx + 7 :]
                        in_think = True
                        changed = True
                    else:
                        safe = max(0, len(buffer) - 7)
                        answer += buffer[:safe]
                        buffer = buffer[safe:]
                elif in_think:
                    idx = buffer.find("</think>")
                    if idx != -1:
                        thinking += buffer[:idx]
                        buffer = buffer[idx + 8 :]
                        in_think = False
                        think_done = True
                        changed = True
                    else:
                        safe = max(0, len(buffer) - 8)
                        thinking += buffer[:safe]
                        buffer = buffer[safe:]
                else:
                    answer += buffer
                    buffer = ""

            if data.get("done"):
                if in_think:
                    thinking += buffer
                else:
                    answer += buffer
                break

        thinking = thinking.strip()
        answer = answer.strip()
    except Exception as e:
        answer = f"Generation error: {str(e)}"

    sources = [d.page_content for d in docs]
    return {
        "generation": answer,
        "thinking": thinking,
        "sources": sources,
        "final_answer": answer,
        "citation_verified": False,
        "steps": _add_step(
            state, "generate", {"answer_len": len(answer), "has_thinking": bool(thinking)}
        ),
    }


def node_verify_citations(state: RAGState, config) -> Dict[str, Any]:
    enable_verify = config["configurable"].get("enable_citation_verify", False)
    llm_uri = config["configurable"]["ollama_uri"]
    model = config["configurable"]["model"]
    answer = state.get("final_answer", "")
    sources = state.get("sources", [])

    if not enable_verify or not sources:
        return {
            "citation_verified": True,
            "steps": _add_step(state, "verify", {"skipped": True}),
        }

    try:
        result = verify_citations(answer, sources, llm_uri, model)
        return {
            "citation_verified": result["verified"],
            "steps": _add_step(
                state,
                "verify",
                {"verified": result["verified"], "issues": result["issues"][:2] if result["issues"] else []},
            ),
        }
    except Exception:
        return {
            "citation_verified": True,
            "steps": _add_step(state, "verify", {"error": "verification failed"}),
        }


def node_finalize(state: RAGState, config) -> Dict[str, Any]:
    cache = config["configurable"].get("cache")
    embeddings = config["configurable"].get("embeddings")
    query = state["query"]
    answer = state.get("final_answer", "")
    thinking = state.get("thinking", "")
    sources = state.get("sources", [])
    cache_hit = state.get("cache_hit", False)

    if not cache_hit and cache is not None and answer:
        try:
            q_emb = cache.get_embedding(query)
            if q_emb is None and embeddings is not None:
                q_emb = embeddings.embed_query(query)
            if q_emb is not None:
                cache.store_answer(query, q_emb, answer, thinking, sources)
        except Exception:
            pass

    return {
        "steps": _add_step(state, "finalize", {"cache_hit": cache_hit}),
    }
