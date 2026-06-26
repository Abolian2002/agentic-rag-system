from typing import List, Optional
from langchain_core.documents import Document
from src.utils.llm import ollama_generate


def generate_query_variants(
    query: str, uri: str, model: str, n: int = 3
) -> List[str]:
    prompt = (
        f"Generate {n} alternative search queries that capture different phrasings "
        f"or sub-aspects of the question below. One query per line, no numbering, no explanation.\n\n"
        f"Original question: {query}\n\n"
        f"Alternative queries:"
    )
    out = ollama_generate(uri, model, prompt, temperature=0.4, timeout=30)
    variants = []
    for line in out.splitlines():
        v = line.strip().lstrip("0123456789.-*) ").strip().strip('"').strip("'")
        if v and v.lower() != query.lower() and len(v) > 5:
            variants.append(v)
    return [query] + variants[:n]


def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]], k: int = 60
) -> List[Document]:
    scores = {}
    doc_map = {}
    for docs in ranked_lists:
        for rank, doc in enumerate(docs):
            key = doc.page_content[:200]
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ordered]


def expand_query_hyde(query: str, uri: str, model: str) -> str:
    prompt = (
        f"Write a concise hypothetical answer (2-3 sentences, in Chinese if the question is in Chinese) "
        f"that would appear in a document relevant to answering this question. "
        f"Do NOT answer the question directly — just write a passage that contains relevant information.\n\n"
        f"Question: {query}\n\nHypothetical passage:"
    )
    expansion = ollama_generate(uri, model, prompt, temperature=0.3, timeout=30)
    return f"{query}\n{expansion}" if expansion else query


def grade_document_relevance(
    query: str, doc_text: str, uri: str, model: str
) -> tuple[bool, str]:
    prompt = (
        "You are a strict relevance grader. Given a question and a document chunk, "
        "determine if the document contains information useful for answering the question.\n\n"
        f"Question: {query}\n\n"
        f"Document chunk:\n{doc_text[:1000]}\n\n"
        "Answer with EXACTLY one word: 'yes' or 'no'. Then on a new line, briefly explain why (one sentence)."
    )
    response = ollama_generate(uri, model, prompt, temperature=0.0, timeout=25).lower()
    lines = response.strip().split("\n", 1)
    answer = lines[0].strip() if lines else "no"
    reason = lines[1].strip() if len(lines) > 1 else ""
    is_relevant = answer.startswith("y") or "yes" in answer[:6]
    return is_relevant, reason


def route_query(query: str, uri: str, model: str) -> dict:
    prompt = (
        "You are a query router for a knowledge base system. Given a user question, "
        "classify it into one of three categories:\n"
        "1. 'rag' - The question asks about specific information in the uploaded documents (needs retrieval)\n"
        "2. 'chat' - General conversation, greeting, or question not requiring document lookup\n"
        "3. 'summary' - The user asks to summarize, list, or get an overview of the documents\n\n"
        f"Question: {query}\n\n"
        "Respond in this exact format:\n"
        "route: <rag|chat|summary>\n"
        "reason: <one sentence explanation>"
    )
    response = ollama_generate(uri, model, prompt, temperature=0.0, timeout=20)
    route = "rag"
    reason = ""
    for line in response.split("\n"):
        line = line.strip().lower()
        if line.startswith("route:"):
            r = line.split(":", 1)[1].strip()
            if r in ("rag", "chat", "summary"):
                route = r
        elif line.startswith("reason:"):
            reason = line.split(":", 1)[1].strip()
    return {"route": route, "reason": reason}


def verify_citations(answer: str, sources: List[str], uri: str, model: str) -> dict:
    if not sources:
        return {"verified": True, "issues": []}
    sources_text = "\n\n".join(
        f"[Source {i+1}]: {s[:500]}" for i, s in enumerate(sources)
    )
    prompt = (
        "You are a citation verifier. Check whether the answer's claims and source references "
        "[Source N] are actually supported by the provided source texts.\n\n"
        f"Answer:\n{answer[:1500]}\n\n"
        f"Sources:\n{sources_text}\n\n"
        "Respond in this format:\n"
        "verified: <yes|no>\n"
        "issues: <list any unsupported claims or incorrect citations, or 'none'>"
    )
    response = ollama_generate(uri, model, prompt, temperature=0.0, timeout=30)
    verified = True
    issues = []
    for line in response.split("\n"):
        line = line.strip().lower()
        if line.startswith("verified:"):
            v = line.split(":", 1)[1].strip()
            verified = v.startswith("y")
        elif line.startswith("issues:"):
            issue_text = line.split(":", 1)[1].strip()
            if issue_text and issue_text != "none":
                issues.append(issue_text)
    return {"verified": verified, "issues": issues}
