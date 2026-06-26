import re
import networkx as nx
from typing import List, Set
from langchain_core.documents import Document


def extract_entities(text: str) -> List[str]:
    entities = set()
    capitalized = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', text)
    entities.update(capitalized)
    tech_patterns = re.findall(
        r'\b(?:[A-Z]{2,}|[A-Z][a-z]+[A-Z][a-zA-Z]*|\w+(?:-\w+)+)\b', text
    )
    entities.update(tech_patterns)
    numeric_entities = re.findall(r'\b[A-Za-z]+\s+\d+(?:\.\d+)?\b', text)
    entities.update(numeric_entities)
    return [e.strip() for e in entities if 2 <= len(e.strip()) <= 50]


def build_knowledge_graph(docs: List[Document]) -> nx.Graph:
    G = nx.Graph()
    for doc in docs:
        entities = extract_entities(doc.page_content)
        if len(entities) < 2:
            continue
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1 :]:
                if e1 != e2:
                    if G.has_edge(e1, e2):
                        G[e1][e2]["weight"] = G[e1][e2].get("weight", 1) + 1
                    else:
                        G.add_edge(e1, e2, weight=1, source=doc.page_content[:300])
        for entity in entities:
            if entity in G.nodes:
                G.nodes[entity]["count"] = G.nodes[entity].get("count", 0) + 1
            else:
                G.add_node(entity, count=1)
    return G


def retrieve_from_graph(
    query: str,
    G: nx.Graph,
    doc_chunks: List[Document],
    top_k: int = 3,
) -> List[Document]:
    if not G or G.number_of_nodes() == 0:
        return []

    query_words = set(re.findall(r'\w+', query.lower()))
    query_entities = [w for w in query_words if len(w) > 2]

    matched_nodes: Set[str] = set()
    for node in G.nodes:
        node_lower = node.lower()
        for word in query_entities:
            if word in node_lower or node_lower in word:
                matched_nodes.add(node)
                break

    if not matched_nodes:
        return []

    related_nodes: Set[str] = set()
    for node in matched_nodes:
        neighbors = sorted(
            G.neighbors(node),
            key=lambda n: G[node][n].get("weight", 1),
            reverse=True,
        )[:10]
        related_nodes.update(neighbors)

    all_entities = matched_nodes | related_nodes
    results = []
    seen = set()

    for chunk in doc_chunks:
        text_lower = chunk.page_content.lower()
        score = sum(
            2 if entity.lower() in text_lower and entity in matched_nodes else 1
            for entity in all_entities
            if entity.lower() in text_lower
        )
        if score > 0 and chunk.page_content not in seen:
            chunk.metadata["graph_score"] = score
            seen.add(chunk.page_content)
            results.append((score, chunk))

    results.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in results[:top_k]]
