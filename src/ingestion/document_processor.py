import os
import json
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import networkx as nx

from src.utils.config import (
    CHROMA_PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
    INDEX_META_PATH,
    CACHE_DIR,
    MAX_CONTEXTUAL_CHUNKS,
)
from src.utils.llm import ollama_generate
from src.retrieval.knowledge_graph import build_knowledge_graph


class MarkdownLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return [
            Document(
                page_content=text,
                metadata={"source": os.path.basename(self.file_path)},
            )
        ]


class DocumentProcessor:
    def __init__(
        self,
        embedding_model: str,
        base_url: str,
        llm_model: Optional[str] = None,
        enable_contextual: bool = False,
    ):
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.llm_model = llm_model
        self.enable_contextual = enable_contextual
        self.ollama_uri = f"{base_url}/api/generate"
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.embeddings = OllamaEmbeddings(
            model=embedding_model, base_url=base_url
        )
        self.vector_store = None
        self.bm25_retriever = None
        self.knowledge_graph = None
        self.doc_chunks: List[Document] = []
        self.index_meta = self._load_meta()

    def _load_meta(self) -> Dict[str, Any]:
        if os.path.exists(INDEX_META_PATH):
            try:
                with open(INDEX_META_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"files": {}, "last_updated": None}

    def _save_meta(self):
        self.index_meta["last_updated"] = datetime.now().isoformat()
        with open(INDEX_META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.index_meta, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _file_hash(file_content: bytes) -> str:
        return hashlib.sha256(file_content).hexdigest()

    def _clean_chunk(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text.strip()

    def _load_file(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext in (".txt", ".md", ".markdown"):
            loader = MarkdownLoader(file_path)
        else:
            return []
        docs = loader.load()
        for d in docs:
            if "source" not in d.metadata:
                d.metadata["source"] = os.path.basename(file_path)
        return docs

    def _contextualize_chunks(
        self, texts: List[Document], documents: List[Document]
    ) -> List[Document]:
        if not self.llm_model:
            return texts

        summaries = {}
        per_source_text: Dict[str, str] = {}
        for d in documents:
            src = d.metadata.get("source", "doc")
            per_source_text[src] = per_source_text.get(src, "") + "\n" + d.page_content
        for src, full in per_source_text.items():
            summaries[src] = full.strip()[:1500]
        global_summary = " ".join(per_source_text.values())[:1500]

        total = min(len(texts), MAX_CONTEXTUAL_CHUNKS)
        progress = st.progress(0.0, text="Contextualizing chunks...")
        for i, chunk in enumerate(texts[:total]):
            src = chunk.metadata.get("source", "doc")
            summary = summaries.get(src, global_summary)
            prompt = (
                "You are helping index a document for search.\n"
                f"<document_summary>\n{summary}\n</document_summary>\n"
                f"<chunk>\n{chunk.page_content}\n</chunk>\n\n"
                "Write a SINGLE short sentence (max 25 words) that situates this chunk "
                "within the document so it can be retrieved on its own. "
                "Output only that sentence, nothing else."
            )
            ctx = ollama_generate(
                self.ollama_uri, self.llm_model, prompt, temperature=0.0, timeout=60
            )
            ctx = ctx.replace("\n", " ").strip()[:300]
            if ctx:
                chunk.metadata["context"] = ctx
                chunk.page_content = f"{ctx}\n\n{chunk.page_content}"
            progress.progress(
                (i + 1) / total, text=f"Contextualizing chunks... {i + 1}/{total}"
            )
        progress.empty()
        return texts

    def _init_chroma(self):
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )

    def process_files(
        self,
        uploaded_files: List[Any],
        progress_callback=None,
    ) -> bool:
        if not os.path.exists("temp"):
            os.makedirs("temp")

        new_documents: List[Document] = []
        new_chunks: List[Document] = []
        files_to_add = []
        files_to_remove = []

        current_hashes = {
            meta["hash"]: fname
            for fname, meta in self.index_meta["files"].items()
        }

        for file in uploaded_files:
            file_content = file.getbuffer()
            file_hash = self._file_hash(file_content)
            file_name = file.name

            if file_name in self.index_meta["files"]:
                if self.index_meta["files"][file_name]["hash"] == file_hash:
                    continue
                else:
                    files_to_remove.append(file_name)

            file_path = os.path.join("temp", file_name)
            with open(file_path, "wb") as f:
                f.write(file_content)

            try:
                docs = self._load_file(file_path)
                if docs:
                    new_documents.extend(docs)
                    files_to_add.append(
                        {
                            "name": file_name,
                            "hash": file_hash,
                            "added_at": datetime.now().isoformat(),
                        }
                    )
            except Exception as e:
                st.error(f"Error processing {file_name}: {str(e)}")
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)

        for fname in files_to_remove:
            del self.index_meta["files"][fname]

        if not new_documents:
            self._load_existing_index()
            return bool(self.doc_chunks)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        split_docs = text_splitter.split_documents(new_documents)

        for chunk in split_docs:
            chunk.page_content = self._clean_chunk(chunk.page_content)
            chunk.metadata["char_count"] = len(chunk.page_content)
            chunk.metadata["doc_type"] = os.path.splitext(
                chunk.metadata.get("source", "")
            )[1].lower().lstrip(".")

        if self.enable_contextual and self.llm_model:
            split_docs = self._contextualize_chunks(split_docs, new_documents)

        new_chunks.extend(split_docs)

        if self.vector_store is None:
            self._init_chroma()

        if new_chunks:
            if progress_callback:
                progress_callback(0.3, "Adding to vector store...")
            self.vector_store.add_documents(new_chunks)

        for finfo in files_to_add:
            self.index_meta["files"][finfo["name"]] = {
                "hash": finfo["hash"],
                "added_at": finfo["added_at"],
                "chunk_count": len(
                    [c for c in new_chunks if c.metadata.get("source") == finfo["name"]]
                ),
            }
        self._save_meta()

        self._load_all_chunks()
        self._build_retrievers()
        return True

    def _load_existing_index(self):
        if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
            self._init_chroma()
            self._load_all_chunks()
            self._build_retrievers()

    def _load_all_chunks(self):
        if self.vector_store is None:
            return
        try:
            count = self.vector_store._collection.count()
            if count == 0:
                self.doc_chunks = []
                return
            result = self.vector_store.get(limit=count, include=["documents", "metadatas"])
            docs = []
            if result and "documents" in result:
                metadatas = result.get("metadatas") or [{}] * len(result["documents"])
                for content, meta in zip(result["documents"], metadatas):
                    if content:
                        docs.append(
                            Document(
                                page_content=content,
                                metadata=meta or {"source": "unknown"},
                            )
                        )
            self.doc_chunks = docs
        except Exception as e:
            try:
                self.doc_chunks = self.vector_store.similarity_search("", k=1000)
            except Exception:
                self.doc_chunks = []

    def _build_bm25(self):
        from langchain_community.retrievers import BM25Retriever
        from rank_bm25 import BM25Okapi

        if not self.doc_chunks:
            return None
        text_contents = [doc.page_content for doc in self.doc_chunks]
        self.bm25_retriever = BM25Retriever.from_texts(
            text_contents,
            bm25_impl=BM25Okapi,
            preprocess_func=lambda text: re.sub(r"\W+", " ", text).lower().split(),
        )
        self.bm25_retriever.k = 6

    def _build_retrievers(self):
        self._build_bm25()
        if self.doc_chunks:
            self.knowledge_graph = build_knowledge_graph(self.doc_chunks)

    def reset(self):
        import shutil

        if os.path.exists(CHROMA_PERSIST_DIR):
            shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)
        if os.path.exists(INDEX_META_PATH):
            os.remove(INDEX_META_PATH)
        self.vector_store = None
        self.bm25_retriever = None
        self.knowledge_graph = None
        self.doc_chunks = []
        self.index_meta = {"files": {}, "last_updated": None}

    def get_stats(self) -> Dict[str, Any]:
        return {
            "document_count": len(self.index_meta.get("files", {})),
            "chunk_count": len(self.doc_chunks),
            "files": [
                {"name": k, "chunks": v.get("chunk_count", 0)}
                for k, v in self.index_meta.get("files", {}).items()
            ],
        }

    def has_documents(self) -> bool:
        return len(self.doc_chunks) > 0

    def suggested_questions(self, n: int = 3) -> List[str]:
        if not self.doc_chunks:
            return []
        import random

        samples = random.sample(self.doc_chunks, min(3, len(self.doc_chunks)))
        questions = []
        for doc in samples:
            prompt = (
                f"Based on this text, generate ONE specific question that a user might ask "
                f"about this content. Output only the question, nothing else.\n\nText: {doc.page_content[:500]}"
            )
            q = ollama_generate(
                self.ollama_uri,
                self.llm_model or "qwen2.5:7b",
                prompt,
                temperature=0.7,
                timeout=20,
            )
            q = q.strip().strip('"').strip("'")
            if q and len(q) > 10:
                questions.append(q[:80])
        return questions[:n]
