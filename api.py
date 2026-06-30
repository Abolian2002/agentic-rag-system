"""
FastAPI Backend for Cortex RAG - Agentic Knowledge Base
替代 Streamlit UI，提供 RESTful API
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import time
import requests
import asyncio
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from langchain_ollama import OllamaEmbeddings

from src.utils.config import (
    OLLAMA_BASE_URL,
    OLLAMA_API_URL,
    DEFAULT_MODEL,
    EMBEDDINGS_MODEL,
    CROSS_ENCODER_MODEL,
)
from src.ingestion.document_processor import DocumentProcessor
from src.retrieval.retriever import RetrieverPipeline
from src.cache.tiered_cache import TieredCache
from src.graph.workflow import run_rag_query

load_dotenv(find_dotenv())

# 加载 CrossEncoder（可选）
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if not hasattr(torch.classes, '__path__') or not torch.classes.__path__:
            torch.classes.__path__ = []
    except Exception:
        pass
    from sentence_transformers import CrossEncoder
    _torch_available = True
except Exception:
    device = "cpu"
    CrossEncoder = None
    _torch_available = False


# Pydantic 模型
class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict[str, str]]] = []
    config: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    answer: str
    thinking: Optional[str] = None
    sources: Optional[List[str]] = []
    workflow_steps: Optional[List[Dict[str, Any]]] = []
    cache_hit: bool = False
    route: str = "rag"
    confidence: Optional[float] = None


class ConfigRequest(BaseModel):
    model: Optional[str] = DEFAULT_MODEL
    rag_enabled: bool = True
    enable_hyde: bool = True
    enable_reranking: bool = True
    enable_graph_rag: bool = True
    enable_fusion: bool = False
    enable_crag: bool = False
    enable_cache: bool = True
    enable_citation_verify: bool = False
    enable_thinking: bool = True
    enable_contextual: bool = False
    temperature: float = 0.3
    max_contexts: int = 4
    show_workflow: bool = True


class ConfigResponse(BaseModel):
    model: str
    rag_enabled: bool
    enable_hyde: bool
    enable_reranking: bool
    enable_graph_rag: bool
    enable_fusion: bool
    enable_crag: bool
    enable_cache: bool
    enable_citation_verify: bool
    enable_thinking: bool
    enable_contextual: bool
    temperature: float
    max_contexts: int
    show_workflow: bool
    cache_stats: Optional[Dict[str, int]] = None


class StatsResponse(BaseModel):
    document_count: int
    chunk_count: int
    files: List[Dict[str, Any]]
    documents_loaded: bool


class StatusResponse(BaseModel):
    ollama_connected: bool
    available_models: List[str]
    documents_loaded: bool


class UploadResponse(BaseModel):
    success: bool
    message: str
    stats: Optional[StatsResponse] = None


# FastAPI 应用
app = FastAPI(
    title="Cortex RAG API",
    description="Agentic RAG Knowledge Base System - RESTful API",
    version="1.0.0",
)

# CORS（允许前端访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件（前端）
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# 全局状态（替代 Streamlit session_state）
class AppState:
    """应用全局状态"""
    def __init__(self):
        self.doc_processor: Optional[DocumentProcessor] = None
        self.retriever_pipeline: Optional[RetrieverPipeline] = None
        self.documents_loaded: bool = False
        self.reranker: Optional[Any] = None
        self.embeddings: Optional[OllamaEmbeddings] = None
        self.cache: Optional[TieredCache] = None
        self.config: ConfigRequest = ConfigRequest()
        self.suggested_questions: List[str] = []
        self.chat_history: List[Dict[str, str]] = []
    
    def reset(self):
        """重置状态"""
        if self.doc_processor:
            self.doc_processor.reset()
        self.doc_processor = None
        self.retriever_pipeline = None
        self.documents_loaded = False
        self.suggested_questions = []
        self.chat_history = []
        if self.cache:
            self.cache.clear()


state = AppState()


# 初始化函数
def init_components():
    """初始化组件"""
    # 加载 reranker
    if _torch_available and CrossEncoder is not None:
        try:
            state.reranker = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
        except Exception:
            state.reranker = None
    
    # 加载 embeddings
    try:
        state.embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL, base_url=OLLAMA_BASE_URL)
    except Exception:
        state.embeddings = None
    
    # 加载缓存
    state.cache = TieredCache()


# 启动时初始化
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    init_components()


# API 接口

@app.get("/", response_class=HTMLResponse)
async def root():
    """返回前端页面"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(), status_code=200)
    return HTMLResponse(content="<h1>Cortex RAG API</h1><p>Frontend not found. Use /docs for API documentation.</p>", status_code=200)


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """检查系统状态"""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
        models = [m["name"] for m in r.json().get("models", [])] if ollama_ok else []
        models = [m for m in models if "embed" not in m.lower()]
    except Exception:
        ollama_ok = False
        models = [DEFAULT_MODEL]
    
    return StatusResponse(
        ollama_connected=ollama_ok,
        available_models=models,
        documents_loaded=state.documents_loaded,
    )


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """获取文档统计"""
    if not state.documents_loaded or not state.doc_processor:
        return StatsResponse(
            document_count=0,
            chunk_count=0,
            files=[],
            documents_loaded=False,
        )
    
    stats = state.doc_processor.get_stats()
    return StatsResponse(
        document_count=stats["document_count"],
        chunk_count=stats["chunk_count"],
        files=stats["files"],
        documents_loaded=True,
    )


@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """获取当前配置"""
    cache_stats = None
    if state.cache and state.config.enable_cache:
        cache_stats = state.cache.stats()
    
    return ConfigResponse(
        model=state.config.model,
        rag_enabled=state.config.rag_enabled,
        enable_hyde=state.config.enable_hyde,
        enable_reranking=state.config.enable_reranking,
        enable_graph_rag=state.config.enable_graph_rag,
        enable_fusion=state.config.enable_fusion,
        enable_crag=state.config.enable_crag,
        enable_cache=state.config.enable_cache,
        enable_citation_verify=state.config.enable_citation_verify,
        enable_thinking=state.config.enable_thinking,
        enable_contextual=state.config.enable_contextual,
        temperature=state.config.temperature,
        max_contexts=state.config.max_contexts,
        show_workflow=state.config.show_workflow,
        cache_stats=cache_stats,
    )


@app.post("/api/config", response_model=ConfigResponse)
async def update_config(config: ConfigRequest):
    """更新配置"""
    state.config = config
    
    # 如果文档已加载且配置改变，需要重建 retriever
    if state.documents_loaded and state.doc_processor:
        state.retriever_pipeline = RetrieverPipeline(
            doc_processor=state.doc_processor,
            reranker=state.reranker,
            ollama_uri=OLLAMA_API_URL,
            model=config.model,
        )
    
    return await get_config()


@app.post("/api/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """上传并处理文档"""
    if not state.embeddings:
        return UploadResponse(
            success=False,
            message="Embeddings model not loaded. Check Ollama connection.",
        )
    
    try:
        # 初始化 DocumentProcessor
        state.doc_processor = DocumentProcessor(
            embedding_model=EMBEDDINGS_MODEL,
            base_url=OLLAMA_BASE_URL,
            llm_model=state.config.model,
            enable_contextual=state.config.enable_contextual,
        )
        
        # 处理文件（模拟 Streamlit 的进度回调）
        # FastAPI 不支持实时进度条，这里简化处理
        success = state.doc_processor.process_files(
            files,
            progress_callback=None,  # 不显示进度
        )
        
        if success and state.doc_processor.has_documents():
            # 创建 RetrieverPipeline
            state.retriever_pipeline = RetrieverPipeline(
                doc_processor=state.doc_processor,
                reranker=state.reranker,
                ollama_uri=OLLAMA_API_URL,
                model=state.config.model,
            )
            state.documents_loaded = True
            
            # 生成建议问题
            state.suggested_questions = state.doc_processor.suggested_questions(n=3)
            
            stats = state.doc_processor.get_stats()
            return UploadResponse(
                success=True,
                message=f"Successfully processed {stats['document_count']} documents, {stats['chunk_count']} chunks.",
                stats=StatsResponse(
                    document_count=stats["document_count"],
                    chunk_count=stats["chunk_count"],
                    files=stats["files"],
                    documents_loaded=True,
                ),
            )
        else:
            return UploadResponse(
                success=False,
                message="Failed to process documents.",
            )
    
    except Exception as e:
        return UploadResponse(
            success=False,
            message=f"Error: {str(e)}",
        )


@app.get("/api/suggested-questions")
async def get_suggested_questions():
    """获取建议问题"""
    return {"questions": state.suggested_questions}


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """执行 RAG 查询"""
    if not state.documents_loaded and state.config.rag_enabled:
        # 如果没有文档，返回 chat 模式
        state.config.rag_enabled = False
    
    # 构建 graph_config
    graph_config = {
        "ollama_uri": OLLAMA_API_URL,
        "model": state.config.model,
        "has_documents": state.documents_loaded and state.config.rag_enabled,
        "retriever": state.retriever_pipeline if state.documents_loaded else None,
        "cache": state.cache if state.config.enable_cache else None,
        "embeddings": state.embeddings,
        "enable_hyde": state.config.enable_hyde,
        "enable_fusion": state.config.enable_fusion,
        "enable_graph_rag": state.config.enable_graph_rag,
        "enable_reranking": state.config.enable_reranking,
        "enable_crag": state.config.enable_crag,
        "enable_cache": state.config.enable_cache,
        "enable_thinking": state.config.enable_thinking,
        "enable_citation_verify": state.config.enable_citation_verify,
        "temperature": state.config.temperature,
        "max_contexts": state.config.max_contexts,
    }
    
    # 如果请求中有自定义配置，覆盖
    if request.config:
        graph_config.update(request.config)
    
    # 执行查询
    try:
        result = run_rag_query(
            query=request.query,
            chat_history=request.chat_history or state.chat_history[-10:],
            config=graph_config,
            thread_id=f"thread_{int(time.time())}",
        )
        
        # 更新聊天历史
        state.chat_history.append({"role": "user", "content": request.query})
        state.chat_history.append({"role": "assistant", "content": result.get("final_answer", "") or result.get("generation", "")})
        
        return QueryResponse(
            answer=result.get("final_answer", "") or result.get("generation", ""),
            thinking=result.get("thinking", ""),
            sources=result.get("sources", []),
            workflow_steps=result.get("steps", []),
            cache_hit=result.get("cache_hit", False),
            route=result.get("route", "rag"),
            confidence=result.get("confidence", None),
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear")
async def clear_all():
    """清空聊天和缓存"""
    state.reset()
    return {"status": "success", "message": "Chat history and cache cleared."}


@app.get("/api/models")
async def get_models():
    """获取可用的 Ollama 模型"""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            models = [m for m in models if "embed" not in m.lower()]
            return {"models": models}
    except Exception:
        pass
    return {"models": [DEFAULT_MODEL]}


# 健康检查
@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy", "documents_loaded": state.documents_loaded}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)