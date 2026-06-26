import streamlit as st
import requests
import json
import os
import time
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

load_dotenv(find_dotenv())

st.set_page_config(
    page_title="Cortex RAG - Agentic Knowledge Base",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource(show_spinner=False)
def load_reranker():
    if not _torch_available or CrossEncoder is None:
        return None
    try:
        return CrossEncoder(CROSS_ENCODER_MODEL, device=device)
    except Exception as e:
        st.warning(f"CrossEncoder not loaded: {e}. Reranking disabled.")
        return None


@st.cache_resource(show_spinner=False)
def load_embeddings():
    try:
        return OllamaEmbeddings(model=EMBEDDINGS_MODEL, base_url=OLLAMA_BASE_URL)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_cache():
    return TieredCache()


@st.cache_resource(show_spinner=False)
def get_doc_processor(_embeddings, llm_model=None, enable_contextual=False):
    return DocumentProcessor(
        embedding_model=EMBEDDINGS_MODEL,
        base_url=OLLAMA_BASE_URL,
        llm_model=llm_model,
        enable_contextual=enable_contextual,
    )


@st.cache_data(show_spinner=False, ttl=30)
def get_ollama_models():
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return [m for m in models if "embed" not in m.lower()]
    except Exception:
        pass
    return [DEFAULT_MODEL]


st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700;12..96,800&family=Manrope:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --bg:        #0a0b12;
        --bg-soft:   #0d1020;
        --panel:     #13162c;
        --panel-2:   #181c38;
        --line:      #23264a;
        --iris:      #8b7bff;
        --iris-2:    #6ea8ff;
        --mint:      #2dd4bf;
        --amber:     #ffb347;
        --text:      #e7e9f5;
        --text-dim:  #8b91b8;
        --text-faint:#5a6090;
        --rose:      #ff6b8a;
    }
    .stApp {
        background:
            radial-gradient(900px 500px at 12% -8%, rgba(139,123,255,0.14), transparent 60%),
            radial-gradient(800px 520px at 100% 0%, rgba(45,212,191,0.10), transparent 55%),
            var(--bg);
        color: var(--text);
        font-family: 'Manrope', 'Segoe UI', sans-serif;
    }
    .stApp::before {
        content: "";
        position: fixed; inset: 0;
        background-image: radial-gradient(rgba(255,255,255,0.025) 1px, transparent 1px);
        background-size: 26px 26px;
        pointer-events: none; z-index: 0;
    }
    .main .block-container { padding-top: 2.2rem; max-width: 960px; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #11142a 0%, #0a0b16 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color: #c2c7e6 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5 {
        font-family: 'Bricolage Grotesque', sans-serif !important;
        color: #d8d4ff !important; letter-spacing: -0.01em;
    }
    [data-testid="stSidebar"] .stMarkdown p { font-size: 0.9rem; }
    .brand-wrap {
        display: flex; align-items: center; justify-content: center;
        gap: 18px; margin: 0 0 2px 0;
    }
    .brand-logo { width: 64px; height: 64px; filter: drop-shadow(0 0 14px rgba(139,123,255,0.4)); }
    @keyframes float-core { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-3px)} }
    .brand-logo { animation: float-core 4s ease-in-out infinite; }
    .rag-title {
        font-family: 'Bricolage Grotesque', sans-serif;
        background: linear-gradient(100deg, #ffffff 0%, #c4b8ff 55%, var(--mint) 100%);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent; color: transparent;
        font-size: 3.1rem; font-weight: 800; letter-spacing: -0.02em;
        margin: 0; line-height: 1; padding: 0;
    }
    .brand-kicker {
        text-align: center; color: var(--iris);
        font-family: 'JetBrains Mono', monospace; font-weight: 600;
        font-size: 0.72rem; letter-spacing: 0.32em; margin: 10px 0 2px 0;
    }
    .brand-tag {
        text-align: center; color: var(--text-faint);
        font-size: 0.86rem; margin-bottom: 0.4rem;
    }
    .chip-row { display:flex; flex-wrap:wrap; gap:7px; justify-content:center; margin: 12px 0 4px; }
    .chip {
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 500;
        color: #aab0e0; background: var(--panel);
        border: 1px solid var(--line); border-radius: 999px;
        padding: 4px 11px; transition: all .2s;
    }
    .chip:hover { border-color: var(--iris); color: #fff; }
    [data-testid="stChatMessageContent"] { font-size: 0.96rem; line-height: 1.65; }
    [data-testid="stChatMessage"] {
        background: rgba(19,22,44,0.45);
        border: 1px solid var(--line);
        border-radius: 14px; padding: 6px 14px; margin: 8px 0;
        backdrop-filter: blur(4px);
    }
    .think-box {
        background: linear-gradient(180deg, #0c0f22, #0a0b16);
        border: 1px solid #2a2f5e; border-left: 3px solid var(--iris);
        border-radius: 10px; margin: 8px 0 12px; overflow: hidden;
        box-shadow: 0 0 0 1px rgba(139,123,255,0.04), 0 8px 24px rgba(0,0,0,0.3);
    }
    .think-box summary {
        list-style:none; cursor:pointer; padding: 10px 15px;
        font-family:'JetBrains Mono',monospace; font-size:0.74rem; font-weight:600;
        color: var(--iris); letter-spacing: 0.06em; user-select:none;
        display:flex; align-items:center; gap:7px;
    }
    .think-box summary::-webkit-details-marker { display:none; }
    .think-box[open] summary { border-bottom: 1px solid #23264a; }
    .think-body {
        padding: 13px 17px; font-size: 0.8rem; color: #9aa0c8;
        font-family:'JetBrains Mono',monospace; white-space:pre-wrap;
        line-height:1.7; max-height:340px; overflow-y:auto;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }
    .think-live { animation: pulse 1.1s ease infinite; color: var(--mint); }
    .source-card {
        background: var(--panel); border: 1px solid var(--line);
        border-left: 3px solid var(--mint); border-radius: 8px;
        padding: 11px 15px; margin: 7px 0; font-size: 0.81rem; color: #aab0d0;
        line-height: 1.55;
    }
    .source-label {
        color: var(--mint); font-family:'JetBrains Mono',monospace;
        font-weight: 600; font-size: 0.68rem; text-transform: uppercase;
        letter-spacing: 0.1em; margin-bottom: 4px; display:block;
    }
    .step-timeline {
        background: var(--panel); border: 1px solid var(--line);
        border-radius: 10px; padding: 12px 16px; margin: 8px 0;
    }
    .step-item {
        display: flex; align-items: flex-start; gap: 10px;
        padding: 6px 0; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace;
    }
    .step-dot {
        width: 8px; height: 8px; border-radius: 50%; margin-top: 5px; flex-shrink: 0;
    }
    .step-dot.ok { background: var(--mint); box-shadow: 0 0 8px var(--mint); }
    .step-dot.active { background: var(--amber); box-shadow: 0 0 8px var(--amber); animation: pulse 1s ease infinite; }
    .step-dot.info { background: var(--iris); }
    .step-name { color: var(--text); font-weight: 600; min-width: 100px; }
    .step-detail { color: var(--text-dim); }
    .stButton > button {
        background: linear-gradient(120deg, var(--iris), #5a61e0);
        color: #fff !important; border: none; border-radius: 10px;
        font-family:'Manrope',sans-serif; font-weight: 600; font-size: 0.86rem;
        transition: transform .15s, box-shadow .2s; box-shadow: 0 4px 14px rgba(139,123,255,0.25);
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(139,123,255,0.4); }
    .stButton > button:active { transform: translateY(0); }
    [data-testid="stChatInput"] textarea,
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
        background: var(--panel) !important; border-color: var(--line) !important;
        border-radius: 10px !important; color: var(--text) !important;
    }
    [data-testid="stChatInput"] {
        background: var(--panel) !important; border: 1px solid var(--line) !important;
        border-radius: 14px;
    }
    [data-testid="stBottomBlockContainer"], [data-testid="stBottom"] > div,
    [data-testid="stChatFloatingInputContainer"] {
        background: transparent !important;
    }
    [data-testid="stBottom"], .stApp > footer { background: transparent !important; }
    .stChatFloatingInputContainer, .block-container + div { background: transparent !important; }
    section.main > div.block-container ~ div { background: transparent !important; }
    .status-badge {
        display:inline-flex; align-items:center; gap:7px;
        padding: 5px 13px; border-radius: 999px;
        font-family:'JetBrains Mono',monospace; font-size: 0.73rem; font-weight: 600;
    }
    .status-ok  { background:#0c2a1c; color:var(--mint); border:1px solid #1f6f4a; }
    .status-err { background:#2e0d12; color:#ff6b8a; border:1px solid #7f2540; }
    .status-warn { background:#2e2a0d; color:var(--amber); border:1px solid #6f5f1f; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.35} }
    .status-ok .dot, .status-err .dot, .status-warn .dot { width:7px;height:7px;border-radius:50%;background:currentColor;animation:blink 1.6s infinite; }
    hr { border-color: var(--line) !important; margin: 0.8rem 0; }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--panel); border: 1px dashed #3a3f6e; border-radius: 12px;
    }
    .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        color: var(--mint) !important; font-family:'JetBrains Mono',monospace; font-size:0.8rem;
    }
    [data-testid="stExpander"] { border:1px solid var(--line); border-radius:10px; background:rgba(13,16,32,0.5); }
    ::-webkit-scrollbar { width: 7px; height:7px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #2a2f5e; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--iris); }
    #MainMenu, footer { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 12px; }
    [data-testid="stDecoration"] { background: linear-gradient(90deg, var(--iris), var(--mint)); }
</style>
""", unsafe_allow_html=True)


LOGO_SVG = """
<svg class="brand-logo" viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="hg" x1="40" y1="40" x2="160" y2="160" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#8b7bff"/><stop offset="0.55" stop-color="#6ea8ff"/><stop offset="1" stop-color="#2dd4bf"/>
    </linearGradient>
    <radialGradient id="hc" cx="0.5" cy="0.5" r="0.5">
      <stop offset="0" stop-color="#c4b8ff"/><stop offset="0.5" stop-color="#8b7bff"/><stop offset="1" stop-color="#2dd4bf"/>
    </radialGradient>
  </defs>
  <circle cx="100" cy="100" r="78" stroke="url(#hg)" stroke-width="1.5" opacity="0.25"/>
  <g stroke="url(#hg)" stroke-width="2" opacity="0.45" stroke-linecap="round">
    <path d="M96 100 L79.5 44"/><path d="M96 100 L40 100"/><path d="M96 100 L79.5 156"/>
    <path d="M96 100 L115.5 42"/><path d="M96 100 L115.5 158"/><path d="M96 100 L51 66"/><path d="M96 100 L51 134"/>
  </g>
  <path d="M146 61 L115.5 42 L79.5 44 L51 66 L40 100 L51 134 L79.5 156 L115.5 158 L146 139"
        stroke="url(#hg)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="68" cy="100" r="2.6" fill="#2dd4bf"/><circle cx="88" cy="72" r="2.6" fill="#ffb347"/><circle cx="88" cy="128" r="2.6" fill="#8b7bff"/>
  <g fill="#0a0b12" stroke="url(#hg)" stroke-width="3">
    <circle cx="146" cy="61" r="7"/><circle cx="115.5" cy="42" r="7"/><circle cx="79.5" cy="44" r="7"/>
    <circle cx="51" cy="66" r="7"/><circle cx="40" cy="100" r="7"/><circle cx="51" cy="134" r="7"/>
    <circle cx="79.5" cy="156" r="7"/><circle cx="115.5" cy="158" r="7"/><circle cx="146" cy="139" r="7"/>
  </g>
  <circle cx="96" cy="100" r="13" fill="url(#hc)"/><circle cx="96" cy="100" r="5" fill="#0a0b12" opacity="0.85"/>
  <circle cx="96" cy="100" r="2.4" fill="#c4b8ff"/>
</svg>
"""


defaults = {
    "messages": [],
    "doc_processor": None,
    "retriever_pipeline": None,
    "rag_enabled": True,
    "documents_loaded": False,
    "enable_hyde": True,
    "enable_reranking": True,
    "enable_graph_rag": True,
    "enable_thinking": True,
    "enable_contextual": False,
    "enable_fusion": False,
    "enable_crag": False,
    "enable_cache": True,
    "enable_citation_verify": False,
    "cache_threshold": 0.92,
    "temperature": 0.3,
    "max_contexts": 4,
    "suggested_questions": [],
    "show_workflow": True,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

reranker = load_reranker()
embeddings = load_embeddings()
cache = load_cache()


with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:11px;margin:-6px 0 4px;">'
        f'<div style="width:38px;height:38px;flex:0 0 38px;">{LOGO_SVG.replace("brand-logo","")}</div>'
        '<div><div style="font-family:\'Bricolage Grotesque\',sans-serif;font-weight:800;'
        'font-size:1.35rem;line-height:1;background:linear-gradient(100deg,#fff,#c4b8ff,#2dd4bf);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Cortex RAG</div>'
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.58rem;'
        'letter-spacing:0.18em;color:#6a70a0;margin-top:3px;">AGENTIC RAG · LANGGRAPH</div></div></div>',
        unsafe_allow_html=True
    )

    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False

    if ollama_ok:
        st.markdown('<span class="status-badge status-ok"><span class="dot"></span>Ollama connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-err"><span class="dot"></span>Ollama offline</span>', unsafe_allow_html=True)
        st.error("Start Ollama with: `ollama serve`")

    st.markdown("---")
    st.markdown("### 🧠 Model")
    available_models = get_ollama_models()
    default_idx = available_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in available_models else 0
    selected_model = st.selectbox("LLM Model", available_models, index=default_idx, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 📁 Documents")
    if st.session_state.documents_loaded and st.session_state.doc_processor:
        stats = st.session_state.doc_processor.get_stats()
        st.success(f"✓ {stats['document_count']} documents, {stats['chunk_count']} chunks")
        if stats['files']:
            with st.expander("📂 Loaded files"):
                for f in stats['files']:
                    st.caption(f"• {f['name']} ({f['chunks']} chunks)")
        if st.button("🔄 Reset Documents", use_container_width=True):
            if st.session_state.doc_processor:
                st.session_state.doc_processor.reset()
            st.session_state.doc_processor = None
            st.session_state.retriever_pipeline = None
            st.session_state.documents_loaded = False
            st.session_state.suggested_questions = []
            cache.clear()
            get_doc_processor.clear()
            st.rerun()
    else:
        st.session_state.enable_contextual = st.checkbox(
            "Contextual Retrieval ✨",
            value=st.session_state.enable_contextual,
            help="Prepend an LLM-generated context sentence to each chunk before indexing."
        )
        uploaded_files = st.file_uploader(
            "Upload PDF / DOCX / TXT / MD",
            type=["pdf", "docx", "txt", "md", "markdown"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded_files:
            doc_proc = get_doc_processor(
                embeddings,
                llm_model=selected_model,
                enable_contextual=st.session_state.enable_contextual,
            )
            with st.spinner("Processing documents..."):
                progress_bar = st.progress(0.0, text="Initializing...")
                success = doc_proc.process_files(
                    uploaded_files,
                    progress_callback=lambda p, msg: progress_bar.progress(p, text=msg),
                )
                progress_bar.empty()
                if success and doc_proc.has_documents():
                    st.session_state.doc_processor = doc_proc
                    st.session_state.retriever_pipeline = RetrieverPipeline(
                        doc_processor=doc_proc,
                        reranker=reranker,
                        ollama_uri=OLLAMA_API_URL,
                        model=selected_model,
                    )
                    st.session_state.documents_loaded = True
                    with st.spinner("Generating suggested questions..."):
                        st.session_state.suggested_questions = doc_proc.suggested_questions(n=3)
                    st.success("Documents ready!")
                    st.rerun()
                else:
                    st.error("Failed to process documents.")

    st.markdown("---")
    st.markdown("### ⚙️ Agentic RAG Settings")
    st.session_state.rag_enabled       = st.checkbox("Enable RAG",             value=st.session_state.rag_enabled)
    st.session_state.enable_hyde       = st.checkbox("HyDE Query Expansion",   value=st.session_state.enable_hyde)
    st.session_state.enable_reranking  = st.checkbox("Neural Reranking",       value=st.session_state.enable_reranking)
    st.session_state.enable_graph_rag  = st.checkbox("GraphRAG",               value=st.session_state.enable_graph_rag)

    st.markdown("##### 🆕 Advanced (LangGraph)")
    st.session_state.enable_fusion = st.checkbox(
        "RAG-Fusion (RRF)", value=st.session_state.enable_fusion,
        help="Generate multiple query variants and merge results with Reciprocal Rank Fusion."
    )
    st.session_state.enable_crag = st.checkbox(
        "Corrective RAG (CRAG)", value=st.session_state.enable_crag,
        help="Grade document relevance; auto-retry with broader search if confidence is low."
    )
    st.session_state.enable_cache = st.checkbox(
        "Tiered Semantic Cache ⚡", value=st.session_state.enable_cache,
        help="Instantly reuse answers for semantically similar questions."
    )
    st.session_state.enable_citation_verify = st.checkbox(
        "Citation Verification", value=st.session_state.enable_citation_verify,
        help="LLM verifies that answer citations match source documents."
    )
    st.session_state.show_workflow = st.checkbox(
        "Show Workflow Steps", value=st.session_state.show_workflow,
        help="Display LangGraph node execution timeline."
    )

    st.markdown("---")
    st.markdown("### 🎛️ Generation")
    st.session_state.enable_thinking = st.checkbox("Show Thinking Process 🧠", value=st.session_state.enable_thinking)
    st.session_state.temperature     = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.05)
    st.session_state.max_contexts    = st.slider("Max Contexts", 1, 8, st.session_state.max_contexts)

    st.markdown("---")
    if st.session_state.enable_cache:
        cs = cache.stats()
        st.caption(f"⚡ Cache: {cs['answers']} answers, {cs['embeddings']} embeddings")
    if st.button("🗑️ Clear Chat & Cache", use_container_width=True):
        st.session_state.messages = []
        cache.clear()
        st.rerun()


def _thinking_html(text: str, live: bool) -> str:
    if live:
        label = '🧠 REASONING<span class="think-live"> ●</span>'
    else:
        label = '🧠 THOUGHT PROCESS &nbsp;·&nbsp; <span style="color:#5a6090">click to expand</span> ✓'
    open_attr = " open" if live else ""
    return (
        f'<details class="think-box"{open_attr}>'
        f'<summary>{label}</summary>'
        f'<div class="think-body">{text}</div>'
        f'</details>'
    )


def _source_html(sources: list) -> str:
    if not sources:
        return ""
    cards = "".join(
        f'<div class="source-card"><div class="source-label">Source {i+1}</div>'
        f'{s[:400]}{"…" if len(s)>400 else ""}</div>'
        for i, s in enumerate(sources)
    )
    return cards


def _steps_html(steps: list) -> str:
    if not steps:
        return ""
    total = len(steps)
    items = []
    for i, s in enumerate(steps):
        step_name = s.get("step", "?")
        info = s.get("info", {})
        is_last = i == total - 1
        dot_class = "ok" if not is_last else "active"
        detail_parts = []
        if step_name == "route" and "route" in info:
            detail_parts.append(f"→ {info['route']}")
        if step_name == "cache_check" and "hit" in info:
            detail_parts.append("⚡ HIT" if info["hit"] else "miss")
            if "similarity" in info:
                detail_parts.append(info["similarity"])
        if step_name == "retrieve":
            detail_parts.append(f"{info.get('found', 0)} docs")
            if info.get("crag"):
                crag = info["crag"]
                if crag["status"] == "ok":
                    detail_parts.append(f"CRAG: {crag['kept']}/{crag['total']} relevant")
                else:
                    detail_parts.append("CRAG: low confidence")
            if info.get("attempt", 1) > 1:
                detail_parts.append(f"retry #{info['attempt']}")
        if step_name == "generate":
            detail_parts.append(f"{info.get('answer_len', 0)} chars")
        if step_name == "verify" and info.get("verified") is not None:
            detail_parts.append("✓ cited" if info["verified"] else "⚠ issues found")
        if step_name == "transform" and info.get("queries"):
            detail_parts.append(f"{len(info['queries'])} queries")
        detail_str = " · ".join(str(d) for d in detail_parts) if detail_parts else ""
        items.append(
            f'<div class="step-item">'
            f'<div class="step-dot {dot_class}"></div>'
            f'<div class="step-name">{step_name}</div>'
            f'<div class="step-detail">{detail_str}</div>'
            f'</div>'
        )
    return (
        '<div class="step-timeline">'
        + "".join(items)
        + '</div>'
    )


def typewriter(text: str, container, speed: float = 0.003):
    display = ""
    for char in text:
        display += char
        container.markdown(display + "▌")
        time.sleep(speed)
    container.markdown(display)


st.markdown(
    f'<div class="brand-wrap">{LOGO_SVG}<div class="rag-title">Cortex RAG</div></div>',
    unsafe_allow_html=True
)
st.markdown('<div class="brand-kicker">AGENTIC RETRIEVAL ENGINE · POWERED BY LANGGRAPH</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-tag">RAG that thinks, routes, retries, and verifies before it answers.</div>', unsafe_allow_html=True)

if st.session_state.documents_loaded and st.session_state.rag_enabled:
    chips = ["🔍 BM25+Chroma", "🕸️ LangGraph"]
    if st.session_state.enable_fusion:     chips.append("🔀 RAG-Fusion")
    if st.session_state.enable_graph_rag:  chips.append("🕸️ GraphRAG")
    if st.session_state.enable_reranking:  chips.append("⚡ Reranked")
    if st.session_state.enable_crag:       chips.append("✅ CRAG+Retry")
    if st.session_state.enable_hyde and not st.session_state.enable_fusion:
        chips.append("🧬 HyDE")
    if st.session_state.enable_cache:      chips.append("💾 Cache")
    if st.session_state.enable_thinking:   chips.append("🧠 Thinking")
    if st.session_state.enable_citation_verify: chips.append("🔎 CiteCheck")
    st.markdown(
        '<div class="chip-row">' + "".join(f'<span class="chip">{c}</span>' for c in chips) + '</div>',
        unsafe_allow_html=True
    )

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

if st.session_state.documents_loaded and not st.session_state.messages and st.session_state.suggested_questions:
    st.markdown("**💡 Suggested questions based on your documents:**")
    cols = st.columns(min(len(st.session_state.suggested_questions), 3))
    for i, q in enumerate(st.session_state.suggested_questions[:3]):
        with cols[i]:
            if st.button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state._pending_question = q
                st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("thinking"):
            st.markdown(_thinking_html(message["thinking"], live=False), unsafe_allow_html=True)
        if message.get("workflow_steps") and st.session_state.show_workflow:
            with st.expander(f"🔄 Workflow Steps ({len(message['workflow_steps'])} nodes)", expanded=False):
                st.markdown(_steps_html(message["workflow_steps"]), unsafe_allow_html=True)
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📄 Sources ({len(message['sources'])})", expanded=False):
                st.markdown(_source_html(message["sources"]), unsafe_allow_html=True)


pending = st.session_state.pop("_pending_question", None)
prompt = st.chat_input("Ask about your documents…") or pending

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    chat_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-10:]
    ]

    has_docs = bool(st.session_state.documents_loaded and st.session_state.retriever_pipeline)

    graph_config = {
        "ollama_uri": OLLAMA_API_URL,
        "model": selected_model,
        "has_documents": has_docs and st.session_state.rag_enabled,
        "retriever": st.session_state.retriever_pipeline if has_docs else None,
        "cache": cache if st.session_state.enable_cache else None,
        "embeddings": embeddings,
        "enable_hyde": st.session_state.enable_hyde,
        "enable_fusion": st.session_state.enable_fusion,
        "enable_graph_rag": st.session_state.enable_graph_rag,
        "enable_reranking": st.session_state.enable_reranking,
        "enable_crag": st.session_state.enable_crag,
        "enable_cache": st.session_state.enable_cache,
        "enable_thinking": st.session_state.enable_thinking,
        "enable_citation_verify": st.session_state.enable_citation_verify,
        "temperature": st.session_state.temperature,
        "max_contexts": st.session_state.max_contexts,
    }

    with st.chat_message("assistant"):
        steps_placeholder = st.empty()
        think_ph = st.empty()
        answer_ph = st.empty()

        with st.spinner("Agent is thinking..."):
            result = run_rag_query(
                query=prompt,
                chat_history=chat_history,
                config=graph_config,
                thread_id=f"thread_{int(time.time())}",
            )

        steps = result.get("steps", [])
        thinking = result.get("thinking", "")
        answer = result.get("final_answer", "") or result.get("generation", "")
        sources = result.get("sources", [])
        cache_hit = result.get("cache_hit", False)
        route = result.get("route", "rag")

        if cache_hit:
            st.markdown(
                '<span class="status-badge status-ok"><span class="dot"></span>⚡ Served from semantic cache</span>',
                unsafe_allow_html=True,
            )
        elif route == "chat":
            st.markdown(
                '<span class="status-badge status-warn"><span class="dot"></span>💬 Chat mode (no retrieval)</span>',
                unsafe_allow_html=True,
            )

        if st.session_state.show_workflow and steps:
            with st.expander(f"🔄 LangGraph Workflow ({len(steps)} nodes executed)", expanded=True):
                st.markdown(_steps_html(steps), unsafe_allow_html=True)

        if thinking and st.session_state.enable_thinking:
            think_ph.markdown(_thinking_html(thinking, live=False), unsafe_allow_html=True)

        if answer:
            typewriter(answer, answer_ph, speed=0.002)
        else:
            answer_ph.error("Failed to generate a response. Make sure Ollama is running.")

        if sources:
            with st.expander(f"📄 Sources ({len(sources)})", expanded=False):
                st.markdown(_source_html(sources), unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "thinking": thinking,
        "sources": sources,
        "workflow_steps": steps,
    })
    st.rerun()
