import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
DEFAULT_MODEL = os.getenv("MODEL", "qwen2.5:7b")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text:latest")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CACHE_DIR = os.getenv("CACHE_DIR", "./.cache")
INDEX_META_PATH = os.path.join(CACHE_DIR, "index_meta.json")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CONTEXTUAL_CHUNKS = 150
MAX_CONTEXTS_DEFAULT = 4
CACHE_THRESHOLD = 0.92
MAX_RETRIEVAL_RETRIES = 1

COLLECTION_NAME = "cortex_rag"
