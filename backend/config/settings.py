"""
Konfigurasi utama untuk RAG Pipeline
Semua parameter model, API keys, dan settings disimpan di sini
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==================== PATH SETTINGS ====================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
INDICES_DIR = DATA_DIR / "indices"
PROCESSED_DIR = DATA_DIR / "processed"

# Buat direktori jika belum ada
INDICES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ==================== PINECONE SETTINGS ====================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_CfFLK_81obZAHtMYz6pzqPYCqBajxMzq9RXQEn8NjxbWyotRfZph6mAREQ5yXpwJekCPm")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hukum-rag")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

# ==================== EMBEDDING SETTINGS ====================
# Menggunakan BGE model untuk embedding bahasa Indonesia yang lebih baik
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
# Alternatif: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 1024  # BGE-M3 dimension, sesuaikan jika ganti model

# ==================== LLM SETTINGS ====================
LLM_MODEL_PATH = str(MODELS_DIR / "llama-3-indo.gguf")
LLM_CONTEXT_LENGTH = 4096  # Dinaikkan karena VM 16 vCPU / 64GB
LLM_GPU_LAYERS = 0  # CPU only
LLM_N_THREADS = 12  # 12 dari 16 vCPU (sisakan 4 untuk embedding/reranker/OS)
LLM_MAX_TOKENS = 1024  # Dinaikkan untuk jawaban lengkap (VM 16 vCPU)
LLM_TEMPERATURE = 0.5
LLM_TOP_P = 0.9

# Hugging Face API (opsional, untuk model cloud)
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")

# ==================== CHUNKING SETTINGS ====================
CHUNK_SIZE = 800  # Diperkecil dari 1000 ke 800 untuk presisi lebih baik
CHUNK_OVERLAP = 150  # Dikurangi sedikit untuk efisiensi
SEPARATORS = ["\n\n", "\n", ".", ";", ",", " ", ""]

# ==================== RETRIEVER SETTINGS ====================
# BM25 Parameters
BM25_K1 = 1.5
BM25_B = 0.75
BM25_TOP_K = 10

# Pinecone/Semantic Search Parameters
SEMANTIC_TOP_K = 10

# Hybrid Fusion Parameters
FUSION_METHOD = "rrf"  # Options: "rrf", "weighted", "interleave"
RRF_K = 60  # Reciprocal Rank Fusion constant
SEMANTIC_WEIGHT = 0.6  # Untuk weighted fusion
LEXICAL_WEIGHT = 0.4

# Final top-k setelah fusion
FINAL_TOP_K = 5

# ==================== PREPROCESSING SETTINGS ====================
# Normalisasi teks hukum
NORMALIZE_UNICODE = True
REMOVE_EXTRA_WHITESPACE = True
NORMALIZE_PASAL = True  # Normalisasi format pasal/ayat

# ==================== API SETTINGS ====================
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", 8000))
DEBUG_MODE = os.getenv("DEBUG_MODE", "True").lower() == "true"

# ==================== LOGGING ====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
