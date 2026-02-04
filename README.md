# RAG Pipeline Hukum Indonesia 🇮🇩⚖️

Pipeline Retrieval-Augmented Generation (RAG) untuk dokumen hukum Indonesia dengan **Hybrid Retriever** (BM25 + Pinecone semantic search) dan Reciprocal Rank Fusion (RRF).

## 🏗️ Arsitektur

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     HYBRID RETRIEVER                            │
│  ┌─────────────────┐            ┌─────────────────────────┐    │
│  │   BM25 (Lokal)  │            │  Pinecone (Cloud)       │    │
│  │   - Exact Match │            │  - Semantic Search      │    │
│  │   - Pasal/Nomor │            │  - BGE-M3 Embeddings    │    │
│  └────────┬────────┘            └───────────┬─────────────┘    │
│           │                                  │                  │
│           └──────────┬───────────────────────┘                  │
│                      │                                          │
│                      ▼                                          │
│           ┌──────────────────────┐                             │
│           │  RRF Fusion          │                             │
│           │  (Rank Aggregation)  │                             │
│           └──────────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CONTEXT BUILDER                             │
│         Top-K chunks + Metadata + Sources                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LLM (Llama-3 Indo GGUF)                        │
│         Legal Prompt Template → Answer Generation               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RESPONSE                                    │
│         Jawaban + Sumber Dokumen                                │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Struktur Proyek

```
proyekrag/
├── backend/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Konfigurasi utama
│   ├── src/
│   │   ├── __init__.py
│   │   ├── document_loader.py   # PDF loader
│   │   ├── legal_preprocessor.py # Normalisasi teks hukum
│   │   ├── chunker.py           # Text chunking dengan overlap
│   │   ├── embeddings.py        # BGE embedding model
│   │   ├── bm25_indexer.py      # BM25 index lokal
│   │   ├── pinecone_indexer.py  # Pinecone vector store
│   │   ├── hybrid_retriever.py  # Hybrid search + RRF
│   │   ├── llm_wrapper.py       # LLM wrapper (local/cloud)
│   │   ├── legal_prompts.py     # Prompt templates
│   │   └── rag_pipeline.py      # Orkestrasi pipeline
│   ├── main.py                  # FastAPI application
│   ├── run.py                   # CLI runner
│   └── requirements.txt
├── data/
│   ├── *.pdf                    # Dokumen PDF sumber
│   ├── processed/               # Metadata chunks
│   └── indices/                 # BM25 index files
├── models/
│   └── llama-3-indo.gguf        # Model LLM lokal
└── frontend/                    # (Coming soon)
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd proyekrag/backend

# Buat virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Konfigurasi

Copy `.env.example` ke `.env` dan sesuaikan:

```bash
cp ../.env.example ../.env
```

Edit `.env`:
```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=hukum-rag
EMBEDDING_MODEL_NAME=BAAI/bge-m3
```

### 3. Indexing Dokumen

```bash
# Via CLI
python run.py index

# Atau tanpa Pinecone (BM25 only)
python run.py index --no-pinecone
```

### 4. Jalankan Server

```bash
# Via CLI
python run.py serve

# Atau langsung
python main.py

# Dengan reload untuk development
python run.py serve --reload
```

Server akan berjalan di `http://localhost:8000`

### 5. Test Query

```bash
# Via CLI
python run.py query "Apa putusan dalam kasus ini?"

# Interactive chat
python run.py chat

# Via API
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"pertanyaan": "Apa putusan dalam kasus ini?"}'
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health status |
| `/chat` | POST | Chat dengan RAG |
| `/chat-stream` | POST | Streaming chat |
| `/chat-basic` | POST | Chat tanpa RAG |
| `/index` | POST | Index dokumen |
| `/stats` | GET | Pipeline statistics |
| `/search` | GET | Search only (debug) |
| `/clear-index` | POST | Clear semua index |

### Contoh Request

```json
POST /chat
{
  "pertanyaan": "Siapa penggugat dalam kasus ini?",
  "top_k": 5,
  "max_tokens": 512,
  "temperature": 0.7,
  "include_context": true
}
```

### Contoh Response

```json
{
  "jawaban": "Berdasarkan dokumen, penggugat dalam kasus ini adalah...",
  "sumber": [
    {
      "source": "putusan_690_pdt.g_2024.pdf",
      "page": 1,
      "doc_type": "putusan",
      "score": 0.85
    }
  ],
  "konteks": "[Sumber 1: ...]",
  "pertanyaan": "Siapa penggugat dalam kasus ini?"
}
```

## ⚙️ Konfigurasi

Semua konfigurasi ada di `config/settings.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Ukuran chunk |
| `CHUNK_OVERLAP` | 200 | Overlap antar chunk |
| `BM25_TOP_K` | 10 | Top-K BM25 |
| `SEMANTIC_TOP_K` | 10 | Top-K Pinecone |
| `FINAL_TOP_K` | 5 | Final results |
| `FUSION_METHOD` | "rrf" | rrf/weighted/interleave |
| `RRF_K` | 60 | RRF constant |
| `LLM_MAX_TOKENS` | 512 | Max LLM tokens |
| `LLM_TEMPERATURE` | 0.7 | LLM temperature |

## 🧠 Model

- **Embedding**: `BAAI/bge-m3` (atau `paraphrase-multilingual-MiniLM-L12-v2`)
- **LLM**: Llama-3 Indo (GGUF format)
- **Vector Store**: Pinecone (cloud)
- **Lexical Search**: BM25Okapi (lokal)

## 📝 License

MIT License
