# Indo Legal RAG (Sistem Tanya Jawab Hukum Indonesia)

Sistem retrieval-augmented generation (RAG) yang dirancang khusus untuk domain hukum Indonesia. Sistem ini menggabungkan pencarian dokumen hibrida (keyword + semantik) dengan kemampuan penalaran LLM Llama-3 untuk menganalisis putusan pengadilan dan memberikan opini hukum yang argumentatif.

## 1. Arsitektur Pipeline

Pipeline RAG ini bekerja dalam beberapa tahapan:

1.  **Ingestion & Chunking**: Dokumen hukum (PDF Putusan) diekstrak teksnya dan dipecah menjadi *chunks* (potongan teks) dengan ukuran optimal untuk menjaga konteks hukum.
2.  **Hybrid Retrieval**:
    *   **Keyword Search (BM25)**: Menangkap kata kunci spesifik (misal: "Pasal 1365", "Wanprestasi").
    *   **Semantic Search (Dense Retrieval)**: Menangkap makna dan konsep hukum yang mungkin tidak menggunakan kata kunci yang sama persis (menggunakan `BAAI/bge-m3`).
3.  **Reranking**: Kandidat dokumen dari kedua metode pencarian digabungkan dan diurutkan ulang (reranked) menggunakan model Cross-Encoder (`BAAI/bge-reranker-v2-m3`) untuk memastikan dokumen paling relevan berada di urutan teratas.
4.  **Context Assembly**: Potongan dokumen terpilih disusun menjadi konteks yang bersih.
5.  **Legal Reasoning & Generation**: Model Llama-3 (versi `llama-3-indo.gguf`) menerima pertanyaan user + konteks dokumen, lalu menghasilkan jawaban dengan format opini hukum yang profesional.

---

## 2. Tech Stack

### Frontend
-   **Framework**: Next.js 14.2 (App Router)
-   **Language**: TypeScript
-   **Styling**: Tailwind CSS + Shadcn/UI (Radix Primitives)
-   **Animation**: Framer Motion
-   **State Management**: React Hooks

### Backend
-   **Framework**: FastAPI (Python 3.10+)
-   **LLM Engine**: `llama-cpp-python` (CPU Optimized)
-   **Vector Search**: Pinecone 
-   **Search Algorithms**: Rank-BM25, Sentence-Transformers (PyTorch)
-   **Orchestration**: LangChain (untuk text splitting dan utilities)

### Infrastruktur & Deployment
-   **Server**: Google Cloud Platform (Virtual Machine - e2-standard-32)
-   **OS**: Linux (Ubuntu)
-   **Process Manager**: PM2 (Production Process Management)
-   **Reverse Proxy**: Nginx (Port 80 -> 3000)

---

## 3. Cara Instalasi (Installation)

### Prasyarat
-   Python 3.10 atau lebih baru
-   Node.js v18+ dan npm
-   Git

### Setup Backend

1.  Masuk ke folder backend:
    ```bash
    cd backend
    ```

2.  Buat virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Untuk Linux/Mac
    # venv\Scripts\activate   # Untuk Windows
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Download Model:
    Pastikan file model `.gguf` (misal `llama-3-indo.gguf`) diletakkan di folder `models/`.

### Setup Frontend

1.  Masuk ke folder frontend:
    ```bash
    cd frontend
    ```

2.  Install dependencies:
    ```bash
    npm install
    ```

### Konfigurasi Environment (.env)

Buat file `.env` di root folder proyek (sejajar dengan folder `backend` dan `frontend`):

```bash
# === Konfigurasi Backend ===
# Vector Database (Opsional jika pakai mode lokal penuh)
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=hukum-rag
PINECONE_ENVIRONMENT=us-east-1

# Embedding
EMBEDDING_MODEL_NAME=BAAI/bge-m3

# API Settings
API_HOST=127.0.0.1
API_PORT=8000
DEBUG_MODE=True
LOG_LEVEL=INFO

# === Konfigurasi Frontend (di frontend/.env.local) ===
# Letakkan di file terpisah: frontend/.env.local
NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8000
```

> **PENTING**: Pastikan `API_HOST=127.0.0.1` demi alasan keamanan agar backend tidak terekspos langsung ke internet.

---

## 4. Cara Menjalankan (Deployment)

Untuk production, disarankan menggunakan **PM2** untuk menjalankan backend dan frontend secara bersamaan.

### Menggunakan PM2 (Recommended)

Kami telah menyediakan file `ecosystem.config.js`.

1.  Start semua service:
    ```bash
    pm2 start ecosystem.config.js
    ```

2.  Cek status log:
    ```bash
    pm2 logs
    ```

3.  Restart services:
    ```bash
    pm2 restart all         # Restart semua
    pm2 restart rag-backend # Restart service backend saja
    ```

### Menjalankan Manual (Development)

**Terminal 1 (Backend):**
```bash
cd backend
source venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```

---

## 5. Dokumentasi API

Backend mengekspos endpoint dokumentasi interaktif yang disediakan oleh FastAPI.

1.  Pastikan backend berjalan.
2.  Buka browser akses: `http://localhost:8000/docs` (Swagger UI).

### Endpoint Utama

*   `POST /chat`
    *   **Body**:
        ```json
        {
          "pertanyaan": "Bagaimana kedudukan ahli waris...",
          "top_k": 5,
          "max_tokens": 2048,
          "temperature": 0.5
        }
        ```
    *   **Response**: JSON berisi jawaban, sumber dokumen, dan metadata.

---

## 6. Troubleshooting

### 1. Error `socket hang up` / `fetch failed` di Frontend
**Penyebab:** Next.js Edge Runtime memiliki batas timeout pendek, atau Backend belum selesai loading model saat Frontend mencoba connect.
**Solusi:**
- Pastikan `/app/api/chat/route.ts` menggunakan `runtime = 'nodejs'` (bukan 'edge').
- Tunggu sekitar 40-60 detik setelah restart backend sebelum mencoba chat pertama kali.

### 2. Backend `WARNING: Invalid HTTP request received`
**Penyebab:** Ada bot/scanner eksternal mencoba akses port 8000.
**Solusi:** Pastikan `API_HOST` di `.env` diset ke `127.0.0.1`. Ini mencegah akses langsung dari luar server (hanya frontend lokal yang bisa akses).

### 3. Model Loading Lama / Timeout
**Penyebab:** Model GGUF besar dan berjalan di CPU.
**Solusi:**
- Sabar, inisialisasi awal bisa memakan waktu 1-2 menit.
- Periksa log backend untuk memantau progress: `pm2 logs rag-backend`.

### 4. Output LLM Duplicate / Aneh
**Penyebab:** Prompt template memiliki token spesial ganda (misal `<|begin_of_text|>`).
**Solusi:** Hapus token duplikat di `backend/src/legal_prompts.py` (sudah diperbaiki di versi terbaru).
