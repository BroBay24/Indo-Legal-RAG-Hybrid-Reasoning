# Dokumentasi Teknis Sistem RAG Hukum Indonesia (Deep Dive)

Dokumen ini menjelaskan detail teknis implementasi, keputusan arsitektur, dan optimasi performa untuk sistem Legal RAG.

## 1. Arsitektur Hybrid Retrieval & Reranking

Sistem ini menggunakan strategi **Hybrid Reciprocal Rank Fusion (RRF)** untuk menggabungkan keunggulan pencarian kata kunci dan pencarian semantik.

### Alur Data
1.  **User Query**
2.  **Parallel Retrieval**:
    *   **BM25 (Sparse)**: Menangani pencarian *lexical* presisi tinggi (misal: "Pasal 1365", "Nomor 814 K/Pdt/2019").
    *   **Vector Search (Dense)**: Menggunakan model `BAAI/bge-m3` untuk menangkap makna semantik dan hubungan konseptual hukum.
3.  **Result Fusion (RRF)**:
    Kedua hasil digabungkan dengan rumus:
    $$ score = \frac{1}{k + rank_{bm25}} + \frac{1}{k + rank_{vector}} $$
    Dimana $k=60$. Ini menyeimbangkan hasil tanpa perlu tuning bobot manual yang agresif.
4.  **Cross-Encoder Reranking**:
    20 Kandidat teratas dari RRF dinilai ulang oleh model `BAAI/bge-reranker-v2-m3`. Model ini membaca *pairs* (Query, Dokumen) secara bersamaan untuk akurasi relevansi tertinggi.
5.  **Context Selection**: 5 Dokumen terbaik (Top-5) diteruskan ke LLM.

## 2. Optimasi Infrastruktur (GCP e2-standard-32)

Sistem berjalan di VM Google Cloud `e2-standard-32` (32 vCPU, 128 GB RAM). Berikut konfigurasi spesifik untuk memaksimalkan hardware ini:

### Alokasi Thread CPU
Kami menggunakan `llama-cpp-python` yang sangat sensitif terhadap manajemen thread. Konfigurasi optimal yang diterapkan:
-   **Total vCPU**: 32
-   **LLM Inference Threads (`n_threads`)**: 24
-   **System/Overhead Threads**: 8

**Alasan**: Memberikan seluruh 32 core ke LLM seringkali menyebabkan *context switching overhead* dan *thread contention* dengan OS atau proses chunking/embedding, yang justru menurunkan performa. Menyisakan 8 core memastikan stabilitas sistem.

### Manajemen Memori & Model
-   **Model**: `llama-3-indo-v1.gguf` (Quantized GGUF).
-   **Context Window (`n_ctx`)**: 8192 tokens (Ditingkatkan dari 4096).
-   **Max Tokens Output**: 2048 tokens.
-   **Batch Size**: 1024.
-   **Offloading**: Sepenuhnya CPU (`n_gpu_layers=0`), memanfaatkan RAM 128GB yang melimpah.

## 3. Penanganan Masalah Produksi (Troubleshooting Log)

Berikut adalah ringkasan teknis dari isu-isu stabilitas yang telah diselesaikan:

### A. Socket Hang Up / Timeout (Frontend)
-   **Gejala**: Request chat gagal setelah 60-120 detik dengan error `fetch failed` atau `socket hang up`.
-   **Penyebab**:
    1.  Next.js Edge Runtime memiliki limit eksekusi yang pendek.
    2.  `AbortController` default membatalkan request sebelum backend (CPU-based) selesai men-generate jawaban panjang.
-   **Solusi**:
    1.  Switch ke `runtime = 'nodejs'` di `route.ts`.
    2.  Set `maxDuration = 300` (5 menit).
    3.  Konfigurasi `experimental.proxyTimeout: 300000` di `next.config.js`.

### B. ECONNREFUSED saat Startup (Race Condition)
-   **Gejala**: Error koneksi saat PM2 restart karena Frontend mencoba connect sebelum Backend selesai memuat model 5GB+ ke memori.
-   **Solusi**: Implementasi *Exponential Backoff Retry* di `route.ts`. Frontend melakukan 5x percobaan koneksi dengan jeda 8 detik, dan mendeteksi error `ECONNREFUSED` yang terbungkus dalam `cause` object Node.js.

### C. LLM Output Quality
-   **Masalah**: Jawaban terlalu umum, halusinasi nama pihak, format tidak standar.
-   **Solusi**:
    1.  **Strict System Prompt**: Memaksa LLM menyebutkan "Nama Asli" pihak berperkara.
    2.  **Llama-3 Template**: Migrasi dari simple raw text ke template Llama-3 (`<|start_header_id|>...`) untuk memisahkan instruksi sistem dan konteks dokumen dengan lebih tegas.
    3.  **Token Increase**: Menaikkan `max_tokens` ke 2048 agar analisis hukum yang kompleks tidak terpotong di tengah kalimat.

## 4. Stack Komponen

| Komponen | Teknologi | Keterangan |
| :--- | :--- | :--- |
| **LLM Engine** | `llama.cpp` (Python bindings) | Inference engine C++ yang efisien untuk CPU. |
| **Embeddings** | `BAAI/bge-m3` | Model state-of-the-art untuk multibahasa/Indonesia. |
| **API Framework** | FastAPI | Asynchronous server, ideal untuk I/O bound tasks. |
| **Frontend** | Next.js 14 (App Router) | React framework untuk SSR dan API Proxying. |
| **Process Manager** | PM2 | Menjaga proses tetap hidup, logging otomatis. |

