"""
Reranker Module: Menggunakan model Cross-Encoder untuk menyusun ulang (rerank) hasil retrieval
"""
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Reranker:
    """
    Reranker menggunakan Cross-Encoder (seperti BGE-Reranker-M3 atau mxbai-rerank).
    Memberikan skor relevansi yang lebih akurat daripada embedding similarity semata.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = None):
        """
        Inisialisasi reranker.
        Args:
           model_name: Nama model reranker di HuggingFace
           device: 'cpu' atau 'cuda' (otomatis jika None)
        """
        try:
           self.model = CrossEncoder(model_name, automodel_args={"torch_dtype": "auto"}, trust_remote_code=True, device=device)
        except Exception as e:
           logger.error(f"[ERROR] Gagal memuat model Reranker: {e}")
           # Fallback ke model lebih ringan jika gagal (opsional)
           self.model = None

    def rerank(self, query: str, chunks: List[Any], top_k: int = 5) -> List[Any]:
        """
        Melakukan reranking terhadap list chunks berdasarkan query.
        
        Args:
            query: Pertanyaan user
            chunks: List object chunk/retrieval result (harus punya atribut .content atau .chunk.content)
            top_k: Jumlah chunk terbaik yang dikembalikan
            
        Returns:
            List chunk yang sudah diurutkan ulang
        """
        if not self.model or not chunks:
            return chunks[:top_k]

        # Siapkan pasangan (query, document_text)
        # Handle format chunk yang mungkin berbeda (objek Chunk vs string vs RetrievalResult)
        pairs = []
        valid_indices = []
        
        for i, item in enumerate(chunks):
            text = ""
            if isinstance(item, str):
                text = item
            elif hasattr(item, 'content'): # Chunk object
                text = item.content
            elif hasattr(item, 'chunk') and hasattr(item.chunk, 'content'): # RetrievalResult object
                text = item.chunk.content
            
            if text:
                pairs.append([query, text])
                valid_indices.append(i)

        if not pairs:
            return chunks[:top_k]

        # Hitung skor
        scores = self.model.predict(pairs)

        # Gabungkan skor dengan chunk asli
        scored_chunks = []
        for i, score in enumerate(scores):
            original_index = valid_indices[i]
            scored_chunks.append({
                "item": chunks[original_index],
                "score": score
            })

        # Urutkan berdasarkan skor tertinggi (descending)
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)

        # Ambil Top-K dan attach score ke item untuk deteksi off-topic
        results = []
        for x in scored_chunks[:top_k]:
            item = x["item"]
            # Attach reranker score ke item agar bisa dideteksi di pipeline
            if hasattr(item, '__dict__'):
                item.rerank_score = x["score"]
            results.append(item)
        
        logger.info(f"[STATS] Reranking selesai. Top score: {scored_chunks[0]['score']:.4f}")
        return results
