"""
BM25 Indexer: Index BM25 lokal untuk pencarian leksikal (exact match)
Optimal untuk pencarian nomor pasal, UU, dan istilah hukum spesifik
"""
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import re

from rank_bm25 import BM25Okapi
import numpy as np

from config import settings
from src.chunker import Chunk

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


class BM25Indexer:
    """
    BM25 Indexer untuk pencarian leksikal dokumen hukum.
    Menggunakan BM25Okapi untuk ranking berbasis term frequency.
    """
    
    def __init__(
        self,
        k1: float = None,
        b: float = None,
        index_path: Optional[Path] = None
    ):
        self.k1 = k1 or settings.BM25_K1
        self.b = b or settings.BM25_B
        self.index_path = index_path or settings.INDICES_DIR / "bm25_index.pkl"
        
        self.bm25 = None
        self.documents: List[Chunk] = []
        self.tokenized_corpus: List[List[str]] = []
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize teks untuk BM25.
        Sederhana: lowercase dan split by whitespace/punctuation.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Lowercase
        text = text.lower()
        
        # Pertahankan nomor pasal/ayat sebagai token utuh
        # Ganti format "pasal 123" menjadi "pasal_123"
        text = re.sub(r'pasal\s+(\d+)', r'pasal_\1', text)
        text = re.sub(r'ayat\s*\((\d+)\)', r'ayat_\1', text)
        text = re.sub(r'uu\s+no\.?\s*(\d+)', r'uu_\1', text)
        
        # Split by non-alphanumeric (kecuali underscore)
        tokens = re.findall(r'\b[\w_]+\b', text)
        
        # Filter tokens terlalu pendek
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens
    
    def build_index(self, chunks: List[Chunk]):
        """
        Membangun BM25 index dari chunks.
        
        Args:
            chunks: List of Chunk objects
        """
        logger.info(f"🔨 Membangun BM25 index dari {len(chunks)} chunks...")
        
        self.documents = chunks
        
        # Tokenize semua dokumen
        self.tokenized_corpus = [
            self._tokenize(chunk.content) 
            for chunk in chunks
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b
        )
        
        logger.info(f"   [OK] BM25 index berhasil dibangun")
        logger.info(f"   [STATS] Vocabulary size: {len(self.bm25.idf)}")
    
    def search(
        self,
        query: str,
        top_k: int = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Pencarian BM25.
        
        Args:
            query: Query text
            top_k: Jumlah hasil yang dikembalikan
            
        Returns:
            List of (Chunk, score) tuples
        """
        if self.bm25 is None:
            logger.warning("[WARNING] BM25 index belum dibangun!")
            return []
        
        top_k = top_k or settings.BM25_TOP_K
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        logger.debug(f"[SEARCH] BM25 search: {query}")
        logger.debug(f"   Tokens: {tokenized_query}")
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return documents with scores
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Filter zero scores
                results.append((self.documents[idx], float(scores[idx])))
        
        logger.debug(f"   Found {len(results)} results")
        return results
    
    def save_index(self, filepath: Optional[Path] = None):
        """
        Simpan BM25 index ke file.
        
        Args:
            filepath: Path untuk menyimpan (opsional)
        """
        filepath = filepath or self.index_path
        
        index_data = {
            "documents": [chunk.to_dict() for chunk in self.documents],
            "tokenized_corpus": self.tokenized_corpus,
            "k1": self.k1,
            "b": self.b,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info(f"[SAVE] BM25 index disimpan ke {filepath}")
    
    def load_index(self, filepath: Optional[Path] = None) -> bool:
        """
        Load BM25 index dari file.
        
        Args:
            filepath: Path ke file index
            
        Returns:
            True jika berhasil, False jika gagal
        """
        filepath = filepath or self.index_path
        
        if not filepath.exists():
            logger.warning(f"[WARNING] Index file tidak ditemukan: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                index_data = pickle.load(f)
            
            # Reconstruct chunks
            self.documents = [
                Chunk(
                    chunk_id=d["chunk_id"],
                    content=d["content"],
                    metadata=d["metadata"]
                )
                for d in index_data["documents"]
            ]
            
            self.tokenized_corpus = index_data["tokenized_corpus"]
            self.k1 = index_data["k1"]
            self.b = index_data["b"]
            
            # Rebuild BM25
            self.bm25 = BM25Okapi(
                self.tokenized_corpus,
                k1=self.k1,
                b=self.b
            )
            
            logger.info(f"[INDEX] BM25 index loaded dari {filepath}")
            logger.info(f"   [STATS] {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error loading index: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistik index."""
        if self.bm25 is None:
            return {"status": "not_built"}
        
        return {
            "status": "built",
            "num_documents": len(self.documents),
            "vocabulary_size": len(self.bm25.idf),
            "avg_doc_length": self.bm25.avgdl,
            "k1": self.k1,
            "b": self.b,
        }


if __name__ == "__main__":
    # Test BM25 Indexer
    print("🧪 Testing BM25 Indexer...")
    
    # Create sample chunks
    sample_chunks = [
        Chunk(
            chunk_id="1",
            content="Pasal 1234 ayat (1) menyatakan bahwa setiap warga negara berhak atas perlindungan hukum yang sama di hadapan hukum.",
            metadata={"source": "test1.pdf"}
        ),
        Chunk(
            chunk_id="2", 
            content="Berdasarkan UU No. 40 Tahun 2007 tentang Perseroan Terbatas, direksi bertanggung jawab penuh atas pengurusan perseroan.",
            metadata={"source": "test2.pdf"}
        ),
        Chunk(
            chunk_id="3",
            content="Menurut Pasal 1365 KUHPerdata, setiap perbuatan melanggar hukum yang menimbulkan kerugian mewajibkan pemberi kerugian untuk mengganti kerugian tersebut.",
            metadata={"source": "test3.pdf"}
        ),
    ]
    
    # Build index
    indexer = BM25Indexer()
    indexer.build_index(sample_chunks)
    
    print(f"\n[STATS] Index stats: {indexer.get_stats()}")
    
    # Test search
    queries = [
        "pasal 1234",
        "UU perseroan terbatas",
        "perbuatan melanggar hukum",
    ]
    
    for query in queries:
        print(f"\n[SEARCH] Query: {query}")
        results = indexer.search(query, top_k=2)
        for chunk, score in results:
            print(f"   Score: {score:.4f} - {chunk.content[:80]}...")
