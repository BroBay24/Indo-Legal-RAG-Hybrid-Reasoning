"""
Embedding Model: Pembungkus untuk model embedding (BGE/Sentence Transformers)
Mendukung BGE-M3 untuk embedding bahasa Indonesia yang lebih baik
"""
import os
from typing import List, Optional, Union
import numpy as np
import logging

from config import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper untuk model embedding.
    Default menggunakan BGE-M3 untuk bahasa Indonesia.
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = "cpu",
        normalize_embeddings: bool = True,
        batch_size: int = 32
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.model = None
        self.dimension = None
        
        self._load_model()
    
    def _load_model(self):
        """Load embedding model."""
        
        try:
            # Coba load dengan sentence-transformers
            from sentence_transformers import SentenceTransformer
            
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            # Get embedding dimension
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            self.dimension = len(test_embedding)
            
            logger.info(f"   [OK] Model loaded. Dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"   [ERROR] Gagal memuat model: {str(e)}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding untuk satu teks.
        
        Args:
            text: Teks input
            
        Returns:
            Numpy array embedding
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        return embedding
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings untuk batch teks.
        
        Args:
            texts: List of texts
            
        Returns:
            Numpy array of embeddings (shape: [n_texts, dimension])
        """
        logger.info(f"[STATS] Membuat embedding untuk {len(texts)} teks...")
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            batch_size=self.batch_size,
            show_progress_bar=True
        )
        
        logger.info(f"   [OK] Embeddings created. Shape: {embeddings.shape}")
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding untuk query (dengan prefix khusus untuk BGE).
        BGE models membutuhkan prefix "query: " untuk query embedding.
        
        Args:
            query: Query text
            
        Returns:
            Numpy array embedding
        """
        # BGE models use instruction prefix for queries
        if "bge" in self.model_name.lower():
            # Instruction untuk BGE
            instruction = "Represent this sentence for searching relevant passages: "
            query = instruction + query
        
        return self.embed_text(query)
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Generate embeddings untuk dokumen.
        
        Args:
            documents: List of document texts
            
        Returns:
            Numpy array of embeddings
        """
        # Untuk BGE, dokumen tidak perlu prefix
        return self.embed_texts(documents)
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.dimension
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Hitung cosine similarity antara dua embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # Jika sudah normalized, dot product = cosine similarity
        if self.normalize_embeddings:
            return float(np.dot(embedding1, embedding2))
        
        # Jika belum normalized, hitung manual
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


class HuggingFaceEmbeddings:
    """
    Alternative wrapper menggunakan LangChain HuggingFaceEmbeddings
    untuk kompatibilitas dengan existing code.
    """
    
    def __init__(self, model_name: str = None):
        from langchain_community.embeddings import HuggingFaceEmbeddings as LCHFEmbeddings
        
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.model = LCHFEmbeddings(model_name=self.model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using LangChain interface."""
        return self.model.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query using LangChain interface."""
        return self.model.embed_query(text)


def get_embeddings(model_name: str = None) -> EmbeddingModel:
    """
    Factory function untuk mendapatkan embedding model.
    
    Args:
        model_name: Nama model (opsional)
        
    Returns:
        EmbeddingModel instance
    """
    return EmbeddingModel(model_name=model_name)


def get_langchain_embeddings(model_name: str = None):
    """
    Factory function untuk mendapatkan LangChain-compatible embeddings.
    
    Args:
        model_name: Nama model (opsional)
        
    Returns:
        HuggingFaceEmbeddings instance
    """
    return HuggingFaceEmbeddings(model_name=model_name)


if __name__ == "__main__":
    # Test embedding model
    print("🧪 Testing Embedding Model...")
    
    # Test dengan model default
    embedder = get_embeddings()
    
    # Test single text
    text = "Pasal 1234 ayat (1) menyatakan bahwa setiap orang berhak atas perlindungan hukum."
    embedding = embedder.embed_text(text)
    print(f"\n[OUTPUT] Single text embedding:")
    print(f"   Text: {text[:50]}...")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   First 5 values: {embedding[:5]}")
    
    # Test query embedding
    query = "apa isi pasal 1234?"
    query_embedding = embedder.embed_query(query)
    print(f"\n[SEARCH] Query embedding:")
    print(f"   Query: {query}")
    print(f"   Embedding shape: {query_embedding.shape}")
    
    # Test similarity
    similarity = embedder.similarity(embedding, query_embedding)
    print(f"\n[STATS] Similarity: {similarity:.4f}")
