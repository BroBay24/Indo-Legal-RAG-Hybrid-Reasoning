"""
Pinecone Indexer: Koneksi ke Pinecone untuk pencarian semantik (vector search)
"""
from typing import List, Dict, Any, Optional, Tuple
import logging
import time

from pinecone import Pinecone, ServerlessSpec
import numpy as np

from config import settings
from src.chunker import Chunk
from src.embeddings import EmbeddingModel, get_embeddings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


class PineconeIndexer:
    """
    Pinecone Indexer untuk pencarian semantik menggunakan vector embeddings.
    """
    
    def __init__(
        self,
        api_key: str = None,
        index_name: str = None,
        embedding_model: Optional[EmbeddingModel] = None,
        dimension: int = None
    ):
        self.api_key = api_key or settings.PINECONE_API_KEY
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        
        # Initialize embedding model
        self.embedding_model = embedding_model or get_embeddings()
        
        # Get index reference
        self.index = None
        self._connect_to_index()
    
    def _connect_to_index(self):
        """Connect to existing Pinecone index or create new one."""
        try:
            # Check if index exists
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"📦 Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=settings.PINECONE_ENVIRONMENT
                    )
                )
                # Wait for index to be ready
                time.sleep(10)
            
            self.index = self.pc.Index(self.index_name)
            logger.info(f"[OK] Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"[ERROR] Error connecting to Pinecone: {str(e)}")
            raise
    
    def upsert_chunks(
        self,
        chunks: List[Chunk],
        batch_size: int = 100,
        namespace: str = ""
    ):
        """
        Upsert chunks ke Pinecone.
        
        Args:
            chunks: List of Chunk objects
            batch_size: Ukuran batch untuk upsert
            namespace: Pinecone namespace
        """
        logger.info(f"[UPLOAD] Uploading {len(chunks)} chunks ke Pinecone...")
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.embed_documents(texts)
        
        # Prepare vectors for upsert
        vectors = []
        for i, chunk in enumerate(chunks):
            # Simplify metadata untuk Pinecone (harus flat)
            metadata = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content[:1000],  # Truncate content
                "source": chunk.metadata.get("source", ""),
                "page": chunk.metadata.get("page", 0),
                "section": chunk.metadata.get("section", ""), # Added section metadata
                "doc_type": chunk.metadata.get("doc_type", ""),
                "case_type": chunk.metadata.get("case_type", ""),
            }
            
            vectors.append({
                "id": chunk.chunk_id,
                "values": embeddings[i].tolist(),
                "metadata": metadata
            })
        
        # Upsert in batches
        total_upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
            total_upserted += len(batch)
            logger.info(f"   [SEND] Uploaded {total_upserted}/{len(vectors)} vectors")
        
        logger.info(f"[OK] Selesai upload {total_upserted} vectors ke Pinecone")
    
    def search(
        self,
        query: str,
        top_k: int = None,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Pencarian semantik di Pinecone.
        
        Args:
            query: Query text
            top_k: Jumlah hasil
            namespace: Pinecone namespace
            filter: Metadata filter
            
        Returns:
            List of (metadata, score) tuples
        """
        top_k = top_k or settings.SEMANTIC_TOP_K
        
        logger.debug(f"[SEARCH] Pinecone search: {query}")
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Search
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
            filter=filter
        )
        
        # Parse results
        search_results = []
        for match in results.matches:
            metadata = match.metadata or {}
            score = match.score
            search_results.append((metadata, score))
        
        logger.debug(f"   Found {len(search_results)} results")
        return search_results
    
    def search_with_chunks(
        self,
        query: str,
        chunks_map: Dict[str, Chunk],
        top_k: int = None,
        namespace: str = ""
    ) -> List[Tuple[Chunk, float]]:
        """
        Pencarian dan return Chunk objects.
        
        Args:
            query: Query text
            chunks_map: Dictionary mapping chunk_id ke Chunk
            top_k: Jumlah hasil
            namespace: Pinecone namespace
            
        Returns:
            List of (Chunk, score) tuples
        """
        results = self.search(query, top_k, namespace)
        
        chunk_results = []
        for metadata, score in results:
            chunk_id = metadata.get("chunk_id")
            if chunk_id and chunk_id in chunks_map:
                chunk_results.append((chunks_map[chunk_id], score))
            else:
                # Buat chunk dari metadata jika tidak ada di map
                chunk = Chunk(
                    chunk_id=chunk_id or "unknown",
                    content=metadata.get("content", ""),
                    metadata=metadata
                )
                chunk_results.append((chunk, score))
        
        return chunk_results
    
    def delete_all(self, namespace: str = ""):
        """Delete semua vectors di namespace."""
        logger.warning(f"Deleting all vectors in namespace: {namespace or 'default'}")
        self.index.delete(delete_all=True, namespace=namespace)
        logger.info("[OK] All vectors deleted")
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistik index."""
        try:
            stats = self.index.describe_index_stats()
            return {
                "status": "connected",
                "index_name": self.index_name,
                "dimension": stats.dimension,
                "total_vectors": stats.total_vector_count,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {}
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


if __name__ == "__main__":
    # Test Pinecone Indexer
    print("🧪 Testing Pinecone Indexer...")
    
    try:
        indexer = PineconeIndexer()
        stats = indexer.get_stats()
        print(f"\n[STATS] Pinecone stats: {stats}")
        
        # Test search if there are vectors
        if stats.get("total_vectors", 0) > 0:
            query = "pasal tentang perlindungan hukum"
            results = indexer.search(query, top_k=3)
            
            print(f"\n[SEARCH] Search results for: {query}")
            for metadata, score in results:
                print(f"   Score: {score:.4f}")
                print(f"   Content: {metadata.get('content', '')[:100]}...")
                print()
                
    except Exception as e:
        print(f"[ERROR] Error: {str(e)}")
        print("   Make sure PINECONE_API_KEY is set correctly")
