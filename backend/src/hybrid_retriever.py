"""
Hybrid Retriever: Menggabungkan BM25 (lexical) + Pinecone (semantic) dengan fusion
Menggunakan Reciprocal Rank Fusion (RRF) untuk menggabungkan hasil
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from config import settings
from src.chunker import Chunk
from src.bm25_indexer import BM25Indexer
from src.pinecone_indexer import PineconeIndexer

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Hasil retrieval dengan metadata"""
    chunk: Chunk
    score: float
    source: str  # "bm25", "semantic", atau "fused"
    rank: int


class HybridRetriever:
    """
    Hybrid Retriever yang menggabungkan pencarian leksikal (BM25) dan
    semantik (Pinecone) menggunakan berbagai metode fusion.
    """
    
    def __init__(
        self,
        bm25_indexer: Optional[BM25Indexer] = None,
        pinecone_indexer: Optional[PineconeIndexer] = None,
        fusion_method: str = None,
        rrf_k: int = None,
        semantic_weight: float = None,
        lexical_weight: float = None
    ):
        self.bm25_indexer = bm25_indexer
        self.pinecone_indexer = pinecone_indexer
        
        self.fusion_method = fusion_method or settings.FUSION_METHOD
        self.rrf_k = rrf_k or settings.RRF_K
        self.semantic_weight = semantic_weight or settings.SEMANTIC_WEIGHT
        self.lexical_weight = lexical_weight or settings.LEXICAL_WEIGHT
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        bm25_top_k: int = None,
        semantic_top_k: int = None,
        use_parallel: bool = True
    ) -> List[RetrievalResult]:
        """
        Menjalankan hybrid retrieval.
        
        Args:
            query: Query text
            top_k: Jumlah hasil akhir
            bm25_top_k: Jumlah hasil dari BM25
            semantic_top_k: Jumlah hasil dari semantic search
            use_parallel: Jalankan BM25 dan semantic secara paralel
            
        Returns:
            List of RetrievalResult
        """
        top_k = top_k or settings.FINAL_TOP_K
        bm25_top_k = bm25_top_k or settings.BM25_TOP_K
        semantic_top_k = semantic_top_k or settings.SEMANTIC_TOP_K
        
        logger.info(f"[SEARCH] Hybrid retrieval: {query}")
        
        bm25_results = []
        semantic_results = []
        
        if use_parallel:
            # Parallel retrieval
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {}
                
                if self.bm25_indexer:
                    futures[executor.submit(
                        self._bm25_search, query, bm25_top_k
                    )] = "bm25"
                
                if self.pinecone_indexer:
                    futures[executor.submit(
                        self._semantic_search, query, semantic_top_k
                    )] = "semantic"
                
                for future in as_completed(futures):
                    source = futures[future]
                    try:
                        results = future.result()
                        if source == "bm25":
                            bm25_results = results
                        else:
                            semantic_results = results
                    except Exception as e:
                        logger.error(f"Error in {source} search: {str(e)}")
        else:
            # Sequential retrieval
            if self.bm25_indexer:
                bm25_results = self._bm25_search(query, bm25_top_k)
            if self.pinecone_indexer:
                semantic_results = self._semantic_search(query, semantic_top_k)
        
        logger.info(f"   BM25: {len(bm25_results)} results")
        logger.info(f"   Semantic: {len(semantic_results)} results")
        
        # Fusion
        fused_results = self._fuse_results(
            bm25_results, 
            semantic_results,
            top_k
        )
        
        logger.info(f"   Fused: {len(fused_results)} results")
        return fused_results
    
    def _bm25_search(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[Chunk, float]]:
        """BM25 search wrapper."""
        if not self.bm25_indexer:
            return []
        return self.bm25_indexer.search(query, top_k)
    
    def _semantic_search(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[Chunk, float]]:
        """Semantic search wrapper."""
        if not self.pinecone_indexer:
            return []
        
        # Get chunks map from BM25 indexer if available
        chunks_map = {}
        if self.bm25_indexer and self.bm25_indexer.documents:
            chunks_map = {
                chunk.chunk_id: chunk 
                for chunk in self.bm25_indexer.documents
            }
        
        return self.pinecone_indexer.search_with_chunks(
            query, chunks_map, top_k
        )
    
    def _fuse_results(
        self,
        bm25_results: List[Tuple[Chunk, float]],
        semantic_results: List[Tuple[Chunk, float]],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Menggabungkan hasil BM25 dan semantic dengan metode fusion.
        
        Args:
            bm25_results: Hasil dari BM25
            semantic_results: Hasil dari semantic search
            top_k: Jumlah hasil akhir
            
        Returns:
            List of fused RetrievalResult
        """
        if self.fusion_method == "rrf":
            return self._rrf_fusion(bm25_results, semantic_results, top_k)
        elif self.fusion_method == "weighted":
            return self._weighted_fusion(bm25_results, semantic_results, top_k)
        elif self.fusion_method == "interleave":
            return self._interleave_fusion(bm25_results, semantic_results, top_k)
        else:
            logger.warning(f"Unknown fusion method: {self.fusion_method}, using RRF")
            return self._rrf_fusion(bm25_results, semantic_results, top_k)
    
    def _rrf_fusion(
        self,
        bm25_results: List[Tuple[Chunk, float]],
        semantic_results: List[Tuple[Chunk, float]],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF).
        RRF score = sum(1 / (k + rank_i)) untuk setiap source
        
        Args:
            bm25_results: BM25 results
            semantic_results: Semantic results
            top_k: Number of final results
            
        Returns:
            Fused results sorted by RRF score
        """
        chunk_scores: Dict[str, Dict[str, Any]] = {}
        
        # Process BM25 results
        for rank, (chunk, score) in enumerate(bm25_results):
            chunk_id = chunk.chunk_id
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {
                    "chunk": chunk,
                    "rrf_score": 0.0,
                    "sources": []
                }
            
            chunk_scores[chunk_id]["rrf_score"] += rrf_score
            chunk_scores[chunk_id]["sources"].append(("bm25", rank + 1, score))
        
        # Process semantic results
        for rank, (chunk, score) in enumerate(semantic_results):
            chunk_id = chunk.chunk_id
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {
                    "chunk": chunk,
                    "rrf_score": 0.0,
                    "sources": []
                }
            
            chunk_scores[chunk_id]["rrf_score"] += rrf_score
            chunk_scores[chunk_id]["sources"].append(("semantic", rank + 1, score))
        
        # Sort by RRF score
        sorted_results = sorted(
            chunk_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )[:top_k]
        
        # Convert to RetrievalResult
        results = []
        for rank, item in enumerate(sorted_results):
            results.append(RetrievalResult(
                chunk=item["chunk"],
                score=item["rrf_score"],
                source="fused",
                rank=rank + 1
            ))
        
        return results
    
    def _weighted_fusion(
        self,
        bm25_results: List[Tuple[Chunk, float]],
        semantic_results: List[Tuple[Chunk, float]],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Weighted score fusion.
        Final score = (semantic_weight * semantic_score) + (lexical_weight * bm25_score)
        """
        chunk_scores: Dict[str, Dict[str, Any]] = {}
        
        # Normalize BM25 scores
        if bm25_results:
            max_bm25 = max(score for _, score in bm25_results)
            for chunk, score in bm25_results:
                norm_score = score / max_bm25 if max_bm25 > 0 else 0
                chunk_id = chunk.chunk_id
                
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = {"chunk": chunk, "score": 0.0}
                
                chunk_scores[chunk_id]["score"] += self.lexical_weight * norm_score
        
        # Semantic scores already normalized (0-1)
        for chunk, score in semantic_results:
            chunk_id = chunk.chunk_id
            
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {"chunk": chunk, "score": 0.0}
            
            chunk_scores[chunk_id]["score"] += self.semantic_weight * score
        
        # Sort and return
        sorted_results = sorted(
            chunk_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]
        
        return [
            RetrievalResult(
                chunk=item["chunk"],
                score=item["score"],
                source="fused",
                rank=rank + 1
            )
            for rank, item in enumerate(sorted_results)
        ]
    
    def _interleave_fusion(
        self,
        bm25_results: List[Tuple[Chunk, float]],
        semantic_results: List[Tuple[Chunk, float]],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Interleave fusion: alternating between sources.
        """
        seen_ids = set()
        results = []
        
        bm25_iter = iter(bm25_results)
        semantic_iter = iter(semantic_results)
        
        while len(results) < top_k:
            # Take from semantic
            try:
                chunk, score = next(semantic_iter)
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    results.append(RetrievalResult(
                        chunk=chunk,
                        score=score,
                        source="semantic",
                        rank=len(results) + 1
                    ))
            except StopIteration:
                pass
            
            if len(results) >= top_k:
                break
            
            # Take from BM25
            try:
                chunk, score = next(bm25_iter)
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    results.append(RetrievalResult(
                        chunk=chunk,
                        score=score,
                        source="bm25",
                        rank=len(results) + 1
                    ))
            except StopIteration:
                pass
            
            # Break if both exhausted
            if len(results) == len(seen_ids) and \
               len(bm25_results) + len(semantic_results) <= len(seen_ids):
                break
        
        return results
    
    def get_context_string(
        self,
        results: List[RetrievalResult],
        max_length: int = 4000,
        include_metadata: bool = True
    ) -> str:
        """
        Membangun context string dari hasil retrieval untuk LLM.
        
        Args:
            results: List of RetrievalResult
            max_length: Maximum context length
            include_metadata: Include source metadata
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results):
            # Format context entry
            if include_metadata:
                source = result.chunk.metadata.get("source", "Unknown")
                page = result.chunk.metadata.get("page", "?")
                header = f"[Sumber {i+1}: {source}, Halaman {page}]"
            else:
                header = f"[Konteks {i+1}]"
            
            entry = f"{header}\n{result.chunk.content}\n"
            
            # Check length
            if current_length + len(entry) > max_length:
                # Truncate if needed
                remaining = max_length - current_length - len(header) - 10
                if remaining > 100:
                    truncated_content = result.chunk.content[:remaining] + "..."
                    entry = f"{header}\n{truncated_content}\n"
                    context_parts.append(entry)
                break
            
            context_parts.append(entry)
            current_length += len(entry)
        
        return "\n".join(context_parts)
    
    def get_sources(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """
        Ekstrak informasi sumber dari hasil retrieval.
        
        Returns:
            List of source dictionaries
        """
        sources = []
        for result in results:
            sources.append({
                "source": result.chunk.metadata.get("source", "Unknown"),
                "page": result.chunk.metadata.get("page"),
                "doc_type": result.chunk.metadata.get("doc_type"),
                "score": result.score,
                "retrieval_source": result.source
            })
        return sources


if __name__ == "__main__":
    # Test Hybrid Retriever
    print("🧪 Testing Hybrid Retriever...")
    
    # Create sample chunks
    sample_chunks = [
        Chunk(
            chunk_id="1",
            content="Pasal 1234 ayat (1) menyatakan bahwa setiap warga negara berhak atas perlindungan hukum.",
            metadata={"source": "test1.pdf", "page": 1}
        ),
        Chunk(
            chunk_id="2",
            content="Berdasarkan UU No. 40 Tahun 2007 tentang Perseroan Terbatas, direksi bertanggung jawab.",
            metadata={"source": "test2.pdf", "page": 5}
        ),
        Chunk(
            chunk_id="3",
            content="Perbuatan melanggar hukum menimbulkan kerugian wajib diganti berdasarkan KUHPerdata.",
            metadata={"source": "test3.pdf", "page": 10}
        ),
    ]
    
    # Build BM25 index
    bm25_indexer = BM25Indexer()
    bm25_indexer.build_index(sample_chunks)
    
    # Create hybrid retriever (tanpa Pinecone untuk test)
    retriever = HybridRetriever(
        bm25_indexer=bm25_indexer,
        pinecone_indexer=None,
        fusion_method="rrf"
    )
    
    # Test retrieval
    query = "pasal tentang perlindungan hukum warga negara"
    results = retriever.retrieve(query, top_k=3)
    
    print(f"\n[SEARCH] Query: {query}")
    print(f"\n[STATS] Results:")
    for result in results:
        print(f"   Rank {result.rank} (score: {result.score:.4f}, source: {result.source})")
        print(f"   Content: {result.chunk.content[:100]}...")
    
    # Test context string
    context = retriever.get_context_string(results)
    print(f"\n[OUTPUT] Context string:\n{context}")
