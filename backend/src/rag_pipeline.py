"""
RAG Pipeline: Orkestrasi komponen RAG (embedding, retriever, LLM)
Pipeline tunggal untuk indexing dan query
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
import json

from config import settings
from src.document_loader import DocumentLoader, LoadedDocument, load_documents
from src.legal_preprocessor import LegalPreprocessor
from src.chunker import DocumentChunker, Chunk, chunk_documents
from src.embeddings import EmbeddingModel, get_embeddings
from src.bm25_indexer import BM25Indexer
from src.pinecone_indexer import PineconeIndexer
from src.hybrid_retriever import HybridRetriever, RetrievalResult
from src.llm_wrapper import LLMWrapper, get_llm
from src.reranker import Reranker
from src.legal_prompts import LegalPromptTemplate, get_prompt_template

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Struktur response RAG"""
    answer: str
    sources: List[Dict[str, Any]]
    context: str
    query: str
    retrieval_results: List[RetrievalResult]


class RAGPipeline:
    """
    Pipeline RAG lengkap yang mengorkestrasi semua komponen:
    - Document loading & preprocessing
    - Chunking
    - Embedding & indexing (BM25 + Pinecone)
    - Hybrid retrieval
    - LLM generation
    """
    
    def __init__(
        self,
        use_local_llm: bool = True,
        use_pinecone: bool = True,
        auto_load_index: bool = True
    ):
        """
        Initialize RAG Pipeline.
        
        Args:
            use_local_llm: Use local GGUF model or HuggingFace API
            use_pinecone: Enable Pinecone semantic search
            auto_load_index: Automatically load existing indices
        """
        self.use_local_llm = use_local_llm
        self.use_pinecone = use_pinecone
        
        # Initialize components
        logger.info("[INFO] Initializing RAG Pipeline...")
        
        # Preprocessor
        self.preprocessor = LegalPreprocessor()
        
        # Document loader & chunker
        self.doc_loader = DocumentLoader()
        self.chunker = DocumentChunker()
        
        # Embedding model
        self.embedding_model = get_embeddings()
        
        # BM25 Indexer
        self.bm25_indexer = BM25Indexer()
        
        # Pinecone Indexer (optional)
        self.pinecone_indexer = None
        if use_pinecone:
            try:
                self.pinecone_indexer = PineconeIndexer(
                    embedding_model=self.embedding_model
                )
            except Exception as e:
                logger.warning(f"   [WARNING] Pinecone connection failed: {str(e)}")
                logger.warning("   Continuing with BM25 only...")
        
        # Hybrid Retriever
        self.retriever = HybridRetriever(
            bm25_indexer=self.bm25_indexer,
            pinecone_indexer=self.pinecone_indexer
        )
        
        # Reranker
        self.reranker = Reranker()
        
        # LLM - Load at startup (not lazy loading)
        self.llm = None
        self._llm_loaded = False
        logger.info("[INFO] Loading LLM at startup...")
        self._ensure_llm_loaded()
        
        # Prompt template (llama3 style untuk output lebih baik)
        self.prompt_template = get_prompt_template(style="llama3", language="id")
        
        # Auto-load existing index
        if auto_load_index:
            self._try_load_index()
        
        logger.info("[OK] RAG Pipeline initialized (LLM ready)")
    
    def _try_load_index(self):
        """Try to load existing BM25 index."""
        if self.bm25_indexer.load_index():
            logger.info("   [INDEX] BM25 index loaded from disk")
        else:
            logger.info("   [EMPTY] No existing index found")
    
    def _ensure_llm_loaded(self):
        """Lazy load LLM only when needed."""
        if not self._llm_loaded:
            try:
                self.llm = get_llm(use_local=self.use_local_llm)
                self._llm_loaded = True
            except Exception as e:
                logger.error(f"   [ERROR] Failed to load LLM: {str(e)}")
                raise
    
    def index_documents(
        self,
        data_path: Optional[str] = None,
        save_index: bool = True,
        upload_to_pinecone: bool = True
    ) -> Dict[str, Any]:
        """
        Full indexing pipeline: load → preprocess → chunk → index.
        
        Args:
            data_path: Path to data directory (optional)
            save_index: Save BM25 index to disk
            upload_to_pinecone: Upload vectors to Pinecone
            
        Returns:
            Indexing statistics
        """
        logger.info("[PROCESS] Starting document indexing...")
        
        stats = {
            "documents_loaded": 0,
            "pages_loaded": 0,
            "chunks_created": 0,
            "bm25_indexed": False,
            "pinecone_indexed": False
        }
        
        # 1. Load documents
        logger.info("[1] Loading documents...")
        if data_path:
            self.doc_loader = DocumentLoader(data_path)
        documents = self.doc_loader.load_all_pdfs()
        stats["documents_loaded"] = len(set(d.source for d in documents))
        stats["pages_loaded"] = len(documents)
        
        if not documents:
            logger.warning("[WARNING] No documents found!")
            return stats
        
        # 2. Chunk documents
        logger.info("[2] Chunking documents...")
        chunks = self.chunker.chunk_documents(documents)
        stats["chunks_created"] = len(chunks)
        
        if save_index:
            self.chunker.save_metadata(chunks)
        
        # 3. Build BM25 index
        logger.info("[3] Building BM25 index...")
        self.bm25_indexer.build_index(chunks)
        stats["bm25_indexed"] = True
        
        if save_index:
            self.bm25_indexer.save_index()
        
        # 4. Upload to Pinecone
        if upload_to_pinecone and self.pinecone_indexer:
            logger.info("[4] Uploading to Pinecone...")
            try:
                self.pinecone_indexer.upsert_chunks(chunks)
                stats["pinecone_indexed"] = True
            except Exception as e:
                logger.error(f"   [ERROR] Pinecone upload failed: {str(e)}")
        
        logger.info("[OK] Indexing complete!")
        logger.info(f"   [STATS] Stats: {stats}")
        
        return stats
    
    def query(
        self,
        question: str,
        top_k: int = None,
        max_tokens: int = None,
        temperature: float = None,
        return_context: bool = True
    ) -> RAGResponse:
        """
        Query pipeline: retrieve → generate answer.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            max_tokens: Max tokens for LLM response
            temperature: LLM temperature
            return_context: Include context in response
            
        Returns:
            RAGResponse with answer and sources
        """
        top_k = top_k or settings.FINAL_TOP_K
        
        logger.info(f"[SEARCH] Processing query: {question}")
        
        # Ensure LLM is loaded
        self._ensure_llm_loaded()
        
        # 0. CEK WARMUP / GREETING (Fast Path)
        # Bypass retrieval untuk query pendek/sapaan agar tidak terjebak reranking context
        lower_q = question.lower().strip()
        warmup_keywords = [
            "tes", "test", "halo", "hallo", "hello", "hi", "hey",
            "pemanasan", "cek", "ping", "coba",
            "selamat pagi", "selamat siang", "selamat sore", "selamat malam",
            "assalamualaikum", "hai",
            "apa yang bisa", "bisa bantu apa", "siapa kamu", "siapa anda",
            "kamu siapa", "anda siapa", "apa fungsi", "apa tugas",
            "bisa apa", "lakukan apa"
        ]
        is_warmup = any(k in lower_q for k in warmup_keywords) and len(question.split()) < 15
        
        if is_warmup:
             logger.info("[WARMUP] Detected warm-up query (Fast Path), bypassing retrieval...")
             # Gunakan jawaban statis yang natural agar tidak bergantung pada LLM
             answer = (
                 "Halo! Saya adalah Asisten Hukum AI untuk sistem RAG Hukum Indonesia. "
                 "Saya dapat membantu Anda dalam:\n\n"
                 "1. **Menganalisis putusan Mahkamah Agung** — menjelaskan pertimbangan hukum, ratio decidendi, dan konsekuensi yuridis.\n"
                 "2. **Menjawab pertanyaan hukum** — berdasarkan dokumen-dokumen hukum yang telah diindeks dalam sistem.\n"
                 "3. **Mencari referensi hukum** — menemukan pasal, undang-undang, atau yurisprudensi yang relevan.\n\n"
                 "Silakan ajukan pertanyaan hukum Anda, dan saya akan memberikan jawaban berdasarkan dokumen yang tersedia."
             )
             return RAGResponse(
                answer=answer,
                sources=[],
                context="",
                query=question,
                retrieval_results=[]
             )

        # 1. Retrieve relevant documents
        logger.info("[1] Retrieving documents...")
        results = self.retriever.retrieve(question, top_k=top_k * 2) # Ambil 2x kandidat untuk rerank
        
        if not results:
            return RAGResponse(
                answer="Maaf, saya tidak menemukan dokumen yang relevan untuk menjawab pertanyaan Anda.",
                sources=[],
                context="",
                query=question,
                retrieval_results=[]
            )

        # 1.5 Reranking
        logger.info("[1].[5] Reranking documents...")
        sorted_results = self.reranker.rerank(
            query=question,
            chunks=results,
            top_k=top_k
        )
        
        # Log all rerank scores for debugging
        if sorted_results:
            logger.info("[DEBUG] Rerank scores:")
            for idx, result in enumerate(sorted_results[:5]):
                score = getattr(result, 'rerank_score', None)
                if score is not None:
                    logger.info(f"   [{idx+1}] Score: {score:.4f}")
                else:
                    logger.info(f"   [{idx+1}] Score: N/A")
        
        # Check relevance score - deteksi pertanyaan off-topic
        # PENTING: Gunakan skor RERANKER (bukan skor retrieval mentah)
        is_off_topic = False
        if sorted_results:
            # Ambil rerank_score yang di-attach oleh reranker
            top_score = getattr(sorted_results[0], 'rerank_score', None)
            
            # Jika tidak ada rerank_score, skip deteksi off-topic
            if top_score is not None:
                # Threshold untuk CrossEncoder: < -7 = sangat tidak relevan
                # CrossEncoder bge-reranker menghasilkan skor dari -inf sampai +inf
                # Relaksasi threshold dari -5.0 ke -7.0 untuk kurangi false positive
                if top_score < -7.0:
                    is_off_topic = True
                    logger.warning(f"[OFF-TOPIC] Top rerank score ({top_score:.3f}) below threshold -7.0")
                else:
                    logger.info(f"[OK] Top rerank score: {top_score:.3f} (above threshold)")
        
        # 2. Build context
        # Jika off-topic, kosongkan context DAN sources
        if is_off_topic:
            context = ""
            sources = []
            logger.warning("[OFF-TOPIC] Context and sources cleared due to low relevance")
        else:
            context = self.retriever.get_context_string(sorted_results)
            sources = self.retriever.get_sources(sorted_results)
        
        # Truncate context if too long (max 6000 chars for better coverage)
        if len(context) > 6000:
            logger.warning(f"[WARNING] Context too long ({len(context)} chars), truncating to 6000")
            context = context[:6000] + "\n[...context dipotong...]"
        
        # 3. Generate answer
        logger.info("[2] Generating answer...")
        
        # Jika context kosong (off-topic atau tidak ada hasil)
        if not context or len(context.strip()) == 0:
            logger.warning("[EMPTY CONTEXT] No relevant context found for query")
            return RAGResponse(
                answer="Pertimbangan hukum spesifik tidak ditemukan dalam potongan dokumen yang tersedia.",
                sources=sources,  # Tetap return sources untuk debugging
                context="",
                query=question,
                retrieval_results=results
            )
        
        prompt = self.prompt_template.format_rag_prompt(
            question=question,
            context=context
        )
        
        logger.info(f"   Context length: {len(context)} chars")
        logger.info(f"   Prompt length: {len(prompt)} chars")
        logger.debug(f"   Context preview: {context[:300]}...")
        
        try:
            answer = self.llm.generate(
                prompt,
                max_tokens=max_tokens or 2048,
                temperature=temperature
            )
            
            if not answer or answer.strip() == "":
                logger.error("[ERROR] LLM returned empty response!")
                logger.error(f"   This might be due to:")
                logger.error(f"   1. Prompt format incompatible with model")
                logger.error(f"   2. Context doesn't contain relevant information")
                logger.error(f"   3. Stop sequences triggered too early")
                
                # Fallback: coba generate tanpa context untuk test
                logger.info("   Attempting fallback generation without context...")
                fallback_prompt = f"Pertanyaan: {question}\n\nJawaban singkat:"
                answer = self.llm.generate(fallback_prompt, max_tokens=200, temperature=0.8)
                
                if not answer:
                    answer = "Maaf, sistem tidak dapat menghasilkan jawaban. Silakan coba dengan pertanyaan yang lebih spesifik."
        except Exception as e:
            logger.error(f"[ERROR] LLM generation error: {str(e)}")
            import traceback
            traceback.print_exc()
            answer = f"Error saat generate jawaban: {str(e)}"
        
        logger.info("[OK] Query processed")
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            context=context if return_context else "",
            query=question,
            retrieval_results=results
        )
    
    def query_stream(
        self,
        question: str,
        top_k: int = None,
        max_tokens: int = None,
        temperature: float = None
    ):
        """
        Streaming query - yields tokens as they're generated.
        
        Yields:
            Tokens from LLM
        """
        top_k = top_k or settings.FINAL_TOP_K
        
        # Ensure LLM is loaded
        self._ensure_llm_loaded()
        
        # Retrieve
        results = self.retriever.retrieve(question, top_k=top_k)
        
        if not results:
            yield "Maaf, saya tidak menemukan dokumen yang relevan."
            return
        
        # Build context and prompt
        context = self.retriever.get_context_string(results)
        prompt = self.prompt_template.format_rag_prompt(
            question=question,
            context=context
        )
        
        # Stream generate
        for token in self.llm.stream_generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        ):
            yield token
    
    def chat_without_rag(
        self,
        question: str,
        max_tokens: int = None,
        temperature: float = None
    ) -> str:
        """
        Chat tanpa RAG (direct LLM).
        
        Args:
            question: User question
            max_tokens: Max tokens
            temperature: Temperature
            
        Returns:
            LLM response
        """
        self._ensure_llm_loaded()
        
        prompt = self.prompt_template.format_chat_prompt(question)
        
        return self.llm.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Return pipeline statistics."""
        stats = {
            "bm25": self.bm25_indexer.get_stats(),
            "embedding_model": self.embedding_model.model_name,
            "embedding_dimension": self.embedding_model.get_dimension(),
            "llm_loaded": self._llm_loaded,
        }
        
        if self.pinecone_indexer:
            stats["pinecone"] = self.pinecone_indexer.get_stats()
        
        if self._llm_loaded and self.llm:
            stats["llm"] = self.llm.get_model_info()
        
        return stats
    
    def clear_index(self, clear_pinecone: bool = False):
        """Clear all indices."""
        logger.warning("Clearing indices...")
        
        # Clear BM25
        self.bm25_indexer = BM25Indexer()
        
        # Delete BM25 index file
        index_file = settings.INDICES_DIR / "bm25_index.pkl"
        if index_file.exists():
            index_file.unlink()
            logger.info("   Deleted BM25 index file")
        
        # Clear Pinecone
        if clear_pinecone and self.pinecone_indexer:
            self.pinecone_indexer.delete_all()
        
        logger.info("[OK] Indices cleared")


def create_pipeline(
    use_local_llm: bool = True,
    use_pinecone: bool = True
) -> RAGPipeline:
    """
    Factory function untuk membuat RAG Pipeline.
    
    Args:
        use_local_llm: Use local GGUF model
        use_pinecone: Enable Pinecone
        
    Returns:
        RAGPipeline instance
    """
    return RAGPipeline(
        use_local_llm=use_local_llm,
        use_pinecone=use_pinecone
    )


if __name__ == "__main__":
    # Test RAG Pipeline
    print("🧪 Testing RAG Pipeline...")
    
    # Create pipeline (tanpa Pinecone untuk test cepat)
    pipeline = create_pipeline(use_local_llm=True, use_pinecone=False)
    
    # Show stats
    print(f"\n[STATS] Pipeline stats:")
    stats = pipeline.get_stats()
    print(json.dumps(stats, indent=2, default=str))
    
    # Index documents
    print("\n[PROCESS] Indexing documents...")
    index_stats = pipeline.index_documents(upload_to_pinecone=False)
    print(f"   Stats: {index_stats}")
    
    # Test query
    if index_stats["chunks_created"] > 0:
        print("\n[SEARCH] Testing query...")
        question = "Apa putusan dalam kasus ini?"
        
        try:
            response = pipeline.query(question, top_k=3)
            print(f"\n[OUTPUT] Question: {response.query}")
            print(f"\n[DOC] Answer: {response.answer}")
            print(f"\n[PROCESS] Sources: {response.sources}")
        except Exception as e:
            print(f"[ERROR] Query error: {str(e)}")
