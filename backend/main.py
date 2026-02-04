"""
Main FastAPI Application: API untuk RAG Pipeline Hukum Indonesia
"""
import os
import sys
from typing import Optional, List
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from src.rag_pipeline import RAGPipeline, create_pipeline

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline: Optional[RAGPipeline] = None
startup_error: Optional[str] = None


# ==================== LIFESPAN ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize and cleanup."""
    global pipeline, startup_error
    
    logger.info("[INFO] Starting RAG API Server...")
    
    # Initialize pipeline
    try:
        pipeline = create_pipeline(
            use_local_llm=True,
            use_pinecone=True
        )
        logger.info("[OK] RAG Pipeline initialized")
    except Exception as e:
        error_msg = f"Failed to initialize pipeline: {str(e)}"
        logger.error(f"[ERROR] {error_msg}")
        startup_error = error_msg
        import traceback
        traceback.print_exc()
    
    yield
    
    # Cleanup
    logger.info("[STOP] Shutting down RAG API Server...")


# ==================== FASTAPI APP ====================
app = FastAPI(
    title="RAG Hukum Indonesia API",
    description="API untuk Retrieval-Augmented Generation dokumen hukum Indonesia dengan hybrid retriever (BM25 + Pinecone)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Sesuaikan untuk production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== REQUEST/RESPONSE MODELS ====================
class ChatRequest(BaseModel):
    """Request untuk chat/query"""
    pertanyaan: str = Field(..., description="Pertanyaan user", min_length=1)
    top_k: Optional[int] = Field(5, description="Jumlah dokumen yang di-retrieve", ge=1, le=20)
    max_tokens: Optional[int] = Field(512, description="Max tokens untuk jawaban", ge=50, le=2048)
    temperature: Optional[float] = Field(0.7, description="Temperature LLM", ge=0, le=2)
    include_context: Optional[bool] = Field(False, description="Sertakan context dalam response")


class ChatResponse(BaseModel):
    """Response untuk chat/query"""
    jawaban: str
    sumber: List[dict]
    konteks: Optional[str] = None
    pertanyaan: str
    debug_info: Optional[dict] = Field(None, description="Informasi debugging (rerank scores, etc)")


class IndexRequest(BaseModel):
    """Request untuk indexing"""
    data_path: Optional[str] = Field(None, description="Path ke folder data (opsional)")
    upload_pinecone: Optional[bool] = Field(True, description="Upload ke Pinecone")


class IndexResponse(BaseModel):
    """Response untuk indexing"""
    status: str
    stats: dict


class BasicChatRequest(BaseModel):
    """Request untuk chat tanpa RAG"""
    pertanyaan: str = Field(..., min_length=1)
    max_tokens: Optional[int] = Field(150, ge=50, le=1024)


class StatsResponse(BaseModel):
    """Response untuk statistics"""
    bm25: dict
    pinecone: Optional[dict] = None
    embedding_model: str
    llm_loaded: bool


# ==================== ENDPOINTS ====================
@app.get("/")
def home():
    """Health check endpoint."""
    return {
        "status": "online",
        "message": "RAG Hukum Indonesia API",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    """Detailed health check."""
    global pipeline
    
    return {
        "status": "healthy",
        "pipeline_initialized": pipeline is not None,
        "components": {
            "bm25": pipeline.bm25_indexer is not None if pipeline else False,
            "pinecone": pipeline.pinecone_indexer is not None if pipeline else False,
            "llm": pipeline._llm_loaded if pipeline else False
        }
    }


@app.post("/chat", response_model=ChatResponse)
def chat_with_rag(request: ChatRequest):
    """
    Endpoint utama untuk chat dengan RAG.
    Mencari dokumen relevan dan generate jawaban.
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail="Pipeline belum diinisialisasi"
        )
    
    try:
        logger.info(f"[RECV] Chat request: {request.pertanyaan[:50]}...")
        
        response = pipeline.query(
            question=request.pertanyaan,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            return_context=request.include_context
        )
        
        # Extract rerank scores for debugging
        debug_info = {
            "rerank_scores": []
        }
        if hasattr(response, 'retrieval_results') and response.retrieval_results:
            for idx, result in enumerate(response.retrieval_results[:5]):
                score = getattr(result, 'rerank_score', None)
                if score is not None:
                    debug_info["rerank_scores"].append({
                        "rank": idx + 1,
                        "score": float(score)
                    })
        
        return ChatResponse(
            jawaban=response.answer,
            sumber=response.sources,
            konteks=response.context if request.include_context else None,
            pertanyaan=response.query,
            debug_info=debug_info if debug_info["rerank_scores"] else None
        )
        
    except Exception as e:
        logger.error(f"[ERROR] Chat error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@app.post("/chat-stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint.
    Returns tokens as they're generated.
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail="Pipeline belum diinisialisasi"
        )
    
    def generate():
        try:
            for token in pipeline.query_stream(
                question=request.pertanyaan,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            ):
                yield token
        except Exception as e:
            yield f"\n\nError: {str(e)}"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )


@app.post("/chat-basic")
def chat_basic(request: BasicChatRequest):
    """
    Chat tanpa RAG - langsung ke LLM.
    Untuk testing model atau pertanyaan umum.
    """
    global pipeline, startup_error
    
    if not pipeline:
        detail_msg = "Pipeline belum diinisialisasi"
        if startup_error:
            detail_msg += f". Error: {startup_error}"
            
        raise HTTPException(
            status_code=503,
            detail=detail_msg
        )
    
    try:
        answer = pipeline.chat_without_rag(
            question=request.pertanyaan,
            max_tokens=request.max_tokens
        )
        
        return {"jawaban": answer}
        
    except Exception as e:
        logger.error(f"[ERROR] Basic chat error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )


@app.post("/index", response_model=IndexResponse)
def index_documents(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Index dokumen PDF ke BM25 dan Pinecone.
    Proses bisa memakan waktu lama.
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail="Pipeline belum diinisialisasi"
        )
    
    try:
        logger.info("[PROCESS] Starting indexing...")
        
        stats = pipeline.index_documents(
            data_path=request.data_path,
            upload_to_pinecone=request.upload_pinecone
        )
        
        return IndexResponse(
            status="success",
            stats=stats
        )
        
    except Exception as e:
        logger.error(f"[ERROR] Indexing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error indexing: {str(e)}"
        )


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    """
    Get pipeline statistics.
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail="Pipeline belum diinisialisasi"
        )
    
    try:
        stats = pipeline.get_stats()
        
        return StatsResponse(
            bm25=stats.get("bm25", {}),
            pinecone=stats.get("pinecone"),
            embedding_model=stats.get("embedding_model", "unknown"),
            llm_loaded=stats.get("llm_loaded", False)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}"
        )


@app.post("/clear-index")
def clear_index(clear_pinecone: bool = False):
    """
    Clear semua index (hati-hati!).
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail="Pipeline belum diinisialisasi"
        )
    
    try:
        pipeline.clear_index(clear_pinecone=clear_pinecone)
        return {"status": "success", "message": "Index cleared"}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing index: {str(e)}"
        )


@app.get("/search")
def search_only(
    query: str,
    top_k: int = 5,
    method: str = "hybrid"  # "hybrid", "bm25", "semantic"
):
    """
    Search tanpa generate jawaban.
    Untuk debugging atau melihat hasil retrieval.
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail="Pipeline belum diinisialisasi"
        )
    
    try:
        if method == "bm25":
            results = pipeline.bm25_indexer.search(query, top_k)
            return {
                "query": query,
                "method": "bm25",
                "results": [
                    {
                        "content": chunk.content[:500],
                        "score": score,
                        "metadata": chunk.metadata
                    }
                    for chunk, score in results
                ]
            }
        elif method == "semantic" and pipeline.pinecone_indexer:
            results = pipeline.pinecone_indexer.search(query, top_k)
            return {
                "query": query,
                "method": "semantic",
                "results": [
                    {
                        "content": metadata.get("content", "")[:500],
                        "score": score,
                        "metadata": metadata
                    }
                    for metadata, score in results
                ]
            }
        else:
            results = pipeline.retriever.retrieve(query, top_k)
            return {
                "query": query,
                "method": "hybrid",
                "results": [
                    {
                        "content": r.chunk.content[:500],
                        "score": r.score,
                        "rank": r.rank,
                        "source": r.source,
                        "metadata": r.chunk.metadata
                    }
                    for r in results
                ]
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching: {str(e)}"
        )


# ==================== CLI RUNNER ====================
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG_MODE
    )