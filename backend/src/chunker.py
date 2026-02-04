"""
Document Chunker: Pemotongan dokumen menjadi chunks dengan overlap
Optimized untuk dokumen hukum Indonesia
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import hashlib
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import settings
from src.document_loader import LoadedDocument
from src.legal_preprocessor import LegalPreprocessor

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Struktur data untuk chunk dokumen"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DocumentChunker:
    """
    Chunker dokumen dengan strategi khusus untuk dokumen hukum.
    Mendukung berbagai metode chunking dan overlap.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: List[str] = None,
        preprocess: bool = True
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.separators = separators or settings.SEPARATORS
        self.preprocess = preprocess
        
        # Initialize splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )
        
        # Preprocessor untuk teks hukum
        if self.preprocess:
            self.preprocessor = LegalPreprocessor()
        
        self.metadata_file = settings.PROCESSED_DIR / "metadata.json"
    
    def chunk_document(self, document: LoadedDocument) -> List[Chunk]:
        """
        Memotong satu dokumen menjadi chunks.
        
        Args:
            document: LoadedDocument object
            
        Returns:
            List of Chunk objects
        """
        content = document.content
        
        # Preprocess jika diaktifkan
        if self.preprocess:
            content = self.preprocessor.preprocess(content)
        
        # Split content
        text_chunks = self.splitter.split_text(content)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            # Generate unique chunk ID
            chunk_id = self._generate_chunk_id(document.source, i, chunk_text)
            
            # Ekstrak entitas hukum dari chunk
            legal_entities = {}
            if self.preprocess:
                legal_entities = self.preprocessor.extract_legal_entities(chunk_text)
            
            # Build metadata (Identifikasi "Section")
            # Logika sederhana: 1000 char pertama dokumen biasanya IDENTITAS/DUDUK PERKARA
            section = "isi"
            if "total_chars" not in document.metadata:
                 document.metadata["total_chars"] = len(content)
            
            # Jika di awal dokumen (kurang dari 15% awal atau chunk index < 3)
            # Biasanya Identitas Pihak ada di halaman 1-2
            if i < 3 or (i * len(chunk_text)) < (len(content) * 0.15):
                section = "header" # Identitas, Kepala Putusan
            elif "MENGADILI" in chunk_text or "AMAR" in chunk_text:
                section = "amar"
            elif "DUDUK PERKARA" in chunk_text:
                section = "duduk_perkara"
            
            chunk_metadata = {
                **document.metadata,
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "char_count": len(chunk_text),
                "legal_entities": legal_entities,
                "section": section # Metadata filtering point
            }
            
            chunk = Chunk(
                chunk_id=chunk_id,
                content=chunk_text,
                metadata=chunk_metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_documents(self, documents: List[LoadedDocument]) -> List[Chunk]:
        """
        Memotong multiple dokumen menjadi chunks.
        
        Args:
            documents: List of LoadedDocument objects
            
        Returns:
            List of all Chunk objects
        """
        logger.info(f"[SPLIT] Memulai chunking {len(documents)} dokumen...")
        
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(f"[OK] Total {len(all_chunks)} chunks dibuat")
        return all_chunks
    
    def _generate_chunk_id(self, source: str, index: int, content: str) -> str:
        """Generate unique chunk ID berdasarkan source dan content."""
        # Kombinasi source, index, dan hash content
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        return f"{source_hash}_{index}_{content_hash}"
    
    def save_metadata(self, chunks: List[Chunk], filepath: Optional[Path] = None):
        """
        Simpan metadata chunks ke file JSON.
        
        Args:
            chunks: List of Chunk objects
            filepath: Path untuk menyimpan file (opsional)
        """
        filepath = filepath or self.metadata_file
        
        metadata = {
            "total_chunks": len(chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunks": [chunk.to_dict() for chunk in chunks]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[SAVE] Metadata disimpan ke {filepath}")
    
    def load_metadata(self, filepath: Optional[Path] = None) -> List[Chunk]:
        """
        Load chunks dari metadata file.
        
        Args:
            filepath: Path ke metadata file
            
        Returns:
            List of Chunk objects
        """
        filepath = filepath or self.metadata_file
        
        if not filepath.exists():
            logger.warning(f"[WARNING] Metadata file tidak ditemukan: {filepath}")
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        chunks = [
            Chunk(
                chunk_id=c["chunk_id"],
                content=c["content"],
                metadata=c["metadata"]
            )
            for c in metadata.get("chunks", [])
        ]
        
        logger.info(f"[INDEX] Loaded {len(chunks)} chunks dari {filepath}")
        return chunks
    
    def to_langchain_documents(self, chunks: List[Chunk]) -> List[Document]:
        """
        Konversi Chunk ke format LangChain Document.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            List of LangChain Document objects
        """
        return [
            Document(
                page_content=chunk.content,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    **chunk.metadata
                }
            )
            for chunk in chunks
        ]


def chunk_documents(
    documents: List[LoadedDocument],
    chunk_size: int = None,
    chunk_overlap: int = None,
    save_metadata: bool = True
) -> List[Chunk]:
    """
    Fungsi helper untuk chunking dokumen.
    
    Args:
        documents: List of LoadedDocument objects
        chunk_size: Ukuran chunk
        chunk_overlap: Ukuran overlap
        save_metadata: Apakah simpan metadata
        
    Returns:
        List of Chunk objects
    """
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = chunker.chunk_documents(documents)
    
    if save_metadata:
        chunker.save_metadata(chunks)
    
    return chunks


if __name__ == "__main__":
    # Test chunking
    from src.document_loader import load_documents
    
    print("🧪 Testing Document Chunker...")
    
    # Load documents
    docs = load_documents()
    
    if docs:
        # Chunk documents
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.chunk_documents(docs[:2])  # Test dengan 2 dokumen
        
        print(f"\n[STATS] Chunking results:")
        print(f"   Total chunks: {len(chunks)}")
        
        if chunks:
            print(f"\n[OUTPUT] Sample chunk:")
            print(f"   ID: {chunks[0].chunk_id}")
            print(f"   Content preview: {chunks[0].content[:200]}...")
            print(f"   Metadata: {chunks[0].metadata}")
