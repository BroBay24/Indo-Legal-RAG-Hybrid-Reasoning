"""
Document Loader: Ekstraksi PDF menjadi teks
Mendukung berbagai format dokumen hukum Indonesia
"""
import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from config import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@dataclass
class LoadedDocument:
    """Struktur data untuk dokumen yang dimuat"""
    content: str
    metadata: Dict[str, Any]
    source: str
    page_number: Optional[int] = None


class DocumentLoader:
    """
    Loader dokumen PDF untuk dokumen hukum Indonesia.
    Mendukung ekstraksi metadata dan content dari file PDF.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = Path(data_path) if data_path else settings.DATA_DIR
        
    def load_pdf(self, file_path: str) -> List[LoadedDocument]:
        """
        Memuat satu file PDF dan mengekstrak konten per halaman.
        
        Args:
            file_path: Path ke file PDF
            
        Returns:
            List of LoadedDocument objects
        """
        logger.info(f"[DOC] Memuat PDF: {file_path}")
        
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            documents = []
            for page in pages:
                # Ekstrak metadata dari nama file
                filename = os.path.basename(file_path)
                doc_metadata = self._extract_metadata_from_filename(filename)
                
                # Gabungkan metadata
                metadata = {
                    **doc_metadata,
                    "source": file_path,
                    "page": page.metadata.get("page", 0) + 1,  # 1-indexed
                    "total_pages": len(pages),
                }
                
                doc = LoadedDocument(
                    content=page.page_content,
                    metadata=metadata,
                    source=file_path,
                    page_number=metadata["page"]
                )
                documents.append(doc)
            
            logger.info(f"   [OK] Berhasil memuat {len(documents)} halaman")
            return documents
            
        except Exception as e:
            logger.error(f"   [ERROR] Error memuat {file_path}: {str(e)}")
            return []
    
    def load_all_pdfs(self, pattern: str = "*.pdf") -> List[LoadedDocument]:
        """
        Memuat semua file PDF dari direktori data.
        
        Args:
            pattern: Glob pattern untuk mencari file
            
        Returns:
            List of all LoadedDocument objects
        """
        pdf_files = glob.glob(str(self.data_path / pattern))
        
        if not pdf_files:
            logger.warning(f"[WARNING] Tidak ada file PDF ditemukan di {self.data_path}")
            return []
        
        logger.info(f"[INDEX] Ditemukan {len(pdf_files)} file PDF")
        
        all_documents = []
        for pdf_file in pdf_files:
            docs = self.load_pdf(pdf_file)
            all_documents.extend(docs)
        
        logger.info(f"[PROCESS] Total {len(all_documents)} halaman dimuat dari semua PDF")
        return all_documents
    
    def _extract_metadata_from_filename(self, filename: str) -> Dict[str, Any]:
        """
        Ekstrak metadata dari nama file dokumen hukum.
        
        Format umum: putusan_[nomor]_[jenis]_[tahun]_[pengadilan]_[tanggal].pdf
        Contoh: putusan_690_pdt.g_2024_pn_jkt.utr_20260117155257.pdf
        """
        metadata = {
            "filename": filename,
            "doc_type": "unknown",
            "case_number": None,
            "case_type": None,
            "year": None,
            "court": None,
        }
        
        # Hapus ekstensi
        name = filename.replace(".pdf", "").lower()
        parts = name.split("_")
        
        try:
            if parts[0] == "putusan":
                metadata["doc_type"] = "putusan"
                
                if len(parts) >= 2:
                    metadata["case_number"] = parts[1]
                
                if len(parts) >= 3:
                    # Parse jenis perkara (pdt.g, k, pid, etc)
                    case_type = parts[2]
                    if "pdt" in case_type:
                        metadata["case_type"] = "perdata"
                    elif "pid" in case_type:
                        metadata["case_type"] = "pidana"
                    elif case_type == "k":
                        metadata["case_type"] = "kasasi"
                    else:
                        metadata["case_type"] = case_type
                
                if len(parts) >= 4:
                    try:
                        metadata["year"] = int(parts[3])
                    except ValueError:
                        pass
                
                if len(parts) >= 5:
                    court_parts = []
                    for p in parts[4:-1]:  # Skip timestamp di akhir
                        if not p.isdigit():
                            court_parts.append(p)
                    if court_parts:
                        metadata["court"] = "_".join(court_parts)
                        
        except Exception as e:
            logger.debug(f"Tidak dapat parse metadata dari filename: {filename}")
        
        return metadata
    
    def to_langchain_documents(self, documents: List[LoadedDocument]) -> List[Document]:
        """
        Konversi LoadedDocument ke format LangChain Document.
        
        Args:
            documents: List of LoadedDocument
            
        Returns:
            List of LangChain Document objects
        """
        return [
            Document(
                page_content=doc.content,
                metadata=doc.metadata
            )
            for doc in documents
        ]


def load_documents(data_path: Optional[str] = None) -> List[LoadedDocument]:
    """
    Fungsi helper untuk memuat semua dokumen.
    
    Args:
        data_path: Path ke direktori data (opsional)
        
    Returns:
        List of LoadedDocument objects
    """
    loader = DocumentLoader(data_path)
    return loader.load_all_pdfs()


if __name__ == "__main__":
    # Test loading
    print("🧪 Testing Document Loader...")
    docs = load_documents()
    
    if docs:
        print(f"\n[STATS] Sample document:")
        print(f"   Source: {docs[0].source}")
        print(f"   Metadata: {docs[0].metadata}")
        print(f"   Content preview: {docs[0].content[:200]}...")
