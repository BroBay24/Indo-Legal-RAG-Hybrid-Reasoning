"""
Legal Preprocessor: Normalisasi dan pembersihan teks dokumen hukum Indonesia
"""
import re
import unicodedata
from typing import List, Optional
import logging

from config import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


class LegalPreprocessor:
    """
    Preprocessor khusus untuk dokumen hukum Indonesia.
    Menangani normalisasi pasal, ayat, huruf, dan format khas dokumen hukum.
    """
    
    # Pattern untuk mendeteksi elemen dokumen hukum
    PASAL_PATTERN = re.compile(r'pasal\s*(\d+)', re.IGNORECASE)
    AYAT_PATTERN = re.compile(r'ayat\s*\(?(\d+)\)?', re.IGNORECASE)
    HURUF_PATTERN = re.compile(r'huruf\s*\(?([a-z])\)?', re.IGNORECASE)
    UU_PATTERN = re.compile(
        r'(?:undang[- ]?undang|uu)\s*(?:nomor|no\.?)\s*(\d+)\s*(?:tahun|th\.?)\s*(\d{4})',
        re.IGNORECASE
    )
    PP_PATTERN = re.compile(
        r'(?:peraturan\s+pemerintah|pp)\s*(?:nomor|no\.?)\s*(\d+)\s*(?:tahun|th\.?)\s*(\d{4})',
        re.IGNORECASE
    )
    
    def __init__(
        self,
        normalize_unicode: bool = True,
        remove_extra_whitespace: bool = True,
        normalize_pasal: bool = True,
        lowercase: bool = False
    ):
        self.normalize_unicode = normalize_unicode
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_pasal = normalize_pasal
        self.lowercase = lowercase
    
    def preprocess(self, text: str) -> str:
        """
        Menjalankan semua preprocessing pada teks.
        
        Args:
            text: Teks input
            
        Returns:
            Teks yang sudah di-preprocess
        """
        if not text:
            return ""
        
        # 1. Normalisasi Unicode
        if self.normalize_unicode:
            text = self._normalize_unicode(text)
        
        # 2. Bersihkan karakter khusus
        text = self._clean_special_chars(text)
        
        # 3. Normalisasi format pasal/ayat
        if self.normalize_pasal:
            text = self._normalize_legal_references(text)
        
        # 4. Hapus whitespace berlebih
        if self.remove_extra_whitespace:
            text = self._remove_extra_whitespace(text)
        
        # 5. Lowercase (opsional)
        if self.lowercase:
            text = text.lower()
        
        return text.strip()
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalisasi Unicode ke bentuk standar (NFC)."""
        return unicodedata.normalize('NFC', text)
    
    def _clean_special_chars(self, text: str) -> str:
        """Bersihkan karakter khusus yang tidak diperlukan."""
        # Hapus karakter kontrol (kecuali newline dan tab)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Ganti multiple dashes dengan single dash
        text = re.sub(r'-{2,}', ' - ', text)
        
        # Ganti bullet points dengan dash
        text = re.sub(r'[•●○◦▪▫]', '-', text)
        
        # Normalisasi quotes
        text = re.sub(r'[""„‟]', '"', text)
        text = re.sub(r'[''‚‛]', "'", text)
        
        return text
    
    def _normalize_legal_references(self, text: str) -> str:
        """Normalisasi referensi hukum (pasal, ayat, UU, dll)."""
        
        # Normalisasi format Pasal
        text = re.sub(
            r'(?i)pasal\s+(\d+)',
            r'Pasal \1',
            text
        )
        
        # Normalisasi format Ayat
        text = re.sub(
            r'(?i)ayat\s*\(?\s*(\d+)\s*\)?',
            r'ayat (\1)',
            text
        )
        
        # Normalisasi format huruf
        text = re.sub(
            r'(?i)huruf\s*\(?\s*([a-z])\s*\)?',
            lambda m: f'huruf ({m.group(1).lower()})',
            text
        )
        
        # Normalisasi UU
        text = re.sub(
            r'(?i)(?:undang[- ]?undang|uu)\s*(?:nomor|no\.?)\s*(\d+)\s*(?:tahun|th\.?)\s*(\d{4})',
            r'UU No. \1 Tahun \2',
            text
        )
        
        # Normalisasi PP
        text = re.sub(
            r'(?i)(?:peraturan\s+pemerintah|pp)\s*(?:nomor|no\.?)\s*(\d+)\s*(?:tahun|th\.?)\s*(\d{4})',
            r'PP No. \1 Tahun \2',
            text
        )
        
        # Normalisasi Perpres
        text = re.sub(
            r'(?i)(?:peraturan\s+presiden|perpres)\s*(?:nomor|no\.?)\s*(\d+)\s*(?:tahun|th\.?)\s*(\d{4})',
            r'Perpres No. \1 Tahun \2',
            text
        )
        
        # Normalisasi Permen
        text = re.sub(
            r'(?i)(?:peraturan\s+menteri|permen)\s*(\w+)\s*(?:nomor|no\.?)\s*(\d+)\s*(?:tahun|th\.?)\s*(\d{4})',
            r'Permen\1 No. \2 Tahun \3',
            text
        )
        
        return text
    
    def _remove_extra_whitespace(self, text: str) -> str:
        """Hapus whitespace berlebih."""
        # Ganti multiple spaces dengan single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Ganti multiple newlines dengan double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Hapus spasi di awal/akhir baris
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text
    
    def extract_legal_entities(self, text: str) -> dict:
        """
        Ekstrak entitas hukum dari teks.
        
        Returns:
            Dictionary berisi list pasal, ayat, UU yang ditemukan
        """
        entities = {
            "pasal": [],
            "ayat": [],
            "uu": [],
            "pp": [],
        }
        
        # Ekstrak Pasal
        for match in self.PASAL_PATTERN.finditer(text):
            pasal_num = match.group(1)
            if pasal_num not in entities["pasal"]:
                entities["pasal"].append(pasal_num)
        
        # Ekstrak Ayat
        for match in self.AYAT_PATTERN.finditer(text):
            ayat_num = match.group(1)
            if ayat_num not in entities["ayat"]:
                entities["ayat"].append(ayat_num)
        
        # Ekstrak UU
        for match in self.UU_PATTERN.finditer(text):
            uu_ref = f"UU No. {match.group(1)} Tahun {match.group(2)}"
            if uu_ref not in entities["uu"]:
                entities["uu"].append(uu_ref)
        
        # Ekstrak PP
        for match in self.PP_PATTERN.finditer(text):
            pp_ref = f"PP No. {match.group(1)} Tahun {match.group(2)}"
            if pp_ref not in entities["pp"]:
                entities["pp"].append(pp_ref)
        
        return entities
    
    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """Preprocess batch of texts."""
        return [self.preprocess(text) for text in texts]


def preprocess_text(
    text: str,
    normalize_unicode: bool = True,
    remove_extra_whitespace: bool = True,
    normalize_pasal: bool = True
) -> str:
    """
    Fungsi helper untuk preprocessing teks.
    
    Args:
        text: Teks input
        normalize_unicode: Apakah normalisasi unicode
        remove_extra_whitespace: Apakah hapus whitespace berlebih
        normalize_pasal: Apakah normalisasi format pasal
        
    Returns:
        Teks yang sudah di-preprocess
    """
    preprocessor = LegalPreprocessor(
        normalize_unicode=normalize_unicode,
        remove_extra_whitespace=remove_extra_whitespace,
        normalize_pasal=normalize_pasal
    )
    return preprocessor.preprocess(text)


if __name__ == "__main__":
    # Test preprocessing
    sample_text = """
    Menimbang bahwa berdasarkan   pasal 1234 ayat (1) huruf a 
    undang-undang nomor 40 tahun 2007 tentang Perseroan Terbatas,
    
    
    tergugat telah melanggar ketentuan PP No 123 tahun 2020.
    
    Berdasarkan Pasal  28  UU  no.  5  th.  1999...
    """
    
    print("🧪 Testing Legal Preprocessor...")
    print("\n[OUTPUT] Original text:")
    print(sample_text)
    
    preprocessor = LegalPreprocessor()
    cleaned = preprocessor.preprocess(sample_text)
    
    print("\n✨ Preprocessed text:")
    print(cleaned)
    
    print("\n[STATS] Extracted entities:")
    entities = preprocessor.extract_legal_entities(cleaned)
    for key, values in entities.items():
        if values:
            print(f"   {key}: {values}")
