"""
Legal Prompts: Template prompt untuk domain hukum Indonesia
Optimized untuk Llama-3 dan model Indonesia
"""
from typing import Optional
from dataclasses import dataclass


@dataclass
class PromptConfig:
    """Konfigurasi prompt"""
    max_context_length: int = 3000
    include_sources: bool = True
    language: str = "id"  # "id" atau "en"


class LegalPromptTemplate:
    """
    Template prompt untuk RAG domain hukum Indonesia.
    Mendukung berbagai format prompt untuk Llama-3.
    """
    
    # System prompts
    SYSTEM_PROMPT_ID = """Anda adalah Asisten Peneliti Hukum Senior (Senior Legal Research Assistant) yang ahli dalam membedah Putusan Mahkamah Agung Republik Indonesia.

TUGAS UTAMA:
Berikan analisis hukum yang mendalam, logis, dan saling terhubung berdasarkan konteks yang diberikan. Jangan hanya menyajikan fakta terputus, tetapi bangunlah sebuah argumen hukum (legal argument).

PROTOKOL BERPIKIR (CHAIN OF THOUGHT):
1. IDENTIFIKASI FAKTA HUKUM (The Facts): Temukan peristiwa atau dokumen kunci yang menjadi objek sengketa (misal: Surat Jual Beli, Surat Hibah, Sertifikat).
2. TEMUKAN CACAT HUKUM (The Defect): Analisis mengapa dokumen/tindakan tersebut dianggap salah oleh Hakim. Cari kata kunci: "tanpa hak", "melawan hukum", "cacat prosedur", "tanpa persetujuan ahli waris".
3. SUSUN RANTAI AKIBAT (The Consequence Chain): Hubungkan cacat tersebut dengan dampaknya terhadap transaksi selanjutnya.
   - Rumus: Karena [A] cacat, maka [B] menjadi tidak sah, sehingga [C] harus dibatalkan demi hukum.
4. SINTESIS JAWABAN: Gabungkan poin 1-3 menjadi narasi hukum yang padat dan meyakinkan.

ATURAN PENULISAN (STYLE GUIDE):
- Gunakan bahasa hukum Indonesia yang baku, formal, dan presisi.
- JANGAN BERHALUSINASI. Jika konteks tidak memuat alasannya, katakan "Pertimbangan hukum spesifik tidak ditemukan dalam potongan dokumen yang tersedia".
- Jika pertanyaan TIDAK BERKAITAN dengan hukum atau konteks dokumen, jawab dengan singkat: "Pertanyaan Anda di luar cakupan analisis dokumen hukum yang tersedia."
- Fokus pada "Ratio Decidendi" (Alasan utama hakim memutus).
- Pastikan menyebutkan dampak hukumnya (misal: "batal demi hukum", "tidak mempunyai kekuatan hukum mengikat").

ATURAN FORMAT (PENTING):
1. JANGAN PERNAH memulai jawaban dengan simbol poin seperti "I)", "1.", "-", atau huruf.
2. JANGAN melakukan copy-paste mentah dari poin-poin dokumen.
3. TUGAS ANDA ADALAH "PARAPHRASING": Baca poin-poin dalam teks, lalu tulis ulang menjadi satu paragraf cerita yang mengalir dan enak dibaca.
4. Pastikan Subjek-Predikat-Objek jelas. Jangan biarkan kalimat menggantung (misal: "yang merupakan..." -> ubah jadi "Surat tersebut tidak sah karena...").

FORMAT JAWABAN YANG DIHARAPKAN:
Berikan jawaban dalam satu paragraf komprehensif yang mencakup:
1. Pernyataan Tegas (Kesimpulan Hakim).
2. Alasan Utama (Penyebab ketidaksahan).
3. Konsekuensi Hukum (Dampak pada transaksi turunan).
"""

    SYSTEM_PROMPT_EN = """You are an AI legal assistant specializing in Indonesian law. Answer questions based on the provided legal documents.

Instructions:
1. Answer ONLY based on the given context
2. If information is not in the context, say so clearly
3. Always cite sources when relevant
4. Use proper legal terminology
5. Provide structured and concise answers"""

    # RAG Prompt Templates
    RAG_TEMPLATE_LLAMA3 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

Konteks dari dokumen hukum:
{context}

Pertanyaan: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    RAG_TEMPLATE_SIMPLE = """{system_prompt}

Konteks:
{context}

Pertanyaan: {question}

Jawaban:"""

    RAG_TEMPLATE_CHATML = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Konteks dari dokumen hukum:
{context}

Pertanyaan: {question}<|im_end|>
<|im_start|>assistant
"""

    # Chat templates (tanpa RAG)
    CHAT_TEMPLATE_LLAMA3 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    CHAT_TEMPLATE_SIMPLE = """{system_prompt}

User: {question}
Assistant:"""

    def __init__(
        self,
        template_style: str = "llama3",
        language: str = "id",
        config: Optional[PromptConfig] = None
    ):
        """
        Initialize prompt template.
        
        Args:
            template_style: "llama3", "chatml", atau "simple"
            language: "id" atau "en"
            config: Optional PromptConfig
        """
        self.template_style = template_style
        self.language = language
        self.config = config or PromptConfig()
        
        # Select templates based on style
        if template_style == "llama3":
            self.rag_template = self.RAG_TEMPLATE_LLAMA3
            self.chat_template = self.CHAT_TEMPLATE_LLAMA3
        elif template_style == "chatml":
            self.rag_template = self.RAG_TEMPLATE_CHATML
            self.chat_template = self.CHAT_TEMPLATE_SIMPLE
        else:
            self.rag_template = self.RAG_TEMPLATE_SIMPLE
            self.chat_template = self.CHAT_TEMPLATE_SIMPLE
        
        # Select system prompt based on language
        self.system_prompt = (
            self.SYSTEM_PROMPT_ID if language == "id" 
            else self.SYSTEM_PROMPT_EN
        )
    
    def format_rag_prompt(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Format RAG prompt dengan context.
        
        Args:
            question: User question
            context: Retrieved context
            system_prompt: Custom system prompt (optional)
            
        Returns:
            Formatted prompt
        """
        system = system_prompt or self.system_prompt
        
        # Truncate context if too long
        if len(context) > self.config.max_context_length:
            context = context[:self.config.max_context_length] + "\n[...konteks dipotong...]"
        
        return self.rag_template.format(
            system_prompt=system,
            context=context,
            question=question
        )
    
    def format_chat_prompt(
        self,
        question: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Format chat prompt tanpa RAG context.
        
        Args:
            question: User question
            system_prompt: Custom system prompt (optional)
            
        Returns:
            Formatted prompt
        """
        system = system_prompt or self.system_prompt
        
        return self.chat_template.format(
            system_prompt=system,
            question=question
        )
    
    def format_multi_turn_prompt(
        self,
        messages: list,
        context: Optional[str] = None
    ) -> str:
        """
        Format multi-turn conversation prompt.
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            context: Optional RAG context
            
        Returns:
            Formatted prompt
        """
        if self.template_style == "llama3":
            return self._format_llama3_multiturn(messages, context)
        elif self.template_style == "chatml":
            return self._format_chatml_multiturn(messages, context)
        else:
            return self._format_simple_multiturn(messages, context)
    
    def _format_llama3_multiturn(
        self,
        messages: list,
        context: Optional[str] = None
    ) -> str:
        """Format Llama-3 style multi-turn."""
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{self.system_prompt}<|eot_id|>"
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                # Add context to first user message if provided
                if context and messages.index(msg) == 0:
                    content = f"Konteks:\n{context}\n\nPertanyaan: {content}"
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            else:
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        
        # Add final assistant header
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return prompt
    
    def _format_chatml_multiturn(
        self,
        messages: list,
        context: Optional[str] = None
    ) -> str:
        """Format ChatML style multi-turn."""
        prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if context and role == "user" and messages.index(msg) == 0:
                content = f"Konteks:\n{context}\n\nPertanyaan: {content}"
            
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        prompt += "<|im_start|>assistant\n"
        
        return prompt
    
    def _format_simple_multiturn(
        self,
        messages: list,
        context: Optional[str] = None
    ) -> str:
        """Format simple multi-turn."""
        prompt = f"{self.system_prompt}\n\n"
        
        if context:
            prompt += f"Konteks:\n{context}\n\n"
        
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt += f"{role}: {msg['content']}\n"
        
        prompt += "Assistant:"
        
        return prompt
    
    @staticmethod
    def get_legal_system_prompts() -> dict:
        """
        Return berbagai system prompts untuk berbagai use case hukum.
        """
        return {
            "general": """Anda adalah asisten AI ahli hukum Indonesia. Jawab pertanyaan berdasarkan dokumen yang diberikan dengan akurat dan profesional.""",
            
            "putusan_analysis": """Anda adalah asisten AI yang menganalisis putusan pengadilan Indonesia. 
Fokus pada:
1. Identifikasi para pihak (Penggugat/Tergugat)
2. Pokok perkara dan tuntutan
3. Pertimbangan hukum hakim
4. Amar putusan
5. Dasar hukum yang digunakan""",
            
            "peraturan_lookup": """Anda adalah asisten AI yang membantu mencari dan menjelaskan peraturan perundang-undangan Indonesia.
Jelaskan dengan bahasa yang mudah dipahami sambil tetap akurat secara hukum.""",
            
            "contract_review": """Anda adalah asisten AI yang membantu review kontrak/perjanjian.
Fokus pada:
1. Identifikasi para pihak
2. Objek perjanjian
3. Hak dan kewajiban
4. Klausul penting (wanprestasi, force majeure, penyelesaian sengketa)
5. Potensi risiko hukum""",
            
            "summarization": """Anda adalah asisten AI yang meringkas dokumen hukum.
Buat ringkasan yang:
1. Mencakup poin-poin utama
2. Menyebutkan dasar hukum relevan
3. Mudah dipahami
4. Tidak lebih dari 3 paragraf"""
        }


def get_prompt_template(
    style: str = "llama3",
    language: str = "id"
) -> LegalPromptTemplate:
    """
    Factory function untuk mendapatkan prompt template.
    
    Args:
        style: Template style ("llama3", "chatml", "simple")
        language: Language ("id", "en")
        
    Returns:
        LegalPromptTemplate instance
    """
    return LegalPromptTemplate(template_style=style, language=language)


if __name__ == "__main__":
    # Test prompt templates
    print("🧪 Testing Legal Prompt Templates...")
    
    template = get_prompt_template(style="llama3", language="id")
    
    # Test RAG prompt
    question = "Apa putusan hakim dalam kasus ini?"
    context = """[Sumber 1: putusan_690_pdt.g_2024.pdf, Halaman 5]
Majelis Hakim memutuskan bahwa tergugat terbukti melakukan wanprestasi 
berdasarkan Pasal 1243 KUHPerdata.

[Sumber 2: putusan_690_pdt.g_2024.pdf, Halaman 10]
Tergugat dihukum untuk membayar ganti rugi sebesar Rp 500.000.000."""
    
    rag_prompt = template.format_rag_prompt(question, context)
    
    print("\n[OUTPUT] RAG Prompt (Llama-3 style):")
    print("-" * 50)
    print(rag_prompt)
    
    # Test multi-turn
    messages = [
        {"role": "user", "content": "Siapa penggugat dalam kasus ini?"},
        {"role": "assistant", "content": "Berdasarkan dokumen, penggugat adalah PT ABC."},
        {"role": "user", "content": "Apa tuntutannya?"}
    ]
    
    multi_prompt = template.format_multi_turn_prompt(messages, context)
    
    print("\n[OUTPUT] Multi-turn Prompt:")
    print("-" * 50)
    print(multi_prompt)
