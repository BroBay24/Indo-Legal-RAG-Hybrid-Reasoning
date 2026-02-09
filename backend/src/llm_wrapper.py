"""
LLM Wrapper: Pembungkus untuk LLM (lokal GGUF dan Hugging Face API)
Mendukung Llama-3 fine-tuned dan model cloud
"""
from typing import Optional, Dict, Any, List, Generator
from abc import ABC, abstractmethod
import logging

from config import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract base class untuk LLM wrapper."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """Generate response dari prompt."""
        pass
    
    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream generate response."""
        pass


class LocalLLM(BaseLLM):
    """
    Wrapper untuk LLM lokal menggunakan llama-cpp-python.
    Mendukung file GGUF (Llama, Mistral, dll).
    """
    
    def __init__(
        self,
        model_path: str = None,
        n_ctx: int = None,
        n_gpu_layers: int = None,
        n_threads: int = None,
        verbose: bool = False
    ):
        self.model_path = model_path or settings.LLM_MODEL_PATH
        self.n_ctx = n_ctx or settings.LLM_CONTEXT_LENGTH
        self.n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else settings.LLM_GPU_LAYERS
        self.n_threads = n_threads or getattr(settings, 'LLM_N_THREADS', 8)
        self.verbose = verbose
        
        self.llm = None
        self._load_model()
    
    def _load_model(self):
        """Load the GGUF model."""
        logger.info(f"[LLM] Memuat model lokal: {self.model_path}")
        
        try:
            from llama_cpp import Llama
            
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                verbose=self.verbose
            )
            
            logger.info(f"   [OK] Model berhasil dimuat (threads={self.n_threads}, ctx={self.n_ctx})")
            
        except FileNotFoundError:
            logger.error(f"   [ERROR] Model tidak ditemukan: {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"   [ERROR] Error memuat model: {str(e)}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        stop: List[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response dari prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            
        Returns:
            Generated text
        """
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        temperature = temperature or settings.LLM_TEMPERATURE
        top_p = top_p or settings.LLM_TOP_P
        # Stop sequences yang lebih spesifik untuk Llama-3
        stop = stop or ["<|eot_id|>", "<|end_of_text|>"]
        
        logger.info(f"[LLM] Generating response (max_tokens={max_tokens}, temp={temperature})")
        logger.debug(f"   Prompt preview: {prompt[:200]}...")
        
        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                echo=False,
                **kwargs
            )
            
            response = output['choices'][0]['text'].strip()
            logger.info(f"   [OK] Generated {len(response)} chars")
            
            # Log response preview untuk debugging
            if response:
                logger.debug(f"   Response preview: {response[:150]}...")
            else:
                logger.warning("[WARNING] LLM returned empty response")
                logger.debug(f"   Full output: {output}")
            
            return response
        except Exception as e:
            logger.error(f"[ERROR] LLM generation failed: {str(e)}")
            raise
    
    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        stop: List[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream generate response.
        
        Yields:
            Token by token response
        """
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        temperature = temperature or settings.LLM_TEMPERATURE
        top_p = top_p or settings.LLM_TOP_P
        stop = stop or ["User:", "Human:", "\n\n\n"]
        
        logger.debug(f"[LLM] Streaming response (max_tokens={max_tokens})")
        
        stream = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            echo=False,
            stream=True,
            **kwargs
        )
        
        for output in stream:
            token = output['choices'][0]['text']
            yield token
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            "type": "local",
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers
        }


class HuggingFaceLLM(BaseLLM):
    """
    Wrapper untuk Hugging Face Inference API.
    Untuk model cloud seperti Llama-3, Mistral, dll.
    """
    
    def __init__(
        self,
        model_id: str = None,
        api_token: str = None,
        max_retries: int = 3
    ):
        self.model_id = model_id or settings.HF_MODEL_ID
        self.api_token = api_token or settings.HF_API_TOKEN
        self.max_retries = max_retries
        
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize HuggingFace client."""
        if not self.api_token:
            logger.warning("[WARNING] HF_API_TOKEN tidak diset, HuggingFace LLM mungkin tidak berfungsi")
            return
        
        try:
            from huggingface_hub import InferenceClient
            
            self.client = InferenceClient(
                model=self.model_id,
                token=self.api_token
            )
            
            logger.info(f"[OK] HuggingFace client initialized: {self.model_id}")
            
        except Exception as e:
            logger.error(f"[ERROR] Error initializing HuggingFace client: {str(e)}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        **kwargs
    ) -> str:
        """Generate using HuggingFace Inference API."""
        if not self.client:
            raise RuntimeError("HuggingFace client not initialized")
        
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        temperature = temperature or settings.LLM_TEMPERATURE
        top_p = top_p or settings.LLM_TOP_P
        
        logger.debug(f"🌐 Calling HuggingFace API (max_tokens={max_tokens})")
        
        response = self.client.text_generation(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        return response.strip()
    
    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream generate using HuggingFace."""
        if not self.client:
            raise RuntimeError("HuggingFace client not initialized")
        
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        temperature = temperature or settings.LLM_TEMPERATURE
        top_p = top_p or settings.LLM_TOP_P
        
        logger.debug(f"🌐 Streaming from HuggingFace API")
        
        for token in self.client.text_generation(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            **kwargs
        ):
            yield token
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            "type": "huggingface",
            "model_id": self.model_id,
            "api_available": self.client is not None
        }


class LLMWrapper:
    """
    Unified wrapper yang mendukung baik local maupun cloud LLM.
    Automatically switches based on configuration.
    """
    
    def __init__(
        self,
        use_local: bool = True,
        local_model_path: str = None,
        hf_model_id: str = None,
        hf_api_token: str = None,
        **kwargs
    ):
        self.use_local = use_local
        self.llm: Optional[BaseLLM] = None
        
        if use_local:
            self.llm = LocalLLM(
                model_path=local_model_path,
                **kwargs
            )
        else:
            self.llm = HuggingFaceLLM(
                model_id=hf_model_id,
                api_token=hf_api_token
            )
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """Generate response."""
        return self.llm.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream generate response."""
        yield from self.llm.stream_generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def generate_with_context(
        self,
        question: str,
        context: str,
        system_prompt: str = None,
        **kwargs
    ) -> str:
        """
        Generate response dengan context (untuk RAG).
        
        Args:
            question: User question
            context: Retrieved context
            system_prompt: System prompt (optional)
            
        Returns:
            Generated answer
        """
        from src.legal_prompts import LegalPromptTemplate
        
        prompt_template = LegalPromptTemplate()
        full_prompt = prompt_template.format_rag_prompt(
            question=question,
            context=context,
            system_prompt=system_prompt
        )
        
        return self.generate(full_prompt, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""
        return self.llm.get_model_info()


def get_llm(use_local: bool = True, **kwargs) -> LLMWrapper:
    """
    Factory function untuk mendapatkan LLM wrapper.
    
    Args:
        use_local: Use local model or cloud
        **kwargs: Additional arguments
        
    Returns:
        LLMWrapper instance
    """
    return LLMWrapper(use_local=use_local, **kwargs)


if __name__ == "__main__":
    # Test LLM Wrapper
    print("🧪 Testing LLM Wrapper...")
    
    try:
        # Test local LLM
        llm = get_llm(use_local=True)
        print(f"\n[STATS] Model info: {llm.get_model_info()}")
        
        # Test simple generation
        prompt = "Jelaskan secara singkat apa itu hukum perdata:"
        print(f"\n[SEARCH] Prompt: {prompt}")
        
        response = llm.generate(prompt, max_tokens=100)
        print(f"\n[OUTPUT] Response: {response}")
        
    except FileNotFoundError:
        print("[WARNING] Model file tidak ditemukan, skip test lokal")
    except Exception as e:
        print(f"[ERROR] Error: {str(e)}")
