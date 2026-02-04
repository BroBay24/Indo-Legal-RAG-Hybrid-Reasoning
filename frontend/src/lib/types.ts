// Types untuk komunikasi dengan FastAPI Backend

export interface ChatRequest {
  pertanyaan: string;
  top_k?: number;
  max_tokens?: number;
  temperature?: number;
  include_context?: boolean;
}

export interface Source {
  source: string;
  page: number;
  doc_type: string;
  score: number;
  retrieval_source: string;
  relevance_score?: number;
  content?: string;
}

export interface ChatResponse {
  jawaban: string;
  sumber: Source[];
  konteks: string | null;
  pertanyaan: string;
  debug_info?: {
    rerank_scores?: Array<{
      rank: number;
      score: number;
    }>;
  };
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  createdAt: Date;
}

// Streaming response types
export interface StreamChunk {
  type: "text" | "sources" | "done" | "error";
  content?: string;
  sources?: Source[];
  error?: string;
}
