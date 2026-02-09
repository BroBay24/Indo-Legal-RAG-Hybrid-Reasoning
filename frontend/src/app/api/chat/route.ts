import { NextRequest } from "next/server";
import { ChatRequest, ChatResponse } from "@/lib/types";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";

// Gunakan nodejs runtime (BUKAN edge) agar timeout bisa diperpanjang
export const runtime = "nodejs";
export const maxDuration = 300; // 5 menit timeout untuk CPU LLM inference

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { messages, pertanyaan } = body;

    // Support 2 format: dari chat-interface (pertanyaan langsung) atau dari messages array
    let userQuestion = pertanyaan;
    if (!userQuestion && messages) {
      const lastUserMessage = messages
        ?.filter((m: { role: string }) => m.role === "user")
        ?.pop();
      userQuestion = lastUserMessage?.content;
    }

    if (!userQuestion) {
      return new Response(
        JSON.stringify({ error: "No user message found" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    const chatRequest: ChatRequest = {
      pertanyaan: userQuestion,
      top_k: 5,
      max_tokens: 2048,
      temperature: 0.5,
      include_context: false,
    };

    // Call FastAPI backend dengan retry (jika backend masih loading)
    const MAX_RETRIES = 5;
    const RETRY_DELAY = 8000; // 8 detik antar retry (backend butuh ~40-60s loading)
    let backendResponse: Response | null = null;
    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      try {
        backendResponse = await fetch(`${BACKEND_URL}/chat`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(chatRequest),
        });

        break; // Success, keluar dari loop
      } catch (err: unknown) {
        lastError = err as Error;
        // Check error + cause untuk ECONNREFUSED (Node wraps it in cause)
        const errStr = String(err);
        const causeStr = (err as { cause?: unknown })?.cause ? String((err as { cause?: unknown }).cause) : "";
        const isConnectionError = 
          errStr.includes("ECONNREFUSED") || 
          errStr.includes("fetch failed") ||
          causeStr.includes("ECONNREFUSED");
        
        if (isConnectionError && attempt < MAX_RETRIES) {
          console.log(`Backend belum siap (attempt ${attempt}/${MAX_RETRIES}), retry dalam ${RETRY_DELAY/1000}s...`);
          await new Promise(r => setTimeout(r, RETRY_DELAY));
          continue;
        }
        throw err;
      }
    }

    if (!backendResponse) {
      throw lastError || new Error("Backend tidak merespon setelah retry");
    }

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text();
      console.error("Backend error:", errorText);
      return new Response(
        JSON.stringify({ error: "Backend request failed", details: errorText }),
        { status: backendResponse.status, headers: { "Content-Type": "application/json" } }
      );
    }

    const data: ChatResponse = await backendResponse.json();

    // Return JSON response langsung
    return new Response(
      JSON.stringify(data),
      {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }
    );
  } catch (error) {
    console.error("API route error:", error);

    // Handle timeout / abort
    if (error instanceof Error && error.name === "AbortError") {
      return new Response(
        JSON.stringify({ error: "Request timeout", details: "LLM membutuhkan waktu terlalu lama. Coba pertanyaan yang lebih singkat." }),
        { status: 504, headers: { "Content-Type": "application/json" } }
      );
    }

    // Handle backend belum siap
    const errStr = String(error);
    if (errStr.includes("ECONNREFUSED") || errStr.includes("fetch failed")) {
      return new Response(
        JSON.stringify({ error: "Backend sedang loading", details: "Server sedang memuat model AI. Tunggu ~30 detik lalu coba lagi." }),
        { status: 503, headers: { "Content-Type": "application/json" } }
      );
    }

    return new Response(
      JSON.stringify({ error: "Internal server error", details: String(error) }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}
