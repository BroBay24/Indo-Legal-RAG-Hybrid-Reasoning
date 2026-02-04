import { NextRequest } from "next/server";
import { ChatRequest, ChatResponse } from "@/lib/types";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export const runtime = "edge";
export const maxDuration = 60; // 60 seconds timeout

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { messages } = body;

    // Ambil pesan terakhir dari user
    const lastUserMessage = messages
      ?.filter((m: { role: string }) => m.role === "user")
      ?.pop();

    if (!lastUserMessage) {
      return new Response(
        JSON.stringify({ error: "No user message found" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    const chatRequest: ChatRequest = {
      pertanyaan: lastUserMessage.content,
      top_k: 5,
      max_tokens: 800,
      temperature: 0.7,
      include_context: false,
    };

    // Call FastAPI backend
    const backendResponse = await fetch(`${BACKEND_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(chatRequest),
    });

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text();
      console.error("Backend error:", errorText);
      return new Response(
        JSON.stringify({ error: "Backend request failed", details: errorText }),
        { status: backendResponse.status, headers: { "Content-Type": "application/json" } }
      );
    }

    const data: ChatResponse = await backendResponse.json();

    // Format response untuk assistant-ui
    // Menggunakan format yang kompatibel dengan assistant-ui
    const encoder = new TextEncoder();
    
    // Build response dengan sources
    let responseText = data.jawaban;
    
    // Append sources info if available
    if (data.sumber && data.sumber.length > 0) {
      responseText += "\n\n---\n**Sumber Referensi:**\n";
      data.sumber.forEach((source, idx) => {
        const fileName = source.source.split("/").pop() || source.source;
        responseText += `${idx + 1}. ${fileName} (Hal. ${source.page}, Skor: ${(source.score * 100).toFixed(1)}%)\n`;
      });
    }

    // Stream response dalam format text/event-stream
    const stream = new ReadableStream({
      start(controller) {
        // Send content as data chunks
        const lines = responseText.split("");
        let currentIndex = 0;
        
        function sendNextChunk() {
          if (currentIndex < lines.length) {
            // Send character by character for streaming effect
            const chunk = lines[currentIndex];
            const sseMessage = `0:${JSON.stringify(chunk)}\n`;
            controller.enqueue(encoder.encode(sseMessage));
            currentIndex++;
            // Use setTimeout to create streaming effect
            setTimeout(sendNextChunk, 10);
          } else {
            // Send finish message
            const finishMessage = `d:{"finishReason":"stop"}\n`;
            controller.enqueue(encoder.encode(finishMessage));
            controller.close();
          }
        }
        
        sendNextChunk();
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      },
    });
  } catch (error) {
    console.error("API route error:", error);
    return new Response(
      JSON.stringify({ error: "Internal server error", details: String(error) }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}
